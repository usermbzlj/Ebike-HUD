"""Train a bump detector on inbox phone clips. Lazy-loads ultralytics / torch."""

from __future__ import annotations

import json
import shutil
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from rpar import SCHEMA_VERSION
from rpar.ml.bump_dataset import (
    default_dataset_dir,
    default_inbox,
    ingest_inbox,
    propose_labels,
    reuse_raw_labels,
)
from rpar.ml.train import write_run_card
from rpar.ml.bump_prompts import WORLD_INFER_PROMPTS, YOLO_NAMES
from rpar.ml.world_bump import TEACHER_NAME, default_teacher_path, default_vocab_path, repo_root, resolve_bump_weights


def ultralytics_available() -> bool:
    return find_spec("ultralytics") is not None


def ensure_teacher_weights(dest: Path | None = None) -> dict[str, Any]:
    """Download YOLO-World-M into models/yolo-world/. No-op if the file already exists."""
    dest = Path(dest) if dest else default_teacher_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        from rpar.ml.world_bump import ensure_bump_vocab

        vocab = ensure_bump_vocab()
        return {"ok": True, "path": str(dest), "downloaded": False, "vocab": vocab}
    if not ultralytics_available():
        return {"ok": False, "reason": "ultralytics_missing", "path": str(dest)}
    from ultralytics import YOLO

    model = YOLO(TEACHER_NAME)
    src = Path(getattr(model, "ckpt_path", "") or getattr(model, "pt_path", "") or "")
    if not src.is_file():
        ckpt = getattr(model, "ckpt", None)
        if isinstance(ckpt, dict) and ckpt.get("path"):
            src = Path(str(ckpt["path"]))
    if not src.is_file():
        # Ultralytics caches as yolov8m-worldv2.pt in cwd or USER/.cache
        for cand in (Path.cwd() / TEACHER_NAME, Path.home() / TEACHER_NAME, dest):
            if cand.is_file() and cand.stat().st_size > 1_000_000:
                src = cand
                break
    if src.is_file() and src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        from rpar.ml.world_bump import ensure_bump_vocab

        vocab = ensure_bump_vocab()
        return {"ok": True, "path": str(dest), "downloaded": True, "vocab": vocab}
    return {"ok": False, "reason": "download_failed", "path": str(dest)}


def write_bump_card(out_dir: Path, **kwargs: Any) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "# bump-world-0.1.0\n\n"
        "Desktop **YOLO-World** open-vocabulary detector prompted for "
        "`pothole` / `speed bump` / `sunken manhole cover`, then optionally fine-tuned "
        "on phone clips dropped in `Video/train/inbox/`.\n\n"
        "Product enums stay spec (`pothole`, `speed_bump`, `manhole_cover`). "
        "Public RDD/Rome class tables are **not** concatenated into this list.\n\n"
        "YOLOPv2 remains drivable-area + vehicles only. Phone sidecar is "
        "`rpar export-bump-tflite` → `model.onnx` on Windows (LiteRT export is Linux/macOS) "
        "or `model.tflite` when Ultralytics allows it (gitignored). Heuristic pits are "
        "replaced only when that graph loads; a missing/broken Interpreter keeps the HUD.\n\n"
        f"- teacher: `{TEACHER_NAME}`\n"
        f"- dataset: `{kwargs.get('dataset_dir', '')}`\n"
        f"- n_clips: {kwargs.get('n_clips', 0)}\n"
        f"- n_train_images: {kwargs.get('n_train_images', 0)}\n"
        f"- trained: {kwargs.get('trained', False)}\n"
        f"- weights: `{kwargs.get('weights', '')}`\n\n"
        "Phone ONNX default `imgsz=640` matches desktop YOLO-World infer. "
        "Not Camera2 1080p60 GT. Fine-tune quality tracks how many real 大坑 / 下沉井盖 / 减速带 clips you deliver.\n"
    )
    dest = out_dir / "MODEL_CARD.md"
    dest.write_text(body, encoding="utf-8")
    return dest


def train_yolo(data_yaml: Path, out_dir: Path, *, epochs: int = 60, imgsz: int = 640, model: str | None = None) -> dict[str, Any]:
    if not ultralytics_available():
        return {"ok": False, "reason": "ultralytics_missing"}
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = model or "yolov8m.pt"
    if model and Path(model).is_file():
        weights = str(model)
    net = YOLO(weights)
    batch = 16
    last_err = None
    for batch in (16, 8, 4):
        try:
            net.train(
                data=str(data_yaml),
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=batch,
                device=0,
                project=str(out_dir),
                name="train",
                exist_ok=True,
                patience=12,
                workers=2,
                pretrained=True,
                verbose=True,
                amp=True,
            )
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                return {"ok": False, "reason": "train_failed", "error": str(exc)[:400]}
    if last_err is not None:
        return {"ok": False, "reason": "train_failed", "error": str(last_err)[:400]}
    best = out_dir / "train" / "weights" / "best.pt"
    pkg = repo_root() / "models" / "bump-world-0.1.0"
    pkg.mkdir(parents=True, exist_ok=True)
    if best.is_file():
        shutil.copy2(best, pkg / "best.pt")
    return {
        "ok": best.is_file(),
        "best": str(best) if best.is_file() else "",
        "package_weights": str(pkg / "best.pt") if (pkg / "best.pt").is_file() else "",
    }


def run_bump_train(
    inbox: Path | None = None,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    sample_fps: float = 2.0,
    epochs: int = 60,
    min_images: int = 16,
    skip_download: bool = False,
    max_frames_per_clip: int = 1600,
    skip_propose: bool = False,
) -> dict[str, Any]:
    inbox = Path(inbox) if inbox else default_inbox()
    dataset_dir = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    out_dir = Path(out_dir) if out_dir else repo_root() / "artifacts" / "bump_train"
    ingested = ingest_inbox(inbox, dataset_dir, sample_fps=sample_fps, max_frames_per_clip=max_frames_per_clip)
    teacher = {"ok": False, "reason": "skipped"}
    if not skip_download:
        teacher = ensure_teacher_weights()
    proposed = reuse_raw_labels(dataset_dir) if skip_propose else propose_labels(dataset_dir)
    n_train = int(proposed.get("n_train_images") or 0)
    trained: dict[str, Any] = {"ok": False, "reason": "not_attempted"}
    if proposed.get("ok") and n_train >= min_images:
        trained = train_yolo(Path(proposed["data_yaml"]), out_dir, epochs=epochs)
    elif proposed.get("ok"):
        trained = {
            "ok": False,
            "reason": "not_enough_images",
            "n_train_images": n_train,
            "min_images": min_images,
            "hint": "Need more phone clips in Video/train/inbox (target ≥20 clips covering 大坑 / 下沉井盖 / 减速带, day and night).",
        }
    pkg = repo_root() / "models" / "bump-world-0.1.0"
    card = write_bump_card(
        pkg,
        dataset_dir=str(dataset_dir),
        n_clips=ingested.get("n_clips"),
        n_train_images=n_train,
        trained=bool(trained.get("ok")),
        weights=trained.get("package_weights") or teacher.get("path") or "",
    )
    write_run_card(
        out_dir / "run_card.json",
        data_version="rideset-inbox",
        seed=7,
        hyperparameters={"epochs": epochs, "imgsz": 640, "teacher": TEACHER_NAME},
        label_map={"0": "pothole", "1": "speed_bump", "2": "manhole_cover"},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(teacher.get("ok") or proposed.get("ok")),
        "inbox": str(inbox),
        "ingested": {k: ingested[k] for k in ("n_clips", "n_frames", "clips", "split") if k in ingested},
        "teacher": teacher,
        "proposed": {k: proposed.get(k) for k in ("ok", "reason", "n_positive_frames", "n_empty_frames", "n_train_images", "n_val_images", "data_yaml")},
        "trained": trained,
        "model_card": str(card),
        "note": "Zero-shot YOLO-World runs as soon as teacher weights exist; fine-tune waits for enough inbox frames.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _names_from_model(model: Any) -> list[str]:
    raw = getattr(model, "names", None) or {}
    if isinstance(raw, dict):
        return [str(raw.get(i, raw.get(str(i), ""))) for i in range(len(raw))]
    return [str(n) for n in raw]


def write_bump_lite_package(out_dir: Path, graph: Path, names: list[str], *, imgsz: int, nms: bool) -> dict[str, Any]:
    import hashlib

    out_dir = Path(out_dir)
    graph = Path(graph)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_name = "model.onnx" if graph.suffix.lower() == ".onnx" else "model.tflite"
    dest = out_dir / dest_name
    if graph.resolve() != dest.resolve():
        shutil.copy2(graph, dest)
    labels = {
        "names": {str(i): n for i, n in enumerate(names)},
        "yolo_names": list(YOLO_NAMES),
        "conf": 0.08,
        "imgsz": int(imgsz),
        "nms": bool(nms),
        "engine": "yolo-bump",
        "graph": dest_name,
        "layout": "NCHW",
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package_id": "bump-world-0.1.0",
        "engine": "yolo-bump",
        "role": "bump_sidecar",
        "files": {dest_name: dest_name, "labels.json": "labels.json"},
        "input_spec": {
            "detect": {"width": int(imgsz), "height": int(imgsz), "layout": "NCHW", "norm": "0-1"},
            "note": "Letterboxed RGB detect. YOLOPv2 / heuristic still own drivable area. "
            "Ultralytics LiteRT export is Linux/macOS only; this host packages ONNX for the phone.",
        },
        "quantization": "fp32",
        "compatible_app": ">=0.1.0",
        "labels": {"semantic_type": list(YOLO_NAMES)},
        "sha256": {},
    }
    for p in out_dir.iterdir():
        if p.is_file() and p.name != "manifest.json" and p.suffix.lower() not in {".tflite", ".onnx", ".pt"}:
            manifest["sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_bump_card(out_dir, weights=dest_name, trained=False, dataset_dir="", n_clips=0, n_train_images=0)
    return {
        "ok": dest.is_file() and dest.stat().st_size > 1_000_000,
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "imgsz": int(imgsz),
        "format": dest_name,
    }


def _export_ultralytics(model: Any, *, imgsz: int, nms: bool) -> tuple[Path | None, bool, str | None]:
    """LiteRT on Linux/macOS; YOLO-World v2 on Windows falls back to ONNX."""
    tflite_err = "tflite_skipped"
    if not __import__("sys").platform.startswith("win"):
        try:
            exported = model.export(format="tflite", imgsz=int(imgsz), nms=bool(nms), keras=False)
            path = Path(str(exported))
            if path.is_file():
                return path, nms, None
            tflite_err = "tflite_path_missing"
        except Exception as exc:
            tflite_err = str(exc)[:400]
    try:
        exported = model.export(
            format="onnx",
            imgsz=int(imgsz),
            simplify=True,
            dynamic=False,
            nms=bool(nms),
        )
        path = Path(str(exported))
        if path.is_file():
            return path, bool(nms), tflite_err
    except Exception:
        try:
            exported = model.export(format="onnx", imgsz=int(imgsz), simplify=True, dynamic=False)
            path = Path(str(exported))
            if path.is_file():
                return path, False, tflite_err
        except Exception as exc:
            return None, False, f"tflite={tflite_err}; onnx={str(exc)[:400]}"
    return None, False, tflite_err


def export_bump_tflite(
    *,
    imgsz: int = 640,
    nms: bool = True,
    weights: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Export a phone detect graph. LiteRT when Ultralytics allows it; ONNX on Windows."""
    if not ultralytics_available():
        return {"ok": False, "reason": "ultralytics_missing"}
    from ultralytics import YOLO

    from rpar.ml.world_bump import ensure_bump_vocab

    vocab = ensure_bump_vocab()
    src = resolve_bump_weights(weights) or default_vocab_path() or default_teacher_path()
    if src is None or not Path(src).is_file():
        return {"ok": False, "reason": "weights_missing", "vocab": vocab}
    out_dir = Path(out_dir) if out_dir else repo_root() / "models" / "bump-world-0.1.0"
    try:
        model = YOLO(str(src))
    except Exception as exc:
        return {"ok": False, "reason": "load_failed", "error": str(exc)[:400], "src": str(src)}
    names = _names_from_model(model)
    if not any(n.strip() for n in names):
        names = list(WORLD_INFER_PROMPTS)
    exported_path, used_nms, warn = _export_ultralytics(model, imgsz=imgsz, nms=nms)
    if exported_path is None or not exported_path.is_file():
        return {"ok": False, "reason": "export_failed", "error": warn or "no_graph", "src": str(src)}
    packed = write_bump_lite_package(out_dir, exported_path, names, imgsz=imgsz, nms=used_nms)
    packed.update(
        {
            "src": str(src),
            "exported": str(exported_path),
            "names": names,
            "nms": used_nms,
            "vocab": vocab,
            "tflite_note": warn,
        }
    )
    return packed
