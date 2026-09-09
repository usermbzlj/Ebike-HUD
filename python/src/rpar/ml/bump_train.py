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
)
from rpar.ml.train import write_run_card
from rpar.ml.world_bump import TEACHER_NAME, default_teacher_path, repo_root


def ultralytics_available() -> bool:
    return find_spec("ultralytics") is not None


def ensure_teacher_weights(dest: Path | None = None) -> dict[str, Any]:
    """Download YOLO-World-M into models/yolo-world/. No-op if the file already exists."""
    dest = Path(dest) if dest else default_teacher_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return {"ok": True, "path": str(dest), "downloaded": False}
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
        return {"ok": True, "path": str(dest), "downloaded": True}
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
        "YOLOPv2 remains drivable-area + vehicles only. This package is **not** a PKC110 "
        "LiteRT bump net; the phone keeps the heuristic until a quantized student is exported.\n\n"
        f"- teacher: `{TEACHER_NAME}`\n"
        f"- dataset: `{kwargs.get('dataset_dir', '')}`\n"
        f"- n_clips: {kwargs.get('n_clips', 0)}\n"
        f"- n_train_images: {kwargs.get('n_train_images', 0)}\n"
        f"- trained: {kwargs.get('trained', False)}\n"
        f"- weights: `{kwargs.get('weights', '')}`\n\n"
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
    weights = model or str(default_teacher_path())
    if not Path(weights).is_file():
        weights = "yolov8m.pt"
    net = YOLO(weights)
    try:
        net.train(
            data=str(data_yaml),
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=8,
            device=0,
            project=str(out_dir),
            name="train",
            exist_ok=True,
            patience=15,
            workers=2,
            pretrained=True,
            verbose=True,
        )
    except Exception as exc:
        return {"ok": False, "reason": "train_failed", "error": str(exc)[:400]}
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
) -> dict[str, Any]:
    inbox = Path(inbox) if inbox else default_inbox()
    dataset_dir = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    out_dir = Path(out_dir) if out_dir else repo_root() / "artifacts" / "bump_train"
    ingested = ingest_inbox(inbox, dataset_dir, sample_fps=sample_fps)
    teacher = {"ok": False, "reason": "skipped"}
    if not skip_download:
        teacher = ensure_teacher_weights()
    proposed = propose_labels(dataset_dir)
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
