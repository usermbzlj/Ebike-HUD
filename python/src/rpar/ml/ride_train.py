"""End-to-end ride training: roadseg distill (main) + VLM/YOLO-World labels + bump student."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import cv2

from rpar import SCHEMA_VERSION
from rpar.ml.bump_dataset import (
    _thin_empty_labels,
    default_dataset_dir,
    default_inbox,
    ingest_inbox,
    materialize_yolo_splits,
    propose_labels,
    reuse_raw_labels,
    write_raw_label,
)
from rpar.ml.bump_train import ensure_teacher_weights, export_bump_tflite, train_yolo, write_bump_card
from rpar.ml.field_distill import distill_from_videos
from rpar.ml.ride_label import LAST_LOAD_ERROR, merge_dets, try_load_qwen, try_load_vlm_boxer, verify_dets
from rpar.ml.world_bump import repo_root, try_load_world_bump
from rpar.ml.yolopv2 import fetch_weights as fetch_yolop, weights_available as yolop_available


def _log(msg: str) -> None:
    print(f"[ride-train] {msg}", flush=True)


def _android_roadseg() -> Path:
    return repo_root() / "android" / "app" / "src" / "main" / "assets" / "models" / "roadseg-field-0.1.0"


def copy_roadseg_to_android(pkg: Path) -> None:
    dest = _android_roadseg()
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("seg_weights.json", "labels.json", "manifest.json", "MODEL_CARD.md"):
        src = pkg / name
        if src.is_file():
            shutil.copy2(src, dest / name)


def collect_train_videos() -> list[Path]:
    root = repo_root() / "Video"
    clips: list[Path] = []
    if root.is_dir():
        clips.extend(sorted(root.glob("*.mp4")))
        inbox = root / "train" / "inbox"
        if inbox.is_dir():
            clips.extend(sorted(inbox.glob("*.mp4")))
    return [p for p in clips if p.is_file() and p.stat().st_size > 10_000]


def frames_per_clip_for(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30) or 30.0
    cap.release()
    dur = n / fps
    return int(max(16, min(120, round(dur / 8.0))))


def distill_main_roadseg(out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir) if out_dir else repo_root() / "models" / "roadseg-field-0.1.0"
    yolop = fetch_yolop()
    if not yolop.get("ok") and not yolop_available():
        return {"ok": False, "reason": "yolop_missing", "fetch": yolop}
    clips = collect_train_videos()
    if not clips:
        return {"ok": False, "reason": "no_videos"}
    max_fpc = max(frames_per_clip_for(p) for p in clips)
    _log(f"distill roadseg from {len(clips)} clips, cap {max_fpc} frames/clip")
    distilled = distill_from_videos(clips, out_dir, frames_per_clip=max_fpc)
    if distilled.get("ok"):
        copy_roadseg_to_android(out_dir)
        distilled["android_assets"] = str(_android_roadseg())
        distilled["frames_per_clip"] = max_fpc
        distilled["n_source_clips"] = len(clips)
    distilled["yolop"] = yolop
    return distilled


def _label_pass(dataset_dir: Path, detect_fn: Callable, dets_map: dict[str, list], tag: str) -> dict[str, list]:
    payload = json.loads((dataset_dir / "index.json").read_text(encoding="utf-8"))
    raw_img = dataset_dir / "images" / "raw"
    rows = payload.get("index") or []
    n = len(rows)
    for i, row in enumerate(rows):
        path = raw_img / row["image"]
        if not path.is_file():
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        dets_map.setdefault(row["image"], []).extend(detect_fn(bgr) or [])
        if (i + 1) % 20 == 0 or i + 1 == n:
            _log(f"{tag} {i + 1}/{n}")
    return dets_map


def _write_merged_labels(dataset_dir: Path, dets_map: dict[str, list]) -> dict[str, Any]:
    payload = json.loads((dataset_dir / "index.json").read_text(encoding="utf-8"))
    n_pos = 0
    n_empty = 0
    for row in payload.get("index") or []:
        dets = merge_dets(dets_map.get(row["image"], []))
        write_raw_label(dataset_dir, row["image"], dets, int(row.get("width") or 1), int(row.get("height") or 1))
        if dets:
            n_pos += 1
        else:
            n_empty += 1
    _thin_empty_labels(dataset_dir, payload, empty_ratio=1.3)
    split = materialize_yolo_splits(dataset_dir)
    out = {"ok": True, "n_positive_frames": n_pos, "n_empty_frames": n_empty, **split}
    (dataset_dir / "proposals.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def qwen_filter_labels(dataset_dir: Path) -> dict[str, Any]:
    verifier = try_load_qwen()
    if verifier is None:
        return {"ok": False, "reason": "qwen_unavailable", "error": LAST_LOAD_ERROR.get("qwen")}
    dataset_dir = Path(dataset_dir)
    payload = json.loads((dataset_dir / "index.json").read_text(encoding="utf-8"))
    raw_img = dataset_dir / "images" / "raw"
    n_in = 0
    n_keep = 0
    rows = [r for r in (payload.get("index") or [])]
    try:
        for i, row in enumerate(rows):
            path = raw_img / row["image"]
            lab = dataset_dir / "labels" / "raw" / (Path(row["image"]).stem + ".txt")
            if not path.is_file() or not lab.is_file() or not lab.read_text(encoding="utf-8").strip():
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            dets = []
            h, w = bgr.shape[:2]
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x0 = (xc - bw / 2) * w
                y0 = (yc - bh / 2) * h
                x1 = (xc + bw / 2) * w
                y1 = (yc + bh / 2) * h
                names = ("pothole", "speed_bump", "manhole_cover")
                dets.append({"name": names[cls] if 0 <= cls < 3 else "pothole", "bbox": (x0, y0, x1, y1), "conf": 0.4})
                n_in += 1
            kept = verify_dets(bgr, dets, verifier)
            n_keep += len(kept)
            write_raw_label(dataset_dir, row["image"], kept, w, h)
            if (i + 1) % 20 == 0:
                _log(f"qwen {i + 1}/{len(rows)} boxes {n_keep}/{n_in}")
    finally:
        verifier.close()
    return {"ok": True, "n_boxes_in": n_in, "n_boxes_out": n_keep}


def run_ride_train(
    *,
    sample_fps: float = 1.25,
    max_frames: int = 1600,
    epochs: int = 40,
    skip_vlm: bool = False,
    skip_qwen: bool = False,
    skip_roadseg: bool = False,
    skip_propose: bool = False,
) -> dict[str, Any]:
    inbox = default_inbox()
    dataset_dir = default_dataset_dir()
    bump_out = repo_root() / "artifacts" / "bump_train"
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "ok": False}

    teacher = ensure_teacher_weights()
    summary["teacher"] = teacher

    _log(f"ingest inbox={inbox}")
    ingested = ingest_inbox(inbox, dataset_dir, sample_fps=sample_fps, max_frames_per_clip=max_frames)
    summary["ingested"] = {k: ingested.get(k) for k in ("n_clips", "n_frames", "clips", "image_split") if k in ingested}
    _log(f"ingested {ingested.get('n_frames')} frames from {ingested.get('n_clips')} clips")

    if not skip_roadseg:
        summary["roadseg"] = distill_main_roadseg()
        _log(f"roadseg {summary['roadseg'].get('ok')} iou={summary['roadseg'].get('road_iou_vs_teacher')}")

    used_world = False
    vlm_name = ""
    if skip_propose:
        proposed = reuse_raw_labels(dataset_dir)
        _log("reuse existing raw labels (skip propose)")
    elif skip_vlm:
        proposed = propose_labels(dataset_dir)
        used_world = bool(proposed.get("ok"))
    else:
        dets_map: dict[str, list] = {}
        world = try_load_world_bump(conf=0.08)
        if world is not None:
            used_world = True
            _log("YOLO-World labeling")
            dets_map = _label_pass(dataset_dir, world.detect_bgr, dets_map, "yolo-world")
            world.close()
        boxer, vlm_name = try_load_vlm_boxer()
        if boxer is not None:
            _log(f"{vlm_name} labeling")
            dets_map = _label_pass(dataset_dir, boxer.detect_bgr, dets_map, vlm_name)
            boxer.close()
        if dets_map:
            proposed = _write_merged_labels(dataset_dir, dets_map)
        else:
            proposed = propose_labels(dataset_dir)
            used_world = bool(proposed.get("ok"))
    summary["proposed"] = proposed
    summary["labelers"] = {
        "yolo_world": used_world,
        "vlm": vlm_name,
        "load_errors": dict(LAST_LOAD_ERROR),
    }
    _log(f"labels pos={proposed.get('n_positive_frames')} train={proposed.get('n_train_images')}")

    if proposed.get("ok") and not skip_qwen and not skip_propose:
        summary["qwen"] = qwen_filter_labels(dataset_dir)
        _log(f"qwen {summary['qwen']}")
        if summary["qwen"].get("ok"):
            proposed.update(materialize_yolo_splits(dataset_dir))
            summary["proposed"] = proposed

    n_train = int(proposed.get("n_train_images") or 0)
    trained: dict[str, Any] = {"ok": False, "reason": "not_attempted"}
    if proposed.get("ok") and n_train >= 16:
        _log(f"train yolov8m {epochs} epochs on {n_train} images")
        yaml_path = Path(proposed["data_yaml"])
        trained = train_yolo(yaml_path, bump_out, epochs=epochs, model="yolov8m.pt")
    summary["trained"] = trained

    exported = {"ok": False}
    if trained.get("ok") or teacher.get("ok"):
        exported = export_bump_tflite(weights=Path(trained["package_weights"]) if trained.get("package_weights") else None)
    summary["exported"] = exported
    write_bump_card(
        repo_root() / "models" / "bump-world-0.1.0",
        dataset_dir=str(dataset_dir),
        n_clips=ingested.get("n_clips"),
        n_train_images=n_train,
        trained=bool(trained.get("ok")),
        weights=exported.get("path") or trained.get("package_weights") or "",
    )
    summary["ok"] = bool(summary.get("roadseg", {}).get("ok") or trained.get("ok") or proposed.get("ok"))
    bump_out.mkdir(parents=True, exist_ok=True)
    (bump_out / "ride_train.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"done ok={summary['ok']}")
    return summary
