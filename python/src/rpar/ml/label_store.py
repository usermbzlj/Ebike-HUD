"""Human YOLO boxes for bump training. Never run keep_box — gold labels stay as drawn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.maskutil import bbox_iou
from rpar.ml.bump_dataset import image_time_split
from rpar.ml.bump_prompts import CLASS_TO_ID, YOLO_NAMES, yolo_line

HUMAN_SIDECAR = "human_labels.json"


def image_stem(clip: str, frame_index: int) -> str:
    return f"{clip}_{int(frame_index):06d}"


def yolo_to_xyxy(xc: float, yc: float, bw: float, bh: float, width: int, height: int) -> tuple[float, float, float, float]:
    box_w = float(bw) * width
    box_h = float(bh) * height
    cx = float(xc) * width
    cy = float(yc) * height
    return (cx - 0.5 * box_w, cy - 0.5 * box_h, cx + 0.5 * box_w, cy + 0.5 * box_h)


def parse_yolo_txt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= len(YOLO_NAMES):
            continue
        bbox = yolo_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), width, height)
        out.append({"class": YOLO_NAMES[cls_id], "bbox": bbox, "source": "file"})
    return out


def write_yolo_txt(path: Path, objects: list[dict[str, Any]], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for obj in objects:
        name = str(obj.get("class") or "")
        if name not in CLASS_TO_ID:
            continue
        bbox = tuple(obj["bbox"])
        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
            continue
        lines.append(yolo_line(CLASS_TO_ID[name], bbox, width, height))
    path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")


def merge_objects(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    iou_thr: float = 0.40,
) -> list[dict[str, Any]]:
    kept = list(existing)
    for item in incoming:
        box = tuple(item["bbox"])
        replaced = False
        for i, prev in enumerate(kept):
            if bbox_iou(box, tuple(prev["bbox"])) >= iou_thr:
                kept[i] = item
                replaced = True
                break
        if not replaced:
            kept.append(item)
    return kept


def sidecar_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / HUMAN_SIDECAR


def load_sidecar(dataset_dir: Path) -> dict[str, Any]:
    path = sidecar_path(dataset_dir)
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "clips": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"schema_version": SCHEMA_VERSION, "clips": {}}
    payload.setdefault("clips", {})
    return payload


def save_sidecar(dataset_dir: Path, payload: dict[str, Any]) -> None:
    path = sidecar_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def frame_objects(dataset_dir: Path, clip: str, frame_index: int, width: int, height: int) -> list[dict[str, Any]]:
    payload = load_sidecar(dataset_dir)
    clip_map = payload.get("clips", {}).get(clip) or {}
    frames = clip_map.get("frames") or {}
    key = str(int(frame_index))
    if key in frames:
        return list(frames[key])
    stem = image_stem(clip, frame_index)
    return parse_yolo_txt(Path(dataset_dir) / "labels" / "raw" / f"{stem}.txt", width, height)


def labeled_frame_indices(dataset_dir: Path, clip: str) -> list[int]:
    found: set[int] = set()
    payload = load_sidecar(dataset_dir)
    frames = ((payload.get("clips") or {}).get(clip) or {}).get("frames") or {}
    for key, objs in frames.items():
        if objs:
            found.add(int(key))
    raw = Path(dataset_dir) / "labels" / "raw"
    prefix = f"{clip}_"
    if raw.is_dir():
        for path in raw.glob(f"{prefix}*.txt"):
            if not path.read_text(encoding="utf-8").strip():
                continue
            stem = path.stem
            try:
                found.add(int(stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return sorted(found)


def _ensure_index_row(
    dataset_dir: Path,
    *,
    clip: str,
    source_name: str,
    frame_index: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index_path = dataset_dir / "index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset_dir": str(dataset_dir),
            "n_clips": 1,
            "n_frames": 0,
            "sample_fps": 4.0,
            "clips": [source_name],
            "clip_ids": [clip],
            "index": [],
            "note": "Human labels from rpar label. MP4s stay untracked.",
        }
    name = f"{image_stem(clip, frame_index)}.jpg"
    index = list(payload.get("index") or [])
    by_image = {row.get("image"): row for row in index}
    row = {
        "image": name,
        "clip": clip,
        "frame_index": int(frame_index),
        "width": int(width),
        "height": int(height),
        "hint": None,
        "source": source_name,
        "human": True,
    }
    by_image[name] = row
    payload["index"] = sorted(by_image.values(), key=lambda r: (str(r.get("clip")), int(r.get("frame_index") or 0)))
    payload["n_frames"] = len(payload["index"])
    payload["image_split"] = image_time_split(payload["index"])
    clip_ids = sorted({str(r.get("clip")) for r in payload["index"]})
    payload["clip_ids"] = clip_ids
    payload["n_clips"] = len(clip_ids)
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def commit_frame(
    dataset_dir: Path,
    *,
    clip: str,
    source_name: str,
    frame_index: int,
    bgr: np.ndarray,
    objects: list[dict[str, Any]],
    replace: bool = False,
) -> dict[str, Any]:
    """Write jpg + YOLO txt for one frame. replace=True overwrites; False merges."""
    h, w = bgr.shape[:2]
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / "images" / "raw"
    lab_dir = dataset_dir / "labels" / "raw"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    stem = image_stem(clip, frame_index)
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    current = [] if replace else parse_yolo_txt(lab_dir / f"{stem}.txt", w, h)
    merged = merge_objects(current, objects)
    tagged = [{**o, "source": "human"} for o in merged]
    write_yolo_txt(lab_dir / f"{stem}.txt", tagged, w, h)
    payload = load_sidecar(dataset_dir)
    clips = payload.setdefault("clips", {})
    clip_map = clips.setdefault(clip, {"source": source_name, "frames": {}})
    clip_map["source"] = source_name
    clip_map.setdefault("frames", {})[str(int(frame_index))] = tagged
    save_sidecar(dataset_dir, payload)
    _ensure_index_row(dataset_dir, clip=clip, source_name=source_name, frame_index=frame_index, width=w, height=h)
    return {"ok": True, "image": f"{stem}.jpg", "n": len(tagged), "objects": tagged}


def clear_frame(dataset_dir: Path, *, clip: str, source_name: str, frame_index: int, bgr: np.ndarray) -> dict[str, Any]:
    return commit_frame(dataset_dir, clip=clip, source_name=source_name, frame_index=frame_index, bgr=bgr, objects=[], replace=True)
