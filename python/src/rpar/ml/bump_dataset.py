"""Ingest phone clips from Video/train/inbox into a session-isolated YOLO bump dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import cv2

from rpar import SCHEMA_VERSION
from rpar.ml.bump_prompts import (
    CLASS_TO_ID,
    YOLO_NAMES,
    clip_class_hint,
    keep_box,
    map_det_name,
    nms_xyxy,
    yolo_line,
)
from rpar.ml.train import split_sessions

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def repo_root(start: Path | None = None) -> Path:
    here = Path(start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "rpar.defaults.yaml").is_file():
            return p
    return Path.cwd()


def default_inbox(start: Path | None = None) -> Path:
    return repo_root(start) / "Video" / "train" / "inbox"


def default_dataset_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "artifacts" / "bump_dataset"


def list_train_clips(inbox: Path | None = None) -> list[Path]:
    d = Path(inbox) if inbox else default_inbox()
    if not d.is_dir():
        return []
    found: list[Path] = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            found.append(p)
    return found


def clip_split(clip_ids: list[str], seed: int = 7) -> tuple[list[str], list[str], list[str]]:
    ids = list(clip_ids)
    if not ids:
        return [], [], []
    if len(ids) == 1:
        return ids, ids, []
    if len(ids) == 2:
        return [ids[0]], [ids[1]], []
    if len(ids) == 3:
        return [ids[0], ids[1]], [ids[2]], []
    man = split_sessions(ids, seed=seed)
    train, val, test = man.train_sessions, man.val_sessions, man.test_sessions
    if not val and train:
        val = [train[-1]]
        train = train[:-1] or list(train)
    if not train:
        train = ids[: max(1, len(ids) - 1)]
        val = ids[len(train) :] or ids[-1:]
    return train, val, test


def extract_clip_frames(
    path: Path,
    dest_images: Path,
    *,
    sample_fps: float = 2.0,
    max_frames: int = 240,
) -> list[dict[str, Any]]:
    dest_images.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(src_fps / max(sample_fps, 0.1))))
    stem = path.stem
    hint = clip_class_hint(path.name)
    rows: list[dict[str, Any]] = []
    idx = 0
    kept = 0
    while kept < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            name = f"{stem}_{idx:06d}.jpg"
            dest = dest_images / name
            cv2.imwrite(str(dest), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            rows.append(
                {
                    "image": name,
                    "clip": stem,
                    "frame_index": idx,
                    "width": int(bgr.shape[1]),
                    "height": int(bgr.shape[0]),
                    "hint": hint,
                    "source": path.name,
                }
            )
            kept += 1
        idx += 1
        if n and idx >= n:
            break
    cap.release()
    if not rows and w and h:
        return []
    return rows


def ingest_inbox(
    inbox: Path | None = None,
    dataset_dir: Path | None = None,
    *,
    sample_fps: float = 2.0,
    max_frames_per_clip: int = 240,
) -> dict[str, Any]:
    inbox = Path(inbox) if inbox else default_inbox()
    dataset_dir = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    inbox.mkdir(parents=True, exist_ok=True)
    raw_dir = dataset_dir / "images" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips = list_train_clips(inbox)
    index: list[dict[str, Any]] = []
    for clip in clips:
        index.extend(extract_clip_frames(clip, raw_dir, sample_fps=sample_fps, max_frames=max_frames_per_clip))
    clip_ids = sorted({row["clip"] for row in index})
    train_ids, val_ids, test_ids = clip_split(clip_ids)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inbox": str(inbox),
        "dataset_dir": str(dataset_dir),
        "n_clips": len(clips),
        "n_frames": len(index),
        "clips": [p.name for p in clips],
        "clip_ids": clip_ids,
        "split": {"train": train_ids, "val": val_ids, "test": test_ids},
        "index": index,
        "note": "YOLO labels are written by propose-bump / train-bump. MP4s stay untracked.",
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _copy_split(dataset_dir: Path, index: list[dict[str, Any]], split_ids: set[str], split: str) -> int:
    img_dir = dataset_dir / "images" / split
    lab_dir = dataset_dir / "labels" / split
    raw_img = dataset_dir / "images" / "raw"
    raw_lab = dataset_dir / "labels" / "raw"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in index:
        if row["clip"] not in split_ids:
            continue
        src = raw_img / row["image"]
        if not src.is_file():
            continue
        shutil.copy2(src, img_dir / row["image"])
        lab_src = raw_lab / (Path(row["image"]).stem + ".txt")
        lab_dst = lab_dir / (Path(row["image"]).stem + ".txt")
        if lab_src.is_file():
            shutil.copy2(lab_src, lab_dst)
        else:
            lab_dst.write_text("", encoding="utf-8")
        n += 1
    return n


def write_data_yaml(dataset_dir: Path, train_rel: str = "images/train", val_rel: str = "images/val") -> Path:
    dataset_dir = Path(dataset_dir)
    body = (
        f"path: {dataset_dir.resolve().as_posix()}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        "names:\n"
        "  0: pothole\n"
        "  1: speed_bump\n"
        "  2: manhole_cover\n"
    )
    dest = dataset_dir / "data.yaml"
    dest.write_text(body, encoding="utf-8")
    return dest


def materialize_yolo_splits(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index = payload.get("index") or []
    split = payload.get("split") or {}
    train_ids = set(split.get("train") or [])
    val_ids = set(split.get("val") or [])
    n_train = _copy_split(dataset_dir, index, train_ids, "train")
    n_val = _copy_split(dataset_dir, index, val_ids, "val")
    yaml_path = write_data_yaml(dataset_dir)
    summary = {
        "data_yaml": str(yaml_path),
        "n_train_images": n_train,
        "n_val_images": n_val,
        "names": list(YOLO_NAMES),
    }
    (dataset_dir / "yolo_split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_raw_label(dataset_dir: Path, image_name: str, dets: list[dict], width: int, height: int) -> Path:
    lab_dir = Path(dataset_dir) / "labels" / "raw"
    lab_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for d in nms_xyxy(dets, iou_thr=0.50):
        kind = map_det_name(str(d.get("name") or d.get("class") or ""))
        if kind is None:
            continue
        bbox = tuple(d["bbox"])
        if not keep_box(kind, bbox, width, height):
            continue
        lines.append(yolo_line(CLASS_TO_ID[kind], bbox, width, height))
    dest = lab_dir / (Path(image_name).stem + ".txt")
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dest


def propose_labels(
    dataset_dir: Path,
    detect_fn: Callable[[Any], list[dict]] | None = None,
    *,
    engine=None,
) -> dict[str, Any]:
    """Write YOLO txt labels. detect_fn(bgr) -> [{name, bbox, conf}]. Lazy-loads the world model if omitted."""
    dataset_dir = Path(dataset_dir)
    payload = json.loads((dataset_dir / "index.json").read_text(encoding="utf-8"))
    raw_img = dataset_dir / "images" / "raw"
    n_pos = 0
    n_empty = 0
    if detect_fn is None:
        if engine is None:
            from rpar.ml.world_bump import try_load_world_bump

            engine = try_load_world_bump()
        if engine is None:
            return {"ok": False, "reason": "bump_model_unavailable", "n_labeled": 0}
        detect_fn = engine.detect_bgr
    for row in payload.get("index") or []:
        path = raw_img / row["image"]
        if not path.is_file():
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        dets = detect_fn(bgr) or []
        hint = row.get("hint")
        if hint:
            for d in dets:
                if map_det_name(str(d.get("name") or "")) == hint:
                    d["conf"] = float(d.get("conf", 0.0)) + 0.05
        write_raw_label(dataset_dir, row["image"], dets, int(bgr.shape[1]), int(bgr.shape[0]))
        if dets:
            n_pos += 1
        else:
            n_empty += 1
    split = materialize_yolo_splits(dataset_dir)
    out = {
        "ok": True,
        "n_positive_frames": n_pos,
        "n_empty_frames": n_empty,
        **split,
    }
    (dataset_dir / "proposals.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
