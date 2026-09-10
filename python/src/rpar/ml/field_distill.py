"""Distill YOLOPv2 teacher masks from Video/ into a dual-scale classmap (PER-001/003).

Not a pothole net. Pytest must not import this path (loads ONNX).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.ml.seg_train import _xyrgb, fit_one_vs_rest, predict_labels, road_iou
from rpar.ml.yolopv2 import Yolopv2Engine, paint_drivable, weights_available
from rpar.roi import FAR, NEAR, crop_xyxy
from rpar.segdecode import CLASS_OCC, CLASS_ROAD
from rpar.segengine import DualScaleSegEngine, FAR_WH, NEAR_WH


def teacher_classmap(engine: Yolopv2Engine, h: int, w: int) -> np.ndarray:
    cm = np.zeros((h, w), dtype=np.uint8)
    if engine.last_da_mask is not None:
        cm[engine.last_da_mask > 0.45] = CLASS_ROAD
    for x1, y1, x2, y2 in engine.last_boxes:
        gy0 = int(y1 + 0.55 * max(8.0, y2 - y1))
        xa, xb = int(max(0, x1)), int(min(w, x2))
        ya, yb = int(max(0, gy0)), int(min(h, y2))
        if xb > xa and yb > ya:
            cm[ya:yb, xa:xb] = CLASS_OCC
    return cm


def _crops(bgr: np.ndarray, cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for box, size in ((FAR, FAR_WH), (NEAR, NEAR_WH)):
        x0, y0, x1, y1 = crop_xyxy(w, h, box)
        rgb = cv2.resize(bgr[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
        lab = cv2.resize(cm[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST)
        xs.append(_xyrgb(rgb))
        ys.append(lab.reshape(-1))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def distill_from_videos(
    clips: list[Path],
    out_dir: Path,
    *,
    frames_per_clip: int = 16,
) -> dict[str, Any]:
    if not weights_available():
        return {"ok": False, "reason": "YOLOPv2.onnx missing"}
    from rpar.field_video import _sync_frame

    engine = Yolopv2Engine()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ious: list[float] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path in clips:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                continue
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if n < 2:
                cap.release()
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30) or 30.0
            fpc = int(max(8, min(int(frames_per_clip), round((n / fps) / 8.0) or 8)))
            print(f"[roadseg] {path.name}: {fpc} teacher frames", flush=True)
            for k in range(fpc):
                idx = int((k + 0.5) * n / fpc)
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(min(n - 1, idx)))
                ok, bgr = cap.read()
                if not ok:
                    continue
                result = engine.infer(_sync_frame(bgr, idx, w, h))
                cm = teacher_classmap(engine, h, w)
                x, y = _crops(bgr, cm)
                xs.append(x)
                ys.append(y)
            cap.release()
        if not xs:
            return {"ok": False, "reason": "no frames"}
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        rng = np.random.default_rng(7)
        n = int(x.shape[0])
        hold_n = min(4000, max(1, n // 10)) if n > 16 else n
        hold = rng.choice(n, size=hold_n, replace=False)
        remain = np.setdiff1d(np.arange(n), hold, assume_unique=False)
        if remain.size == 0:
            idx = hold
        else:
            idx = rng.choice(remain, size=min(40000, int(remain.size)), replace=False)
        weights = fit_one_vs_rest(x[idx], y[idx])
        acc = float((predict_labels(x[hold], weights) == y[hold]).mean())
        student = DualScaleSegEngine(weights)
        # one sanity IoU vs last teacher map
        cap = cv2.VideoCapture(str(clips[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ok, bgr = cap.read()
        cap.release()
        iou = 0.0
        if ok:
            h, w = bgr.shape[:2]
            engine.infer(_sync_frame(bgr, 30, w, h))
            gt = teacher_classmap(engine, h, w)
            pred = student.classmap(bgr)
            iou = road_iou(pred, gt)
            poly = []
            painted = paint_drivable(bgr, (pred == CLASS_ROAD).astype(np.float32), [], poly)
            cv2.imwrite(str(out_dir / "student_preview.jpg"), painted, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        payload = {
            "schema_version": SCHEMA_VERSION,
            "teacher": "yolopv2-onnx",
            "pixel_acc": acc,
            "road_iou_vs_teacher": iou,
            "n_pixels": int(idx.size),
            "n_clips": len(clips),
            "weights": weights.tolist(),
            "note": "Dual-scale lstsq on YOLOPv2 teacher masks from local Video/. Not PKC110 GT.",
        }
        (out_dir / "seg_weights.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_field_package(out_dir, None)
        return {"ok": True, **{k: v for k, v in payload.items() if k != "weights"}, "package": str(out_dir)}
    finally:
        engine.close()


def write_field_package(out_dir: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write manifest/labels/sha256 around existing seg_weights.json (no TFLite required)."""
    import hashlib

    from rpar.ml.train import write_model_package

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "seg_weights.json"
    if payload is not None:
        weights_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not weights_path.is_file():
        return {"ok": False, "reason": "missing seg_weights.json"}
    write_model_package(out_dir, "roadseg-field-0.1.0", engine="classmap")
    labels = json.loads((out_dir / "labels.json").read_text(encoding="utf-8"))
    (out_dir / "MODEL_CARD.md").write_text(
        "# roadseg-field-0.1.0\n\n"
        "Dual-scale classmap distilled from YOLOPv2 teacher masks on local `Video/` clips. "
        "Road + vehicle occlusion only. Not a pothole detector and not Camera2 1080p60 GT.\n",
        encoding="utf-8",
    )
    tflite = out_dir / "model.tflite"
    if tflite.is_file() and tflite.stat().st_size < 200:
        tflite.unlink()
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["engine"] = "classmap"
    man["package_id"] = "roadseg-field-0.1.0"
    man["quantization"] = "lstsq-json"
    man["files"] = {"seg_weights.json": "seg_weights.json", "labels.json": "labels.json"}
    man["labels"] = labels
    man["input_spec"] = {
        "far": {"width": 96, "height": 48, "layout": "RGB+xy", "norm": "unit"},
        "near": {"width": 96, "height": 48, "layout": "RGB+xy", "norm": "unit"},
        "note": "Dual-scale ROI head; not a single 640x640 full-frame.",
    }
    man["sha256"] = {}
    for p in out_dir.iterdir():
        if p.is_file() and p.name not in {"manifest.json", "student_preview.jpg"}:
            man["sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return {"ok": True, "package": str(out_dir), "files": sorted(man["sha256"].keys())}
