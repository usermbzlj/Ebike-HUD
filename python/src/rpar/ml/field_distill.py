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
            for k in range(frames_per_clip):
                idx = int((k + 0.5) * n / frames_per_clip)
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
        idx = rng.choice(x.shape[0], size=min(12000, x.shape[0]), replace=False)
        hold = rng.choice(x.shape[0], size=min(2000, x.shape[0]), replace=False)
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
        (out_dir / "MODEL_CARD.md").write_text(
            "# roadseg-field-0.1.0\n\n"
            "Dual-scale classmap distilled from YOLOPv2 teacher masks on local `Video/` clips. "
            "Road + vehicle occlusion only. Not a pothole detector and not Camera2 1080p60 GT.\n",
            encoding="utf-8",
        )
        return {"ok": True, **{k: v for k, v in payload.items() if k != "weights"}, "package": str(out_dir)}
    finally:
        engine.close()
