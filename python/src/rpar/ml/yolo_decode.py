"""YOLO detect tensor decode + letterbox. No ultralytics / TensorFlow import."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpar.ml.bump_prompts import nms_xyxy


def letterbox_meta(height: int, width: int, imgsz: int = 640) -> dict[str, float | int]:
    h = max(int(height), 1)
    w = max(int(width), 1)
    size = max(int(imgsz), 1)
    gain = min(size / h, size / w)
    new_w = int(round(w * gain))
    new_h = int(round(h * gain))
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    return {"gain": float(gain), "pad_x": float(pad_x), "pad_y": float(pad_y), "imgsz": size}


def xyxy_letterbox_to_orig(
    box: tuple[float, float, float, float],
    meta: dict[str, float | int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    gain = float(meta["gain"]) or 1.0
    pad_x = float(meta["pad_x"])
    pad_y = float(meta["pad_y"])
    x0 = (box[0] - pad_x) / gain
    y0 = (box[1] - pad_y) / gain
    x1 = (box[2] - pad_x) / gain
    y1 = (box[3] - pad_y) / gain
    w = max(int(width), 1)
    h = max(int(height), 1)
    return (
        float(np.clip(min(x0, x1), 0, w - 1)),
        float(np.clip(min(y0, y1), 0, h - 1)),
        float(np.clip(max(x0, x1), 1, w)),
        float(np.clip(max(y0, y1), 1, h)),
    )


def _xywh_to_xyxy(xc: float, yc: float, bw: float, bh: float) -> tuple[float, float, float, float]:
    return (xc - bw * 0.5, yc - bh * 0.5, xc + bw * 0.5, yc + bh * 0.5)


def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def _as_nxk(arr: np.ndarray, n_classes: int) -> tuple[str, np.ndarray]:
    """Return ('nms', N×6) or ('raw', N×(4+nc))."""
    a = _squeeze_batch(arr)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    rows, cols = int(a.shape[0]), int(a.shape[1])
    nc = max(int(n_classes), 1)
    raw_ch = 4 + nc
    if cols == 6 and rows <= 4000:
        return "nms", a
    if rows == 6 and cols > 6:
        return "nms", a.T
    if cols == raw_ch:
        return "raw", a
    if rows == raw_ch:
        return "raw", a.T
    if cols > rows and rows <= 64:
        return "raw", a.T
    return "raw", a


def decode_yolo_output(
    output: Any,
    names: list[str] | dict[Any, str],
    *,
    conf_thr: float = 0.08,
    iou_thr: float = 0.50,
    n_classes: int | None = None,
) -> list[dict]:
    if isinstance(names, dict):
        ordered = [str(names.get(i, names.get(str(i), ""))) for i in range(len(names))]
        if not any(ordered):
            ordered = [str(v) for v in names.values()]
        name_list = ordered
    else:
        name_list = [str(n) for n in names]
    nc = int(n_classes if n_classes is not None else max(len(name_list), 1))
    layout, pred = _as_nxk(output, nc)
    dets: list[dict] = []
    if layout == "nms":
        for row in pred:
            if row.shape[0] < 6:
                continue
            conf = float(row[4])
            if conf < conf_thr:
                continue
            cls_id = int(round(float(row[5])))
            if cls_id < 0 or cls_id >= len(name_list):
                continue
            name = name_list[cls_id]
            if not name.strip():
                continue
            dets.append(
                {
                    "name": name,
                    "cls": cls_id,
                    "conf": conf,
                    "bbox": (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                }
            )
    else:
        ch = int(pred.shape[1])
        if ch < 5:
            return []
        cls_scores = pred[:, 4:]
        if cls_scores.size == 0:
            return []
        cls_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
        for i in range(int(pred.shape[0])):
            conf = float(confs[i])
            if conf < conf_thr:
                continue
            cls_id = int(cls_ids[i])
            if cls_id < 0 or cls_id >= len(name_list):
                continue
            name = name_list[cls_id]
            if not name.strip():
                continue
            xc, yc, bw, bh = (float(pred[i, 0]), float(pred[i, 1]), float(pred[i, 2]), float(pred[i, 3]))
            dets.append(
                {
                    "name": name,
                    "cls": cls_id,
                    "conf": conf,
                    "bbox": _xywh_to_xyxy(xc, yc, bw, bh),
                }
            )
    return nms_xyxy(dets, iou_thr=iou_thr)


def decode_to_original(
    output: Any,
    names: list[str] | dict[Any, str],
    width: int,
    height: int,
    *,
    imgsz: int = 640,
    conf_thr: float = 0.08,
    iou_thr: float = 0.50,
) -> list[dict]:
    meta = letterbox_meta(height, width, imgsz)
    dets = decode_yolo_output(output, names, conf_thr=conf_thr, iou_thr=iou_thr)
    out: list[dict] = []
    for d in dets:
        box = xyxy_letterbox_to_orig(tuple(d["bbox"]), meta, width, height)
        item = dict(d)
        item["bbox"] = box
        out.append(item)
    return out
