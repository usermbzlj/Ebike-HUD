"""YOLOPv2 ONNX sidecar: drivable area + vehicles → RoadObservation (PER-001/002).

Weights are MIT-licensed third-party ONNX (Kazuhito00/YOLOPv2-ONNX-Sample from CAIC-AD/YOLOPv2).
Not a pothole/manhole net. Empty output must not blank the HUD.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from rpar.enums import InferenceBackend
from rpar.maskutil import rect_polygon, simplify_polygon
from rpar.models import FrameQualityMap, PerceptionResult, SynchronizedFrame
from rpar.segdecode import polygons_for_label

WEIGHTS_NAME = "YOLOPv2.onnx"
RELEASE_URL = "https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample/releases/download/v0.0.0/YOLOPv2.onnx"
# BDD-ish + COCO vehicle ids; unknown schemes fall back to box geometry.
_VEHICLE_CLS = {1, 2, 3, 4, 5, 6, 7}


def default_weights_path() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        cand = p / "models" / "yolopv2" / WEIGHTS_NAME
        if cand.is_file():
            return cand
    return Path.cwd() / "models" / "yolopv2" / WEIGHTS_NAME


def fetch_weights(dest: Path | None = None) -> dict[str, Any]:
    """Download YOLOPv2.onnx into models/yolopv2/ if missing."""
    dest = Path(dest) if dest else default_weights_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return {"ok": True, "path": str(dest), "downloaded": False, "bytes": dest.stat().st_size}
    import urllib.request

    tmp = dest.with_suffix(".onnx.part")
    try:
        urllib.request.urlretrieve(RELEASE_URL, tmp)
        if tmp.is_file() and tmp.stat().st_size > 1_000_000:
            tmp.replace(dest)
            return {"ok": True, "path": str(dest), "downloaded": True, "bytes": dest.stat().st_size}
    except Exception as exc:
        if tmp.is_file():
            tmp.unlink()
        return {"ok": False, "reason": str(exc)[:300], "path": str(dest)}
    return {"ok": False, "reason": "download_failed", "path": str(dest)}


def weights_available(path: Path | None = None) -> bool:
    p = Path(path) if path else default_weights_path()
    return p.is_file() and p.stat().st_size > 1_000_000


def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640), color=(114, 114, 114), stride: int = 32) -> tuple[np.ndarray, tuple[float, float]]:
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (dw, dh)


def _sigmoid(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-arr))


def _make_grid(nx: int, ny: int) -> np.ndarray:
    xv, yv = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-9)
        order = order[np.where(ovr <= iou_threshold)[0] + 1]
    return np.asarray(keep, dtype=np.int32)


def split_for_trace_model(pred: list[np.ndarray], anchor_grid: list[np.ndarray]) -> np.ndarray:
    z = []
    st = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        p = pred[i].reshape(bs, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2)
        y = _sigmoid(p)
        gr = _make_grid(nx, ny)
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + gr) * st[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
        z.append(y.reshape(bs, -1, 85))
    return np.concatenate(z, 1)


def nms_dets(prediction: np.ndarray, conf_thres: float = 0.3, iou_thres: float = 0.45) -> list[np.ndarray]:
    xc = prediction[..., 4] > conf_thres
    output = [np.zeros((0, 6), dtype=np.float32) for _ in range(prediction.shape[0])]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if x.shape[0] == 0:
            continue
        x = x.copy()
        x[:, 5:] *= x[:, 4:5]
        box = _xywh2xyxy(x[:, :4])
        conf = np.max(x[:, 5:], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1).reshape((-1, 1)).astype(np.float32)
        x = np.concatenate((box, conf, j), 1)
        x = x[x[:, 4] > conf_thres]
        if x.shape[0] == 0:
            continue
        i = _nms(x[:, :4], x[:, 4], iou_thres)
        output[xi] = x[i[:300]]
    return output


def scale_coords(img1_hw: tuple[int, int], coords: np.ndarray, img0_hw: tuple[int, int]) -> np.ndarray:
    gain = min(img1_hw[0] / img0_hw[0], img1_hw[1] / img0_hw[1])
    pad_w = (img1_hw[1] - img0_hw[1] * gain) / 2
    pad_h = (img1_hw[0] - img0_hw[0] * gain) / 2
    coords = coords.copy()
    coords[:, [0, 2]] -= pad_w
    coords[:, [1, 3]] -= pad_h
    coords[:, :4] /= gain
    h0, w0 = img0_hw
    coords[:, 0] = np.clip(coords[:, 0], 0, w0)
    coords[:, 1] = np.clip(coords[:, 1], 0, h0)
    coords[:, 2] = np.clip(coords[:, 2], 0, w0)
    coords[:, 3] = np.clip(coords[:, 3], 0, h0)
    return coords


def da_mask(seg: np.ndarray, pad_wh: tuple[float, float], out_hw: tuple[int, int]) -> np.ndarray:
    temp = np.asarray(seg[0][0], dtype=np.float32)
    pad_w, pad_h = int(pad_wh[0]), int(pad_wh[1])
    sh, sw = temp.shape[:2]
    y1, y2 = pad_h, sh - pad_h if pad_h else sh
    x1, x2 = pad_w, sw - pad_w if pad_w else sw
    if y2 <= y1 + 4 or x2 <= x1 + 4:
        cropped = temp
    else:
        cropped = temp[y1:y2, x1:x2]
    road = 1.0 - cropped
    h, w = out_hw
    return cv2.resize(road, (w, h), interpolation=cv2.INTER_LINEAR)


def lane_mask(seg: np.ndarray, pad_wh: tuple[float, float], out_hw: tuple[int, int]) -> np.ndarray:
    temp = np.asarray(seg[0][0], dtype=np.float32)
    pad_w, pad_h = int(pad_wh[0]), int(pad_wh[1])
    sh, sw = temp.shape[:2]
    y1, y2 = pad_h, sh - pad_h if pad_h else sh
    x1, x2 = pad_w, sw - pad_w if pad_w else sw
    cropped = temp if y2 <= y1 + 4 or x2 <= x1 + 4 else temp[y1:y2, x1:x2]
    h, w = out_hw
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def is_vehicle_box(cls_id: float, box: tuple[float, float, float, float], frame_hw: tuple[int, int]) -> bool:
    h, w = frame_hw
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    area = bw * bh
    if area < 0.0010 * w * h:
        return False
    if area > 0.22 * w * h:
        return False
    if y2 < 0.22 * h:
        return False
    aspect = bw / bh
    if aspect > 4.2 or aspect < 0.28:
        return False
    if x1 < 0.03 * w and bw < 0.14 * w and bh > 0.35 * h:
        return False
    if x2 > 0.97 * w and bw < 0.14 * w and bh > 0.35 * h:
        return False
    if bh < 0.025 * h and bw < 0.035 * w:
        return False
    cid = int(cls_id)
    if 0 <= cid <= 20 and cid not in _VEHICLE_CLS:
        return False
    return True


def mask_to_polygon(mask: np.ndarray, min_frac: float = 0.02) -> list[tuple[float, float]]:
    binary = (mask > 0.45).astype(np.uint8) * 255
    if not binary.any():
        return []
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    h, w = binary.shape
    polys = polygons_for_label((binary > 0).astype(np.uint8), 1, min_area=max(80.0, min_frac * w * h))
    if not polys:
        return []
    return max(polys, key=lambda p: cv2.contourArea(np.array(p, dtype=np.float32)))


def paint_drivable(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    boxes: list[tuple[float, float, float, float]] | None = None,
    road_poly: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """High-alpha demo wash: green drivable area + amber vehicle boxes."""
    img = bgr.copy()
    if mask is not None and mask.size:
        m = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        sel = ((m > 115).astype(np.uint8)) * 255
        color = np.zeros_like(img)
        color[:, :] = (40, 210, 70)
        wash = cv2.bitwise_and(color, color, mask=sel)
        base = cv2.bitwise_and(img, img, mask=sel)
        mixed = cv2.addWeighted(wash, 0.48, base, 0.52, 0)
        inv = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(sel))
        img = cv2.add(mixed, inv)
    if road_poly and len(road_poly) >= 3:
        pts = np.array(road_poly, dtype=np.int32)
        cv2.polylines(img, [pts], True, (60, 255, 110), 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in boxes or []:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (20, 140, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (p1[0], max(0, p1[1] - 22)), (p1[0] + 78, p1[1]), (8, 10, 14), -1)
        cv2.putText(img, "vehicle", (p1[0] + 4, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 236, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "YOLOPv2  road + vehicle", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (236, 244, 248), 2, cv2.LINE_AA)
    return img


class Yolopv2Engine:
    """Desktop ONNX runtime. Tensors never leave this class (PER-014)."""

    def __init__(self, weights: Path | None = None, score_th: float = 0.38, nms_th: float = 0.45) -> None:
        self.weights = Path(weights) if weights else default_weights_path()
        self.score_th = score_th
        self.nms_th = nms_th
        self._session = None
        self.last_da_mask: np.ndarray | None = None
        self.last_lane_mask: np.ndarray | None = None
        self.last_boxes: list[tuple[float, float, float, float]] = []

    def _ort(self):
        if self._session is None:
            if not weights_available(self.weights):
                raise FileNotFoundError(self.weights)
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self.weights), providers=["CPUExecutionProvider"])
        return self._session

    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        t0 = perf_counter()
        bgr = frame.bgr
        h, w = bgr.shape[:2]
        inp, pad_wh = letterbox(bgr, (640, 640))
        blob = inp[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None, ...]
        sess = self._ort()
        name = sess.get_inputs()[0].name
        results = sess.run(None, {name: blob})
        dets = [results[0][0], results[0][1], results[0][2]]
        anchors = [results[1], results[2], results[3]]
        pred = split_for_trace_model(dets, anchors)
        nmsed = nms_dets(pred, self.score_th, self.nms_th)
        occ: list[list[tuple[float, float]]] = []
        boxes_out: list[tuple[float, float, float, float]] = []
        if nmsed and len(nmsed[0]):
            scaled = scale_coords(inp.shape[:2], nmsed[0][:, :4].copy(), (h, w))
            for det, (x1, y1, x2, y2) in zip(nmsed[0], scaled):
                box = (float(x1), float(y1), float(x2), float(y2))
                cls_id = float(det[5]) if det.shape[0] > 5 else 2.0
                if not is_vehicle_box(cls_id, box, (h, w)):
                    continue
                bw = max(8.0, box[2] - box[0])
                bh = max(8.0, box[3] - box[1])
                gy0 = box[1] + bh * 0.55
                occ.append(rect_polygon(box[0], gy0, box[2], box[3]))
                boxes_out.append(box)
        road_m = da_mask(results[4], pad_wh, (h, w))
        self.last_da_mask = road_m
        try:
            self.last_lane_mask = lane_mask(results[5], pad_wh, (h, w))
        except Exception:
            self.last_lane_mask = None
        self.last_boxes = boxes_out
        road_poly = mask_to_polygon(road_m)
        return PerceptionResult(
            timestamp_ns=frame.meta.sensor_timestamp_ns,
            source_frame_id=frame.meta.frame_id,
            road_polygon=simplify_polygon(road_poly, 4.0) if road_poly else [],
            occluded_polygons=occ,
            observations=[],
            backend=InferenceBackend.CPU,
            latency_ms=(perf_counter() - t0) * 1000.0,
            input_sizes=[(640, 640)],
            dual_scale=False,
        )

    def capability(self) -> dict[str, Any]:
        return {
            "backend": "yolopv2-onnx",
            "outputs": "RoadObservation",
            "tensors_to_ui": False,
            "tasks": ["drivable_area", "vehicle_occlusion"],
            "not": ["pothole", "manhole", "speed_bump"],
            "weights": str(self.weights),
        }

    def close(self) -> None:
        self._session = None
