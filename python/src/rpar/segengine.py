"""Dual-scale classmap engine + heuristic hybrid (PER-001/003/014). Empty sidecar never blanks HUD."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from rpar.maskutil import bbox_iou
from rpar.ml.seg_train import _xyrgb, predict_labels
from rpar.models import FrameQualityMap, PerceptionResult, SynchronizedFrame
from rpar.perception import HeuristicPerceptionEngine, PerceptionEngine
from rpar.roi import FAR, NEAR, crop_xyxy
from rpar.segdecode import classmap_to_result


FAR_WH = (96, 48)
NEAR_WH = (96, 48)


def stitch_dual_scale(bgr: np.ndarray, predict_fn) -> np.ndarray:
    """Run the same ROI head on far then near; near overwrites the overlap (PER-003)."""
    h, w = bgr.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for box, size in ((FAR, FAR_WH), (NEAR, NEAR_WH)):
        x0, y0, x1, y1 = crop_xyxy(w, h, box)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        labels = predict_fn(small)
        if labels.shape[:2] != (size[1], size[0]):
            labels = labels.reshape(size[1], size[0])
        up = cv2.resize(labels.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y1, x0:x1] = up
    return canvas


class DualScaleSegEngine:
    """Numpy 1×1 class head on far/near RGB+xy features. TFLite is the Android twin."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)

    def _predict_small(self, bgr_small: np.ndarray) -> np.ndarray:
        pred = predict_labels(_xyrgb(bgr_small), self.weights)
        h, w = bgr_small.shape[:2]
        return pred.reshape(h, w)

    def classmap(self, bgr: np.ndarray) -> np.ndarray:
        return stitch_dual_scale(bgr, self._predict_small)

    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        t0 = perf_counter()
        cm = self.classmap(frame.bgr)
        out = classmap_to_result(
            cm,
            frame,
            quality,
            latency_ms=(perf_counter() - t0) * 1000.0,
            input_sizes=[FAR_WH, NEAR_WH],
        )
        return out

    def capability(self) -> dict[str, Any]:
        return {
            "backend": "dual_scale_classmap",
            "outputs": "RoadObservation",
            "tensors_to_ui": False,
            "input": {"far": list(FAR_WH), "near": list(NEAR_WH)},
        }

    def close(self) -> None:
        return None


def merge_perception(primary: PerceptionResult, sidecar: PerceptionResult, score: float | None = None) -> PerceptionResult:
    """Keep typed heuristic instances; take sidecar road/occlusion; never return an empty HUD."""
    road = sidecar.road_polygon if len(sidecar.road_polygon) >= 3 else primary.road_polygon
    occ = sidecar.occluded_polygons if sidecar.occluded_polygons else primary.occluded_polygons
    obs = []
    for o in primary.observations:
        if score is None:
            obs.append(o)
            continue
        prev = o.calibrated_confidence if o.calibrated_confidence is not None else o.model_confidence
        obs.append(replace(o, calibrated_confidence=float(np.clip(0.55 * float(prev) + 0.45 * score, 0, 1))))
    for extra in sidecar.observations:
        if all(bbox_iou(extra.bbox, p.bbox) < 0.30 for p in obs):
            obs.append(extra)
    return PerceptionResult(
        timestamp_ns=primary.timestamp_ns,
        source_frame_id=primary.source_frame_id,
        road_polygon=road,
        occluded_polygons=occ,
        observations=obs,
        backend=primary.backend,
        latency_ms=max(primary.latency_ms, sidecar.latency_ms),
        input_sizes=list(dict.fromkeys(list(primary.input_sizes) + list(sidecar.input_sizes))),
        dual_scale=True,
    )


class HybridPerceptionEngine:
    def __init__(self, primary: PerceptionEngine, sidecar: PerceptionEngine | None) -> None:
        self.primary = primary
        self.sidecar = sidecar

    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        h = self.primary.infer(frame, quality)
        if self.sidecar is None:
            return h
        try:
            s = self.sidecar.infer(frame, quality)
        except Exception:
            return h
        if len(s.road_polygon) < 3 and not s.observations and not s.occluded_polygons:
            return h
        return merge_perception(h, s)

    def capability(self) -> dict[str, Any]:
        cap = dict(self.primary.capability())
        cap["hybrid"] = self.sidecar is not None
        if self.sidecar is not None:
            cap["sidecar"] = self.sidecar.capability()
        return cap

    def close(self) -> None:
        self.primary.close()
        if self.sidecar is not None:
            self.sidecar.close()


def engine_from_seg_weights(cfg, weights: np.ndarray) -> HybridPerceptionEngine:
    return HybridPerceptionEngine(HeuristicPerceptionEngine(cfg), DualScaleSegEngine(weights))
