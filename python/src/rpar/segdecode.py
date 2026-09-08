"""Convert dense class maps to RoadObservation (PER-014). Tensors never leak to UI."""

from __future__ import annotations

import cv2
import numpy as np

from rpar.enums import GeometryType, InferenceBackend, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.maskutil import mask_rle_from_polygon, polygon_area, polygon_bbox, simplify_polygon
from rpar.models import FrameQualityMap, PerceptionResult, RoadObservation, SynchronizedFrame
from rpar.quality import mask_visibility

CLASS_BG = 0
CLASS_ROAD = 1
CLASS_ANOMALY = 2
CLASS_OCC = 3


def _argmax_map(classmap: np.ndarray) -> np.ndarray:
    if classmap.ndim == 3:
        return np.argmax(classmap, axis=-1).astype(np.uint8)
    return classmap.astype(np.uint8)


def polygons_for_label(classmap: np.ndarray, label: int, min_area: float) -> list[list[tuple[float, float]]]:
    binary = (classmap == label).astype(np.uint8) * 255
    if not binary.any():
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[tuple[float, float]]] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        hull = cv2.convexHull(c)
        poly = [(float(p[0][0]), float(p[0][1])) for p in hull]
        poly = simplify_polygon(poly, 3.0)
        if len(poly) >= 3:
            out.append(poly)
    return out


def classmap_to_result(
    classmap: np.ndarray,
    frame: SynchronizedFrame,
    quality: FrameQualityMap | None = None,
    *,
    latency_ms: float = 0.0,
    input_sizes: list[tuple[int, int]] | None = None,
    sx: float = 1.0,
    sy: float = 1.0,
) -> PerceptionResult:
    labels = _argmax_map(classmap)
    h, w = labels.shape
    min_area = max(24.0, 0.0004 * w * h)
    road_polys = polygons_for_label(labels, CLASS_ROAD, min_area)
    occ = polygons_for_label(labels, CLASS_OCC, min_area)
    anomaly = polygons_for_label(labels, CLASS_ANOMALY, min_area)

    def _scale(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * sx, y * sy) for x, y in poly]

    road_poly = _scale(max(road_polys, key=polygon_area, default=[]))
    occ_s = [_scale(p) for p in occ]
    obs: list[RoadObservation] = []
    vis = quality.global_quality.visibility_class if quality else VisibilityClass.UNKNOWN
    for poly in anomaly:
        sp = _scale(poly)
        bbox = polygon_bbox(sp)
        qv = mask_visibility(quality, bbox) if quality is not None else 0.6
        obs.append(
            RoadObservation(
                timestamp_ns=frame.meta.sensor_timestamp_ns,
                source_frame_id=frame.meta.frame_id,
                semantic_type=SemanticType.UNKNOWN_ANOMALY,
                geometry_type=GeometryType.UNKNOWN,
                state=ObjectState.UNKNOWN,
                severity=Severity.UNKNOWN,
                mask_rle=mask_rle_from_polygon(sp),
                polygon=sp,
                bbox=bbox,
                model_confidence=0.7,
                quality_at_mask=float(np.clip(qv, 0, 1)),
                visibility=vis,
                calibrated_confidence=0.64,
            )
        )
    sizes = input_sizes or [(w, h)]
    return PerceptionResult(
        timestamp_ns=frame.meta.sensor_timestamp_ns,
        source_frame_id=frame.meta.frame_id,
        road_polygon=road_poly,
        occluded_polygons=occ_s,
        observations=obs,
        backend=InferenceBackend.CPU,
        latency_ms=latency_ms,
        input_sizes=sizes,
        dual_scale=True,
    )


class ClassmapEngine:
    """PER-014: dense labels → RoadObservation. Never returns raw tensors to UI."""

    def __init__(self, classmap_fn) -> None:
        self.classmap_fn = classmap_fn

    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        return classmap_to_result(self.classmap_fn(frame), frame, quality)

    def capability(self) -> dict:
        return {"backend": "classmap", "outputs": "RoadObservation", "tensors_to_ui": False}

    def close(self) -> None:
        return None
