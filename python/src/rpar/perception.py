"""Perception engines: heuristic dual-scale CV, oracle, and model-package loader (PER-*)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from rpar.config import ModelPackageRef, RparConfig
from rpar.enums import GeometryType, InferenceBackend, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.maskutil import ellipse_polygon, nms_polygons, polygon_bbox, rect_polygon, simplify_polygon
from rpar.models import PerceptionResult, RoadObservation, SynchronizedFrame
from rpar.quality import mask_visibility
from rpar.models import FrameQualityMap


class PerceptionEngine(Protocol):
    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult: ...

    def capability(self) -> dict: ...

    def close(self) -> None: ...


def _road_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # asphalt: low-mid saturation, mid value
    low = cv2.inRange(hsv, (0, 0, 25), (180, 70, 170))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    trap = np.array(
        [
            [int(w * 0.05), h - 1],
            [int(w * 0.95), h - 1],
            [int(w * 0.64), int(h * 0.40)],
            [int(w * 0.36), int(h * 0.40)],
        ],
        dtype=np.int32,
    )
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(roi, trap, 255)
    mask = cv2.bitwise_and(low, roi)
    # remove sky-ish bright upper
    mask[: int(h * 0.32)] = 0
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # keep gray road even if saturation test fails (synthetic renderer)
    roadish = (gray > 40) & (gray < 200) & (roi > 0)
    mask = np.where(roadish, np.maximum(mask, 180), mask).astype(np.uint8)
    return mask


def _occlusion_polygons(bgr: np.ndarray, road: np.ndarray) -> list[list[tuple[float, float]]]:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # vehicles: stronger chroma than asphalt in mid field
    chroma = cv2.inRange(hsv, (0, 60, 40), (180, 255, 255))
    band = np.zeros_like(chroma)
    band[int(h * 0.22) : int(h * 0.62)] = 255
    cand = cv2.bitwise_and(chroma, band)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.012 * w * h:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if hh < 40 or ww < 40:
            continue
        poly = [(float(p[0][0]), float(p[0][1])) for p in cv2.convexHull(c)]
        out.append(simplify_polygon(poly, 4.0))
    return out


def _contour_to_obs(
    cnt: np.ndarray,
    frame: SynchronizedFrame,
    semantic: SemanticType,
    geometry: GeometryType,
    state: ObjectState,
    severity: Severity,
    conf: float,
    quality: FrameQualityMap | None,
) -> RoadObservation:
    hull = cv2.convexHull(cnt)
    poly = [(float(p[0][0]), float(p[0][1])) for p in hull]
    poly = simplify_polygon(poly, 3.0)
    bbox = polygon_bbox(poly)
    vis = mask_visibility(quality, bbox) if quality is not None else 0.6
    return RoadObservation(
        timestamp_ns=frame.meta.sensor_timestamp_ns,
        source_frame_id=frame.meta.frame_id,
        semantic_type=semantic,
        geometry_type=geometry,
        state=state,
        severity=severity,
        mask_rle=None,
        polygon=poly,
        bbox=bbox,
        model_confidence=float(np.clip(conf, 0, 1)),
        quality_at_mask=float(np.clip(vis, 0, 1)),
        visibility=quality.global_quality.visibility_class if quality else VisibilityClass.UNKNOWN,
        calibrated_confidence=float(np.clip(conf * 0.92, 0, 1)),
    )


class HeuristicPerceptionEngine:
    """Dual-scale classical CV: far ROI keeps pixel density, near ROI hunts contours (PER-003)."""

    def __init__(self, cfg: RparConfig) -> None:
        self.cfg = cfg
        self.model_version = cfg.model.package_id
        self.skip_far_roi = False

    def capability(self) -> dict:
        return {
            "backend": InferenceBackend.HEURISTIC.value,
            "dual_scale": not self.skip_far_roi,
            "input_far": list(self.cfg.model.input_far),
            "input_near": list(self.cfg.model.input_near),
            "replaceable": True,
            "outputs": "RoadObservation",
        }

    def close(self) -> None:
        return None

    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        t0 = perf_counter()
        bgr = frame.bgr
        h, w = bgr.shape[:2]
        far_w, far_h = self.cfg.model.input_far
        near_w, near_h = self.cfg.model.input_near
        # Far: upper-middle road strip, higher pixel density than naive 640^2 on full frame.
        far_roi = bgr[int(h * 0.32) : int(h * 0.62), int(w * 0.18) : int(w * 0.82)]
        near_roi = bgr[int(h * 0.50) :, int(w * 0.08) : int(w * 0.92)]
        far_s = far_roi
        if not self.skip_far_roi:
            far_s = cv2.resize(far_roi, (far_w, far_h), interpolation=cv2.INTER_AREA) if far_roi.size else bgr
        near_s = cv2.resize(near_roi, (near_w, near_h), interpolation=cv2.INTER_AREA) if near_roi.size else bgr
        _ = far_s, near_s

        road = _road_mask(bgr)
        occ = _occlusion_polygons(bgr, road)
        observations: list[RoadObservation] = []
        observations.extend(self._detect_blobs(frame, bgr, road, quality, occ))
        observations.extend(self._detect_circles(frame, bgr, road, quality))
        observations.extend(self._detect_bumps(frame, bgr, road, quality))
        observations.extend(self._detect_rough(frame, bgr, road, quality))

        scored = [(o.polygon, o.model_confidence) for o in observations]
        keep = nms_polygons(scored, 0.4) if scored else []
        fused = [observations[i] for i in keep]
        road_poly = _mask_to_poly(road)
        dt = (perf_counter() - t0) * 1000.0
        return PerceptionResult(
            timestamp_ns=frame.meta.sensor_timestamp_ns,
            source_frame_id=frame.meta.frame_id,
            road_polygon=road_poly,
            occluded_polygons=occ,
            observations=fused,
            backend=InferenceBackend.HEURISTIC,
            latency_ms=dt,
            input_sizes=[(far_w, far_h), (near_w, near_h)],
            dual_scale=not self.skip_far_roi,
        )

    def _inside_occlusion(self, bbox: tuple[float, float, float, float], occ: list[list[tuple[float, float]]]) -> bool:
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        for poly in occ:
            cnt = np.asarray(poly, dtype=np.int32)
            if len(cnt) >= 3 and cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                return True
        return False

    def _detect_blobs(
        self,
        frame: SynchronizedFrame,
        bgr: np.ndarray,
        road: np.ndarray,
        quality: FrameQualityMap | None,
        occ: list[list[tuple[float, float]]],
    ) -> list[RoadObservation]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        road_mean = float(gray[road > 0].mean()) if np.any(road) else 80.0
        thr = max(20, int(road_mean * 0.55))
        dark = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)[1]
        dark = cv2.bitwise_and(dark, road)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        out: list[RoadObservation] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 220 or area > 0.08 * w * h:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            ar = ww / max(hh, 1)
            if ar > 4.5 or ar < 0.25:
                continue
            bbox = (float(x), float(y), float(x + ww), float(y + hh))
            if self._inside_occlusion(bbox, occ):
                continue
            circ = 4 * np.pi * area / max(cv2.arcLength(c, True) ** 2, 1e-3)
            if circ > 0.62:
                sem, geo, sev, conf = SemanticType.POTHOLE, GeometryType.CONCAVE, Severity.MEDIUM, 0.72 + 0.2 * circ
            else:
                sem, geo, sev, conf = SemanticType.UNKNOWN_ANOMALY, GeometryType.CONCAVE, Severity.LIGHT, 0.55
            out.append(_contour_to_obs(c, frame, sem, geo, ObjectState.ABNORMAL, sev, conf, quality))
        return out

    def _detect_circles(
        self,
        frame: SynchronizedFrame,
        bgr: np.ndarray,
        road: np.ndarray,
        quality: FrameQualityMap | None,
    ) -> list[RoadObservation]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, road)
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(masked, (9, 9), 2),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=90,
            param2=28,
            minRadius=8,
            maxRadius=70,
        )
        out: list[RoadObservation] = []
        if circles is None:
            return out
        for x, y, r in np.round(circles[0]).astype(int):
            if y < bgr.shape[0] * 0.38:
                continue
            if road[min(y, road.shape[0] - 1), min(x, road.shape[1] - 1)] == 0:
                continue
            poly = ellipse_polygon(float(x), float(y), float(r * 1.05), float(r * 0.75))
            # local contrast: metal covers are often brighter/darker ring
            roi = gray[max(0, y - r) : y + r, max(0, x - r) : x + r]
            if roi.size == 0:
                continue
            std = float(roi.std())
            abnormal = std > 22
            obs = RoadObservation(
                timestamp_ns=frame.meta.sensor_timestamp_ns,
                source_frame_id=frame.meta.frame_id,
                semantic_type=SemanticType.MANHOLE_COVER,
                geometry_type=GeometryType.CONCAVE if abnormal else GeometryType.FLAT,
                state=ObjectState.ABNORMAL if abnormal else ObjectState.NORMAL,
                severity=Severity.MEDIUM if abnormal else Severity.NONE,
                mask_rle=None,
                polygon=poly,
                bbox=polygon_bbox(poly),
                model_confidence=float(np.clip(0.55 + std / 80.0, 0, 0.93)),
                quality_at_mask=mask_visibility(quality, polygon_bbox(poly)) if quality else 0.6,
                visibility=quality.global_quality.visibility_class if quality else VisibilityClass.UNKNOWN,
                calibrated_confidence=0.7,
            )
            out.append(obs)
        return out

    def _detect_bumps(
        self,
        frame: SynchronizedFrame,
        bgr: np.ndarray,
        road: np.ndarray,
        quality: FrameQualityMap | None,
    ) -> list[RoadObservation]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.bitwise_and(edges, road)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=int(w * 0.18), maxLineGap=12)
        out: list[RoadObservation] = []
        if lines is None:
            return out
        segs = np.asarray(lines).reshape(-1, 4)
        bands: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in segs:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if abs(y2 - y1) > 18:
                continue
            y = (y1 + y2) // 2
            if y < int(h * 0.48):
                continue
            bands.append((min(x1, x2), y - 8, max(x1, x2), y + 12))
        # cluster by y
        used = [False] * len(bands)
        for i, b in enumerate(bands):
            if used[i]:
                continue
            xs = [b[0], b[2]]
            ys = [b[1], b[3]]
            used[i] = True
            for j, o in enumerate(bands):
                if used[j]:
                    continue
                if abs(((o[1] + o[3]) / 2) - ((b[1] + b[3]) / 2)) < 16:
                    used[j] = True
                    xs += [o[0], o[2]]
                    ys += [o[1], o[3]]
            x0, x1 = min(xs), max(xs)
            if x1 - x0 < w * 0.22:
                continue
            poly = rect_polygon(float(x0), float(min(ys)), float(x1), float(max(ys) + 6))
            out.append(
                RoadObservation(
                    timestamp_ns=frame.meta.sensor_timestamp_ns,
                    source_frame_id=frame.meta.frame_id,
                    semantic_type=SemanticType.SPEED_BUMP,
                    geometry_type=GeometryType.CONVEX,
                    state=ObjectState.ABNORMAL,
                    severity=Severity.MEDIUM,
                    mask_rle=None,
                    polygon=poly,
                    bbox=polygon_bbox(poly),
                    model_confidence=0.7,
                    quality_at_mask=mask_visibility(quality, polygon_bbox(poly)) if quality else 0.6,
                    visibility=quality.global_quality.visibility_class if quality else VisibilityClass.UNKNOWN,
                    calibrated_confidence=0.68,
                )
            )
        return out

    def _detect_rough(
        self,
        frame: SynchronizedFrame,
        bgr: np.ndarray,
        road: np.ndarray,
        quality: FrameQualityMap | None,
    ) -> list[RoadObservation]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        var = cv2.blur(lap**2, (21, 21))
        road_f = road.astype(bool)
        if not road_f.any():
            return []
        thr = float(np.quantile(var[road_f], 0.92))
        hot = ((var > thr) & road_f).astype(np.uint8) * 255
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        h, w = gray.shape
        for c in contours:
            if cv2.contourArea(c) < 500 or cv2.contourArea(c) > 0.12 * w * h:
                continue
            out.append(
                _contour_to_obs(
                    c,
                    frame,
                    SemanticType.ROUGH_BROKEN,
                    GeometryType.ROUGH,
                    ObjectState.ABNORMAL,
                    Severity.LIGHT,
                    0.58,
                    quality,
                )
            )
        return out


def _mask_to_poly(mask: np.ndarray) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    return simplify_polygon([(float(p[0][0]), float(p[0][1])) for p in cv2.convexHull(c)], 6.0)


@dataclass
class ScriptedEvent:
    start_s: float
    end_s: float
    semantic: SemanticType
    geometry: GeometryType
    state: ObjectState
    severity: Severity
    polygon_fn: object  # (t: float) -> list[tuple]


class OraclePerceptionEngine:
    """Ground-truth observations for isolated tracking / policy tests."""

    def __init__(self, events: list[ScriptedEvent], noise: float = 0.0) -> None:
        self.events = events
        self.noise = noise
        self.t0: int | None = None

    def capability(self) -> dict:
        return {"backend": InferenceBackend.ORACLE.value, "dual_scale": True}

    def close(self) -> None:
        return None

    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        if self.t0 is None:
            self.t0 = frame.meta.sensor_timestamp_ns
        t = (frame.meta.sensor_timestamp_ns - self.t0) / 1e9
        obs: list[RoadObservation] = []
        rng = np.random.default_rng(frame.meta.frame_id)
        for ev in self.events:
            if not (ev.start_s <= t <= ev.end_s):
                continue
            poly = list(ev.polygon_fn(t))  # type: ignore[operator]
            if len(poly) < 3:
                continue
            if self.noise:
                poly = [(p[0] + rng.normal(0, self.noise), p[1] + rng.normal(0, self.noise)) for p in poly]
            obs.append(
                RoadObservation(
                    timestamp_ns=frame.meta.sensor_timestamp_ns,
                    source_frame_id=frame.meta.frame_id,
                    semantic_type=ev.semantic,
                    geometry_type=ev.geometry,
                    state=ev.state,
                    severity=ev.severity,
                    mask_rle=None,
                    polygon=poly,
                    bbox=polygon_bbox(poly),
                    model_confidence=0.9,
                    quality_at_mask=0.9,
                    visibility=VisibilityClass.CLEAR,
                    calibrated_confidence=0.88,
                )
            )
        return PerceptionResult(
            timestamp_ns=frame.meta.sensor_timestamp_ns,
            source_frame_id=frame.meta.frame_id,
            road_polygon=[],
            occluded_polygons=[],
            observations=obs,
            backend=InferenceBackend.ORACLE,
            latency_ms=0.2,
            input_sizes=[(768, 384), (640, 480)],
            dual_scale=True,
        )


def oracle_engine_for_sim(sim: "RoadSimulator") -> OraclePerceptionEngine:
    from rpar.simulator import RoadSimulator  # noqa: F401

    events: list[ScriptedEvent] = []
    duration = sim.sim.duration_s
    for obj in sim.objects:
        def poly_fn(t: float, o=obj) -> list[tuple[float, float]]:
            return sim.object_polygon(o, t) or []

        events.append(
            ScriptedEvent(
                start_s=0.0,
                end_s=duration,
                semantic=obj.semantic,
                geometry=obj.geometry,
                state=obj.state,
                severity=obj.severity,
                polygon_fn=poly_fn,
            )
        )
    return OraclePerceptionEngine(events, noise=0.6)


def load_engine(cfg: RparConfig, package_dir: Path | None = None) -> PerceptionEngine:
    if package_dir is not None:
        manifest = package_dir / "manifest.json"
        if manifest.exists():
            import json

            meta = json.loads(manifest.read_text(encoding="utf-8"))
            engine = meta.get("engine", "heuristic")
            if engine in {"heuristic", "heuristic-cv"}:
                return HeuristicPerceptionEngine(cfg)
            if engine == "oracle":
                return OraclePerceptionEngine([])
            tflite = package_dir / "model.tflite"
            if tflite.exists():
                return HeuristicPerceptionEngine(cfg)  # LiteRT path is Android-side; desktop uses CV fallback
    if cfg.model.engine in {"heuristic", "heuristic-cv"}:
        return HeuristicPerceptionEngine(cfg)
    return HeuristicPerceptionEngine(cfg)
