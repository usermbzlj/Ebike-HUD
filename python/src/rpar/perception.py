"""Perception engines: heuristic dual-scale CV, oracle, and model-package loader (PER-*).

The heuristic engine is classical CV and is documented as such. It really runs on two
scales: the far corridor crop at native pixel density and the near crop downscaled
(PER-003). Its thresholds live in `PerceptionConfig`, never as literals in UI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from rpar.config import PerceptionConfig, RparConfig
from rpar.enums import GeometryType, InferenceBackend, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.maskutil import ellipse_polygon, mask_rle_from_polygon, nms_polygons, polygon_bbox, rect_polygon, simplify_polygon
from rpar.models import FrameQualityMap, PerceptionResult, RoadObservation, SynchronizedFrame
from rpar.quality import mask_visibility
from rpar.roi import crop_xyxy


class PerceptionEngine(Protocol):
    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult: ...

    def capability(self) -> dict: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------- road support
def road_trapezoid(h: int, w: int) -> np.ndarray:
    roi = np.zeros((h, w), dtype=np.uint8)
    trap = np.array(
        [
            [int(w * 0.05), h - 1],
            [int(w * 0.95), h - 1],
            [int(w * 0.64), int(h * 0.40)],
            [int(w * 0.36), int(h * 0.40)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(roi, trap, 255)
    return roi


def _road_region(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (region, asphalt).

    `asphalt` marks pixels that look like plain road surface (used for luma statistics).
    `region` is the filled support of the largest asphalt component: dark potholes and
    bright markings *inside* the road stay inside the region instead of punching holes.
    """
    h, w = bgr.shape[:2]
    roi = road_trapezoid(h, w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    low_sat = cv2.inRange(hsv, (0, 0, 20), (180, 80, 235))
    asphalt = cv2.bitwise_and(low_sat, roi)
    asphalt[: int(h * 0.32)] = 0
    asphalt = cv2.medianBlur(asphalt, 5)
    asphalt = cv2.morphologyEx(asphalt, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(asphalt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(asphalt)
    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) >= 0.02 * w * h:
            cv2.drawContours(region, [big], -1, 255, -1)
    if not region.any():
        region = roi.copy()
        region[: int(h * 0.32)] = 0
    # asphalt statistics exclude very dark / very bright pixels (holes, markings, glints)
    plain = (gray > 30) & (gray < 215)
    asphalt = np.where(plain, asphalt, 0).astype(np.uint8)
    return region, asphalt


def _luma(gray: np.ndarray, mask: np.ndarray, fallback: float = 80.0) -> float:
    sel = mask.astype(bool)
    if not sel.any():
        return fallback
    return float(gray[sel].mean())


_BLOCKING_VISIBILITY = {
    VisibilityClass.GLARE,
    VisibilityClass.UNDEREXPOSED,
    VisibilityClass.OCCLUDED,
    VisibilityClass.LENS_DROP,
}


def night_score(bgr: np.ndarray, gray: np.ndarray, asphalt: np.ndarray, cfg: PerceptionConfig) -> float:
    """0 = daylight, >= night_score_threshold = artificial light / dusk.

    Auto exposure keeps night asphalt almost as bright as day asphalt, so luma alone is not
    enough. Streetlights add a strong colour cast (high saturation) and the sky band goes dark;
    the three cues are summed.
    """
    h = gray.shape[0]
    luma = _luma(gray, asphalt)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    upper = float(gray[: max(1, int(h * 0.25))].mean())
    luma_term = float(np.clip((cfg.night_road_luma + 30.0 - luma) / 60.0, 0.0, 1.0))
    sat_term = float(np.clip((sat - 25.0) / 40.0, 0.0, 1.0))
    sky_term = float(np.clip((110.0 - upper) / 60.0, 0.0, 1.0))
    return luma_term + sat_term + 0.5 * sky_term


def scene_is_dark(bgr: np.ndarray, gray: np.ndarray, asphalt: np.ndarray, cfg: PerceptionConfig) -> bool:
    return night_score(bgr, gray, asphalt, cfg) >= cfg.night_score_threshold


def _low_light(
    bgr: np.ndarray, gray: np.ndarray, asphalt: np.ndarray, quality: FrameQualityMap | None, cfg: PerceptionConfig
) -> bool:
    """Night sensor noise and glare must not spawn instance labels."""
    if scene_is_dark(bgr, gray, asphalt, cfg):
        return True
    vis = quality.global_quality.visibility_class if quality else VisibilityClass.UNKNOWN
    return vis in _BLOCKING_VISIBILITY


def manhole_is_sunken(gray: np.ndarray, bbox: tuple[float, float, float, float], delta: float) -> bool:
    """A cover whose disc centre is darker than its rim by `delta` reads as settled."""
    h_img, w_img = gray.shape[:2]
    x0 = int(np.clip(np.floor(bbox[0]), 0, w_img - 1))
    y0 = int(np.clip(np.floor(bbox[1]), 0, h_img - 1))
    x1 = int(np.clip(np.ceil(bbox[2]), x0 + 1, w_img))
    y1 = int(np.clip(np.ceil(bbox[3]), y0 + 1, h_img))
    crop = gray[y0:y1, x0:x1]
    if crop.size < 16:
        return False
    hh, ww = crop.shape[:2]
    cy, cx = hh / 2.0, ww / 2.0
    inner_r = max(2.0, 0.28 * min(hh, ww))
    yy, xx = np.ogrid[:hh, :ww]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    inner = dist2 <= inner_r**2
    ring = (dist2 <= (inner_r * 2.05) ** 2) & ~inner
    if not inner.any() or not ring.any():
        return False
    return float(crop[inner].mean()) < float(crop[ring].mean()) - float(delta)


def _occlusion_polygons(bgr: np.ndarray, region: np.ndarray, cfg: PerceptionConfig) -> list[list[tuple[float, float]]]:
    """Vehicles standing on the road: chromatic or very dark blobs whose base sits in the road region."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = cv2.inRange(hsv, (0, 70, 40), (180, 255, 255))
    dark = cv2.inRange(gray, 0, 28)
    cand = cv2.bitwise_or(chroma, dark)
    band = np.zeros_like(cand)
    band[int(h * 0.22) : int(h * 0.72)] = 255
    cand = cv2.bitwise_and(cand, band)
    support = cv2.dilate(region, np.ones((25, 25), np.uint8))
    cand = cv2.bitwise_and(cand, support)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[tuple[float, float]]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.occluder_min_frac * w * h:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if hh < 40 or ww < 40:
            continue
        aspect = ww / max(hh, 1)
        if aspect < 0.4 or aspect > 3.2:
            continue
        base_y = min(y + hh, h - 1)
        base_x = min(max(x + ww // 2, 0), w - 1)
        if region[base_y, base_x] == 0 and region[min(base_y + 10, h - 1), base_x] == 0:
            continue
        poly = [(float(p[0][0]), float(p[0][1])) for p in cv2.convexHull(c)]
        out.append(simplify_polygon(poly, 4.0))
    return out


def _contour_to_obs(
    poly: list[tuple[float, float]],
    frame: SynchronizedFrame,
    semantic: SemanticType,
    geometry: GeometryType,
    state: ObjectState,
    severity: Severity,
    conf: float,
    quality: FrameQualityMap | None,
) -> RoadObservation:
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
        # Heuristic scores are not calibrated probabilities; expose that honestly.
        calibrated_confidence=None,
    )


def _hull_poly(cnt: np.ndarray, eps: float = 3.0) -> list[tuple[float, float]]:
    hull = cv2.convexHull(cnt)
    return simplify_polygon([(float(p[0][0]), float(p[0][1])) for p in hull], eps)


def _lift(poly: list[tuple[float, float]], ox: float, oy: float, inv_scale: float) -> list[tuple[float, float]]:
    return [(x * inv_scale + ox, y * inv_scale + oy) for x, y in poly]


@dataclass
class _Scale:
    """One processing scale: crop offset and the downscale factor applied to the crop."""

    name: str
    ox: int
    oy: int
    scale: float  # crop pixels per processed pixel (>= 1 means downscaled)
    bgr: np.ndarray
    region: np.ndarray
    asphalt: np.ndarray

    @property
    def area_scale(self) -> float:
        return 1.0 / (self.scale * self.scale)


class HeuristicPerceptionEngine:
    """Dual-scale classical CV: far ROI keeps native density, near ROI is downscaled (PER-003)."""

    def __init__(self, cfg: RparConfig) -> None:
        self.cfg = cfg
        self.p: PerceptionConfig = cfg.perception
        self.model_version = cfg.model.package_id
        self.skip_far_roi = False
        self._last_sizes: list[tuple[int, int]] = []

    def capability(self) -> dict:
        return {
            "backend": InferenceBackend.HEURISTIC.value,
            "dual_scale": not self.skip_far_roi,
            "far_roi": list(self.p.far_roi),
            "near_roi": list(self.p.near_roi),
            "near_max_width": self.p.near_max_width,
            "replaceable": True,
            "outputs": "RoadObservation",
            "calibrated": False,
        }

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------ scales
    def _scales(self, bgr: np.ndarray, region: np.ndarray, asphalt: np.ndarray) -> list[_Scale]:
        h, w = bgr.shape[:2]
        out: list[_Scale] = []
        boxes = [] if self.skip_far_roi else [("far", self.p.far_roi, None)]
        boxes.append(("near", self.p.near_roi, self.p.near_max_width))
        for name, box, max_w in boxes:
            x0, y0, x1, y1 = crop_xyxy(w, h, box)
            crop = bgr[y0:y1, x0:x1]
            reg = region[y0:y1, x0:x1]
            asp = asphalt[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            s = 1.0
            if max_w and crop.shape[1] > max_w:
                s = crop.shape[1] / float(max_w)
                size = (max_w, max(1, int(round(crop.shape[0] / s))))
                crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
                reg = cv2.resize(reg, size, interpolation=cv2.INTER_NEAREST)
                asp = cv2.resize(asp, size, interpolation=cv2.INTER_NEAREST)
            out.append(_Scale(name, x0, y0, s, crop, reg, asp))
        return out

    # ------------------------------------------------------------------ infer
    def infer(self, frame: SynchronizedFrame, quality: FrameQualityMap | None = None) -> PerceptionResult:
        t0 = perf_counter()
        bgr = frame.bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        region, asphalt = _road_region(bgr)
        occ = _occlusion_polygons(bgr, region, self.p)
        night = _low_light(bgr, gray, asphalt, quality, self.p)
        scales = self._scales(bgr, region, asphalt)
        self._last_sizes = [(s.bgr.shape[1], s.bgr.shape[0]) for s in scales]

        observations: list[RoadObservation] = []
        for sc in scales:
            g = cv2.cvtColor(sc.bgr, cv2.COLOR_BGR2GRAY)
            local: list[RoadObservation] = []
            local.extend(self._detect_blobs(frame, g, sc, night, quality))
            if not night:
                local.extend(self._detect_circles(frame, g, sc, quality))
            inv = sc.scale
            for o in local:
                poly = _lift(o.polygon, sc.ox, sc.oy, inv)
                o.polygon = poly
                o.bbox = polygon_bbox(poly)
                if self._inside_occlusion(o.bbox, occ):
                    continue
                observations.append(o)
        if not night:
            observations.extend(self._detect_bumps(frame, gray, region, quality))
            observations.extend(self._detect_joints(frame, gray, region, quality))
            observations.extend(self._detect_rough(frame, gray, region, asphalt, quality))

        scored = [(o.polygon, o.model_confidence) for o in observations]
        keep = nms_polygons(scored, 0.4) if scored else []
        fused = [observations[i] for i in keep]
        info = self._detect_puddles(frame, gray, region, asphalt, quality, occ)
        if not night:
            info += self._detect_gravel(frame, gray, region, quality, occ)
        if info:
            info_keep = nms_polygons([(o.polygon, o.model_confidence) for o in info], 0.4)
            fused.extend(info[i] for i in info_keep)
        for o in fused:
            if o.mask_rle is None:
                o.mask_rle = mask_rle_from_polygon(o.polygon)
        road_poly = _mask_to_poly(region)
        dt = (perf_counter() - t0) * 1000.0
        return PerceptionResult(
            timestamp_ns=frame.meta.sensor_timestamp_ns,
            source_frame_id=frame.meta.frame_id,
            road_polygon=road_poly,
            occluded_polygons=occ,
            observations=fused,
            backend=InferenceBackend.HEURISTIC,
            latency_ms=dt,
            input_sizes=list(self._last_sizes),
            dual_scale=not self.skip_far_roi,
        )

    @staticmethod
    def _inside_occlusion(bbox: tuple[float, float, float, float], occ: list[list[tuple[float, float]]]) -> bool:
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        for poly in occ:
            cnt = np.asarray(poly, dtype=np.int32)
            if len(cnt) >= 3 and cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                return True
        return False

    # ------------------------------------------------------------------ detectors (per scale)
    def _detect_blobs(
        self,
        frame: SynchronizedFrame,
        gray: np.ndarray,
        sc: _Scale,
        night: bool,
        quality: FrameQualityMap | None,
    ) -> list[RoadObservation]:
        p = self.p
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        road_mean = _luma(gray, sc.asphalt)
        thr = max(18 if night else 20, int(road_mean * (0.42 if night else 0.58)))
        dark = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)[1]
        dark = cv2.bitwise_and(dark, sc.region)
        k = 7 if night else 5
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        out: list[RoadObservation] = []
        min_area = (p.pothole_min_area_night_px if night else p.pothole_min_area_px) * sc.area_scale
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > 0.05 * w * h:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            ar = ww / max(hh, 1)
            if ar > 3.2 or ar < 0.35:
                continue
            # skip blobs glued to the crop border: they are usually shadows or the road edge
            if x <= 1 or y <= 1 or x + ww >= w - 1:
                continue
            circ = 4 * np.pi * area / max(cv2.arcLength(c, True) ** 2, 1e-3)
            pad = 8
            neigh = gray[max(0, y - pad) : y + hh + pad, max(0, x - pad) : x + ww + pad]
            core = gray[y : y + hh, x : x + ww]
            if core.size == 0 or neigh.size == 0:
                continue
            contrast = float(neigh.mean() - core.mean())
            if contrast < p.anomaly_contrast:
                continue
            need_contrast = p.pothole_contrast_night if night else p.pothole_contrast
            if night and (circ < 0.72 or contrast < need_contrast):
                continue
            if circ > p.pothole_circularity and contrast >= need_contrast:
                # Only a dark, shadowed interior reads as a hole; a fresh repair patch is merely darker.
                sev = Severity.HEAVY if area > 4.0 * min_area and contrast > 1.6 * need_contrast else Severity.MEDIUM
                conf = float(np.clip(0.45 + 0.25 * circ + 0.2 * min(1.0, contrast / (2.0 * need_contrast)), 0, 0.92))
                sem, geo = SemanticType.POTHOLE, GeometryType.CONCAVE
            elif night:
                continue
            else:
                sem, geo, sev, conf = SemanticType.UNKNOWN_ANOMALY, GeometryType.CONCAVE, Severity.LIGHT, 0.5
            out.append(_contour_to_obs(_hull_poly(c), frame, sem, geo, ObjectState.ABNORMAL, sev, conf, quality))
        return out

    def _detect_circles(
        self,
        frame: SynchronizedFrame,
        gray: np.ndarray,
        sc: _Scale,
        quality: FrameQualityMap | None,
    ) -> list[RoadObservation]:
        masked = cv2.bitwise_and(gray, sc.region)
        h, w = gray.shape
        min_r = max(6, int(8 * (w / 1920.0) * 1.5))
        max_r = max(min_r + 4, int(70 * (w / 1920.0) * 1.5))
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(masked, (9, 9), 2),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(w * 0.03)),
            param1=90,
            param2=30,
            minRadius=min_r,
            maxRadius=max_r,
        )
        out: list[RoadObservation] = []
        if circles is None:
            return out
        for x, y, r in np.round(circles[0]).astype(int):
            if sc.region[min(max(y, 0), h - 1), min(max(x, 0), w - 1)] == 0:
                continue
            poly = ellipse_polygon(float(x), float(y), float(r * 1.05), float(r * 0.75))
            bbox = polygon_bbox(poly)
            roi = gray[max(0, y - r) : y + r, max(0, x - r) : x + r]
            if roi.size == 0:
                continue
            sunken = manhole_is_sunken(gray, bbox, self.p.manhole_sunken_delta)
            std = float(roi.std())
            out.append(
                _contour_to_obs(
                    poly,
                    frame,
                    SemanticType.MANHOLE_COVER,
                    GeometryType.CONCAVE if sunken else GeometryType.FLAT,
                    ObjectState.ABNORMAL if sunken else ObjectState.NORMAL,
                    Severity.MEDIUM if sunken else Severity.NONE,
                    float(np.clip(0.5 + std / 90.0, 0, 0.9)),
                    quality,
                )
            )
        return out

    # ------------------------------------------------------------------ detectors (full frame)
    def _transverse_bands(
        self,
        gray: np.ndarray,
        region: np.ndarray,
        *,
        canny: tuple[int, int],
        min_len_frac: float,
        max_slope_px: int,
        cluster_px: int,
        half_h: tuple[int, int],
        min_span_frac: float,
        y_min_frac: float,
    ) -> list[tuple[int, int, int, int]]:
        h, w = gray.shape
        edges = cv2.Canny(gray, canny[0], canny[1])
        edges = cv2.bitwise_and(edges, region)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=int(w * min_len_frac), maxLineGap=12
        )
        if lines is None:
            return []
        segs = np.asarray(lines).reshape(-1, 4)
        bands: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in segs:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if abs(y2 - y1) > max_slope_px:
                continue
            y = (y1 + y2) // 2
            if y < int(h * y_min_frac):
                continue
            bands.append((min(x1, x2), y - half_h[0], max(x1, x2), y + half_h[1]))
        used = [False] * len(bands)
        out: list[tuple[int, int, int, int]] = []
        for i, b in enumerate(bands):
            if used[i]:
                continue
            xs = [b[0], b[2]]
            ys = [b[1], b[3]]
            used[i] = True
            for j, o in enumerate(bands):
                if used[j]:
                    continue
                if abs(((o[1] + o[3]) / 2) - ((b[1] + b[3]) / 2)) < cluster_px:
                    used[j] = True
                    xs += [o[0], o[2]]
                    ys += [o[1], o[3]]
            x0, x1 = min(xs), max(xs)
            if x1 - x0 < w * min_span_frac:
                continue
            out.append((x0, min(ys), x1, max(ys)))
        return out

    def _detect_bumps(
        self, frame: SynchronizedFrame, gray: np.ndarray, region: np.ndarray, quality: FrameQualityMap | None
    ) -> list[RoadObservation]:
        bands = self._transverse_bands(
            gray, region, canny=(40, 120), min_len_frac=0.18, max_slope_px=18, cluster_px=16,
            half_h=(8, 12), min_span_frac=0.22, y_min_frac=0.48,
        )
        out: list[RoadObservation] = []
        for x0, y0, x1, y1 in bands:
            # A speed bump is a raised band: its top edge is brighter than the asphalt above it
            # and its bottom edge darker (shadow). Flat markings are bright on both edges.
            top = gray[max(0, y0 - 6) : max(1, y0), x0:x1]
            core = gray[y0:y1, x0:x1]
            below = gray[y1 : min(gray.shape[0], y1 + 8), x0:x1]
            if core.size == 0 or top.size == 0 or below.size == 0:
                continue
            if float(core.mean()) - float(below.mean()) < 6.0:
                continue
            poly = rect_polygon(float(x0), float(y0), float(x1), float(y1 + 6))
            out.append(
                _contour_to_obs(
                    poly, frame, SemanticType.SPEED_BUMP, GeometryType.CONVEX, ObjectState.ABNORMAL,
                    Severity.MEDIUM, 0.6, quality,
                )
            )
        return out

    def _detect_joints(
        self, frame: SynchronizedFrame, gray: np.ndarray, region: np.ndarray, quality: FrameQualityMap | None
    ) -> list[RoadObservation]:
        """PER-007: thin transverse seams, not convex speed bumps."""
        bands = self._transverse_bands(
            gray, region, canny=(50, 140), min_len_frac=0.14, max_slope_px=6, cluster_px=6,
            half_h=(2, 3), min_span_frac=0.16, y_min_frac=0.50,
        )
        out: list[RoadObservation] = []
        for x0, y0, x1, y1 in bands:
            if y1 - y0 > 10:
                continue
            poly = rect_polygon(float(x0), float(y0), float(x1), float(y1))
            out.append(
                _contour_to_obs(
                    poly, frame, SemanticType.ROAD_JOINT, GeometryType.FLAT, ObjectState.NORMAL,
                    Severity.NONE, 0.6, quality,
                )
            )
        return out

    def _detect_rough(
        self,
        frame: SynchronizedFrame,
        gray: np.ndarray,
        region: np.ndarray,
        asphalt: np.ndarray,
        quality: FrameQualityMap | None,
    ) -> list[RoadObservation]:
        """Only local texture outliers. Never label the roughest 8% of a smooth road (night FP storm)."""
        road_f = region.astype(bool)
        if not road_f.any():
            return []
        if _luma(gray, asphalt) < self.p.night_road_luma + 10.0:
            return []
        vis = quality.global_quality.visibility_class if quality else VisibilityClass.UNKNOWN
        if vis in _BLOCKING_VISIBILITY or vis == VisibilityClass.BLUR:
            return []
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        var = cv2.blur(lap**2, (21, 21))
        vals = var[road_f]
        med = float(np.median(vals))
        p99 = float(np.quantile(vals, 0.99))
        if med < 1e-3 or p99 < max(28.0, med * 6.0):
            return []
        hot = ((var > max(p99 * 0.92, med * 7.0)) & road_f).astype(np.uint8) * 255
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        scored: list[tuple[float, np.ndarray]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 900 or area > 0.06 * w * h:
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [c], -1, 1, -1)
            peak = float(var[mask.astype(bool)].mean()) if mask.any() else 0.0
            ratio = peak / max(med, 1e-3)
            if ratio < 7.0:
                continue
            scored.append((ratio, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for ratio, c in scored[:2]:
            conf = float(np.clip(0.55 + 0.04 * (ratio - 7.0), 0.55, 0.85))
            out.append(
                _contour_to_obs(
                    _hull_poly(c), frame, SemanticType.ROUGH_BROKEN, GeometryType.ROUGH,
                    ObjectState.ABNORMAL, Severity.LIGHT, conf, quality,
                )
            )
        return out

    def _detect_puddles(
        self,
        frame: SynchronizedFrame,
        gray: np.ndarray,
        region: np.ndarray,
        asphalt: np.ndarray,
        quality: FrameQualityMap | None,
        occ: list[list[tuple[float, float]]],
    ) -> list[RoadObservation]:
        """PER-013: bright wet reflections as info-layer, never geometric risk."""
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        road_mean = _luma(gray, asphalt)
        thr = min(245, max(int(road_mean + 45), int(road_mean * 1.35 + 20)))
        near = cv2.dilate(region, np.ones((17, 17), np.uint8))
        bright = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)[1]
        bright = cv2.bitwise_and(bright, near)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        out: list[RoadObservation] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 180 or area > 0.06 * w * h:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            if y < h * 0.40:
                continue
            ar = ww / max(hh, 1)
            if ar > 3.8 or ar < 0.28:
                continue
            bbox = (float(x), float(y), float(x + ww), float(y + hh))
            if self._inside_occlusion(bbox, occ):
                continue
            circ = 4 * np.pi * area / max(cv2.arcLength(c, True) ** 2, 1e-3)
            if circ < 0.38:
                continue
            out.append(
                _contour_to_obs(
                    _hull_poly(c), frame, SemanticType.PUDDLE, GeometryType.FLAT, ObjectState.UNKNOWN,
                    Severity.NONE, 0.5 + 0.25 * min(circ, 1.0), quality,
                )
            )
        return out

    def _detect_gravel(
        self,
        frame: SynchronizedFrame,
        gray: np.ndarray,
        region: np.ndarray,
        quality: FrameQualityMap | None,
        occ: list[list[tuple[float, float]]],
    ) -> list[RoadObservation]:
        """PER-013: scattered high-texture debris as info-layer."""
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        tex = cv2.absdiff(gray, blur)
        road_f = region > 0
        if not road_f.any():
            return []
        mid = (gray > 50) & (gray < 185) & road_f & (tex > 16)
        hot = mid.astype(np.uint8) * 255
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        out: list[RoadObservation] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 280 or area > 0.07 * w * h:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            bbox = (float(x), float(y), float(x + ww), float(y + hh))
            if self._inside_occlusion(bbox, occ):
                continue
            circ = 4 * np.pi * area / max(cv2.arcLength(c, True) ** 2, 1e-3)
            if circ > 0.72:
                continue
            out.append(
                _contour_to_obs(
                    _hull_poly(c), frame, SemanticType.GRAVEL, GeometryType.ROUGH, ObjectState.UNKNOWN,
                    Severity.NONE, 0.5, quality,
                )
            )
        return out


def _mask_to_poly(mask: np.ndarray) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    return simplify_polygon([(float(p[0][0]), float(p[0][1])) for p in cv2.convexHull(c)], 6.0)


# ---------------------------------------------------------------------------- oracle
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

    def __init__(self, events: list[ScriptedEvent], noise: float = 0.0, sim: object | None = None) -> None:
        self.events = events
        self.noise = noise
        self.t0: int | None = None
        self.sim = sim

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
        for o in obs:
            if o.mask_rle is None:
                o.mask_rle = mask_rle_from_polygon(o.polygon)
        occ: list[list[tuple[float, float]]] = []
        if self.sim is not None and hasattr(self.sim, "classmap_at"):
            from rpar.segdecode import CLASS_OCC, polygons_for_label

            cm = self.sim.classmap_at(frame.meta.frame_id)
            occ = polygons_for_label(cm, CLASS_OCC, max(24.0, 0.0004 * frame.meta.width * frame.meta.height))
        return PerceptionResult(
            timestamp_ns=frame.meta.sensor_timestamp_ns,
            source_frame_id=frame.meta.frame_id,
            road_polygon=[],
            occluded_polygons=occ,
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
    return OraclePerceptionEngine(events, noise=0.6, sim=sim)


# ---------------------------------------------------------------------------- loaders
def load_engine(cfg: RparConfig, package_dir: Path | None = None) -> PerceptionEngine:
    heuristic = HeuristicPerceptionEngine(cfg)
    if package_dir is not None:
        manifest = package_dir / "manifest.json"
        weights_path = package_dir / "seg_weights.json"
        if weights_path.exists():
            import json

            from rpar.segengine import DualScaleSegEngine, HybridPerceptionEngine

            meta = json.loads(weights_path.read_text(encoding="utf-8"))
            return HybridPerceptionEngine(heuristic, DualScaleSegEngine(np.asarray(meta["weights"])))
        if manifest.exists():
            import json

            meta = json.loads(manifest.read_text(encoding="utf-8"))
            engine = meta.get("engine", "heuristic")
            if engine == "oracle":
                return OraclePerceptionEngine([])
            if engine in {"classmap", "litert-classmap"}:
                from rpar.segengine import DualScaleSegEngine, HybridPerceptionEngine

                wp = package_dir / "seg_weights.json"
                if wp.exists():
                    body = json.loads(wp.read_text(encoding="utf-8"))
                    return HybridPerceptionEngine(heuristic, DualScaleSegEngine(np.asarray(body["weights"])))
            if engine in {"heuristic", "heuristic-cv"}:
                return heuristic
            if engine in {"yolopv2", "yolop"}:
                return load_field_engine(cfg, prefer_yolop=True)
            if (package_dir / "YOLOPv2.onnx").exists():
                return load_field_engine(cfg, prefer_yolop=True, weights=package_dir / "YOLOPv2.onnx")
    if cfg.model.engine in {"yolopv2", "yolop"}:
        return load_field_engine(cfg, prefer_yolop=True)
    return heuristic


def load_field_engine(
    cfg: RparConfig,
    *,
    prefer_yolop: bool = True,
    prefer_bump: bool = False,
    weights: Path | None = None,
) -> PerceptionEngine:
    """Heuristic + optional YOLOPv2 road sidecar + optional YOLO-World bump net."""
    heuristic = HeuristicPerceptionEngine(cfg)
    engine: PerceptionEngine = heuristic
    if prefer_yolop:
        from rpar.ml.yolopv2 import Yolopv2Engine, weights_available
        from rpar.segengine import DualScaleSegEngine, HybridPerceptionEngine

        path = Path(weights) if weights else None
        if weights_available(path):
            engine = HybridPerceptionEngine(heuristic, Yolopv2Engine(path))
        else:
            for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
                cand = parent / "models" / "roadseg-field-0.1.0" / "seg_weights.json"
                if cand.is_file():
                    import json

                    meta = json.loads(cand.read_text(encoding="utf-8"))
                    w = meta.get("weights")
                    if w:
                        engine = HybridPerceptionEngine(heuristic, DualScaleSegEngine(np.asarray(w)))
                    break
    if prefer_bump:
        from rpar.ml.world_bump import try_load_world_bump
        from rpar.segengine import BumpHybridEngine

        bump = try_load_world_bump()
        if bump is not None:
            engine = BumpHybridEngine(engine, bump)
    return engine
