"""Frame quality, local visibility and infer-frame scheduler (QUAL-001..010).

Sharpness is exposure-normalised so a dark but sharp night frame is still usable; the
underexposure score handles darkness separately. Lens contamination is detected
temporally (`LensContaminationDetector`): drops on the lens stay still while the
road scene moves.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from rpar.config import QualityConfig
from rpar.enums import PerceptionStatus, VisibilityClass
from rpar.models import FrameQuality, FrameQualityMap, QualityTile, SynchronizedFrame

ANALYSIS_WIDTH = 480


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_normalized_lap(lap: float, mean_gray: float) -> float:
    """Laplacian variance scales with contrast^2; normalise to a mid-grey exposure."""
    ref = float(np.clip(mean_gray, 24.0, 255.0))
    return float(lap * (128.0 / ref) ** 2)


def _motion_blur_score(gray: np.ndarray, lap_adj: float) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    if mag.mean() < 1e-3:
        return 1.0
    anisotropy = abs(np.mean(np.abs(gx)) - np.mean(np.abs(gy))) / (np.mean(mag) + 1e-6)
    spread = float(mag.std() / (mag.mean() + 1e-6))
    # High anisotropy + low Laplacian typically means directional smear.
    blur = np.clip((18.0 - lap_adj) / 18.0, 0.0, 1.0)
    return float(np.clip(0.55 * blur + 0.45 * np.clip(anisotropy * 1.8, 0, 1) * (1.0 - np.clip(spread / 2.0, 0, 1)), 0, 1))


def headlight_cone_mask(h: int, w: int) -> np.ndarray:
    """QUAL-008: near-field bike-headlight footprint on the road plane."""
    mask = np.zeros((h, w), dtype=np.uint8)
    trap = np.array(
        [
            [int(w * 0.28), h - 1],
            [int(w * 0.72), h - 1],
            [int(w * 0.58), int(h * 0.52)],
            [int(w * 0.42), int(h * 0.52)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, trap, 1)
    return mask


def fit_headlight_mean(gray: np.ndarray) -> float:
    cone = headlight_cone_mask(gray.shape[0], gray.shape[1])
    vals = gray[cone > 0]
    return float(vals.mean()) if vals.size else 0.0


def _glare_score(bgr: np.ndarray, headlight_mean: float | None = None) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    bright = (v > 245) & (sat < 80)
    # Upper half glare (sun / headlights) is more diagnostic than road sparkle.
    h = bgr.shape[0]
    upper = bright[: h // 2].mean() if h > 2 else bright.mean()
    score = float(np.clip(upper * 3.5 + bright.mean() * 1.5, 0, 1))
    if headlight_mean is None or headlight_mean <= 1.0:
        return score
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cone = headlight_cone_mask(gray.shape[0], gray.shape[1]) > 0
    expected = cone & (gray > headlight_mean * 0.65) & (gray < headlight_mean * 1.40)
    extra = bright & ~expected
    return float(np.clip(extra.mean() * 4.0 + (upper * 2.0), 0, 1))


def _exposure_scores(gray: np.ndarray) -> tuple[float, float]:
    mean = float(gray.mean()) / 255.0
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    under = float(np.clip(hist[:3].sum() * 1.4 + (0.22 - mean) * 2.0, 0, 1))
    over = float(np.clip(hist[-2:].sum() * 1.6 + (mean - 0.78) * 2.2, 0, 1))
    return under, over


def _defocus_score(gray: np.ndarray, lap: float) -> float:
    # High-frequency energy concentrated vs overall contrast.
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(edges.mean()) / 255.0
    if lap < 8 and edge_density < 0.04:
        return float(np.clip((10.0 - lap) / 10.0, 0, 1))
    return float(np.clip((14.0 - lap) / 28.0, 0, 1) * (1.0 - np.clip(edge_density * 6, 0, 0.7)))


def _road_roi_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    trap = np.array(
        [
            [int(w * 0.08), h - 1],
            [int(w * 0.92), h - 1],
            [int(w * 0.62), int(h * 0.42)],
            [int(w * 0.38), int(h * 0.42)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, trap, 1)
    return mask


def _classify(q: FrameQuality, lap_adj: float, road_visible: float, occluded: float, cfg: QualityConfig) -> VisibilityClass:
    if occluded > 0.45:
        return VisibilityClass.OCCLUDED
    if q.glare >= cfg.glare_block:
        return VisibilityClass.GLARE
    if q.underexposure >= cfg.underexposure_block:
        return VisibilityClass.UNDEREXPOSED
    if q.overexposure >= cfg.overexposure_block:
        return VisibilityClass.OVEREXPOSED
    if q.motion_blur >= cfg.motion_blur_block or lap_adj < cfg.laplacian_usable:
        return VisibilityClass.BLUR
    if road_visible < cfg.min_road_visible:
        return VisibilityClass.UNKNOWN
    return VisibilityClass.CLEAR


def evaluate_frame(
    bgr: np.ndarray,
    cfg: QualityConfig,
    headlight_mean: float | None = None,
) -> FrameQualityMap:
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (ANALYSIS_WIDTH, max(1, int(ANALYSIS_WIDTH * h / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap = _laplacian_var(gray)
    lap_adj = exposure_normalized_lap(lap, float(gray.mean()))
    motion = _motion_blur_score(gray, lap_adj)
    under, over = _exposure_scores(gray)
    glare = _glare_score(small, headlight_mean=headlight_mean)
    defocus = _defocus_score(gray, lap_adj)
    road = _road_roi_mask(small.shape[0], small.shape[1])
    road_visible = float((gray[road > 0] > 18).mean()) if road.any() else 0.0
    # crude vehicle/occlusion proxy: saturated blobs inside the mid-height road corridor
    occluded = 0.0
    sh, sw = small.shape[:2]
    mid_rows = slice(int(sh * 0.28), int(sh * 0.62))
    mid = small[mid_rows]
    mid_road = road[mid_rows] > 0
    if mid.size and mid_road.any():
        hsv_m = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
        chroma = cv2.inRange(hsv_m, (0, 70, 50), (180, 255, 255)) > 0
        occluded = float((chroma & mid_road).sum() / max(1, mid_road.sum()))

    sharpness_n = float(np.clip(lap_adj / 80.0, 0, 1))
    usable = (
        lap_adj >= cfg.laplacian_usable
        and motion < cfg.motion_blur_block
        and glare < cfg.glare_block
        and under < cfg.underexposure_block
        and over < cfg.overexposure_block
        and road_visible >= cfg.min_road_visible
    )
    gq = FrameQuality(
        sharpness=sharpness_n,
        motion_blur=motion,
        defocus=defocus,
        underexposure=under,
        overexposure=over,
        glare=glare,
        usable=bool(usable),
        visibility_class=VisibilityClass.CLEAR,
        road_visible_ratio=road_visible,
        reason="" if usable else "low_quality",
    )
    gq.visibility_class = _classify(gq, lap_adj, road_visible, occluded, cfg)

    tiles: list[QualityTile] = []
    tr, tc = cfg.tile_rows, cfg.tile_cols
    for r in range(tr):
        for c in range(tc):
            y0, y1 = int(r * sh / tr), int((r + 1) * sh / tr)
            x0, x1 = int(c * sw / tc), int((c + 1) * sw / tc)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            plap = exposure_normalized_lap(_laplacian_var(patch), float(patch.mean()))
            pmean = float(patch.mean()) / 255.0
            vis = VisibilityClass.CLEAR
            score = float(np.clip(plap / 60.0, 0, 1))
            if pmean < 0.12:
                vis, score = VisibilityClass.UNDEREXPOSED, min(score, 0.25)
            elif pmean > 0.9:
                vis, score = VisibilityClass.OVEREXPOSED, min(score, 0.25)
            elif plap < 8:
                vis, score = VisibilityClass.BLUR, min(score, 0.3)
            tiles.append(
                QualityTile(
                    x0=int(x0 * w / sw),
                    y0=int(y0 * h / sh),
                    x1=int(x1 * w / sw),
                    y1=int(y1 * h / sh),
                    visibility=vis,
                    score=score,
                )
            )

    return FrameQualityMap(
        global_quality=gq,
        tiles=tiles,
        occupancy_occluded_ratio=occluded,
        selected_for_infer=gq.usable,
        selected_age_ms=0.0,
        degrade_reason=None if gq.usable else gq.reason or gq.visibility_class.value,
    )


class LensContaminationDetector:
    """QUAL-009: tiles that never change while the rest of the frame moves are lens drops / dirt.

    Feed one downscaled grey frame per step. `update` returns the number of tiles that have
    been static for at least `static_seconds` during a moving scene.
    """

    def __init__(self, cfg: QualityConfig, static_seconds: float | None = None) -> None:
        self.cfg = cfg
        self.static_seconds = static_seconds if static_seconds is not None else cfg.lens_static_seconds
        self._prev: np.ndarray | None = None
        self._static_since: np.ndarray | None = None
        self.static_tiles = 0
        self.flagged: bool = False

    def reset(self) -> None:
        self._prev = None
        self._static_since = None
        self.static_tiles = 0
        self.flagged = False

    def update(self, gray_small: np.ndarray, now_ns: int) -> int:
        tr, tc = self.cfg.tile_rows, self.cfg.tile_cols
        g = gray_small.astype(np.float32)
        if self._prev is None or self._prev.shape != g.shape:
            self._prev = g
            self._static_since = np.full((tr, tc), -1, dtype=np.int64)
            self.static_tiles = 0
            self.flagged = False
            return 0
        diff = np.abs(g - self._prev)
        self._prev = g
        h, w = diff.shape
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        energy = np.zeros((tr, tc), dtype=np.float32)
        edges = np.zeros((tr, tc), dtype=np.float32)
        for r in range(tr):
            for c in range(tc):
                ys = slice(int(r * h / tr), int((r + 1) * h / tr))
                xs = slice(int(c * w / tc), int((c + 1) * w / tc))
                patch = diff[ys, xs]
                energy[r, c] = float(patch.mean()) if patch.size else 0.0
                edges[r, c] = float(lap[ys, xs].mean()) if patch.size else 0.0
        # Median is ~0 when only the road corridor moves; use a high percentile.
        scene = float(np.percentile(energy, 92))
        assert self._static_since is not None
        if scene < 1.5:
            # Scene itself is still (stopped at a light): no evidence either way.
            return self.static_tiles
        # A contaminated tile has structure (drop edges) that does not move. Plain sky has no
        # structure and is ignored; the moving road has structure that moves.
        static = (energy < max(0.35, 0.08 * scene)) & (edges > 0.85)
        newly = static & (self._static_since < 0)
        self._static_since[newly] = now_ns
        self._static_since[~static] = -1
        held = static & (self._static_since >= 0) & ((now_ns - self._static_since) / 1e9 >= self.static_seconds)
        self.static_tiles = int(held.sum())
        self.flagged = self.static_tiles >= self.cfg.lens_static_tiles
        return self.static_tiles


def mask_visibility(quality: FrameQualityMap, bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    scores = []
    for t in quality.tiles:
        ix0, iy0 = max(x0, t.x0), max(y0, t.y0)
        ix1, iy1 = min(x1, t.x1), min(y1, t.y1)
        if ix1 > ix0 and iy1 > iy0:
            scores.append(t.score)
    if not scores:
        return float(quality.global_quality.sharpness * (1.0 - quality.global_quality.motion_blur))
    return float(np.mean(scores))


def perception_status(
    qmap: FrameQualityMap,
    consecutive_bad_s: float,
    cfg: QualityConfig | None = None,
) -> PerceptionStatus:
    limited_s = cfg.bad_streak_limited_s if cfg is not None else 0.45
    g = qmap.global_quality
    if g.visibility_class == VisibilityClass.LENS_DROP:
        return PerceptionStatus.LENS_CONTAMINATION
    if g.visibility_class == VisibilityClass.OCCLUDED or qmap.occupancy_occluded_ratio > 0.5:
        return PerceptionStatus.OCCLUDED
    if consecutive_bad_s > limited_s:
        return PerceptionStatus.PERCEPTION_LIMITED
    if not g.usable and g.visibility_class == VisibilityClass.BLUR:
        return PerceptionStatus.SEVERE_BLUR
    if g.visibility_class in {VisibilityClass.GLARE, VisibilityClass.UNDEREXPOSED, VisibilityClass.OVEREXPOSED}:
        return PerceptionStatus.DEGRADED_VISIBILITY
    if not g.usable:
        return PerceptionStatus.DEGRADED_VISIBILITY
    return PerceptionStatus.NORMAL


class QualityScheduler:
    """Keep ~0.5 s of frames and pick the newest usable frame within max age (QUAL-003)."""

    def __init__(self, cfg: QualityConfig) -> None:
        self.cfg = cfg
        self._buf: deque[tuple[SynchronizedFrame, FrameQualityMap]] = deque()

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, frame: SynchronizedFrame, qmap: FrameQualityMap) -> None:
        self._buf.append((frame, qmap))
        horizon = int(self.cfg.buffer_seconds * 1e9)
        now = frame.meta.sensor_timestamp_ns
        while self._buf and now - self._buf[0][0].meta.sensor_timestamp_ns > horizon:
            self._buf.popleft()

    def select(self, now_ns: int) -> tuple[SynchronizedFrame, FrameQualityMap, bool, float]:
        """Returns frame, quality, is_fresh_evidence, age_ms."""
        if not self._buf:
            raise RuntimeError("empty quality buffer")
        latest_f, latest_q = self._buf[-1]
        max_age = int(self.cfg.max_selected_age_ms * 1e6)
        chosen = None
        for f, q in reversed(self._buf):
            age = now_ns - f.meta.sensor_timestamp_ns
            if age > max_age:
                break
            if q.global_quality.usable:
                chosen = (f, q, age)
                break
        if chosen is None:
            age = now_ns - latest_f.meta.sensor_timestamp_ns
            latest_q.selected_for_infer = False
            latest_q.selected_age_ms = age / 1e6
            latest_q.degrade_reason = "no_fresh_usable_frame"
            return latest_f, latest_q, False, age / 1e6
        f, q, age = chosen
        q.selected_for_infer = True
        q.selected_age_ms = age / 1e6
        q.degrade_reason = None
        return f, q, True, age / 1e6
