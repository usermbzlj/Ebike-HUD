"""Road-plane distance, TTC, corridor and LEFT/CENTER/RIGHT (GEO-001..009)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace

import numpy as np

from rpar.config import GeometryConfig
from rpar.enums import Direction
from rpar.maskutil import ground_contact, polygon_area
from rpar.models import Intrinsics, MountProfile
from rpar.tracking import TrackInternal
from rpar.transforms import (
    apply_h,
    default_intrinsics,
    ground_homography,
    invert_h,
    pixel_to_ground,
    project_vehicle_point,
)


class GeometryEngine:
    def __init__(self, mount: MountProfile, cfg: GeometryConfig, intrinsics: Intrinsics | None = None) -> None:
        self.mount = mount
        self.cfg = cfg
        self.k = intrinsics or default_intrinsics()
        self._h = ground_homography(self.mount, self.k)
        self._h_inv = invert_h(self._h)
        self._dist_hist: dict[int, deque[tuple[int, float]]] = defaultdict(lambda: deque(maxlen=12))
        self._ttc_hist: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=8))
        self.valid = bool(mount.valid)
        self.invalid_reason: str | None = None if mount.valid else "no_mount_profile"

    def set_mount(self, mount: MountProfile, intrinsics: Intrinsics | None = None) -> None:
        self.mount = mount
        if intrinsics is not None:
            self.k = intrinsics
        self._h = ground_homography(self.mount, self.k)
        self._h_inv = invert_h(self._h)
        self.valid = bool(mount.valid)
        self.invalid_reason = None if mount.valid else "no_mount_profile"

    def health_check(
        self,
        pitch_err_deg: float = 0.0,
        roll_err_deg: float = 0.0,
        horizon_y_px: float | None = None,
    ) -> bool:
        ok = abs(pitch_err_deg) <= self.cfg.pitch_health_deg and abs(roll_err_deg) <= self.cfg.roll_health_deg
        expected_h = projected_horizon_y(self.mount, self.k)
        if ok and horizon_y_px is not None and expected_h is not None and abs(horizon_y_px - expected_h) > 48:
            ok = False
        if not ok:
            self.valid = False
            self.invalid_reason = "install_health_fail"
        elif self.mount.valid:
            self.valid = True
            self.invalid_reason = None
        else:
            self.valid = False
            self.invalid_reason = "no_mount_profile"
        return self.valid

    def apply_known_distance_markers(self) -> MountProfile:
        markers = []
        if self.mount.known_distance_5m_px is not None:
            markers.append((5.0, float(self.mount.known_distance_5m_px)))
        if self.mount.known_distance_10m_px is not None:
            markers.append((10.0, float(self.mount.known_distance_10m_px)))
        if self.mount.known_distance_20m_px is not None:
            markers.append((20.0, float(self.mount.known_distance_20m_px)))
        if len(markers) < 2:
            return self.mount
        fitted = fit_mount_from_distance_markers(self.mount, self.k, markers)
        self.set_mount(fitted, self.k)
        return fitted

    def contact_to_road(self, uv: tuple[float, float]) -> np.ndarray | None:
        xy = pixel_to_ground(np.array(uv, dtype=np.float64), self.mount, self.k)
        if xy is None or not np.isfinite(xy).all() or xy[1] < 0.4 or xy[1] > 80:
            return None
        return xy

    def distance_for_track(self, tr: TrackInternal, now_ns: int) -> tuple[float | None, float, bool]:
        if not self.valid:
            return None, 0.0, False
        contact = ground_contact(tr.polygon) if tr.polygon else (tr.mean[0], tr.mean[1])
        road = self.contact_to_road(contact)
        if road is None:
            return None, 0.0, False
        tr.road_xy = road
        dist = float(np.hypot(road[0], road[1]))
        # prefer forward component
        dist = float(max(0.5, road[1]))
        self._dist_hist[tr.track_id].append((now_ns, dist))
        conf = 0.82
        if dist > 30:
            conf = 0.45
        elif dist > 20:
            conf = 0.62
        if tr.quality_at_mask < 0.4:
            conf *= 0.7
        return dist, float(np.clip(conf, 0, 1)), True

    def ttc(self, tr: TrackInternal, speed_mps: float | None, now_ns: int) -> float | None:
        hist = self._dist_hist.get(tr.track_id)
        if not hist or len(hist) < 3:
            return None
        if speed_mps is None or speed_mps < self.cfg.min_speed_for_ttc_mps:
            return None
        t = np.array([h[0] for h in hist], dtype=np.float64) / 1e9
        d = np.array([h[1] for h in hist], dtype=np.float64)
        t = t - t[0]
        if t[-1] < 0.08:
            return None
        A = np.vstack([t, np.ones_like(t)]).T
        slope, _ = np.linalg.lstsq(A, d, rcond=None)[0]
        closing = -slope
        if closing < 0.4:
            return None
        fused = 0.55 * closing + 0.45 * speed_mps
        dist = d[-1]
        raw = dist / max(fused, 0.2)
        self._ttc_hist[tr.track_id].append(raw)
        return float(np.median(self._ttc_hist[tr.track_id]))

    def corridor_edges(self, look_ahead_m: float = 28.0) -> tuple[np.ndarray, np.ndarray]:
        hw = self.cfg.corridor_half_width_m
        left = np.array([[-hw, 2.0], [-hw, look_ahead_m]], dtype=np.float64)
        right = np.array([[hw, 2.0], [hw, look_ahead_m]], dtype=np.float64)
        return left, right

    def direction_for(self, tr: TrackInternal) -> Direction:
        if tr.road_xy is None:
            # fallback would be screen thirds — forbidden as primary. Still need a value.
            return Direction.UNKNOWN
        x, y = float(tr.road_xy[0]), float(tr.road_xy[1])
        width_m = 0.0
        if tr.polygon and self.valid:
            xs = []
            for p in tr.polygon:
                g = self.contact_to_road(p)
                if g is not None:
                    xs.append(g[0])
            if xs:
                width_m = max(xs) - min(xs)
        hw = self.cfg.corridor_half_width_m
        if width_m >= self.cfg.across_min_width_m and min(abs(x) , hw) < hw:
            return Direction.ACROSS
        if abs(x) <= hw * 0.72:
            return Direction.CENTER_FRONT
        if x < 0:
            return Direction.LEFT_FRONT
        return Direction.RIGHT_FRONT

    def path_relevance(self, tr: TrackInternal) -> float:
        if tr.road_xy is None:
            return 0.2
        x, y = float(tr.road_xy[0]), float(tr.road_xy[1])
        hw = self.cfg.corridor_half_width_m
        lat = np.exp(-0.5 * (x / (hw * 1.15)) ** 2)
        along = np.clip((40.0 - y) / 40.0, 0, 1)
        return float(np.clip(0.75 * lat + 0.25 * along, 0, 1))

    def display_distance(self, dist: float | None, valid: bool, confidence: float) -> str | None:
        if not valid or dist is None or not self.valid:
            return None
        if dist < 10:
            return f"{int(round(dist))} m"
        if dist <= 30:
            rounded = int(round(dist / 2.0) * 2)
            return f"约 {rounded} m"
        if confidence < 0.5:
            return "远处"
        return f"约 {int(round(dist))} m"

    def band(self, dist: float | None) -> str:
        if dist is None:
            return "unknown"
        if dist < 12:
            return "near"
        if dist < 25:
            return "mid"
        return "far"

    def project_corridor_pixels(self) -> list[tuple[float, float]]:
        left, right = self.corridor_edges()
        pts_m = [left[0], left[1], right[1], right[0]]
        pix = []
        for x, y in pts_m:
            uv = project_vehicle_point(np.array([x, y, 0.0]), self.mount, self.k)
            if uv is None:
                return []
            pix.append((float(uv[0]), float(uv[1])))
        return pix


def projected_horizon_y(mount: MountProfile, k: Intrinsics) -> float | None:
    uv = project_vehicle_point(np.array([0.0, 80.0, 0.0]), mount, k)
    return None if uv is None else float(uv[1])


def fit_mount_from_distance_markers(
    mount: MountProfile,
    k: Intrinsics,
    markers: list[tuple[float, float]],
) -> MountProfile:
    """CAL-003: fit pitch and height so (0, d, 0) projects to the measured pixel Y."""
    if len(markers) < 2:
        return mount
    best = (mount.pitch_deg, mount.camera_height_m)
    best_err = float("inf")
    for pitch in np.linspace(mount.pitch_deg - 8.0, mount.pitch_deg + 8.0, 33):
        for height in np.linspace(max(0.55, mount.camera_height_m - 0.45), mount.camera_height_m + 0.45, 19):
            trial = replace(mount, pitch_deg=float(pitch), camera_height_m=float(height))
            err = 0.0
            ok = True
            for dist_m, y_px in markers:
                uv = project_vehicle_point(np.array([0.0, dist_m, 0.0]), trial, k)
                if uv is None:
                    ok = False
                    break
                err += (float(uv[1]) - float(y_px)) ** 2
            if not ok:
                continue
            if err < best_err:
                best_err = err
                best = (float(pitch), float(height))
    return replace(mount, pitch_deg=best[0], camera_height_m=best[1], valid=True)


def should_mark_passed(tr: TrackInternal, dist: float | None, prev_dist: float | None, near_m: float) -> bool:
    if dist is None:
        return False
    if dist < near_m and tr.mean[3] > 40:  # contact point moving down the image fast
        return True
    if prev_dist is not None and dist > prev_dist + 4.0 and dist < 8.0:
        return True
    if tr.road_xy is not None and tr.road_xy[1] < 1.6:
        return True
    return False
