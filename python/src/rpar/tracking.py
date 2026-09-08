"""Object association, Kalman motion, lifecycle state machine (TRK-001..004)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rpar.config import TrackingConfig
from rpar.enums import GeometryType, LifecycleState, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.maskutil import bbox_iou, ground_contact, polygon_bbox, polygon_centroid
from rpar.models import MaskRle, RoadObservation


@dataclass
class TrackInternal:
    track_id: int
    hits: int
    misses: int
    created_ns: int
    last_ns: int
    confirmed_ns: int | None = None
    alerted: bool = False
    last_alert_ns: int | None = None
    last_severity: Severity = Severity.UNKNOWN
    semantic: SemanticType = SemanticType.UNKNOWN_ANOMALY
    geometry: GeometryType = GeometryType.UNKNOWN
    object_state: ObjectState = ObjectState.UNKNOWN
    severity: Severity = Severity.UNKNOWN
    polygon: list[tuple[float, float]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    model_confidence: float = 0.0
    quality_at_mask: float = 0.0
    visibility: VisibilityClass = VisibilityClass.UNKNOWN
    state: LifecycleState = LifecycleState.CANDIDATE
    expire_reason: str = ""
    # Kalman in image contact-point + size: [x, y, vx, vy, w, h]
    mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cov: np.ndarray = field(default_factory=lambda: np.eye(6) * 40.0)
    road_xy: np.ndarray | None = None
    distance_hist: list[tuple[int, float]] = field(default_factory=list)
    fade: float = 1.0
    source_frame_id: int = 0
    hold_until_ns: int | None = None
    mask_rle: MaskRle | None = None


class KalmanImage:
    def __init__(self, process_noise: float, meas_noise: float) -> None:
        self.q = process_noise
        self.r = meas_noise

    def predict(self, mean: np.ndarray, cov: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        f = np.eye(6)
        f[0, 2] = dt
        f[1, 3] = dt
        q = np.diag([self.q, self.q, self.q * 4, self.q * 4, self.q * 0.5, self.q * 0.5]) * max(dt, 1e-3)
        mean_p = f @ mean
        cov_p = f @ cov @ f.T + q
        return mean_p, cov_p

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros((4, 6))
        h[0, 0] = h[1, 1] = h[2, 4] = h[3, 5] = 1.0
        r = np.eye(4) * self.r
        y = z - h @ mean
        s = h @ cov @ h.T + r
        k = cov @ h.T @ np.linalg.inv(s)
        mean_u = mean + k @ y
        cov_u = (np.eye(6) - k @ h) @ cov
        return mean_u, cov_u


def _obs_meas(obs: RoadObservation) -> np.ndarray:
    cx, cy = ground_contact(obs.polygon) if obs.polygon else polygon_centroid(obs.polygon)
    x0, y0, x1, y1 = obs.bbox
    return np.array([cx, cy, max(4.0, x1 - x0), max(4.0, y1 - y0)], dtype=np.float64)


class TrackEngine:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self._next_id = 1
        self.tracks: dict[int, TrackInternal] = {}
        self.kf = KalmanImage(cfg.process_noise, cfg.meas_noise)
        self._history: list[tuple[int, LifecycleState, str]] = []

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._history.clear()

    def _spawn(self, obs: RoadObservation, now: int) -> TrackInternal:
        z = _obs_meas(obs)
        tr = TrackInternal(
            track_id=self._next_id,
            hits=1,
            misses=0,
            created_ns=now,
            last_ns=now,
            semantic=obs.semantic_type,
            geometry=obs.geometry_type,
            object_state=obs.state,
            severity=obs.severity,
            polygon=list(obs.polygon),
            bbox=obs.bbox,
            model_confidence=obs.model_confidence,
            quality_at_mask=obs.quality_at_mask,
            visibility=obs.visibility,
            mean=np.array([z[0], z[1], 0, 0, z[2], z[3]], dtype=np.float64),
            source_frame_id=obs.source_frame_id,
            mask_rle=obs.mask_rle,
        )
        self._next_id += 1
        self.tracks[tr.track_id] = tr
        self._history.append((tr.track_id, tr.state, "spawn"))
        return tr

    def _match(
        self,
        observations: list[RoadObservation],
        optical_flow: dict[int, tuple[float, float]] | None = None,
        prev_xy: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[list[tuple[int | None, int]], set[int]]:
        """Greedy IoU + center matching, with optional LK flow (TRK-002)."""
        unused = set(self.tracks.keys())
        assigned: list[tuple[int | None, int]] = []
        used_obs: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        flow = optical_flow or {}
        prev = prev_xy or {}
        for tid, tr in self.tracks.items():
            for j, obs in enumerate(observations):
                iou = bbox_iou(tr.bbox, obs.bbox)
                cx, cy = ground_contact(obs.polygon) if obs.polygon else polygon_centroid(obs.polygon)
                dist = float(np.hypot(cx - tr.mean[0], cy - tr.mean[1]))
                if tid in flow and tid in prev:
                    fx, fy = flow[tid]
                    px, py = prev[tid]
                    dist = min(dist, float(np.hypot(cx - (px + fx), cy - (py + fy))))
                same = obs.semantic_type == tr.semantic or obs.semantic_type == SemanticType.UNKNOWN_ANOMALY
                score = iou * (1.2 if same else 0.7) - dist / 400.0
                if iou >= self.cfg.iou_match or dist < self.cfg.center_match_px:
                    pairs.append((score, tid, j))
        pairs.sort(reverse=True)
        taken_tr: set[int] = set()
        for _, tid, j in pairs:
            if tid in taken_tr or j in used_obs:
                continue
            assigned.append((tid, j))
            taken_tr.add(tid)
            used_obs.add(j)
            unused.discard(tid)
        for j, _ in enumerate(observations):
            if j not in used_obs:
                assigned.append((None, j))
        return assigned, unused

    def update(
        self,
        observations: list[RoadObservation],
        now_ns: int,
        allow_new_high_conf: bool = True,
        quality_ok: bool = True,
        dt_s: float = 0.033,
        camera_yaw_rate: float = 0.0,
        frame_w: float = 1920.0,
        optical_flow: dict[int, tuple[float, float]] | None = None,
    ) -> list[TrackInternal]:
        prev_xy = {tid: (float(tr.mean[0]), float(tr.mean[1])) for tid, tr in self.tracks.items()}
        for tr in self.tracks.values():
            tr.mean, tr.cov = self.kf.predict(tr.mean, tr.cov, max(dt_s, 1e-3))
            # TRK-002: rotational camera flow so handlebar yaw is not object motion
            tr.mean[0] += float(camera_yaw_rate) * dt_s * (frame_w * 0.55)

        matches, unmatched_tracks = self._match(observations, optical_flow=optical_flow, prev_xy=prev_xy)
        updated: set[int] = set()
        for tid, j in matches:
            obs = observations[j]
            if tid is None:
                if allow_new_high_conf:
                    self._spawn(obs, now_ns)
                continue
            tr = self.tracks[tid]
            z = _obs_meas(obs)
            tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, z)
            tr.hits += 1
            tr.misses = 0
            tr.last_ns = now_ns
            tr.polygon = list(obs.polygon)
            tr.bbox = obs.bbox if obs.bbox != (0, 0, 0, 0) else polygon_bbox(obs.polygon)
            tr.model_confidence = 0.7 * tr.model_confidence + 0.3 * obs.model_confidence
            tr.quality_at_mask = obs.quality_at_mask
            tr.visibility = obs.visibility
            tr.source_frame_id = obs.source_frame_id
            tr.mask_rle = obs.mask_rle
            if obs.semantic_type != SemanticType.UNKNOWN_ANOMALY:
                tr.semantic = obs.semantic_type
            if obs.geometry_type != GeometryType.UNKNOWN:
                tr.geometry = obs.geometry_type
            if obs.state != ObjectState.UNKNOWN:
                tr.object_state = obs.state
            if obs.severity != Severity.UNKNOWN:
                tr.severity = obs.severity
            tr.fade = 1.0
            self._advance_lifecycle(tr, now_ns, observed=True, quality_ok=quality_ok)
            updated.add(tid)

        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]
            tr.misses += 1
            if not quality_ok:
                tr.hold_until_ns = now_ns + int(self.cfg.low_quality_hold_s * 1e9)
                tr.fade = max(0.25, tr.fade * 0.82)
                if tr.state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED, LifecycleState.TRACKED}:
                    age = (now_ns - tr.last_ns) / 1e9
                    if age <= self.cfg.low_quality_hold_s:
                        continue
            self._advance_lifecycle(tr, now_ns, observed=False, quality_ok=quality_ok)

        dead = [tid for tid, tr in self.tracks.items() if tr.state in {LifecycleState.EXPIRED} and tr.fade < 0.05]
        for tid in dead:
            del self.tracks[tid]
        return list(self.tracks.values())

    def mark_passed(self, track_id: int, now_ns: int, reason: str) -> None:
        tr = self.tracks.get(track_id)
        if not tr:
            return
        tr.state = LifecycleState.PASSED
        tr.expire_reason = reason
        tr.last_ns = now_ns
        self._history.append((track_id, tr.state, reason))

    def _advance_lifecycle(self, tr: TrackInternal, now_ns: int, observed: bool, quality_ok: bool) -> None:
        age = (now_ns - tr.created_ns) / 1e9
        need = self.cfg.min_confirm_hits
        if tr.semantic == SemanticType.UNKNOWN_ANOMALY:
            need += self.cfg.unknown_anomaly_extra_hits
        if tr.semantic == SemanticType.ROUGH_BROKEN:
            need += 3
        if tr.state == LifecycleState.CANDIDATE:
            if not observed:
                if (now_ns - tr.last_ns) / 1e9 > self.cfg.candidate_max_age_s:
                    tr.state = LifecycleState.EXPIRED
                    tr.expire_reason = "candidate_timeout"
                    self._history.append((tr.track_id, tr.state, tr.expire_reason))
                return
            if tr.hits >= 2 and age >= self.cfg.confirm_window_s * 0.4:
                tr.state = LifecycleState.TRACKED
                self._history.append((tr.track_id, tr.state, "associated"))
        if tr.state == LifecycleState.TRACKED and observed and quality_ok:
            if tr.hits >= need and age >= self.cfg.confirm_window_s:
                tr.state = LifecycleState.CONFIRMED
                tr.confirmed_ns = now_ns
                self._history.append((tr.track_id, tr.state, "stable_visible"))
        if tr.state in {LifecycleState.TRACKED, LifecycleState.CONFIRMED, LifecycleState.ALERTED} and not observed:
            if (now_ns - tr.last_ns) / 1e9 > self.cfg.low_quality_hold_s + 0.25:
                tr.state = LifecycleState.EXPIRED
                tr.expire_reason = "lost"
                tr.fade = 0.0
                self._history.append((tr.track_id, tr.state, tr.expire_reason))

    def history(self) -> list[tuple[int, LifecycleState, str]]:
        return list(self._history)
