"""Object association, Kalman motion, lifecycle state machine (TRK-001..004).

Lifecycle: CANDIDATE -> TRACKED -> CONFIRMED -> ALERTED, plus PASSED and EXPIRED.
Only active tracks (CANDIDATE/TRACKED/CONFIRMED/ALERTED) take part in association.
EXPIRED tracks are dropped in the same update; PASSED tracks linger for
`passed_remove_s` so the HUD can fade them, then are dropped too.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rpar.config import TrackingConfig
from rpar.enums import GeometryType, LifecycleState, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.maskutil import bbox_iou, ground_contact, polygon_bbox, polygon_centroid
from rpar.models import MaskRle, RoadObservation

ACTIVE_STATES = frozenset(
    {LifecycleState.CANDIDATE, LifecycleState.TRACKED, LifecycleState.CONFIRMED, LifecycleState.ALERTED}
)

# Visual families: an observation may only continue a track whose class is confusable with it.
_FAMILY = {
    SemanticType.POTHOLE: "hole",
    SemanticType.MANHOLE_COVER: "hole",
    SemanticType.UNKNOWN_ANOMALY: "hole",
    SemanticType.ROUGH_BROKEN: "hole",
    SemanticType.REPAIR_PATCH: "hole",
    SemanticType.SPEED_BUMP: "band",
    SemanticType.ROAD_JOINT: "band",
    SemanticType.PUDDLE: "info",
    SemanticType.GRAVEL: "info",
}


def classes_compatible(a: SemanticType, b: SemanticType) -> bool:
    if a == b:
        return True
    return _FAMILY.get(a) == _FAMILY.get(b)


def is_bump_track(semantic: SemanticType, geometry: GeometryType) -> bool:
    return semantic in {SemanticType.POTHOLE, SemanticType.SPEED_BUMP} or (
        semantic == SemanticType.MANHOLE_COVER and geometry == GeometryType.CONCAVE
    )


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
    semantic_votes: Counter = field(default_factory=Counter)
    severity_hist: list[int] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def predicted_bbox(self) -> tuple[float, float, float, float]:
        cx, cy, _, _, w, h = self.mean
        return (float(cx - w / 2.0), float(cy - h), float(cx + w / 2.0), float(cy))


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
        i_kh = np.eye(6) - k @ h
        # Joseph form keeps the covariance symmetric positive semi-definite.
        cov_u = i_kh @ cov @ i_kh.T + k @ r @ k.T
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

    # ------------------------------------------------------------------ spawn / match
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
            bbox=obs.bbox if obs.bbox != (0, 0, 0, 0) else polygon_bbox(obs.polygon),
            model_confidence=obs.model_confidence,
            quality_at_mask=obs.quality_at_mask,
            visibility=obs.visibility,
            mean=np.array([z[0], z[1], 0, 0, z[2], z[3]], dtype=np.float64),
            source_frame_id=obs.source_frame_id,
            mask_rle=obs.mask_rle,
        )
        tr.semantic_votes[obs.semantic_type] += 1
        if int(obs.severity) >= 0:
            tr.severity_hist.append(int(obs.severity))
        self._next_id += 1
        self.tracks[tr.track_id] = tr
        self._history.append((tr.track_id, tr.state, "spawn"))
        return tr

    def _center_threshold(self, tr: TrackInternal) -> float:
        # Scale-aware: far (small) objects must match tighter than near (large) ones.
        size = max(float(tr.mean[4]), float(tr.mean[5]))
        return float(np.clip(0.75 * size, 16.0, self.cfg.center_match_px))

    def _match(
        self,
        observations: list[RoadObservation],
        optical_flow: dict[int, tuple[float, float]] | None = None,
        prev_xy: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[list[tuple[int | None, int]], set[int]]:
        """Greedy IoU + centre matching on predicted boxes, with optional LK flow (TRK-002)."""
        active = {tid: tr for tid, tr in self.tracks.items() if tr.active}
        unused = set(active.keys())
        assigned: list[tuple[int | None, int]] = []
        used_obs: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        flow = optical_flow or {}
        prev = prev_xy or {}
        for tid, tr in active.items():
            pred_box = tr.predicted_bbox()
            thr = self._center_threshold(tr)
            for j, obs in enumerate(observations):
                if not classes_compatible(tr.semantic, obs.semantic_type):
                    continue
                iou = max(bbox_iou(tr.bbox, obs.bbox), bbox_iou(pred_box, obs.bbox))
                cx, cy = ground_contact(obs.polygon) if obs.polygon else polygon_centroid(obs.polygon)
                dist = float(np.hypot(cx - tr.mean[0], cy - tr.mean[1]))
                if tid in flow and tid in prev:
                    fx, fy = flow[tid]
                    px, py = prev[tid]
                    dist = min(dist, float(np.hypot(cx - (px + fx), cy - (py + fy))))
                same = obs.semantic_type == tr.semantic
                if iou >= self.cfg.iou_match or dist < thr:
                    score = iou * (1.2 if same else 0.8) - dist / (4.0 * self.cfg.center_match_px)
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

    # ------------------------------------------------------------------ update
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
        focal_px: float | None = None,
    ) -> list[TrackInternal]:
        # Rotational flow from handlebar yaw is not object motion (TRK-002). A yaw rate of w rad/s
        # moves image points by ~ f * w * dt pixels; fall back to ~85 deg HFOV when no intrinsics.
        f_px = float(focal_px) if focal_px and focal_px > 0 else float(frame_w) * 0.55
        prev_xy = {tid: (float(tr.mean[0]), float(tr.mean[1])) for tid, tr in self.tracks.items()}
        for tr in self.tracks.values():
            if not tr.active:
                continue
            tr.mean, tr.cov = self.kf.predict(tr.mean, tr.cov, max(dt_s, 1e-3))
            tr.mean[0] += float(camera_yaw_rate) * dt_s * f_px

        matches, unmatched_tracks = self._match(observations, optical_flow=optical_flow, prev_xy=prev_xy)
        for tid, j in matches:
            obs = observations[j]
            if tid is None:
                if allow_new_high_conf:
                    self._spawn(obs, now_ns)
                continue
            tr = self.tracks[tid]
            self._absorb(tr, obs, now_ns)
            self._advance_lifecycle(tr, now_ns, observed=True, quality_ok=quality_ok)

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

        self._prune(now_ns)
        return list(self.tracks.values())

    def _absorb(self, tr: TrackInternal, obs: RoadObservation, now_ns: int) -> None:
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
        tr.fade = 1.0
        # Class by majority vote; UNKNOWN never outvotes a typed observation.
        tr.semantic_votes[obs.semantic_type] += 1
        typed = {k: v for k, v in tr.semantic_votes.items() if k != SemanticType.UNKNOWN_ANOMALY}
        pool = typed or dict(tr.semantic_votes)
        tr.semantic = max(pool.items(), key=lambda kv: (kv[1], kv[0] == tr.semantic))[0]
        if obs.semantic_type == tr.semantic or obs.semantic_type == SemanticType.UNKNOWN_ANOMALY:
            if obs.geometry_type != GeometryType.UNKNOWN:
                tr.geometry = obs.geometry_type
            if obs.state != ObjectState.UNKNOWN:
                tr.object_state = obs.state
        if int(obs.severity) >= 0:
            tr.severity_hist.append(int(obs.severity))
            del tr.severity_hist[:-5]
            tr.severity = Severity(int(np.median(tr.severity_hist)))

    def _prune(self, now_ns: int) -> None:
        dead = []
        for tid, tr in self.tracks.items():
            if tr.state == LifecycleState.EXPIRED:
                dead.append(tid)
            elif tr.state == LifecycleState.PASSED and (now_ns - tr.last_ns) / 1e9 > self.cfg.passed_remove_s:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

    def mark_passed(self, track_id: int, now_ns: int, reason: str) -> None:
        tr = self.tracks.get(track_id)
        if not tr:
            return
        tr.state = LifecycleState.PASSED
        tr.expire_reason = reason
        tr.last_ns = now_ns
        self._history.append((track_id, tr.state, reason))

    def _advance_lifecycle(self, tr: TrackInternal, now_ns: int, observed: bool, quality_ok: bool) -> None:
        if not tr.active:
            return
        age = (now_ns - tr.created_ns) / 1e9
        need = self.cfg.min_confirm_hits
        bump = is_bump_track(tr.semantic, tr.geometry)
        if tr.semantic == SemanticType.UNKNOWN_ANOMALY:
            need += self.cfg.unknown_anomaly_extra_hits
        if tr.semantic == SemanticType.ROUGH_BROKEN:
            need += 3
        if bump:
            need = min(need, max(2, int(self.cfg.bump_confirm_hits)))
        if tr.state == LifecycleState.CANDIDATE:
            tracked_age = self.cfg.confirm_window_s * (0.25 if bump else 0.4)
            if tr.hits >= 2 and age >= tracked_age:
                tr.state = LifecycleState.TRACKED
                self._history.append((tr.track_id, tr.state, "associated"))
            elif not observed:
                if (now_ns - tr.last_ns) / 1e9 > self.cfg.candidate_max_age_s:
                    tr.state = LifecycleState.EXPIRED
                    tr.expire_reason = "candidate_timeout"
                    self._history.append((tr.track_id, tr.state, tr.expire_reason))
                return
        if tr.state == LifecycleState.TRACKED and quality_ok and observed:
            conf_age = self.cfg.confirm_window_s * (0.5 if bump else 1.0)
            if tr.hits >= need and age >= conf_age:
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
