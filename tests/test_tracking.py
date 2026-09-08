from __future__ import annotations

from rpar.config import TrackingConfig
from rpar.enums import GeometryType, LifecycleState, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.maskutil import ellipse_polygon
from rpar.models import RoadObservation
from rpar.tracking import TrackEngine


def _obs(fid: int, ts: int, cx: float, cy: float, sem=SemanticType.POTHOLE) -> RoadObservation:
    poly = ellipse_polygon(cx, cy, 20, 12)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return RoadObservation(
        timestamp_ns=ts,
        source_frame_id=fid,
        semantic_type=sem,
        geometry_type=GeometryType.CONCAVE,
        state=ObjectState.ABNORMAL,
        severity=Severity.MEDIUM,
        mask_rle=None,
        polygon=poly,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        model_confidence=0.85,
        quality_at_mask=0.8,
        visibility=VisibilityClass.CLEAR,
        calibrated_confidence=0.8,
    )


def test_id_persists_across_frames():
    eng = TrackEngine(TrackingConfig(min_confirm_hits=3, confirm_window_s=0.3))
    tid = None
    for i in range(12):
        ts = int(i * 50e6)
        tracks = eng.update([_obs(i, ts, 400 + i * 3, 600 + i * 2)], ts, allow_new_high_conf=True, quality_ok=True, dt_s=0.05)
        if tracks:
            if tid is None:
                tid = tracks[0].track_id
            else:
                assert tracks[0].track_id == tid
    assert any(t.state == LifecycleState.CONFIRMED for t in eng.tracks.values())


def test_blur_does_not_spawn_new_high_conf():
    eng = TrackEngine(TrackingConfig())
    ts0 = 0
    eng.update([_obs(0, ts0, 400, 600)], ts0, allow_new_high_conf=True, quality_ok=True, dt_s=0.05)
    n0 = len(eng.tracks)
    eng.update([_obs(1, int(80e6), 900, 200)], int(80e6), allow_new_high_conf=False, quality_ok=False, dt_s=0.08)
    assert len(eng.tracks) == n0


def test_unknown_needs_more_hits():
    cfg = TrackingConfig(min_confirm_hits=2, unknown_anomaly_extra_hits=3, confirm_window_s=0.2)
    eng = TrackEngine(cfg)
    for i in range(3):
        ts = int(i * 80e6)
        eng.update([_obs(i, ts, 400, 500, SemanticType.UNKNOWN_ANOMALY)], ts, True, True, 0.08)
    assert all(t.state != LifecycleState.CONFIRMED for t in eng.tracks.values())
