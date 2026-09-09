from __future__ import annotations

from rpar.config import TrackingConfig, load_config
from rpar.enums import GeometryType, LifecycleState, ObjectState, PerceptionStatus, SemanticType, Severity, VisibilityClass
from rpar.geometry import GeometryEngine
from rpar.maskutil import ellipse_polygon
from rpar.models import RoadObservation
from rpar.perception import oracle_engine_for_sim
from rpar.pipeline import RealtimePipeline, allow_new_observations, occlusion_cover_ratio
from rpar.simulator import RoadSimulator, SimConfig
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


def test_occlusion_sets_hud_without_freezing_all_tracks():
    assert allow_new_observations(PerceptionStatus.OCCLUDED, usable=True, fresh=True) is True
    assert allow_new_observations(PerceptionStatus.NORMAL, usable=True, fresh=True, occlusion_ratio=0.2) is True
    assert allow_new_observations(PerceptionStatus.NORMAL, usable=True, fresh=True, occlusion_ratio=0.50) is False
    assert allow_new_observations(PerceptionStatus.NORMAL, usable=True, fresh=True, occlusion_ratio=0.01) is True
    assert allow_new_observations(PerceptionStatus.SEVERE_BLUR, usable=True, fresh=True) is False
    poly = [(0.0, 0.0), (80.0, 0.0), (80.0, 50.0), (0.0, 50.0)]
    assert occlusion_cover_ratio([poly], 100, 100) >= 0.39

    sim = RoadSimulator(SimConfig(width=320, height=180, fps=15, duration_s=1.2, blur_windows=[], occlude_windows=[(0.4, 1.15)]))
    cfg = load_config()
    pipe = RealtimePipeline(cfg, oracle_engine_for_sim(sim), GeometryEngine(sim.mount, cfg.geometry, sim.k))
    during: set[int] = set()
    saw_occ = False
    for i in range(sim.n_frames()):
        frame, _ = sim.frame_at(i)
        t = i / sim.sim.fps
        view = pipe.step(frame)
        ids = {tr.track_id for tr in view.tracks if tr.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}}
        if 0.5 <= t <= 1.1:
            during |= ids
            if view.status == PerceptionStatus.OCCLUDED:
                saw_occ = True
    assert saw_occ
    # A vehicle-sized occluder must not wipe every confirmed track (potholes beside a car).
    assert during
