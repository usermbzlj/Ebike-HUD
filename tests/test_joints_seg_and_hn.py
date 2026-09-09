from __future__ import annotations

from pathlib import Path

from rpar.config import QualityConfig, load_config
from rpar.enums import GeometryType, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.geometry import GeometryEngine
from rpar.impact import harvest_impact_windows
from rpar.ml.eval import HARD_NEGATIVES, hard_negative_report
from rpar.ml.seg_train import train_dual_scale_classmap
from rpar.perception import oracle_engine_for_sim
from rpar.pipeline import RealtimePipeline
from rpar.quality import evaluate_frame
from rpar.simulator import RoadSimulator, SimConfig, WorldObject


def _run_lens(lens_drops: bool) -> tuple[bool, int]:
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=3.2, blur_windows=[], lens_drops=lens_drops))
    cfg = load_config()
    pipe = RealtimePipeline(cfg, oracle_engine_for_sim(sim), GeometryEngine(sim.mount, cfg.geometry, sim.k))
    saw_lens = False
    for i in range(sim.n_frames()):
        frame, _ = sim.frame_at(i)
        view = pipe.step(frame)
        if view.status.value == "LENS_CONTAMINATION":
            saw_lens = True
    return saw_lens, pipe.lens.static_tiles


def test_lens_drops_are_detected_temporally():
    # Drops stay put while the road streams past: motionless tiles with structure => LENS_CONTAMINATION.
    flagged, tiles = _run_lens(True)
    assert flagged, f"static drop tiles never flagged (static_tiles={tiles})"
    # The same moving scene without drops must not trip the detector (plain sky has no structure).
    clean, _ = _run_lens(False)
    assert clean is False


def test_single_frame_quality_has_no_lens_opinion():
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.2, blur_windows=[], lens_drops=True))
    frame, _ = sim.frame_at(0)
    q = evaluate_frame(frame.bgr, QualityConfig())
    assert q.global_quality.visibility_class != VisibilityClass.LENS_DROP


def test_road_joint_does_not_speak():
    sim = RoadSimulator(SimConfig(width=640, height=360, fps=15, duration_s=0.8, blur_windows=[]))
    sim.objects = [
        WorldObject("joint", SemanticType.ROAD_JOINT, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 12.0, 0.0, 0.12, 3.2, (88, 88, 92)),
    ]
    cfg = load_config()
    pipe = RealtimePipeline(cfg, oracle_engine_for_sim(sim), GeometryEngine(sim.mount, cfg.geometry, sim.k))
    fired = 0
    saw = False
    for i in range(sim.n_frames()):
        frame, _ = sim.frame_at(i)
        view = pipe.step(frame)
        fired += sum(1 for a in view.alerts if a.fired)
        saw = saw or any(t.semantic_type == SemanticType.ROAD_JOINT for t in view.tracks)
    assert saw
    assert fired == 0


def test_hard_negative_slices_are_named():
    hn = hard_negative_report(load_config())
    assert set(hn["slices"]) == set(HARD_NEGATIVES)
    assert hn["pass"] is True
    assert hn["n_alerts_fired"] == 0


def test_dual_scale_classmap_trainer_has_road():
    model = train_dual_scale_classmap(seed=3)
    assert model["n_pixels"] > 100
    assert model["has_road_polygon"] is True
    assert model["road_iou"] >= 0.15
    assert model["input"]["far"] == [48, 24]


def test_impact_windows_never_flag_alerts():
    samples = [
        {"timestamp_ns": 1, "z": 9.81},
        {"timestamp_ns": 2, "z": 9.81 + 8.0},
    ]
    rows = harvest_impact_windows(samples, near_track_ids=[7])
    assert len(rows) == 1
    assert rows[0]["used_for_alert"] is False
    assert rows[0]["near_track_ids"] == [7]
