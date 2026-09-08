from __future__ import annotations

from rpar.config import load_config
from rpar.enums import Direction, LifecycleState, UiMode
from rpar.geometry import GeometryEngine
from rpar.perception import OraclePerceptionEngine, ScriptedEvent
from rpar.pipeline import RealtimePipeline
from rpar.simulator import RoadSimulator, SimConfig, WorldObject
from rpar.enums import GeometryType, ObjectState, SemanticType, Severity
from rpar.maskutil import ellipse_polygon


def test_oracle_direction_and_distance():
    cfg = load_config()
    sim = RoadSimulator(SimConfig(width=960, height=540, fps=20, duration_s=2.0, speed_mps=8.0))
    # A right-side pothole that stays in view.
    sim.objects = [
        WorldObject(
            "pot_r",
            SemanticType.POTHOLE,
            GeometryType.CONCAVE,
            ObjectState.ABNORMAL,
            Severity.HEAVY,
            18.0,
            1.6,
            1.0,
            0.9,
            (10, 10, 10),
        )
    ]

    fps = sim.sim.fps

    def poly_at(t: float):
        idx = min(sim.n_frames() - 1, max(0, int(round(t * fps))))
        _, gt = sim.frame_at(idx)
        if gt["objects"]:
            return gt["objects"][0]["polygon"]
        return ellipse_polygon(600, 400, 30, 18)

    engine = OraclePerceptionEngine(
        [
            ScriptedEvent(
                0.0,
                2.0,
                SemanticType.POTHOLE,
                GeometryType.CONCAVE,
                ObjectState.ABNORMAL,
                Severity.HEAVY,
                poly_at,
            )
        ]
    )
    pipe = RealtimePipeline(cfg, engine, GeometryEngine(sim.mount, cfg.geometry, sim.k))
    dir_ok = dist_err = n = 0
    first_conf = None
    for i in range(sim.n_frames()):
        frame, gt = sim.frame_at(i)
        view = pipe.step(frame, ui_mode=UiMode.RESEARCH)
        if not gt["objects"]:
            continue
        g = gt["objects"][0]
        confirmed = [t for t in view.tracks if t.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED, LifecycleState.TRACKED}]
        if not confirmed:
            continue
        tr = confirmed[0]
        n += 1
        if tr.direction == Direction.RIGHT_FRONT:
            dir_ok += 1
        if tr.distance_m is not None and tr.distance_valid:
            dist_err += abs(tr.distance_m - g["distance_m"])
            if first_conf is None and tr.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}:
                first_conf = g["distance_m"]
    assert n >= 10
    assert dir_ok / n >= 0.95
    assert dist_err / n <= 2.5
    assert first_conf is None or first_conf >= 8.0
