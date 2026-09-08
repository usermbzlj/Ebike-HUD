from __future__ import annotations

import json
from pathlib import Path

from rpar.capture import record_simulated_session
from rpar.config import load_config
from rpar.error_cases import harvest_error_cases
from rpar.geometry import GeometryEngine
from rpar.pipeline import RealtimePipeline
from rpar.segdecode import CLASS_ANOMALY, CLASS_ROAD, ClassmapEngine, classmap_to_result
from rpar.enums import GeometryType, ObjectState, SemanticType, Severity
from rpar.simulator import RoadSimulator, SimConfig, WorldObject


def test_simulator_classmap_has_road_and_anomaly():
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.6, blur_windows=[]))
    sim.objects = [
        WorldObject(
            "near_pot",
            SemanticType.POTHOLE,
            GeometryType.CONCAVE,
            ObjectState.ABNORMAL,
            Severity.HEAVY,
            8.0,
            0.1,
            1.2,
            1.0,
            (18, 18, 18),
        )
    ]
    cm = sim.classmap_at(1)
    assert (cm == CLASS_ROAD).mean() > 0.05
    assert int((cm == CLASS_ANOMALY).sum()) > 40
    frame, _ = sim.frame_at(1)
    result = classmap_to_result(cm, frame)
    assert len(result.road_polygon) >= 3
    assert result.observations
    assert all(o.polygon for o in result.observations)


def test_classmap_engine_feeds_pipeline():
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=1.0, blur_windows=[]))
    engine = ClassmapEngine(lambda frame: sim.classmap_at(int(frame.meta.frame_id)))
    pipe = RealtimePipeline(load_config(), engine, GeometryEngine(sim.mount, load_config().geometry, sim.k))
    view = None
    for i in range(sim.n_frames()):
        frame, _ = sim.frame_at(i)
        view = pipe.step(frame)
    assert view is not None
    assert view.road_polygon
    assert pipe.last_observations or view.tracks


def test_recorded_session_writes_road_masks(tmp_path: Path):
    root = record_simulated_session(
        tmp_path,
        sim_cfg=SimConfig(width=320, height=180, fps=15, duration_s=0.7, blur_windows=[]),
    )
    road_lines = [ln for ln in (root / "perception" / "road.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert road_lines
    row = json.loads(road_lines[-1])
    assert "road_polygon" in row


def test_error_lib_harvests_alerts_and_marks(tmp_path: Path):
    root = record_simulated_session(
        tmp_path / "sess",
        sim_cfg=SimConfig(width=320, height=180, fps=15, duration_s=0.8, blur_windows=[]),
    )
    (root / "events" / "marks.jsonl").write_text(
        json.dumps({"timestamp_ns": 1, "note": "false_positive_shadow"}) + "\n",
        encoding="utf-8",
    )
    out = harvest_error_cases(root, tmp_path / "lib")
    assert (tmp_path / "lib" / "error_cases.json").exists()
    assert out["n_items"] >= 1
    kinds = {it.get("kind") for it in out["items"]}
    assert "mark" in kinds
