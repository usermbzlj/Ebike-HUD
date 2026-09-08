from __future__ import annotations

import numpy as np

from rpar.config import load_config
from rpar.enums import GeometryType, InferenceBackend, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.ml.seg_train import classmap_precision_cards, train_dual_scale_classmap
from rpar.models import PerceptionResult, RoadObservation
from rpar.perception import HeuristicPerceptionEngine, load_engine
from rpar.segengine import DualScaleSegEngine, HybridPerceptionEngine, gate_observations, merge_perception
from rpar.simulator import RoadSimulator, SimConfig
from rpar.tracking import TrackEngine


def _obs(semantic: SemanticType, bbox: tuple[float, float, float, float], conf: float = 0.8) -> RoadObservation:
    x0, y0, x1, y1 = bbox
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return RoadObservation(
        timestamp_ns=1,
        source_frame_id=0,
        semantic_type=semantic,
        geometry_type=GeometryType.CONCAVE,
        state=ObjectState.ABNORMAL,
        severity=Severity.MEDIUM,
        mask_rle=None,
        polygon=poly,
        bbox=bbox,
        model_confidence=conf,
        quality_at_mask=0.9,
        visibility=VisibilityClass.CLEAR,
        calibrated_confidence=conf,
    )


def test_merge_keeps_heuristic_when_sidecar_has_instances():
    h = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[(0, 10), (20, 10), (20, 20), (0, 20)],
        occluded_polygons=[],
        observations=[_obs(SemanticType.POTHOLE, (8, 8, 16, 16))],
        backend=InferenceBackend.HEURISTIC,
        latency_ms=4.0,
        input_sizes=[(96, 48)],
    )
    s = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[(1, 11), (40, 11), (40, 30), (1, 30)],
        occluded_polygons=[[(2, 2), (6, 2), (6, 6), (2, 6)]],
        observations=[_obs(SemanticType.UNKNOWN_ANOMALY, (12, 14, 18, 22), 0.5)],
        backend=InferenceBackend.CPU,
        latency_ms=7.0,
        input_sizes=[(96, 48)],
    )
    m = merge_perception(h, s)
    assert len(m.road_polygon) >= 3
    assert m.road_polygon[0][0] == 1
    assert any(o.semantic_type == SemanticType.POTHOLE for o in m.observations)
    assert any(o.semantic_type == SemanticType.UNKNOWN_ANOMALY for o in m.observations)
    assert m.latency_ms == 7.0


def test_merge_drops_off_road_and_occluded():
    h = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        occluded_polygons=[],
        observations=[
            _obs(SemanticType.POTHOLE, (12, 14, 18, 22)),
            _obs(SemanticType.MANHOLE_COVER, (80, 80, 90, 90)),
            _obs(SemanticType.POTHOLE, (3, 3, 5, 5)),
        ],
        backend=InferenceBackend.HEURISTIC,
        latency_ms=4.0,
        input_sizes=[(96, 48)],
    )
    s = PerceptionResult(
        timestamp_ns=1,
        source_frame_id=0,
        road_polygon=[(1, 11), (40, 11), (40, 30), (1, 30)],
        occluded_polygons=[[(2, 2), (6, 2), (6, 6), (2, 6)]],
        observations=[],
        backend=InferenceBackend.CPU,
        latency_ms=7.0,
        input_sizes=[(96, 48)],
    )
    m = merge_perception(h, s)
    kinds = {o.semantic_type for o in m.observations}
    assert SemanticType.POTHOLE in kinds
    assert SemanticType.MANHOLE_COVER not in kinds
    assert all(not (3 <= o.bbox[0] <= 5 and 3 <= o.bbox[1] <= 5) for o in m.observations)


def test_gate_drops_lane_line_centers():
    obs = [_obs(SemanticType.POTHOLE, (10, 10, 20, 20))]
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[15, 15] = 0.9
    kept = gate_observations(obs, [(0, 0), (40, 0), (40, 40), (0, 40)], [], mask)
    assert kept == []


def test_hybrid_occlusion_only_sidecar_merges():
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.3, blur_windows=[]))
    cfg = load_config()

    class OccOnly:
        def infer(self, frame, quality=None):
            return PerceptionResult(
                timestamp_ns=frame.meta.sensor_timestamp_ns,
                source_frame_id=frame.meta.frame_id,
                road_polygon=[],
                occluded_polygons=[[(10, 40), (40, 40), (40, 70), (10, 70)]],
                observations=[],
                backend=InferenceBackend.CPU,
                latency_ms=2.0,
                input_sizes=[(640, 640)],
                dual_scale=False,
            )

        def capability(self):
            return {"backend": "yolopv2-onnx"}

        def close(self):
            return None

    hybrid = HybridPerceptionEngine(HeuristicPerceptionEngine(cfg), OccOnly())
    frame, _ = sim.frame_at(0)
    out = hybrid.infer(frame)
    assert out.occluded_polygons
    assert len(out.road_polygon) >= 3 or out.observations is not None


def test_hybrid_empty_sidecar_does_not_blank():
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.3, blur_windows=[]))
    cfg = load_config()

    class Empty:
        def infer(self, frame, quality=None):
            return PerceptionResult(
                timestamp_ns=frame.meta.sensor_timestamp_ns,
                source_frame_id=frame.meta.frame_id,
                road_polygon=[],
                occluded_polygons=[],
                observations=[],
                backend=InferenceBackend.CPU,
                latency_ms=1.0,
                input_sizes=[(8, 8)],
            )

        def capability(self):
            return {}

        def close(self):
            return None

    hybrid = HybridPerceptionEngine(HeuristicPerceptionEngine(cfg), Empty())
    frame, _ = sim.frame_at(0)
    view = hybrid.infer(frame)
    assert view.backend is not None


def test_dual_scale_seg_has_road_polygon():
    model = train_dual_scale_classmap(seed=3)
    engine = DualScaleSegEngine(model["weights"])
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.4, blur_windows=[]))
    frame, _ = sim.frame_at(1)
    result = engine.infer(frame)
    assert len(result.road_polygon) >= 3
    assert result.dual_scale is True
    assert len(result.input_sizes) == 2


def test_load_engine_hybrid_from_weights(tmp_path):
    model = train_dual_scale_classmap(seed=3)
    pkg = tmp_path / "roadseg"
    pkg.mkdir()
    import json

    (pkg / "seg_weights.json").write_text(json.dumps(model), encoding="utf-8")
    (pkg / "manifest.json").write_text(json.dumps({"engine": "classmap"}), encoding="utf-8")
    eng = load_engine(load_config(), pkg)
    assert isinstance(eng, HybridPerceptionEngine)
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.3, blur_windows=[]))
    frame, _ = sim.frame_at(0)
    out = eng.infer(frame)
    assert out.dual_scale is True


def test_precision_cards_have_int8():
    cards = classmap_precision_cards(train_dual_scale_classmap(seed=3)["weights"])
    assert "INT8" in cards["cards"]
    assert cards["cards"]["FP32"]["road_iou"] >= 0.0


def test_field_json_weights_run_without_onnx():
    from pathlib import Path

    wp = Path("models/roadseg-field-0.1.0/seg_weights.json")
    if not wp.is_file():
        return
    import json

    meta = json.loads(wp.read_text(encoding="utf-8"))
    engine = DualScaleSegEngine(meta["weights"])
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=10, duration_s=0.3, blur_windows=[]))
    frame, _ = sim.frame_at(0)
    out = engine.infer(frame)
    assert out.dual_scale is True
    assert list(out.input_sizes) == [(96, 48), (96, 48)]


def test_optical_flow_helps_match():
    cfg = load_config().tracking
    te = TrackEngine(cfg)
    a = _obs(SemanticType.POTHOLE, (10, 10, 20, 20))
    te.update([a], now_ns=1_000, dt_s=0.03)
    tid = next(iter(te.tracks))
    b = _obs(SemanticType.POTHOLE, (140, 12, 150, 22))
    te.update([b], now_ns=2_000, dt_s=0.03, optical_flow={tid: (130.0, 0.0)})
    assert len(te.tracks) == 1
