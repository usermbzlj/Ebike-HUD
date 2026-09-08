from __future__ import annotations

import numpy as np

from rpar.config import RparConfig
from rpar.enums import UiMode
from rpar.geometry import GeometryEngine
from rpar.ml.synth_train import patch_features
from rpar.perception import HeuristicPerceptionEngine
from rpar.pipeline import RealtimePipeline
from rpar.roi import dual_scale_feature_vector, far_polygon, near_polygon
from rpar.simulator import RoadSimulator, SimConfig
from rpar.transforms import default_intrinsics, default_mount


def test_roi_polygons_scale_with_frame_size():
    small = far_polygon(640, 360)
    large = far_polygon(1920, 1080)
    assert small[0][0] != large[0][0]
    assert abs(large[0][0] - 1920 * 0.18) < 1e-6
    assert abs(large[0][1] - 1080 * 0.32) < 1e-6
    near = near_polygon(640, 360)
    assert abs(near[0][0] - 640 * 0.08) < 1e-6
    assert abs(near[2][1] - 360 * 1.0) < 1e-6


def test_patch_features_constant_patch():
    g = np.full((24, 48), 128.0)
    f = patch_features(g)
    assert f.shape == (6,)
    assert abs(f[0] - 128.0 / 255.0) < 1e-9
    assert abs(f[1]) < 1e-9
    assert abs(f[2]) < 1e-9
    assert abs(f[3]) < 1e-9


def test_dual_scale_feature_vector_is_12d():
    gray = np.full((180, 320), 90.0, dtype=np.float32)
    gray[100:140, 40:200] = 30.0
    vec = dual_scale_feature_vector(gray)
    assert vec.shape == (12,)
    assert np.isfinite(vec).all()


def test_research_primitives_use_frame_roi_not_1080p_literals():
    sim = RoadSimulator(SimConfig(width=640, height=360, fps=15, duration_s=0.2))
    frame, _ = sim.frame_at(0)
    assert frame.meta.iso is None
    assert frame.meta.availability.get("iso") is False
    cfg = RparConfig()
    pipe = RealtimePipeline(
        cfg,
        HeuristicPerceptionEngine(cfg),
        GeometryEngine(default_mount(640, 360), cfg.geometry, default_intrinsics(640, 360)),
    )
    view = pipe.step(frame, ui_mode=UiMode.RESEARCH)
    rois = [p for p in view.primitives if p.kind == "roi"]
    assert len(rois) == 2
    assert abs(rois[0].polygon[0][0] - 640 * 0.18) < 1.0
    assert (346.0, 346.0) not in rois[0].polygon
    kinds = {p.kind for p in view.primitives}
    assert "roi" in kinds
