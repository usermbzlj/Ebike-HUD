from __future__ import annotations

import numpy as np

from rpar.ab_compare import model_ab_same_input
from rpar.alerts import AlertPolicy
from rpar.config import AlertConfig, load_config
from rpar.enums import (
    GeometryType,
    ObjectState,
    PerceptionStatus,
    SemanticType,
    Severity,
    StabilizationMode,
)
from rpar.impact import estimate_impact_score
from rpar.models import FrameMeta, SynchronizedFrame
from rpar.perception import HeuristicPerceptionEngine
from rpar.quality import evaluate_frame, fit_headlight_mean, headlight_cone_mask
from tests.test_alerts import _obj


def _frame(bgr: np.ndarray, frame_id: int = 0) -> SynchronizedFrame:
    h, w = bgr.shape[:2]
    meta = FrameMeta(
        frame_id=frame_id,
        sensor_timestamp_ns=1_000_000_000 + frame_id * 16_666_667,
        image_timestamp_ns=1_000_000_000 + frame_id * 16_666_667,
        exposure_time_ns=None,
        iso=None,
        focal_length_mm=None,
        focus_distance_diopters=None,
        af_state=None,
        ae_state=None,
        awb_state=None,
        crop_region=None,
        stabilization_mode=StabilizationMode.OFF,
        width=w,
        height=h,
    )
    return SynchronizedFrame(
        meta=meta,
        bgr=bgr,
        pose=None,
        angular_velocity=(0.0, 0.0, 0.0),
        linear_accel=(0.0, 0.0, 9.81),
        location=None,
        speed_mps=10.0,
    )


def test_puddle_detected_and_never_alerts():
    import cv2

    bgr = np.zeros((360, 640, 3), dtype=np.uint8)
    bgr[:] = (55, 52, 50)
    bgr[150:, :] = (95, 95, 95)
    cv2.ellipse(bgr, (320, 280), (36, 18), 0, 0, 360, (235, 235, 240), -1)
    eng = HeuristicPerceptionEngine(load_config())
    result = eng.infer(_frame(bgr))
    puddles = [o for o in result.observations if o.semantic_type == SemanticType.PUDDLE]
    assert puddles, [o.semantic_type.value for o in result.observations]
    assert puddles[0].geometry_type == GeometryType.FLAT
    assert puddles[0].severity == Severity.NONE
    pol = AlertPolicy(AlertConfig(score_threshold=0.01, min_effective=0.0, min_visibility=0.0, min_severity=0))
    d = pol.evaluate(
        _obj(
            semantic_type=SemanticType.PUDDLE,
            geometry_type=GeometryType.FLAT,
            object_state=ObjectState.UNKNOWN,
            severity=Severity.NONE,
        ),
        PerceptionStatus.NORMAL,
        10**10,
        True,
    )
    assert d.fired is False
    assert "info_layer" in d.reasons


def test_gravel_info_layer_rejected():
    pol = AlertPolicy(AlertConfig(score_threshold=0.01, min_effective=0.0, min_visibility=0.0, min_severity=0))
    d = pol.evaluate(
        _obj(
            semantic_type=SemanticType.GRAVEL,
            geometry_type=GeometryType.ROUGH,
            object_state=ObjectState.UNKNOWN,
            severity=Severity.NONE,
        ),
        PerceptionStatus.NORMAL,
        10**10,
        True,
    )
    assert d.fired is False
    assert "info_layer" in d.reasons


def test_depth_confidence_reserved_and_impact_interface():
    obj = _obj()
    assert obj.depth_confidence is None or obj.depth_confidence >= 0.0
    assert estimate_impact_score(None, 10.0, 3.0) is None
    assert estimate_impact_score((0.0, 0.0, 9.81), 10.0, 3.0) is None
    spike = estimate_impact_score((0.0, 0.0, 9.81 + 8.0), 10.0, 3.0)
    assert spike is not None and 0 < spike <= 1.0
    assert estimate_impact_score((0.0, 0.0, 20.0), 10.0, 20.0) is None


def test_model_ab_same_input():
    ab = model_ab_same_input(load_config(), duration_s=0.6)
    assert ab["method"] == "same_input"
    assert ab["n_frames"] > 0
    assert ab["A"]["engine"] == "heuristic"
    assert ab["B"]["engine"] == "oracle"
    assert "n_alerts" in ab["delta"]


def test_headlight_field_calibration_reduces_cone_glare():
    bgr = np.zeros((180, 320, 3), dtype=np.uint8)
    bgr[:] = 30
    mask = headlight_cone_mask(180, 320)
    bgr[mask > 0] = 210
    gray = bgr[:, :, 1]
    mean = fit_headlight_mean(gray)
    assert mean > 100
    cfg = load_config().quality
    raw = evaluate_frame(bgr, cfg)
    cal = evaluate_frame(bgr, cfg, headlight_mean=mean)
    assert cal.global_quality.glare <= raw.global_quality.glare + 1e-6
