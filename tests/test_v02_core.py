"""v0.2 product gates: road support, tracker lifecycle, IMU frame, config parity."""

from __future__ import annotations

import json

import numpy as np

from rpar.config import android_config_json, default_config_path, load_config, repo_root
from rpar.enums import GeometryType, LifecycleState, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.field_video import m2_clip_gates
from rpar.impact import estimate_impact_score, vertical_residual
from rpar.maskutil import ellipse_polygon
from rpar.models import FrameMeta, RoadObservation, SynchronizedFrame
from rpar.perception import HeuristicPerceptionEngine, _road_region
from rpar.tracking import TrackEngine
from rpar.config import TrackingConfig


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


def test_road_region_keeps_dark_pothole_inside():
    h, w = 360, 640
    bgr = np.full((h, w, 3), 90, dtype=np.uint8)
    cv_y, cv_x, r = 260, 320, 28
    bgr[cv_y - r : cv_y + r, cv_x - r : cv_x + r] = 18
    region, asphalt = _road_region(bgr)
    assert region[cv_y, cv_x] > 0
    assert asphalt[cv_y, cv_x] == 0


def test_heuristic_sees_synthetic_pothole():
    h, w = 540, 960
    bgr = np.full((h, w, 3), 96, dtype=np.uint8)
    cy, cx, r = 400, 480, 36
    yy, xx = np.ogrid[:h, :w]
    pit = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    bgr[pit] = (16, 16, 16)
    frame = SynchronizedFrame(
        meta=FrameMeta(
            frame_id=1,
            sensor_timestamp_ns=1_000_000_000,
            image_timestamp_ns=1_000_000_000,
            exposure_time_ns=None,
            iso=None,
            focal_length_mm=None,
            focus_distance_diopters=None,
            af_state=None,
            ae_state=None,
            awb_state=None,
            crop_region=None,
            stabilization_mode=None,
            width=w,
            height=h,
            availability={},
        ),
        bgr=bgr,
        pose=None,
        angular_velocity=None,
        linear_accel=None,
        location=None,
        speed_mps=8.0,
    )
    result = HeuristicPerceptionEngine(load_config()).infer(frame)
    kinds = {o.semantic_type for o in result.observations}
    assert SemanticType.POTHOLE in kinds or SemanticType.UNKNOWN_ANOMALY in kinds
    assert result.dual_scale is True
    assert len(result.input_sizes) == 2


def test_expired_fp_does_not_absorb_later_real_hit():
    eng = TrackEngine(TrackingConfig(candidate_max_age_s=0.2, min_confirm_hits=3, confirm_window_s=0.4))
    eng.update([_obs(0, 0, 400, 600)], 0, True, True, 0.05)
    for i in range(1, 8):
        ts = int(i * 80e6)
        eng.update([], ts, True, True, 0.08)
    assert not any(t.state == LifecycleState.EXPIRED for t in eng.tracks.values())
    ts = int(8 * 80e6)
    tracks = eng.update([_obs(8, ts, 400, 600)], ts, True, True, 0.08)
    live = [t for t in tracks if t.state != LifecycleState.EXPIRED]
    assert live
    assert live[0].hits == 1
    assert live[0].state == LifecycleState.CANDIDATE


def test_cross_family_does_not_match():
    eng = TrackEngine(TrackingConfig())
    eng.update([_obs(0, 0, 400, 600, SemanticType.POTHOLE)], 0, True, True, 0.05)
    ts = int(50e6)
    eng.update([_obs(1, ts, 405, 605, SemanticType.SPEED_BUMP)], ts, True, True, 0.05)
    assert len(eng.tracks) == 2


def test_impact_ignores_tilted_phone_gravity():
    # Device Z is camera-forward; gravity mostly along Y when mounted on the handlebar.
    residual = vertical_residual((0.2, 9.81, 0.3), gravity_xyz=(0.2, 9.81, 0.3))
    assert residual < 0.5
    score = estimate_impact_score((0.2, 9.81, 0.3), 8.0, 4.0, gravity_xyz=(0.2, 9.81, 0.3))
    assert score is None


def test_android_config_export_matches_yaml():
    cfg = load_config(default_config_path())
    exported = json.loads(android_config_json(cfg))
    phone = repo_root() / "android" / "app" / "src" / "main" / "assets" / "rpar.defaults.json"
    assert phone.exists(), "run: python -m rpar.apps.cli export-android-config"
    on_phone = json.loads(phone.read_text(encoding="utf-8"))
    assert exported["schema_version"] == on_phone["schema_version"]
    assert exported["alert"]["bump_score_threshold"] == on_phone["alert"]["bump_score_threshold"]
    assert exported["perception"]["night_score_threshold"] == on_phone["perception"]["night_score_threshold"]
    assert exported["camera"]["preferred_hfov_deg"] == on_phone["camera"]["preferred_hfov_deg"]


def test_field_gates_reject_blind_night_and_spam_day():
    night_blind = m2_clip_gates(
        {
            "alias": "night_25013",
            "lighting": "night",
            "run": {
                "frames": 80,
                "overlay_ok": True,
                "n_confirmed_tracks": 0,
                "n_alerts_fired": 0,
                "confirmed_per_min": 0.0,
                "mean_infer_fps": 0.0,
                "infer_count": 0,
                "road_frame_share": 0.2,
            },
        }
    )
    assert night_blind["checks"]["night_did_infer"] is False
    assert night_blind["pass"] is False
    day_spam = m2_clip_gates(
        {
            "alias": "day_25007",
            "lighting": "day",
            "run": {
                "frames": 180,
                "overlay_ok": True,
                "n_confirmed_tracks": 12,
                "confirmed_per_min": 90.0,
                "n_alerts_fired": 8,
                "road_frame_share": 0.4,
                "mean_infer_fps": 8.0,
                "infer_count": 40,
            },
        }
    )
    assert day_spam["checks"]["day_not_spam"] is False
    assert day_spam["pass"] is False
