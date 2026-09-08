from __future__ import annotations

import numpy as np

from rpar.config import load_config
from rpar.enums import SemanticType
from rpar.models import FrameMeta, SynchronizedFrame
from rpar.perception import HeuristicPerceptionEngine


def _frame(bgr: np.ndarray) -> SynchronizedFrame:
    h, w = bgr.shape[:2]
    return SynchronizedFrame(
        meta=FrameMeta(
            frame_id=0,
            sensor_timestamp_ns=1,
            image_timestamp_ns=1,
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
        speed_mps=10.0,
    )


def test_smooth_night_asphalt_has_no_rough_broken():
    rng = np.random.default_rng(0)
    h, w = 180, 320
    bgr = np.full((h, w, 3), 48, dtype=np.uint8)
    bgr[90:, :, :] = (rng.integers(38, 62, size=(90, w, 3))).astype(np.uint8)
    eng = HeuristicPerceptionEngine(load_config())
    out = eng.infer(_frame(bgr))
    rough = [o for o in out.observations if o.semantic_type == SemanticType.ROUGH_BROKEN]
    assert rough == []
    spam = {SemanticType.ROUGH_BROKEN, SemanticType.POTHOLE, SemanticType.UNKNOWN_ANOMALY, SemanticType.GRAVEL}
    assert [o for o in out.observations if o.semantic_type in spam] == []


def test_night_field_frame_does_not_carpet_rough():
    import cv2

    from rpar.field_video import list_clips, repo_video_dir

    night = next((c for c in list_clips(repo_video_dir()) if c.get("alias") == "night_25013"), None)
    if night is None:
        return
    cap = cv2.VideoCapture(night["path"])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, int(n * 0.4))))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return
    eng = HeuristicPerceptionEngine(load_config())
    out = eng.infer(_frame(bgr))
    spam = {SemanticType.ROUGH_BROKEN, SemanticType.POTHOLE, SemanticType.UNKNOWN_ANOMALY, SemanticType.GRAVEL}
    assert [o for o in out.observations if o.semantic_type in spam] == []
    assert len(out.observations) <= 8
