from __future__ import annotations

import numpy as np

from rpar.config import QualityConfig
from rpar.enums import VisibilityClass
from rpar.quality import evaluate_frame


def test_sharp_vs_blurred():
    rng = np.random.default_rng(0)
    sharp = np.zeros((360, 640, 3), dtype=np.uint8)
    sharp[:] = 80
    sharp[200:340, 80:560] = 70
    sharp[220:230, 100:540] = 200  # lane
    sharp[250:300, 300:360] = 20  # dark blob
    blur = sharp.copy()
    import cv2

    blur = cv2.GaussianBlur(blur, (31, 31), 12)
    cfg = QualityConfig()
    qs = evaluate_frame(sharp, cfg)
    qb = evaluate_frame(blur, cfg)
    assert qs.global_quality.sharpness > qb.global_quality.sharpness
    assert qb.global_quality.visibility_class in {VisibilityClass.BLUR, VisibilityClass.UNKNOWN} or qb.global_quality.usable is False
