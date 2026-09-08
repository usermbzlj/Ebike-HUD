"""Shared dual-scale road ROI fractions (far + near) in a unified image frame."""

from __future__ import annotations

import numpy as np

# Normalized crop of the driving corridor: far keeps density at 15–30 m; near is contour/severity.
FAR = (0.18, 0.32, 0.82, 0.62)
NEAR = (0.08, 0.50, 0.92, 1.00)


def crop_xyxy(w: int, h: int, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    ix0 = int(w * x0)
    iy0 = int(h * y0)
    ix1 = max(ix0 + 1, int(w * x1))
    iy1 = max(iy0 + 1, int(h * y1))
    return ix0, iy0, min(w, ix1), min(h, iy1)


def roi_polygon(w: int, h: int, box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    return [
        (float(w * x0), float(h * y0)),
        (float(w * x1), float(h * y0)),
        (float(w * x1), float(h * y1)),
        (float(w * x0), float(h * y1)),
    ]


def far_polygon(w: int, h: int) -> list[tuple[float, float]]:
    return roi_polygon(w, h, FAR)


def near_polygon(w: int, h: int) -> list[tuple[float, float]]:
    return roi_polygon(w, h, NEAR)


def dual_scale_feature_vector(
    gray: np.ndarray,
    far_wh: tuple[int, int] = (48, 24),
    near_wh: tuple[int, int] = (40, 32),
) -> np.ndarray:
    """12-D far/near scorer input matching the packaged linear LiteRT graph."""
    import cv2

    from rpar.ml.synth_train import patch_features

    h, w = gray.shape[:2]
    fx0, fy0, fx1, fy1 = crop_xyxy(w, h, FAR)
    nx0, ny0, nx1, ny1 = crop_xyxy(w, h, NEAR)
    far = gray[fy0:fy1, fx0:fx1]
    near = gray[ny0:ny1, nx0:nx1]
    far_s = (
        cv2.resize(far, far_wh, interpolation=cv2.INTER_AREA)
        if far.size
        else np.zeros((far_wh[1], far_wh[0]), dtype=np.float64)
    )
    near_s = (
        cv2.resize(near, near_wh, interpolation=cv2.INTER_AREA)
        if near.size
        else np.zeros((near_wh[1], near_wh[0]), dtype=np.float64)
    )
    return np.concatenate([patch_features(far_s), patch_features(near_s)])
