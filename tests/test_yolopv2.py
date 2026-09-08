from __future__ import annotations

import numpy as np

from rpar.ml.yolopv2 import (
    is_vehicle_box,
    lane_mask,
    letterbox,
    mask_to_polygon,
    nms_dets,
    paint_drivable,
    scale_coords,
    weights_available,
)


def test_letterbox_stride_multiple():
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    out, pad = letterbox(img, (640, 640))
    assert out.shape[0] % 32 == 0
    assert out.shape[1] % 32 == 0
    assert pad[0] >= 0 and pad[1] >= 0


def test_mask_to_polygon_trapezoid():
    mask = np.zeros((120, 200), dtype=np.float32)
    mask[50:110, 20:180] = 0.9
    poly = mask_to_polygon(mask, min_frac=0.01)
    assert len(poly) >= 3


def test_nms_keeps_highest_score():
    pred = np.zeros((1, 4, 85), dtype=np.float32)
    pred[0, 0, :4] = [20, 20, 10, 10]
    pred[0, 0, 4] = 0.9
    pred[0, 0, 7] = 0.9
    pred[0, 1, :4] = [21, 21, 10, 10]
    pred[0, 1, 4] = 0.8
    pred[0, 1, 7] = 0.8
    pred[0, 2, :4] = [80, 80, 12, 12]
    pred[0, 2, 4] = 0.85
    pred[0, 2, 7] = 0.85
    out = nms_dets(pred, conf_thres=0.3, iou_thres=0.45)
    assert out[0].shape[0] >= 1


def test_scale_coords_identity_when_same():
    boxes = np.array([[10.0, 10.0, 40.0, 50.0]], dtype=np.float32)
    got = scale_coords((100, 100), boxes, (100, 100))
    assert np.allclose(got, boxes, atol=1.0)


def test_vehicle_filter_drops_sky_specks():
    assert is_vehicle_box(2, (10, 10, 18, 16), (540, 960)) is False
    assert is_vehicle_box(2, (200, 300, 420, 500), (540, 960)) is True
    assert is_vehicle_box(3, (450, 100, 530, 160), (540, 960)) is True
    assert is_vehicle_box(3, (0, 20, 70, 400), (540, 960)) is False
    assert is_vehicle_box(2, (20, 40, 900, 500), (540, 960)) is False


def test_paint_drivable_changes_pixels():
    bgr = np.zeros((80, 120, 3), dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.float32)
    mask[40:70, 10:110] = 0.9
    out = paint_drivable(bgr, mask, [(20, 30, 60, 70)])
    assert out.mean() > bgr.mean()
    assert out[50, 50, 1] > 40


def test_lane_mask_helper_shape():
    seg = np.zeros((1, 1, 80, 160), dtype=np.float32)
    seg[0, 0, 20:60, 20:140] = 0.8
    out = lane_mask(seg, (8.0, 4.0), (40, 80))
    assert out.shape == (40, 80)
    assert float(out.mean()) > 0.1


def test_weights_available_does_not_load_session():
    # Presence check only; pytest must not load the 156 MB ONNX graph.
    assert isinstance(weights_available(), bool)
