from __future__ import annotations

import numpy as np

from rpar.ml.bump_prompts import map_det_name
from rpar.ml.yolo_decode import decode_to_original, decode_yolo_output, letterbox_meta, xyxy_letterbox_to_orig


NAMES = ["pothole", "speed bump", "sunken manhole cover"]


def test_letterbox_meta_1080p():
    meta = letterbox_meta(1080, 1920, 320)
    assert abs(float(meta["gain"]) - 320 / 1920) < 1e-6
    assert float(meta["pad_x"]) == 0.0
    assert float(meta["pad_y"]) >= 69.0


def test_decode_nms_rows():
    out = np.zeros((1, 8, 6), dtype=np.float32)
    out[0, 0] = [100, 120, 180, 200, 0.91, 2]
    out[0, 1] = [102, 122, 178, 198, 0.40, 2]
    dets = decode_yolo_output(out, NAMES, conf_thr=0.08)
    assert len(dets) == 1
    assert dets[0]["name"] == "sunken manhole cover"
    assert map_det_name(dets[0]["name"]) == "manhole_cover"


def test_decode_raw_channels_first():
    out = np.zeros((1, 7, 32), dtype=np.float32)
    out[0, 0, 3] = 160
    out[0, 1, 3] = 200
    out[0, 2, 3] = 80
    out[0, 3, 3] = 40
    out[0, 6, 3] = 0.88
    dets = decode_yolo_output(out, NAMES, conf_thr=0.08)
    assert len(dets) == 1
    assert dets[0]["name"] == "sunken manhole cover"
    x0, y0, x1, y1 = dets[0]["bbox"]
    assert abs(x0 - 120) < 1e-3
    assert abs(y1 - 220) < 1e-3


def test_map_back_to_original_pixels():
    out = np.zeros((1, 1, 6), dtype=np.float32)
    meta = letterbox_meta(360, 640, 320)
    # A 320-letterbox box around the lower-middle of the canvas.
    out[0, 0] = [140, 180, 200, 230, 0.7, 0]
    dets = decode_to_original(out, NAMES, 640, 360, imgsz=320, conf_thr=0.08)
    assert len(dets) == 1
    box = dets[0]["bbox"]
    orig = xyxy_letterbox_to_orig((140, 180, 200, 230), meta, 640, 360)
    assert abs(box[0] - orig[0]) < 1e-4
    assert box[2] > box[0]
    assert box[3] > box[1]


def test_empty_background_class_skipped():
    names = ["pothole", ""]
    out = np.zeros((1, 2, 6), dtype=np.float32)
    out[0, 0] = [10, 10, 40, 40, 0.9, 1]
    assert decode_yolo_output(out, names) == []
