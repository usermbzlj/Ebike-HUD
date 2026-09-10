from __future__ import annotations

import numpy as np

from rpar.ml.bump_dataset import reuse_raw_labels
from rpar.ml.bump_prompts import keep_box
from rpar.ml.label_sam import box_ok, mask_to_xyxy
from rpar.ml.label_store import (
    commit_frame,
    image_stem,
    merge_objects,
    parse_yolo_txt,
    write_yolo_txt,
    yolo_to_xyxy,
)


def test_yolo_roundtrip_keeps_car_sized_box(tmp_path):
    # Human labels must not go through keep_box: that filter dropped real pits and kept car rears.
    box = (760.0, 400.0, 1160.0, 760.0)
    assert keep_box("pothole", box, 1920, 1080) is False
    path = tmp_path / "lab.txt"
    write_yolo_txt(path, [{"class": "pothole", "bbox": box}], 1920, 1080)
    got = parse_yolo_txt(path, 1920, 1080)
    assert len(got) == 1
    assert got[0]["class"] == "pothole"
    back = got[0]["bbox"]
    assert abs(back[0] - box[0]) < 2 and abs(back[2] - box[2]) < 2


def test_yolo_to_xyxy_center():
    x0, y0, x1, y1 = yolo_to_xyxy(0.5, 0.5, 0.25, 0.10, 200, 100)
    assert abs((x0 + x1) / 2 - 100) < 1e-6
    assert abs(x1 - x0 - 50) < 1e-6
    assert abs(y1 - y0 - 10) < 1e-6


def test_merge_replaces_overlap():
    old = [{"class": "pothole", "bbox": (10, 10, 40, 40)}]
    new = [{"class": "pothole", "bbox": (12, 12, 44, 42), "source": "human"}]
    merged = merge_objects(old, new)
    assert len(merged) == 1
    assert merged[0]["source"] == "human"


def test_mask_to_xyxy_and_box_ok():
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:20, 30:50] = True
    box = mask_to_xyxy(mask)
    assert box == (30.0, 10.0, 50.0, 20.0)
    assert box_ok(None, (30, 10, 50, 20), 80, 40) is True
    assert box_ok((30, 10, 50, 20), (0, 0, 2, 2), 80, 40) is False


def test_commit_frame_writes_raw_jpg_and_txt(tmp_path):
    import cv2

    bgr = np.full((180, 320, 3), 40, dtype=np.uint8)
    out = commit_frame(
        tmp_path,
        clip="ride",
        source_name="ride.mp4",
        frame_index=12,
        bgr=bgr,
        objects=[{"class": "speed_bump", "bbox": (20, 80, 200, 110)}],
    )
    assert out["ok"] is True
    stem = image_stem("ride", 12)
    assert (tmp_path / "images" / "raw" / f"{stem}.jpg").is_file()
    body = (tmp_path / "labels" / "raw" / f"{stem}.txt").read_text(encoding="utf-8")
    assert body.startswith("1 ")
    reused = reuse_raw_labels(tmp_path)
    assert reused["n_positive_frames"] == 1
    assert reused["n_train_images"] + reused["n_val_images"] >= 1
    img = cv2.imread(str(tmp_path / "images" / "raw" / f"{stem}.jpg"))
    assert img is not None and img.shape[0] == 180
