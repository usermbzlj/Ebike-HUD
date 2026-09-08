from __future__ import annotations

from pathlib import Path

import numpy as np

from rpar.maskutil import mask_rle_from_polygon, rle_decode, rle_encode, rect_polygon
from rpar.models import LocationSample
from rpar.session import copy_resumable, export_split_zip
from rpar.timebase import interpolate_location


def test_rle_roundtrip_square():
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 1
    counts = rle_encode(mask)
    back = rle_decode(counts, 10, 8)
    assert np.array_equal(mask, back)


def test_instance_rle_from_polygon():
    poly = rect_polygon(10.0, 20.0, 40.0, 50.0)
    rle = mask_rle_from_polygon(poly)
    assert rle is not None
    assert rle.width >= 1 and rle.height >= 1
    decoded = rle_decode(rle.counts, rle.width, rle.height)
    assert decoded.sum() > 0


def test_location_interpolation_midpoint():
    a = LocationSample(0, 31.0, 121.0, 8.0, 10.0, 0.0, 4.0, 0.4)
    b = LocationSample(1_000_000_000, 32.0, 122.0, 10.0, 20.0, 20.0, 4.0, 0.4)
    mid = interpolate_location([a, b], 500_000_000)
    assert mid is not None
    assert mid.interpolated is True
    assert abs(mid.latitude - 31.5) < 1e-9
    assert abs((mid.speed_mps or 0) - 15.0) < 1e-9


def test_split_zip_and_resumable_copy(tmp_path: Path):
    src = tmp_path / "sess"
    (src / "video").mkdir(parents=True)
    (src / "a.txt").write_text("hello", encoding="utf-8")
    (src / "video" / "seg.bin").write_bytes(b"x" * 4000)
    parts = export_split_zip(src, tmp_path / "out", max_bytes=2500)
    assert len(parts) >= 2
    assert all(p.exists() and p.stat().st_size > 0 for p in parts)
    raw = tmp_path / "blob.bin"
    raw.write_bytes(b"abcdef" * 100)
    dest = tmp_path / "blob.copy"
    dest.write_bytes(b"abcdef" * 10)
    copy_resumable(raw, dest)
    assert dest.read_bytes() == raw.read_bytes()
