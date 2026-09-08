from __future__ import annotations

import numpy as np

from rpar.timebase import (
    QuaternionInterpolator,
    TimeInterpolator,
    alignment_error_p95_ms,
    detect_gaps,
    match_capture_results,
    percentile_intervals_ms,
    s_to_ns,
)


def test_linear_interpolation():
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    v = np.array([[0.0, 10.0], [10.0, 20.0]])
    ip = TimeInterpolator(t, v)
    mid = ip.at(500_000_000)
    assert abs(mid[0] - 5.0) < 1e-9
    assert abs(mid[1] - 15.0) < 1e-9


def test_slerp_no_flip():
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    q = np.array([[0, 0, 0, 1.0], [0, 0, 0, -1.0]])  # same rotation, opposite sign
    qi = QuaternionInterpolator(t, q)
    mid = qi.at(500_000_000)
    assert abs(np.linalg.norm(mid) - 1) < 1e-6


def test_match_and_reorder():
    frames = [10, 20, 15, 30]
    results = [10, 20, 30]
    pairs, anom = match_capture_results(frames, results, max_dt_ns=2)
    codes = {a.code for a in anom}
    assert "REORDER" in codes
    assert len(pairs) == 3


def test_alignment_p95_under_10ms():
    cam = [s_to_ns(i / 60) for i in range(120)]
    imu = [s_to_ns(i / 200) for i in range(400)]
    p95 = alignment_error_p95_ms(cam, imu)
    assert p95 <= 10.0


def test_gap_detection():
    ts = [0, 5_000_000, 10_000_000, 80_000_000]
    gaps = detect_gaps(ts, expected_hz=200, stall_factor=4)
    assert any(g.code == "SENSOR_STALL" for g in gaps)


def test_percentiles():
    ts = [i * 5_000_000 for i in range(50)]
    st = percentile_intervals_ms(ts)
    assert 4.5 < st["p50_ms"] < 5.5
