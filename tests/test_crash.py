from __future__ import annotations

import json
from pathlib import Path

from rpar.crash import linear_accel_residual, scan_accel_jsonl, severe_impact
from rpar.session import verify_session


def test_gravity_is_not_crash():
    assert severe_impact(0.0, 0.0, 9.81) is False
    assert linear_accel_residual(0.0, 0.0, 9.81) < 1.0


def test_pothole_two_g_is_not_crash():
    assert severe_impact(0.0, 0.0, 9.81 + 20.0) is False


def test_four_g_residual_is_crash():
    assert severe_impact(0.0, 0.0, 9.81 + 40.0) is True


def test_scan_accel_jsonl(tmp_path: Path):
    p = tmp_path / "accelerometer.jsonl"
    rows = [
        {"timestamp_ns": 1, "x": 0.0, "y": 0.0, "z": 9.81},
        {"timestamp_ns": 2, "x": 0.0, "y": 0.0, "z": 55.0},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    scan = scan_accel_jsonl(p)
    assert scan["n_samples"] == 2
    assert scan["n_hits"] == 1
    assert scan["used_for_alert"] is False


def test_verify_session_crash_warning(tmp_path: Path):
    root = tmp_path / "sess"
    (root / "imu").mkdir(parents=True)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    acc = root / "imu" / "accelerometer.jsonl"
    acc.write_text(json.dumps({"timestamp_ns": 3, "x": 0, "y": 0, "z": 60}) + "\n", encoding="utf-8")
    report = verify_session(root)
    assert report["cam012_crash_scan"]["n_hits"] == 1
    assert any("CAM-012" in w for w in report["warnings"])
