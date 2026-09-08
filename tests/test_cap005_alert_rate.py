from __future__ import annotations

import json
from pathlib import Path

from rpar.alert_rate import alert_rate_report
from rpar.capability import cpu_bench_window
from rpar.transforms import display_compensate


def test_cpu_bench_window_records_percentiles():
    win = cpu_bench_window(duration_s=0.05, n=16)
    assert win["n_iters"] >= 1
    assert win["p50_ms"] > 0
    assert win["p95_ms"] >= win["p50_ms"]
    assert win["stable_10min"]["status"] == "short_probe"
    assert win["stable_10min"]["requested_s"] == 600.0


def test_display_compensate_shifts_on_yaw():
    poly = [(100.0, 200.0), (120.0, 200.0)]
    assert display_compensate(poly, 0.0, 40.0, 1920) == poly
    shifted = display_compensate(poly, 2.0, 40.0, 1920)
    assert shifted[0][0] > poly[0][0]
    assert shifted[0][1] == poly[0][1]


def test_alert_rate_insufficient_hours(tmp_path: Path):
    root = tmp_path / "sess"
    (root / "events").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "start_elapsed_realtime_ns": 0,
                "end_elapsed_realtime_ns": int(0.5 * 3600 * 1e9),
            }
        ),
        encoding="utf-8",
    )
    (root / "events" / "alerts.jsonl").write_text(
        json.dumps({"fired": True}) + "\n" + json.dumps({"fired": False}) + "\n",
        encoding="utf-8",
    )
    report = alert_rate_report([root])
    assert report["hours_ok"] is False
    assert report["n_alerts_fired"] == 1
    assert report["pass"] is False
    assert report["alerts_per_hour"] == 2.0
