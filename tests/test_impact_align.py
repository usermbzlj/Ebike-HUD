from __future__ import annotations

import json

from rpar.field_video import m2_clip_gates, m2_field_report
from rpar.impact import (
    align_tracks_to_future_impact,
    harvest_impact_windows,
    impact_label,
    separates_relief_from_patch,
    synthetic_future_impact_proof,
)


def test_future_align_uses_pass_window_and_never_alerts():
    tracks = [
        {
            "track_id": 1,
            "timestamp_ns": 0,
            "lifecycle_state": "CONFIRMED",
            "semantic_type": "speed_bump",
            "geometry_type": "convex",
            "distance_m": 10.0,
            "speed_mps": 10.0,
        },
        {
            "track_id": 2,
            "timestamp_ns": 0,
            "lifecycle_state": "CONFIRMED",
            "semantic_type": "repair_patch",
            "geometry_type": "flat",
            "distance_m": 25.0,
            "speed_mps": 10.0,
        },
        {
            "track_id": 3,
            "timestamp_ns": 0,
            "lifecycle_state": "TRACKED",
            "semantic_type": "pothole",
            "geometry_type": "concave",
            "distance_m": 10.0,
            "speed_mps": 10.0,
        },
    ]
    accel = []
    for i in range(80):
        t = i * 50_000_000
        z = 9.81 + 8.0 if 18 <= i <= 22 else 9.81
        accel.append({"timestamp_ns": t, "z": z})
    rows = align_tracks_to_future_impact(tracks, accel)
    assert {r["track_id"] for r in rows} == {1, 2}
    assert all(r["used_for_alert"] is False for r in rows)
    bump = next(r for r in rows if r["track_id"] == 1)
    patch = next(r for r in rows if r["track_id"] == 2)
    assert bump["expected_pass_s"] == 1.0
    assert bump["peak_ms2"] > 5.0
    assert bump["impact_label"] != "none"
    assert patch["peak_ms2"] < bump["peak_ms2"]
    sep = separates_relief_from_patch(rows)
    assert sep["pass"] is True
    assert sep["used_for_alert"] is False


def test_synthetic_pass_imu_separates_relief_from_patch():
    proof = synthetic_future_impact_proof()
    assert proof["used_for_alert"] is False
    assert proof["n_aligned"] >= 4
    assert proof["separation"]["pass"] is True
    assert proof["separation"]["mean_relief_peak_ms2"] > proof["separation"]["mean_patch_peak_ms2"]
    assert impact_label(0.2, 10.0) == "none"
    assert impact_label(8.0, 10.0) in {"medium", "strong"}


def test_harvest_windows_still_never_alert():
    rows = harvest_impact_windows([{"timestamp_ns": 2, "z": 9.81 + 8.0}], near_track_ids=[7])
    assert rows[0]["used_for_alert"] is False


def test_recorded_session_writes_impact_align(tmp_path):
    from rpar.capture import record_simulated_session
    from rpar.session import verify_session
    from rpar.simulator import SimConfig

    root = record_simulated_session(
        tmp_path,
        sim_cfg=SimConfig(width=320, height=180, fps=10, duration_s=1.0, blur_windows=[]),
    )
    payload = json.loads((root / "perception" / "impact_align.json").read_text(encoding="utf-8"))
    assert payload["used_for_alert"] is False
    assert verify_session(root)["ok"] is True


def test_m2_night_gate_rejects_rough_broken_storm():
    night_bad = {
        "ok": True,
        "alias": "night_25013",
        "lighting": "night",
        "run": {
            "frames": 90,
            "overlay_ok": True,
            "n_confirmed_tracks": 12,
            "n_alerts_fired": 0,
            "n_rough_broken_confirmed": 12,
            "road_frame_share": 0.8,
            "confirmed_per_min": 8.0,
        },
    }
    night_ok = {
        "ok": True,
        "alias": "night_25013",
        "lighting": "night",
        "run": {
            "frames": 90,
            "overlay_ok": True,
            "n_confirmed_tracks": 2,
            "n_alerts_fired": 0,
            "n_rough_broken_confirmed": 0,
            "n_info_confirmed": 2,
            "confirmed_semantics": {"puddle": 2},
            "road_frame_share": 0.7,
            "confirmed_per_min": 2.0,
        },
    }
    day_ok = {
        "ok": True,
        "alias": "day_25007",
        "lighting": "day",
        "run": {
            "frames": 90,
            "overlay_ok": True,
            "n_confirmed_tracks": 1,
            "n_alerts_fired": 0,
            "n_rough_broken_confirmed": 0,
            "road_frame_share": 0.6,
            "confirmed_per_min": 0.4,
        },
    }
    assert m2_clip_gates(night_bad)["pass"] is False
    assert m2_clip_gates(night_ok)["pass"] is True
    assert m2_clip_gates(day_ok)["pass"] is True
    night_bump = {
        "ok": True,
        "alias": "night_25013",
        "lighting": "night",
        "run": {
            "frames": 90,
            "overlay_ok": True,
            "n_confirmed_tracks": 1,
            "n_alerts_fired": 1,
            "n_rough_broken_confirmed": 0,
            "n_info_confirmed": 0,
            "n_bump_confirmed": 1,
            "confirmed_semantics": {"pothole": 1},
            "road_frame_share": 0.7,
            "confirmed_per_min": 0.7,
        },
    }
    assert m2_clip_gates(night_bump)["pass"] is True
    assert "night_no_alerts" not in m2_clip_gates(night_bump)["checks"]
    report = m2_field_report([night_ok, day_ok])
    assert report["pass"] is True
    assert "first_confirm_distance_with_geometric_GT" in report["not_claimed"]
    assert m2_field_report([night_bad, day_ok])["pass"] is False
