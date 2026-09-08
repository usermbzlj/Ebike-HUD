from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rpar.enums import RunMode
from rpar.models import FrameMeta, MountProfile
from rpar.session import SessionManifest, SessionWriter, estimate_hours_remaining, verify_session
from rpar.transforms import default_mount


def test_session_checksum_and_segments(tmp_path: Path):
    mount = default_mount()
    man = SessionManifest(
        schema_version="1.0",
        session_id="test_sess",
        device={"manufacturer": "OPPO", "model": "PKC110", "os_version": "test"},
        camera_profile="rear_main_1080p60_sdr",
        mount_profile_id=mount.profile_id,
        calibration_hash="sha256:demo",
        model_packages=["heuristic-cv-0.1.0"],
        start_elapsed_realtime_ns=1000,
        end_elapsed_realtime_ns=None,
        video_segments=0,
        capability_report_hash="sha256:demo",
        privacy_mode="LOCAL_ONLY",
        run_mode=RunMode.REALTIME_PERCEPTION.value,
        config_snapshot={"schema_version": "1.0"},
    )
    w = SessionWriter(tmp_path / "session_test", man, mount)
    w.open_segment(fps=10, size=(64, 48), segment_frames=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    meta = FrameMeta(0, 1000, 1000, None, None, None, None, None, None, None, None, None, 64, 48, {"exposure": False})
    for i in range(3):
        meta.frame_id = i
        w.write_frame(frame, meta, None, None, None, None, {"blur": 0.1})
    w.finalize(2000)
    report = verify_session(tmp_path / "session_test")
    assert report["ok"] is True
    assert (tmp_path / "session_test" / "manifest.json").exists()
    assert (tmp_path / "session_test" / "checksums.sha256").exists()
    data = json.loads((tmp_path / "session_test" / "manifest.json").read_text(encoding="utf-8"))
    assert data["privacy_mode"] == "LOCAL_ONLY"
    assert data["run_mode"] == "REALTIME_PERCEPTION"


def test_record_session_observations_and_quality(tmp_path: Path):
    from rpar.capture import record_simulated_session
    from rpar.simulator import SimConfig

    root = record_simulated_session(
        tmp_path,
        sim_cfg=SimConfig(width=320, height=180, fps=15, duration_s=0.8, blur_windows=[]),
    )
    obs_lines = [ln for ln in (root / "perception" / "observations.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    runtime = [json.loads(ln) for ln in (root / "diagnostics" / "runtime.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert obs_lines, "FULL_LOG must persist RoadObservation"
    inferred = [row.get("inferred") for row in runtime]
    assert any(inferred)
    assert not all(inferred)
    assert all("degrade_reason" in row or "selected_for_infer" in row for row in runtime)
    imu = json.loads((root / "diagnostics" / "imu_intervals.json").read_text(encoding="utf-8"))
    assert "gyro" in imu and imu["gyro"]["hz"] > 0


def test_storage_estimate():
    hours = estimate_hours_remaining(11.25 * 1024**3, bitrate_mbps=25.0)
    assert 0.8 < hours < 1.3


def test_verify_session_recovers_part_mp4(tmp_path: Path):
    video = tmp_path / "video"
    video.mkdir()
    part = video / "segment_000.part.mp4"
    part.write_bytes(b"truncated-segment")
    report = verify_session(tmp_path)
    recovered = video / "segment_000.mp4.recovered"
    assert recovered.exists()
    assert recovered.read_bytes() == b"truncated-segment"
    assert any("unfinalized" in w for w in report["warnings"])
