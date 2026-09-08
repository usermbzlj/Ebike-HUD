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


def test_storage_estimate():
    hours = estimate_hours_remaining(11.25 * 1024**3, bitrate_mbps=25.0)
    assert 0.8 < hours < 1.3
