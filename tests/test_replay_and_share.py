from __future__ import annotations

import json
from pathlib import Path

from rpar.capture import record_simulated_session
from rpar.privacy import sanitize_crash_log, sanitize_event
from rpar.replay import SessionReplay, scan_time_offset_ms
from rpar.session import iter_jsonl, verify_session
from rpar.share import export_share_bundle, redact_bgr
from rpar.simulator import SimConfig
import numpy as np


def test_iter_jsonl_missing(tmp_path: Path):
    assert list(iter_jsonl(tmp_path / "nope.jsonl")) == []


def test_record_replay_offset_and_share(tmp_path: Path):
    root = record_simulated_session(
        tmp_path / "sessions",
        sim_cfg=SimConfig(width=320, height=180, fps=15, duration_s=0.8, blur_windows=[(0.3, 0.5)]),
    )
    report = verify_session(root)
    assert report["ok"] is True
    assert (root / "imu" / "gyro.jsonl").stat().st_size > 0
    assert (root / "video" / "overlay_preview.mp4").exists()
    gyro_n = sum(1 for _ in iter_jsonl(root / "imu" / "gyro.jsonl"))
    assert gyro_n >= 20

    rep = SessionReplay(root)
    rep.open()
    st = rep.at(2)
    assert st["n_frames"] >= 8
    assert len(st["gyro"]) == 3
    assert "imu_series" in st
    clip = rep.export_clip(tmp_path / "clip.mp4", 0, 4)
    assert clip.exists() and clip.stat().st_size > 100
    rep.close()

    off = scan_time_offset_ms(root)
    assert off["ok"] is True
    assert -200 <= off["best_offset_ms"] <= 200

    share_dir = tmp_path / "share"
    out = export_share_bundle(root, share_dir)
    assert out["privacy_mode"] == "SHARE_REDACTED"
    man = json.loads((share_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["privacy_mode"] == "SHARE_REDACTED"
    orig = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert orig["privacy_mode"] == "LOCAL_ONLY"


def test_redact_and_crash_sanitize():
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[40:50, 20:70] = 255
    out = redact_bgr(img)
    assert out.shape == img.shape
    leaked = '{"latitude": 31.2304, "path": "C:\\\\Users\\\\a\\\\ride.mp4", "msg": "ok"}'
    cleaned = sanitize_crash_log(leaked)
    assert "31.2304" not in cleaned
    assert ".mp4" not in cleaned
    ev = sanitize_event({"latitude": 31.2, "code": "SYNC_STALL", "detail": "gap"})
    assert ev["latitude"] is None
    assert ev["code"] == "SYNC_STALL"
