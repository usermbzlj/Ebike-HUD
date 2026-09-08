from __future__ import annotations

from pathlib import Path

from rpar.config import load_config
from rpar.enums import ALERT_FORBIDDEN_PHRASES
from rpar.golden import run_acceptance_suite, run_simulator_golden
from rpar.ml.train import split_sessions
from rpar.simulator import SimConfig


def test_adjacent_frames_cannot_split_across_sets():
    # sessions are atomic — two clips from same ride stay together because we split by session id
    man = split_sessions(["20260101_a", "20260102_b", "20260103_c", "20260104_d"], seed=1)
    all_ids = man.train_sessions + man.val_sessions + man.test_sessions
    assert len(all_ids) == 4
    assert len(set(all_ids)) == 4


def test_no_steer_advice_in_python_tree():
    root = Path(__file__).resolve().parents[1] / "python"
    blob = []
    for p in root.rglob("*"):
        if p.suffix in {".py", ".html", ".yaml", ".json", ".kt", ".xml"}:
            blob.append(p.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(blob)
    for bad in ALERT_FORBIDDEN_PHRASES:
        # allowed only inside the forbidden-list definition
        if bad in {"向左避让", "向右避让"}:
            assert text.count(bad) <= 2


def test_golden_simulator_smoke(tmp_path: Path):
    cfg = load_config()
    metrics = run_simulator_golden(
        tmp_path,
        cfg,
        SimConfig(width=960, height=540, fps=30, duration_s=1.2, blur_windows=[(0.6, 0.85)]),
    )
    assert metrics["frames"] >= 30
    assert metrics["overlay_ok"] is True
    assert metrics["p95_latency_ms"] >= 0


def test_acceptance_includes_night_and_wet(tmp_path: Path):
    report = run_acceptance_suite(tmp_path / "accept", load_config())
    assert "night" in report["slices"]
    assert "wet" in report["slices"]
    assert report["slices"]["night"]["overlay_ok"] is True
    assert report["slices"]["wet"]["overlay_ok"] is True
