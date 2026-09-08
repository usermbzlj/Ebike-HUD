from __future__ import annotations

from pathlib import Path

from rpar.capability import desktop_capability_stub
from rpar.config import load_config
from rpar.golden import run_oracle_golden
from rpar.ml.eval import hard_negative_report, precision_cards, reliability_diagram, scene_matrix_report
from rpar.ml.train import write_model_package
from rpar.simulator import SimConfig


def test_reliability_diagram_ece():
    conf = [0.1, 0.2, 0.8, 0.9]
    y = [0, 0, 1, 1]
    r = reliability_diagram(conf, y, bins=2)
    assert r["n"] == 4
    assert r["ece"] is not None
    assert r["ece"] >= 0


def test_oracle_golden_meets_geo_gates(tmp_path: Path):
    metrics = run_oracle_golden(
        tmp_path,
        load_config(),
        SimConfig(width=960, height=540, fps=20, duration_s=1.6, blur_windows=[(0.7, 0.95)]),
    )
    assert metrics["direction_accuracy"] is not None
    assert metrics["distance_mae_m"] is not None
    assert metrics["gates"]["direction_ge_95"] is True
    assert metrics["gates"]["distance_mae_5_15_le_2_5"] is True
    assert metrics["gates"]["pass"] is True
    assert metrics["first_confirm_distance_m"]


def test_hard_negatives_do_not_speak():
    hn = hard_negative_report(load_config())
    assert hn["n_alerts_fired"] == 0
    assert hn["pass"] is True


def test_model_package_has_tflite_and_sha(tmp_path: Path):
    d = write_model_package(tmp_path / "pkg", "heuristic-cv-0.1.0")
    assert (d / "model.tflite").exists()
    man = (d / "manifest.json").read_text(encoding="utf-8")
    assert "model.tflite" in man
    assert "sha256" in man


def test_eval_bundle_scene_slices(tmp_path: Path):
    scenes = scene_matrix_report(tmp_path / "scenes", load_config())
    assert set(scenes["slices"]) >= {"day", "night", "follow", "glare", "rain", "vibration"}
    cards = precision_cards(tmp_path / "precision")
    assert "INT8" in cards


def test_desktop_capability_cpu_microbench():
    cap = desktop_capability_stub()
    results = cap["acceleration"]["results"]
    assert results
    assert results[0]["backend"] == "CPU"
    assert results[0]["status"] == "ok"
    assert "p95_ms" in results[0]


def test_desktop_capability_concurrent_schema():
    cap = desktop_capability_stub()
    combo = cap["camera"]["concurrent_streams"][0]
    assert combo["requested_fps"] == 60
    assert "measured_yuv_fps" in combo
    cam0 = cap["camera"]["cameras"][0]
    assert cam0["concurrent_streams"][0]["combo"].startswith("preview+yuv+record@1080p")
    assert cam0["ae_target_fps_ranges"]
