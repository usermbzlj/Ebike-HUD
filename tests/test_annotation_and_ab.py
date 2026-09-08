from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from rpar.ab_compare import damping_ab_report
from rpar.annotation import hit_test_tracks, tracks_to_annotation_task, tracks_to_cvat_xml
from rpar.capture import record_simulated_session
from rpar.config import load_config
from rpar.ml.active import build_active_queue
from rpar.ml.synth_train import train_dual_scale_linear, write_training_bundle
from rpar.simulator import SimConfig
from rpar.transforms import chessboard_overlay_error_px, default_intrinsics, default_mount


def test_chessboard_overlay_under_8px():
    err = chessboard_overlay_error_px(default_mount(), default_intrinsics(), n=4)
    assert err <= load_config().render.overlay_error_px


def test_cvat_and_hit_and_active(tmp_path: Path):
    root = record_simulated_session(
        tmp_path / "sessions",
        sim_cfg=SimConfig(width=320, height=180, fps=12, duration_s=0.7, blur_windows=[]),
    )
    tracks = root / "perception" / "tracks.jsonl"
    xml = tracks_to_cvat_xml(tracks, tmp_path / "cvat.xml")
    assert xml.exists()
    ET.parse(xml)
    task = tracks_to_annotation_task(tracks, tmp_path / "task.json")
    assert "items" in task
    q = build_active_queue(root)
    assert "queue" in q
    hit = hit_test_tracks([{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "track_id": 9}], 5, 5)
    assert hit and hit["track_id"] == 9
    miss = hit_test_tracks([{"polygon": [[0, 0], [2, 0], [2, 2], [0, 2]], "track_id": 1}], 50, 50)
    assert miss is None


def test_synth_train_and_damping_ab(tmp_path: Path):
    model = train_dual_scale_linear(seed=3, n_pos=24, n_neg=24)
    assert model["train_acc"] >= 0.7
    assert "INT8" in model["weights"]
    bundle = write_training_bundle(tmp_path / "train")
    assert (tmp_path / "train" / "splits.json").exists()
    assert bundle["split"]["isolation"] == "session_date_route"
    assert "tflite" in bundle
    assert "ok" in bundle["tflite"]
    a = record_simulated_session(tmp_path / "a", sim_cfg=SimConfig(width=240, height=136, fps=10, duration_s=0.5))
    b = record_simulated_session(tmp_path / "b", sim_cfg=SimConfig(width=240, height=136, fps=10, duration_s=0.5))
    ab = damping_ab_report(a, b)
    assert ab["method"] == "auto_stats"
    assert "blur_share" in ab["A"]


def test_model_ab_cli_payload(tmp_path: Path):
    from rpar.ab_compare import write_model_ab
    out = write_model_ab(tmp_path / "model_ab.json")
    assert out["method"] == "same_input"
    assert (tmp_path / "model_ab.json").exists()
