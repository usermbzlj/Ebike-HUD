from __future__ import annotations

from pathlib import Path

import pytest

from rpar.field_video import extract_preview, list_clips, repo_video_dir, run_field_videos, write_catalog
from rpar.golden import run_video_file


def _clips() -> list[dict]:
    return [c for c in list_clips(repo_video_dir()) if c.get("ok")]


@pytest.mark.skipif(not _clips(), reason="no mp4 in Video/")
def test_catalog_maps_spec_durations():
    clips = _clips()
    aliases = {c["alias"] for c in clips}
    assert "day_25007" in aliases or "night_25013" in aliases or aliases
    day = next((c for c in clips if c["alias"] == "day_25007"), None)
    night = next((c for c in clips if c["alias"] == "night_25013"), None)
    if day:
        assert day["lighting"] == "day"
        assert abs(day["duration_s"] - 68.824) <= 1.0
    if night:
        assert night["lighting"] == "night"
        assert abs(night["duration_s"] - 55.014) <= 1.0
    cat = write_catalog()
    assert cat["n_clips"] == len(clips)
    assert Path(cat["directory"], "catalog.json").exists()


@pytest.mark.skipif(not _clips(), reason="no mp4 in Video/")
def test_heuristic_runs_on_real_clip(tmp_path: Path):
    clip = _clips()[0]
    dest = tmp_path / "one"
    metrics = run_video_file(Path(clip["path"]), dest, max_frames=12)
    assert metrics["frames"] >= 8
    assert metrics["overlay_ok"] is True
    assert metrics["width"] >= 320
    preview = extract_preview(Path(clip["path"]), tmp_path / "preview.jpg")
    assert preview is not None and preview.exists()


@pytest.mark.skipif(len(_clips()) < 2, reason="need at least two field clips")
def test_field_suite_smoke(tmp_path: Path):
    bundle = run_field_videos(out_dir=tmp_path / "field", max_frames=8)
    assert bundle["n_ok"] >= 2
    assert (tmp_path / "field" / "field_video.json").exists()
