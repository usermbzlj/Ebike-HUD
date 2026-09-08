"""Discover and run the local Video/ field clips (spec 1.2 / 11.4). MP4s stay untracked."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.config import RparConfig, load_config
from rpar.golden import run_video_file

# Spec §1.2 sample durations. Copies may be transcoded (fps/size differ; duration is the fingerprint).
_SPEC_CLIPS = (
    {"alias": "day_25007", "spec_name": "25007.mp4", "duration_s": 68.824, "lighting": "day"},
    {"alias": "night_25013", "spec_name": "25013.mp4", "duration_s": 55.014, "lighting": "night"},
)


def repo_video_dir(start: Path | None = None) -> Path:
    here = Path(start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        cand = p / "Video"
        if cand.is_dir():
            return cand
    return Path.cwd() / "Video"


def _rel_path(path: Path, video_dir: Path | None = None) -> str:
    root = (video_dir or repo_video_dir()).resolve().parent
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path, limit: int = 2_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        buf = f.read(limit)
        h.update(buf)
        if path.stat().st_size > limit:
            f.seek(max(0, path.stat().st_size - 65536))
            h.update(f.read())
            h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


def probe_clip(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "ok": False, "error": "unreadable"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    dur = n / fps if fps else 0.0
    means: list[float] = []
    for idx in [0, n // 4, n // 2, 3 * n // 4, max(0, n - 1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, im = cap.read()
        if ok:
            means.append(float(im.mean()))
    cap.release()
    luma = float(np.mean(means)) if means else 0.0
    lighting = "night" if luma < 105 else "day"
    alias = "extra"
    spec_name = None
    for spec in _SPEC_CLIPS:
        if abs(dur - spec["duration_s"]) <= 1.0:
            alias = spec["alias"]
            spec_name = spec["spec_name"]
            lighting = spec["lighting"]
            break
    return {
        "ok": True,
        "path": str(path),
        "relpath": _rel_path(path),
        "name": path.name,
        "alias": alias,
        "spec_name": spec_name,
        "width": w,
        "height": h,
        "fps": round(fps, 3),
        "frames": n,
        "duration_s": round(dur, 3),
        "mean_luma": round(luma, 2),
        "lighting": lighting,
        "bytes": path.stat().st_size,
        "sha256_prefix": _sha256(path),
        "note": "Likely a transcoded copy if size/fps differ from the 1080p60 / 1080p SDR originals.",
    }


def list_clips(video_dir: Path | None = None) -> list[dict[str, Any]]:
    d = Path(video_dir) if video_dir else repo_video_dir()
    if not d.is_dir():
        return []
    return [probe_clip(p) for p in sorted(d.glob("*.mp4")) if p.is_file()]


def write_catalog(video_dir: Path | None = None) -> dict[str, Any]:
    d = Path(video_dir) if video_dir else repo_video_dir()
    clips = list_clips(d)

    def public(clip: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in clip.items() if k != "path"}

    published = [public(c) for c in clips]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "directory": "Video",
        "n_clips": len(published),
        "clips": published,
        "spec_map": {
            "25007.mp4": next((c for c in published if c.get("alias") == "day_25007"), None),
            "25013.mp4": next((c for c in published if c.get("alias") == "night_25013"), None),
        },
    }
    out = d / "catalog.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def extract_preview(path: Path, dest: Path, at_ratio: float = 0.35) -> Path | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, int(n * at_ratio))))
    ok, im = cap.read()
    cap.release()
    if not ok:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return dest


def run_field_videos(
    video_dir: Path | None = None,
    out_dir: Path | None = None,
    cfg: RparConfig | None = None,
    max_frames: int = 180,
) -> dict[str, Any]:
    """Run the heuristic pipeline on every local field clip and write overlays + metrics."""
    cfg = cfg or load_config()
    d = Path(video_dir) if video_dir else repo_video_dir()
    out_dir = Path(out_dir) if out_dir else Path("artifacts") / "field_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = write_catalog(d)
    live = {c["name"]: c for c in list_clips(d)}
    results: list[dict[str, Any]] = []
    for clip in catalog["clips"]:
        if not clip.get("ok"):
            results.append(clip)
            continue
        src = Path(live[clip["name"]]["path"])
        alias = clip["alias"]
        dest = out_dir / alias
        dest.mkdir(parents=True, exist_ok=True)
        extract_preview(src, dest / "preview.jpg")
        metrics = run_video_file(src, dest, cfg, max_frames=max_frames)
        row = {**clip, "run": metrics}
        (dest / "metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(row)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "catalog": catalog,
        "max_frames": max_frames,
        "results": results,
        "n_ok": sum(1 for r in results if r.get("run", {}).get("frames", 0) > 0),
    }
    (out_dir / "field_video.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle
