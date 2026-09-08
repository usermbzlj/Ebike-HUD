"""Harvest mis-alerts and manual marks into a repeatable error-case library (spec 11.4)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from rpar.session import iter_jsonl


def harvest_error_cases(session_dir: Path, out_dir: Path) -> dict[str, Any]:
    session_dir = Path(session_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for a in iter_jsonl(session_dir / "events" / "alerts.jsonl"):
        if a.get("fired"):
            items.append(
                {
                    "kind": "alert",
                    "timestamp_ns": a.get("timestamp_ns"),
                    "track_id": a.get("track_id"),
                    "phrase": a.get("phrase"),
                    "semantic_type": a.get("semantic_type"),
                    "snapshot": a.get("snapshot") or a,
                }
            )
    for m in iter_jsonl(session_dir / "events" / "marks.jsonl"):
        items.append({"kind": "mark", **m})
    clip_dir = session_dir / "events" / "clips"
    copied = 0
    if clip_dir.exists():
        dest_clips = out_dir / "clips"
        dest_clips.mkdir(parents=True, exist_ok=True)
        for f in sorted(clip_dir.iterdir()):
            if f.is_file():
                shutil.copy2(f, dest_clips / f.name)
                copied += 1
    payload = {
        "session_id": session_dir.name,
        "n_items": len(items),
        "n_clips": copied,
        "items": items,
        "note": "Replay the parent session to inspect video, IMU and decision snapshots.",
    }
    (out_dir / "error_cases.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
