"""Crash-log sanitization and share-pack privacy (SEC-002/003/006, REC-007)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {
    "latitude",
    "longitude",
    "altitude",
    "file_path",
    "absolute_path",
    "path",
    "video_path",
    "frame_bgr",
    "jpeg",
    "bitmap",
    "preview_bytes",
}

PATH_RE = re.compile(r"([A-Za-z]:)?[/\\][^\s\"']+\.(mp4|jpg|png|jsonl|json|bin)", re.I)


def sanitize_value(key: str, value: Any) -> Any:
    lk = key.lower()
    if lk in SENSITIVE_KEYS or any(s in lk for s in ("lat", "lon", "filepath", "abspath")):
        if isinstance(value, (int, float)):
            return None
        return "[redacted]"
    if isinstance(value, str) and PATH_RE.search(value):
        return PATH_RE.sub("[path]", value)
    if isinstance(value, dict):
        return sanitize_event(value)
    if isinstance(value, list):
        return [sanitize_value(key, v) for v in value]
    return value


def sanitize_event(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: sanitize_value(k, v) for k, v in obj.items()}


def sanitize_crash_log(text: str) -> str:
    """Crash logs must not contain video frames, precise GPS, or full file paths."""
    text = PATH_RE.sub("[path]", text)
    text = re.sub(r'"latitude"\s*:\s*-?\d+(\.\d+)?', '"latitude": null', text)
    text = re.sub(r'"longitude"\s*:\s*-?\d+(\.\d+)?', '"longitude": null', text)
    return text


def coarsen_location(sample: dict[str, Any], decimals: int = 2) -> dict[str, Any]:
    out = dict(sample)
    for k in ("latitude", "longitude"):
        if isinstance(out.get(k), (int, float)):
            out[k] = round(float(out[k]), decimals)
    out["horizontal_accuracy_m"] = max(float(out.get("horizontal_accuracy_m") or 0), 150.0)
    out["privacy"] = "coarsened"
    return out


def write_sanitized_jsonl(src: Path, dest: Path, fn) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as out:
        if src.exists():
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                out.write(json.dumps(fn(json.loads(line)), ensure_ascii=False) + "\n")
                n += 1
    return n
