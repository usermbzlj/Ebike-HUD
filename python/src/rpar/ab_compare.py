"""Physical-damping A/B from two sessions (CAL-006): blur share and gyro peaks, not opinion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpar.report import build_report
from rpar.session import iter_jsonl


def _gyro_peak(session_dir: Path) -> float | None:
    mags = []
    for g in iter_jsonl(Path(session_dir) / "imu" / "gyro.jsonl"):
        mags.append(abs(float(g.get("x") or 0)) + abs(float(g.get("y") or 0)) + abs(float(g.get("z") or 0)))
    return max(mags) if mags else None


def damping_ab_report(session_a: Path, session_b: Path) -> dict[str, Any]:
    a = build_report(session_a)
    b = build_report(session_b)
    a["gyro_peak"] = _gyro_peak(session_a)
    b["gyro_peak"] = _gyro_peak(session_b)
    blur_a = a.get("blur_share")
    blur_b = b.get("blur_share")
    winner = None
    if isinstance(blur_a, (int, float)) and isinstance(blur_b, (int, float)):
        winner = "B" if blur_b < blur_a else ("A" if blur_a < blur_b else "tie")
    return {
        "A": a,
        "B": b,
        "winner_lower_blur_share": winner,
        "method": "auto_stats",
        "note": "Compare blur_share and gyro_peak; do not use subjective smoothness.",
    }


def write_damping_ab(session_a: Path, session_b: Path, out_path: Path) -> dict[str, Any]:
    payload = damping_ab_report(session_a, session_b)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
