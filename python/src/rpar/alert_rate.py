"""M4 false-voice rate from session alert logs. Does not invent 10 h of riding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpar.session import iter_jsonl

FALSE_VOICE_HOURS_MIN = 10.0
FALSE_VOICE_PER_HOUR_MAX = 0.5


def session_hours(root: Path) -> float:
    man_p = Path(root) / "manifest.json"
    if man_p.is_file():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        start = man.get("start_elapsed_realtime_ns")
        end = man.get("end_elapsed_realtime_ns")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            return float(end - start) / 1e9 / 3600.0
    cam = Path(root) / "camera" / "frame_metadata.jsonl"
    ts: list[int] = []
    if cam.is_file():
        for row in iter_jsonl(cam):
            t = row.get("sensor_timestamp_ns") or row.get("timestamp_ns")
            if t:
                ts.append(int(t))
            if len(ts) >= 50_000:
                break
    if len(ts) >= 2:
        return max(0.0, (max(ts) - min(ts)) / 1e9 / 3600.0)
    return 0.0


def count_fired_alerts(root: Path) -> int:
    alerts = Path(root) / "events" / "alerts.jsonl"
    n = 0
    if alerts.is_file():
        for row in iter_jsonl(alerts):
            if row.get("fired"):
                n += 1
    return n


def alert_rate_report(session_dirs: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hours = 0.0
    fired = 0
    for p in session_dirs:
        root = Path(p)
        h = session_hours(root)
        n = count_fired_alerts(root)
        hours += h
        fired += n
        rows.append({"session": str(root), "hours": h, "n_alerts_fired": n})
    rate = (fired / hours) if hours > 0 else None
    hours_ok = hours >= FALSE_VOICE_HOURS_MIN
    rate_ok = bool(rate is not None and rate <= FALSE_VOICE_PER_HOUR_MAX)
    return {
        "schema_version": "1.0",
        "n_sessions": len(rows),
        "hours": hours,
        "n_alerts_fired": fired,
        "alerts_per_hour": rate,
        "hours_ok": hours_ok,
        "rate_ok": rate_ok,
        "pass": hours_ok and rate_ok,
        "threshold_hours": FALSE_VOICE_HOURS_MIN,
        "threshold_per_hour": FALSE_VOICE_PER_HOUR_MAX,
        "sessions": rows,
        "note": "M4 gate needs ≥10 h of PKC110 riding. This tool only aggregates existing session logs.",
    }


def write_alert_rate(session_dirs: list[Path], out: Path) -> dict[str, Any]:
    report = alert_rate_report(session_dirs)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
