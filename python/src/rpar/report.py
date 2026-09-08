"""Ride statistics HTML/JSON reports (REP-004)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rpar.session import iter_jsonl, verify_session


def build_report(session_dir: Path) -> dict[str, Any]:
    root = Path(session_dir)
    v = verify_session(root)
    tracks = list(iter_jsonl(root / "perception" / "tracks.jsonl"))
    alerts = list(iter_jsonl(root / "events" / "alerts.jsonl"))
    diag = list(iter_jsonl(root / "diagnostics" / "runtime.jsonl"))
    vis = Counter(t.get("lifecycle_state") for t in tracks)
    fired = [a for a in alerts if a.get("fired")]
    blur = [d.get("blur") for d in diag if d.get("blur") is not None]
    gyro = list(iter_jsonl(root / "imu" / "gyro.jsonl"))
    gyro_mags = [abs(float(g.get("x") or 0)) + abs(float(g.get("y") or 0)) + abs(float(g.get("z") or 0)) for g in gyro]
    report = {
        "session": v.get("manifest"),
        "verify_ok": v.get("ok"),
        "verify_errors": v.get("errors"),
        "track_states": dict(vis),
        "n_track_rows": len(tracks),
        "n_alerts_fired": len(fired),
        "n_alert_decisions": len(alerts),
        "mean_blur": sum(blur) / len(blur) if blur else None,
        "blur_share": (sum(1 for b in blur if (b or 0) > 0.5) / len(blur)) if blur else None,
        "mean_infer_fps": _mean([d.get("infer_fps") for d in diag]),
        "p95_latency_ms": _p95([d.get("latency_p95_ms") for d in diag]),
        "mean_thermal_c": _mean([d.get("thermal_c") for d in diag]),
        "mean_battery_pct": _mean([d.get("battery_pct") for d in diag]),
        "gyro_peak": max(gyro_mags) if gyro_mags else None,
        "usable_frame_share": (sum(1 for b in blur if (b or 0) < 0.5) / len(blur)) if blur else None,
        "camera_interval": v.get("camera_interval"),
        "n_inferred": sum(1 for d in diag if d.get("inferred")),
        "mean_memory_mb": _mean([d.get("memory_mb") for d in diag]),
        "thermal_series": [
            {"timestamp_ns": d.get("timestamp_ns"), "thermal_c": d.get("thermal_c"), "battery_pct": d.get("battery_pct")}
            for d in diag
            if d.get("thermal_c") is not None or d.get("battery_pct") is not None
        ][:: max(1, len(diag) // 240 or 1)],
        "visibility": dict(Counter(t.get("visibility_confidence") is not None for t in tracks)),
    }
    try:
        from rpar.impact import align_session_impact

        impact = align_session_impact(root)
        report["impact_m5"] = {
            "n_aligned": impact.get("n_aligned"),
            "labels": impact.get("labels"),
            "used_for_alert": False,
            "separation": impact.get("separation"),
        }
    except Exception as exc:
        report["impact_m5"] = {"error": str(exc)[:200], "used_for_alert": False}
    return report


def _mean(vals: list) -> float | None:
    xs = [float(v) for v in vals if isinstance(v, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _p95(vals: list) -> float | None:
    xs = sorted(float(v) for v in vals if isinstance(v, (int, float)))
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def write_report(session_dir: Path, out_json: Path, out_html: Path | None = None) -> dict[str, Any]:
    report = build_report(session_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if out_html:
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in report.items())
        out_html.write_text(
            f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>RPAR ride report</title>
<style>body{{font-family:Segoe UI,sans-serif;background:#07090d;color:#e8eef2;padding:24px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #223;padding:8px;text-align:left}}</style>
<h1>Road Perception AR · 骑行报告</h1><table>{rows}</table></html>""",
            encoding="utf-8",
        )
    return report
