"""Active-learning queue from uncertain tracks, alerts, and unknown semantics (ML-010)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpar.session import iter_jsonl


def build_active_queue(session_dir: Path, max_n: int = 200) -> dict[str, Any]:
    root = Path(session_dir)
    tracks = list(iter_jsonl(root / "perception" / "tracks.jsonl"))
    alerts = list(iter_jsonl(root / "events" / "alerts.jsonl"))
    fired_ids = {int(a.get("track_id") or -1) for a in alerts if a.get("fired")}
    scored: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for t in tracks:
        tid = int(t.get("track_id") or 0)
        fid = int(t.get("source_frame_id") or 0)
        key = (tid, fid)
        if key in seen:
            continue
        seen.add(key)
        eff = float(t.get("effective_confidence") or 0.5)
        unc = 1.0 - abs(eff - 0.5) * 2.0
        reasons: list[str] = []
        if 0.28 <= eff <= 0.72:
            reasons.append("uncertain_confidence")
            unc += 0.15
        sem = str(t.get("semantic_type") or "")
        if sem in {"unknown_anomaly", "unknown"}:
            reasons.append("unknown_anomaly")
            unc += 0.35
        if tid in fired_ids:
            reasons.append("alert_event")
            unc += 0.25
        vis = t.get("visibility_confidence")
        if isinstance(vis, (int, float)) and vis < 0.4:
            reasons.append("low_visibility")
            unc += 0.2
        if not reasons:
            continue
        scored.append(
            {
                "track_id": tid,
                "frame_id": fid,
                "timestamp_ns": t.get("timestamp_ns"),
                "semantic_type": sem,
                "score": round(unc, 4),
                "reasons": reasons,
                "source": "session_events",
            }
        )
    scored.sort(key=lambda r: r["score"], reverse=True)
    queue = scored[: max(1, max_n)]
    return {
        "session": str(root),
        "n_candidates": len(scored),
        "queue": queue,
        "priority_sources": ["uncertain_confidence", "unknown_anomaly", "alert_event", "low_visibility"],
    }


def write_active_queue(session_dir: Path, out_path: Path, max_n: int = 200) -> dict[str, Any]:
    payload = build_active_queue(session_dir, max_n=max_n)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
