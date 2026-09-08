"""Physical-damping A/B (CAL-006) and same-input model A/B (MOD-003)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpar.config import RparConfig, load_config
from rpar.enums import UiMode
from rpar.geometry import GeometryEngine
from rpar.perception import HeuristicPerceptionEngine, PerceptionEngine, oracle_engine_for_sim
from rpar.pipeline import RealtimePipeline
from rpar.report import build_report
from rpar.session import iter_jsonl
from rpar.simulator import RoadSimulator, SimConfig


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


def _summarize_views(views: list) -> dict[str, Any]:
    n_alerts = 0
    n_confirmed = 0
    first_confirm_m = None
    for v in views:
        n_alerts += sum(1 for a in v.alerts if a.fired)
        for t in v.tracks:
            if t.lifecycle_state.value in {"CONFIRMED", "ALERTED"}:
                n_confirmed += 1
                if first_confirm_m is None and t.distance_m is not None:
                    first_confirm_m = float(t.distance_m)
    return {
        "n_alerts": n_alerts,
        "n_confirmed_track_frames": n_confirmed,
        "first_confirm_m": first_confirm_m,
    }


def _run_engine_on_frames(
    cfg: RparConfig,
    engine: PerceptionEngine,
    sim: RoadSimulator,
    frames: list,
    version: str,
) -> dict[str, Any]:
    pipe = RealtimePipeline(cfg, engine, GeometryEngine(sim.mount, cfg.geometry, sim.k), model_version=version)
    views = [pipe.step(f, ui_mode=UiMode.RESEARCH) for f in frames]
    out = _summarize_views(views)
    out["engine"] = version
    out["backend"] = views[-1].backend.value if views else None
    return out


def model_ab_same_input(cfg: RparConfig | None = None, duration_s: float = 1.0) -> dict[str, Any]:
    """MOD-003: two engines on identical simulator frames and the same metrics."""
    cfg = cfg or load_config()
    sim = RoadSimulator(SimConfig(width=320, height=180, fps=12, duration_s=duration_s))
    frames = [sim.frame_at(i)[0] for i in range(sim.n_frames())]
    a = _run_engine_on_frames(cfg, HeuristicPerceptionEngine(cfg), sim, frames, "heuristic")
    b = _run_engine_on_frames(cfg, oracle_engine_for_sim(sim), sim, frames, "oracle")
    return {
        "method": "same_input",
        "n_frames": len(frames),
        "A": a,
        "B": b,
        "delta": {
            "n_alerts": (b["n_alerts"] or 0) - (a["n_alerts"] or 0),
            "n_confirmed_track_frames": (b["n_confirmed_track_frames"] or 0) - (a["n_confirmed_track_frames"] or 0),
        },
        "note": "Same frames, same metrics; A=heuristic CV, B=oracle.",
    }


def write_model_ab(out_path: Path, cfg: RparConfig | None = None) -> dict[str, Any]:
    payload = model_ab_same_input(cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
