"""Scene-sliced eval, hard-negative report, confidence calibration (ML-003/004/008/009)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rpar import SCHEMA_VERSION
from rpar.config import RparConfig, load_config
from rpar.enums import LifecycleState, ObjectState, SemanticType, UiMode
from rpar.geometry import GeometryEngine
from rpar.golden import _accumulate, run_oracle_golden
from rpar.perception import oracle_engine_for_sim
from rpar.pipeline import RealtimePipeline
from rpar.simulator import RoadSimulator, SimConfig, WorldObject
from rpar.enums import GeometryType, Severity


SCENE_SLICES = ("day", "night", "wet", "backlight", "follow", "glare", "rough")
HARD_NEGATIVES = ("tree_shadow", "patch", "marking", "reflection", "manhole_normal", "vehicle_shadow")


def reliability_diagram(confidences: list[float], correct: list[int], bins: int = 10) -> dict[str, Any]:
    if not confidences:
        return {"bins": [], "ece": None, "n": 0}
    c = np.clip(np.asarray(confidences, dtype=np.float64), 0, 1)
    y = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        m = (c >= edges[i]) & (c < edges[i + 1] if i < bins - 1 else c <= edges[i + 1])
        if not np.any(m):
            rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "acc": None, "conf": None, "n": 0})
            continue
        acc = float(y[m].mean())
        conf = float(c[m].mean())
        rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "acc": acc, "conf": conf, "n": int(m.sum())})
        ece += abs(acc - conf) * (m.sum() / c.size)
    return {"bins": rows, "ece": float(ece), "n": int(c.size)}


def _hard_negative_world() -> list[WorldObject]:
    return [
        WorldObject("patch_flat", SemanticType.REPAIR_PATCH, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 18.0, -1.8, 1.6, 1.1, (42, 42, 48)),
        WorldObject("manhole_ok", SemanticType.MANHOLE_COVER, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 20.0, 1.4, 0.7, 0.7, (70, 72, 74)),
    ]


def hard_negative_report(cfg: RparConfig | None = None) -> dict[str, Any]:
    """Flat patches and normal covers must not produce voice alerts (ML-004 / ALT-004)."""
    cfg = cfg or load_config()
    sim = RoadSimulator(SimConfig(width=640, height=360, duration_s=1.4, fps=15, blur_windows=[]))
    sim.objects = _hard_negative_world()
    pipe = RealtimePipeline(cfg, oracle_engine_for_sim(sim), GeometryEngine(sim.mount, cfg.geometry, sim.k))
    fired = 0
    confirmed = 0
    confs: list[float] = []
    correct: list[int] = []
    for i in range(sim.n_frames()):
        frame, _gt = sim.frame_at(i)
        view = pipe.step(frame)
        fired += sum(1 for a in view.alerts if a.fired)
        for tr in view.tracks:
            if tr.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}:
                confirmed += 1
            confs.append(tr.effective_confidence)
            # hard negatives should stay low-risk / unalerted
            correct.append(0 if tr.alert_score < cfg.alert.score_threshold else 1)
    cal = reliability_diagram(confs, [1 - x for x in correct])
    return {
        "schema_version": SCHEMA_VERSION,
        "slices": {name: {"kind": "hard_negative"} for name in HARD_NEGATIVES},
        "n_alerts_fired": fired,
        "n_confirmed_rows": confirmed,
        "pass": fired == 0,
        "calibration": cal,
    }


def scene_matrix_report(out_dir: Path, cfg: RparConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "day": SimConfig(width=640, height=360, duration_s=1.2, fps=15, blur_windows=[]),
        "night": SimConfig(width=640, height=360, duration_s=1.2, fps=15, night=True, blur_windows=[]),
        "wet": SimConfig(width=640, height=360, duration_s=1.2, fps=15, wet=True, blur_windows=[]),
        "backlight": SimConfig(width=640, height=360, duration_s=1.2, fps=15, glare_windows=[(0.4, 1.1)]),
        "follow": SimConfig(width=640, height=360, duration_s=1.2, fps=15, occlude_windows=[(0.5, 1.0)]),
        "glare": SimConfig(width=640, height=360, duration_s=1.2, fps=15, glare_windows=[(0.7, 1.2)]),
        "rough": SimConfig(width=640, height=360, duration_s=1.2, fps=15, blur_windows=[(0.4, 0.9)]),
    }
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "slices": {}}
    for name in SCENE_SLICES:
        dest = out_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        sim = RoadSimulator(mapping[name])
        pipe = RealtimePipeline(cfg, oracle_engine_for_sim(sim), GeometryEngine(sim.mount, cfg.geometry, sim.k))
        m = _accumulate(pipe, sim, UiMode.RESEARCH, dest / "overlay.mp4")
        payload = m.to_dict()
        (dest / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report["slices"][name] = payload
    (out_dir / "scene_matrix.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def precision_cards(out_dir: Path) -> dict[str, Any]:
    """ML-008: separate cards so FP32 desktop numbers never stand in for quantized mobile."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = {
        "FP32": {"engine": "heuristic-cv", "note": "desktop reference; not a phone number"},
        "FP16": {"engine": "unset", "note": "requires sideloaded LiteRT package"},
        "INT8": {"engine": "unset", "note": "requires sideloaded LiteRT package; evaluate on PKC110 input"},
    }
    for name, body in cards.items():
        payload = {"schema_version": SCHEMA_VERSION, "precision": name, **body}
        (out_dir / f"{name.lower()}_card.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cards


def ab_compare(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> dict[str, Any]:
    keys = ("direction_accuracy", "distance_mae_m", "alerts", "mean_infer_fps", "p95_latency_ms")
    delta = {}
    for k in keys:
        a, b = metrics_a.get(k), metrics_b.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[k] = b - a
        else:
            delta[k] = None
    return {"a": metrics_a, "b": metrics_b, "delta_b_minus_a": delta}


def write_eval_bundle(out_dir: Path, cfg: RparConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = run_oracle_golden(out_dir / "oracle", cfg)
    scenes = scene_matrix_report(out_dir / "scenes", cfg)
    hn = hard_negative_report(cfg)
    (out_dir / "hard_negatives.json").write_text(json.dumps(hn, indent=2), encoding="utf-8")
    cards = precision_cards(out_dir / "precision")
    cal = hn.get("calibration") or {}
    (out_dir / "reliability.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    bundle = {"oracle": oracle, "scenes": scenes, "hard_negatives": hn, "precision_cards": cards, "calibration": cal}
    (out_dir / "eval_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle
