"""Scene-sliced eval, hard-negative report, confidence calibration (ML-003/004/008/009)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rpar import SCHEMA_VERSION
from rpar.config import RparConfig, load_config
from rpar.enums import GeometryType, LifecycleState, ObjectState, SemanticType, Severity, UiMode
from rpar.geometry import GeometryEngine
from rpar.golden import _accumulate, run_oracle_golden
from rpar.perception import oracle_engine_for_sim
from rpar.pipeline import RealtimePipeline
from rpar.simulator import RoadSimulator, SimConfig, WorldObject


SCENE_SLICES = ("day", "night", "wet", "backlight", "follow", "glare", "rough", "rain", "vibration")
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


def _hard_negative_worlds() -> dict[str, list[WorldObject]]:
    return {
        "tree_shadow": [
            WorldObject("tree_shadow", SemanticType.UNKNOWN_ANOMALY, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 16.0, -1.4, 4.0, 0.45, (22, 22, 24)),
        ],
        "patch": [
            WorldObject("patch_flat", SemanticType.REPAIR_PATCH, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 18.0, -1.8, 1.6, 1.1, (42, 42, 48)),
        ],
        "marking": [
            WorldObject("marking", SemanticType.REPAIR_PATCH, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 15.0, 0.0, 0.18, 2.8, (210, 210, 220)),
        ],
        "reflection": [
            WorldObject("wet_glint", SemanticType.PUDDLE, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 12.0, 0.6, 1.4, 0.9, (200, 200, 210)),
        ],
        "manhole_normal": [
            WorldObject("manhole_ok", SemanticType.MANHOLE_COVER, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 20.0, 1.4, 0.7, 0.7, (70, 72, 74)),
        ],
        "vehicle_shadow": [
            WorldObject("veh_shadow", SemanticType.UNKNOWN_ANOMALY, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 10.0, 0.2, 3.5, 2.4, (18, 18, 20)),
        ],
    }


def _hard_negative_world() -> list[WorldObject]:
    out: list[WorldObject] = []
    for objs in _hard_negative_worlds().values():
        out.extend(objs)
    return out


def _slice_hard_negative(name: str, objs: list[WorldObject], cfg: RparConfig) -> dict[str, Any]:
    sim = RoadSimulator(SimConfig(width=640, height=360, duration_s=1.0, fps=15, blur_windows=[]))
    sim.objects = objs
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
            correct.append(0 if tr.alert_score < cfg.alert.score_threshold else 1)
    return {
        "kind": "hard_negative",
        "n_alerts_fired": fired,
        "n_confirmed_rows": confirmed,
        "pass": fired == 0,
        "calibration": reliability_diagram(confs, [1 - x for x in correct]),
    }


def hard_negative_report(cfg: RparConfig | None = None) -> dict[str, Any]:
    """Flat patches, shadows, markings and normal covers must not produce voice alerts (ML-004 / ALT-004)."""
    cfg = cfg or load_config()
    worlds = _hard_negative_worlds()
    slices = {name: _slice_hard_negative(name, objs, cfg) for name, objs in worlds.items()}
    fired = sum(int(s["n_alerts_fired"]) for s in slices.values())
    confirmed = sum(int(s["n_confirmed_rows"]) for s in slices.values())
    cal = slices.get("patch", {}).get("calibration") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "slices": slices,
        "n_alerts_fired": fired,
        "n_confirmed_rows": confirmed,
        "pass": fired == 0 and all(bool(s["pass"]) for s in slices.values()),
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
        "rain": SimConfig(width=640, height=360, duration_s=1.2, fps=15, rain=True, wet=True, blur_windows=[(0.4, 0.8)]),
        "vibration": SimConfig(width=640, height=360, duration_s=1.2, fps=15, vibration=True, blur_windows=[(0.2, 1.0)]),
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
        "FP16": {"engine": "classmap-weight-cast", "note": "see train-seg precision_cards; re-eval on PKC110"},
        "INT8": {"engine": "classmap-weight-cast", "note": "weight-cast INT8 on synthetic frame; not PKC110 Camera2"},
    }
    try:
        from rpar.ml.seg_train import classmap_precision_cards, train_dual_scale_classmap

        seg = classmap_precision_cards(np.asarray(train_dual_scale_classmap(seed=3)["weights"]))
        for name, body in seg.get("cards", {}).items():
            cards[name] = {**cards.get(name, {}), **body, "engine": "dual_scale_classmap"}
    except Exception as exc:
        cards["error"] = str(exc)[:200]
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
    from rpar.impact import synthetic_future_impact_proof

    impact = synthetic_future_impact_proof()
    impact_pub = {k: v for k, v in impact.items() if k != "rows"}
    (out_dir / "impact_m5.json").write_text(json.dumps(impact, indent=2), encoding="utf-8")
    bundle = {
        "oracle": oracle,
        "scenes": scenes,
        "hard_negatives": hn,
        "precision_cards": cards,
        "calibration": cal,
        "impact_m5": impact_pub,
    }
    (out_dir / "eval_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle
