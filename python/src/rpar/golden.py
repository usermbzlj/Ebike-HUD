"""Golden-video regression: run pipeline on simulator or mp4, emit metrics + overlay video."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar.config import RparConfig, load_config
from rpar.enums import Direction, LifecycleState, SemanticType, UiMode
from rpar.geometry import GeometryEngine
from rpar.overlay import compose
from rpar.perception import HeuristicPerceptionEngine, load_field_engine, oracle_engine_for_sim
from rpar.pipeline import RealtimePipeline
from rpar.simulator import RoadSimulator, SimConfig, write_preview_video
from rpar.transforms import default_intrinsics, default_mount


@dataclass
class GoldenMetrics:
    frames: int
    confirmed: int
    candidates: int
    alerts: int
    id_switches_est: int
    direction_correct: int
    direction_total: int
    distance_err: list[float]
    first_confirm_distance: dict[str, float]
    blur_new_confirmed: int
    mean_infer_fps: float
    p95_latency_ms: float
    overlay_ok: bool
    action_zone_hits: int = 0
    action_zone_n: int = 0
    false_solid: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        mae = float(np.mean(self.distance_err)) if self.distance_err else None
        first_vals = [v for v in self.first_confirm_distance.values() if isinstance(v, (int, float))]
        return {
            "frames": self.frames,
            "confirmed": self.confirmed,
            "candidates": self.candidates,
            "alerts": self.alerts,
            "id_switches_est": self.id_switches_est,
            "direction_accuracy": (self.direction_correct / self.direction_total) if self.direction_total else None,
            "distance_mae_m": mae,
            "first_confirm_distance_m": self.first_confirm_distance,
            "first_confirm_median_m": float(np.median(first_vals)) if first_vals else None,
            "blur_new_confirmed": self.blur_new_confirmed,
            "mean_infer_fps": self.mean_infer_fps,
            "p95_latency_ms": self.p95_latency_ms,
            "overlay_ok": self.overlay_ok,
            "action_zone_recall": (self.action_zone_hits / self.action_zone_n) if self.action_zone_n else None,
            "false_solid_count": self.false_solid,
            "false_solid_per_min": (self.false_solid / max(self.duration_s / 60.0, 1e-6)) if self.duration_s else None,
        }


def acceptance_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    dir_acc = metrics.get("direction_accuracy")
    mae = metrics.get("distance_mae_m")
    first = metrics.get("first_confirm_distance_m") or {}
    first_vals = [v for v in first.values() if isinstance(v, (int, float))]
    passed = bool(
        dir_acc is not None and dir_acc >= 0.95 and mae is not None and mae <= 2.5 and metrics.get("overlay_ok")
    )
    return {
        "direction_ge_95": bool(dir_acc is not None and dir_acc >= 0.95),
        "distance_mae_5_15_le_2_5": bool(mae is not None and mae <= 2.5),
        "has_first_confirm": bool(first_vals),
        "overlay_ok": bool(metrics.get("overlay_ok")),
        "pass": passed,
    }


def _gt_direction(x_m: float, half_w: float = 0.85) -> Direction:
    if abs(x_m) <= half_w * 0.72:
        return Direction.CENTER_FRONT
    return Direction.LEFT_FRONT if x_m < 0 else Direction.RIGHT_FRONT


def run_simulator_golden(
    out_dir: Path,
    cfg: RparConfig | None = None,
    sim_cfg: SimConfig | None = None,
    ui_mode: UiMode = UiMode.RESEARCH,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    sim_cfg = sim_cfg or SimConfig(duration_s=3.5, blur_windows=[(1.2, 1.55)])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = RoadSimulator(sim_cfg)
    pipe = RealtimePipeline(cfg, HeuristicPerceptionEngine(cfg), GeometryEngine(sim.mount, cfg.geometry, sim.k))
    write_preview_video(str(out_dir / "raw.mp4"), sim)
    metrics = _accumulate(pipe, sim, ui_mode, out_dir / "overlay.mp4")
    payload = metrics.to_dict()
    payload["gates"] = acceptance_gates(payload)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _accumulate(pipe, sim, ui_mode, overlay_path: Path | None = None) -> GoldenMetrics:
    vw = None
    if overlay_path is not None:
        vw = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            sim.sim.fps,
            (sim.sim.width, sim.sim.height),
        )
    first_confirm: dict[str, float] = {}
    dist_err: list[float] = []
    dir_ok = dir_n = 0
    alerts = 0
    confirmed = candidates = 0
    blur_new = 0
    last_ids: dict[str, int] = {}
    switches = 0
    frames = sim.n_frames()
    match_px = 140.0
    zone_ids: set[str] = set()
    zone_confirmed: set[str] = set()
    confirmed_tracks: set[int] = set()
    matched_tracks: set[int] = set()
    for i in range(frames):
        frame, gt = sim.frame_at(i)
        view = pipe.step(frame, ui_mode=ui_mode)
        if vw is not None:
            vw.write(compose(frame.bgr, view, ui_mode))
        alerts += sum(1 for a in view.alerts if a.fired)
        for tr in view.tracks:
            if tr.lifecycle_state == LifecycleState.CONFIRMED:
                confirmed += 1
                confirmed_tracks.add(tr.track_id)
            if tr.lifecycle_state == LifecycleState.ALERTED:
                confirmed_tracks.add(tr.track_id)
            if tr.lifecycle_state == LifecycleState.CANDIDATE:
                candidates += 1
        blur = any(a <= gt["t"] <= b for a, b in sim.sim.blur_windows)
        if blur:
            for tr in view.tracks:
                if tr.lifecycle_state == LifecycleState.CONFIRMED and tr.temporal_confidence < 0.25:
                    blur_new += 1
        for g in gt["objects"]:
            if 5.0 <= float(g["distance_m"]) <= 15.0:
                zone_ids.add(g["id"])
            gx = np.mean([p[0] for p in g["polygon"]])
            gy = np.mean([p[1] for p in g["polygon"]])
            best = None
            best_d = 1e9
            for tr in view.tracks:
                if not tr.polygon:
                    continue
                tx = np.mean([p[0] for p in tr.polygon])
                ty = np.mean([p[1] for p in tr.polygon])
                d = (tx - gx) ** 2 + (ty - gy) ** 2
                if d < best_d:
                    best_d, best = d, tr
            if best is None or best_d > match_px**2:
                continue
            matched_tracks.add(best.track_id)
            if best.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED, LifecycleState.TRACKED}:
                if g["id"] not in first_confirm and best.lifecycle_state in {
                    LifecycleState.CONFIRMED,
                    LifecycleState.ALERTED,
                }:
                    first_confirm[g["id"]] = g["distance_m"]
                    zone_confirmed.add(g["id"])
                if best.distance_m is not None and best.distance_valid:
                    dist_err.append(abs(best.distance_m - g["distance_m"]))
                want = _gt_direction(g["x_m"], pipe.cfg.geometry.corridor_half_width_m)
                if g["semantic"] == "speed_bump":
                    want = Direction.CENTER_FRONT
                dir_n += 1
                if want == Direction.CENTER_FRONT and best.direction in {Direction.CENTER_FRONT, Direction.ACROSS}:
                    dir_ok += 1
                elif best.direction == want:
                    dir_ok += 1
                prev = last_ids.get(g["id"])
                if prev is not None and prev != best.track_id:
                    switches += 1
                last_ids[g["id"]] = best.track_id
    if vw is not None:
        vw.release()
    return GoldenMetrics(
        frames=frames,
        confirmed=confirmed,
        candidates=candidates,
        alerts=alerts,
        id_switches_est=switches,
        direction_correct=dir_ok,
        direction_total=dir_n,
        distance_err=dist_err,
        first_confirm_distance=first_confirm,
        blur_new_confirmed=blur_new,
        mean_infer_fps=pipe.last_view.infer_fps if pipe.last_view else 0.0,
        p95_latency_ms=pipe.last_view.latency_p95_ms if pipe.last_view else 0.0,
        overlay_ok=bool(overlay_path and overlay_path.exists() and overlay_path.stat().st_size > 1000),
        action_zone_hits=len(zone_ids & zone_confirmed) if zone_ids else len(zone_confirmed),
        action_zone_n=len(zone_ids) if zone_ids else 0,
        false_solid=len(confirmed_tracks - matched_tracks),
        duration_s=frames / max(float(sim.sim.fps), 1.0),
    )


def run_oracle_golden(
    out_dir: Path,
    cfg: RparConfig | None = None,
    sim_cfg: SimConfig | None = None,
    ui_mode: UiMode = UiMode.RESEARCH,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    sim_cfg = sim_cfg or SimConfig(duration_s=2.5, fps=30, blur_windows=[(1.0, 1.35)])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = RoadSimulator(sim_cfg)
    pipe = RealtimePipeline(cfg, oracle_engine_for_sim(sim), GeometryEngine(sim.mount, cfg.geometry, sim.k))
    metrics = _accumulate(pipe, sim, ui_mode, out_dir / "overlay.mp4")
    payload = metrics.to_dict()
    payload["gates"] = acceptance_gates(payload)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_acceptance_suite(out_dir: Path, cfg: RparConfig | None = None) -> dict[str, Any]:
    """Scene-sliced golden reports (ML-003 / 11.4)."""
    cfg = cfg or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = {
        "day": SimConfig(duration_s=2.2, fps=30, blur_windows=[]),
        "night": SimConfig(duration_s=2.2, fps=30, night=True, blur_windows=[]),
        "blur": SimConfig(duration_s=2.2, fps=30, blur_windows=[(0.7, 1.15)]),
        "glare": SimConfig(duration_s=2.2, fps=30, glare_windows=[(0.8, 1.2)]),
        "follow": SimConfig(duration_s=2.2, fps=30, occlude_windows=[(0.6, 1.1)]),
        "wet": SimConfig(duration_s=2.2, fps=30, wet=True, blur_windows=[]),
    }
    report: dict[str, Any] = {"schema_version": "1.0", "slices": {}}
    for name, sc in slices.items():
        dest = out_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        sim = RoadSimulator(sc)
        pipe = RealtimePipeline(cfg, oracle_engine_for_sim(sim), GeometryEngine(sim.mount, cfg.geometry, sim.k))
        m = _accumulate(pipe, sim, UiMode.RESEARCH, dest / "overlay.mp4")
        payload = m.to_dict()
        payload["gates"] = acceptance_gates(payload)
        (dest / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report["slices"][name] = payload
    (out_dir / "scene_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_video_file(
    path: Path,
    out_dir: Path,
    cfg: RparConfig | None = None,
    max_frames: int = 400,
    *,
    engine=None,
    prefer_yolop: bool = False,
    still_ratios: tuple[float, ...] = (0.25, 0.45, 0.65),
    ui_mode: UiMode = UiMode.RIDING,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    mount = default_mount(w, h)
    k = default_intrinsics(w, h)
    own_engine = engine is None
    eng = engine if engine is not None else load_field_engine(cfg, prefer_yolop=prefer_yolop)
    cap_info0 = eng.capability() if hasattr(eng, "capability") else {}
    sidecar0 = cap_info0.get("sidecar") if isinstance(cap_info0.get("sidecar"), dict) else {}
    version = cfg.model.package_id
    if cap_info0.get("hybrid"):
        version = f"{version}+{sidecar0.get('backend', 'sidecar')}"
    pipe = RealtimePipeline(cfg, eng, GeometryEngine(mount, cfg.geometry, k), model_version=version)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    from rpar.models import FrameMeta, SynchronizedFrame

    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is None or max_frames <= 0:
        limit = n_src if n_src > 0 else 10_000_000
    else:
        limit = max_frames
    still_base = n_src if (max_frames is None or max_frames <= 0) and n_src > 0 else limit
    i = 0
    t0 = 1_000_000_000_000
    unique_tracks: set[int] = set()
    confirmed_ids: set[int] = set()
    confirmed_sem: dict[int, str] = {}
    confirmed_dist_ids: set[int] = set()
    confirmed_dir_ids: set[int] = set()
    confirmed_rows = 0
    alerts = 0
    road_frames = 0
    occ_frames = 0
    blurs: list[float] = []
    glares: list[float] = []
    lumas: list[float] = []
    still_at = {max(0, int(still_base * r) - 1) for r in still_ratios}
    stills: list[str] = []
    while i < limit:
        ok, bgr = cap.read()
        if not ok:
            break
        ts = t0 + int(i * 1e9 / max(fps, 1))
        lumas.append(float(bgr.mean()))
        frame = SynchronizedFrame(
            meta=FrameMeta(
                frame_id=i,
                sensor_timestamp_ns=ts,
                image_timestamp_ns=ts,
                exposure_time_ns=None,
                iso=None,
                focal_length_mm=None,
                focus_distance_diopters=None,
                af_state=None,
                ae_state=None,
                awb_state=None,
                crop_region=None,
                stabilization_mode=None,
                width=w,
                height=h,
                availability={"exposure": False, "iso": False},
            ),
            bgr=bgr,
            pose=None,
            angular_velocity=None,
            linear_accel=None,
            location=None,
            speed_mps=10.5,
        )
        view = pipe.step(frame, ui_mode=ui_mode)
        composed = compose(bgr, view, ui_mode)
        vw.write(composed)
        if i in still_at:
            still_path = out_dir / f"overlay_{i:04d}.jpg"
            cv2.imwrite(str(still_path), composed, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            stills.append(str(still_path))
            ride = pipe.view_with_mode(view, UiMode.RIDING)
            ride_path = out_dir / f"riding_{i:04d}.jpg"
            cv2.imwrite(str(ride_path), compose(bgr, ride, UiMode.RIDING), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            stills.append(str(ride_path))
        if i and i % 60 == 0:
            print(f"  {path.name}: {i}/{limit} frames", flush=True)
        alerts += sum(1 for a in view.alerts if a.fired)
        blurs.append(view.blur)
        glares.append(view.glare)
        if view.road_polygon and len(view.road_polygon) >= 3:
            road_frames += 1
        if view.occluded_polygons:
            occ_frames += 1
        for tr in view.tracks:
            unique_tracks.add(tr.track_id)
            if tr.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}:
                confirmed_rows += 1
                confirmed_ids.add(tr.track_id)
                confirmed_sem.setdefault(tr.track_id, tr.semantic_type.value)
                if tr.distance_m is not None and tr.distance_valid:
                    confirmed_dist_ids.add(tr.track_id)
                if tr.direction != Direction.UNKNOWN:
                    confirmed_dir_ids.add(tr.track_id)
        i += 1
    cap.release()
    vw.release()
    cap_info = eng.capability() if hasattr(eng, "capability") else {}
    if own_engine:
        eng.close()
    luma = float(np.mean(lumas)) if lumas else 0.0
    sidecar = cap_info.get("sidecar") if isinstance(cap_info.get("sidecar"), dict) else cap_info
    duration_s = (i / fps) if fps else 0.0
    n_confirmed_tracks = len(confirmed_ids)
    sem_counts = dict(Counter(confirmed_sem.values()))
    return {
        "frames": i,
        "out": str(out_dir / "overlay.mp4"),
        "stills": stills,
        "engine": sidecar.get("backend", cap_info.get("backend")),
        "hybrid": bool(cap_info.get("hybrid")),
        "width": w,
        "height": h,
        "src_fps": fps,
        "duration_s": duration_s,
        "mean_luma": luma,
        "lighting": "night" if luma < 105 else "day",
        "n_unique_tracks": len(unique_tracks),
        "n_confirmed_rows": confirmed_rows,
        "n_confirmed_tracks": n_confirmed_tracks,
        "n_unconfirmed_tracks": max(0, len(unique_tracks) - n_confirmed_tracks),
        "n_alerts_fired": alerts,
        "confirmed_semantics": sem_counts,
        "n_rough_broken_confirmed": sum(1 for s in confirmed_sem.values() if s == SemanticType.ROUGH_BROKEN.value),
        "n_unknown_confirmed": sum(1 for s in confirmed_sem.values() if s == SemanticType.UNKNOWN_ANOMALY.value),
        "n_confirmed_with_distance": len(confirmed_dist_ids),
        "n_confirmed_with_direction": len(confirmed_dir_ids),
        "road_frame_share": (road_frames / i) if i else 0.0,
        "occlusion_frame_share": (occ_frames / i) if i else 0.0,
        "confirmed_per_min": (n_confirmed_tracks / duration_s * 60.0) if duration_s > 0 else 0.0,
        "mean_blur": float(np.mean(blurs)) if blurs else None,
        "mean_glare": float(np.mean(glares)) if glares else None,
        "mean_infer_fps": pipe.last_view.infer_fps if pipe.last_view else 0.0,
        "p95_latency_ms": pipe.last_view.latency_p95_ms if pipe.last_view else 0.0,
        "overlay_ok": (out_dir / "overlay.mp4").exists() and (out_dir / "overlay.mp4").stat().st_size > 1000,
    }
