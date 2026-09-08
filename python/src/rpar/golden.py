"""Golden-video regression: run pipeline on simulator or mp4, emit metrics + overlay video."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rpar.config import RparConfig, load_config
from rpar.enums import Direction, LifecycleState, UiMode
from rpar.geometry import GeometryEngine
from rpar.models import Intrinsics
from rpar.overlay import compose
from rpar.perception import HeuristicPerceptionEngine
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

    def to_dict(self) -> dict[str, Any]:
        mae = float(np.mean(self.distance_err)) if self.distance_err else None
        return {
            "frames": self.frames,
            "confirmed": self.confirmed,
            "candidates": self.candidates,
            "alerts": self.alerts,
            "id_switches_est": self.id_switches_est,
            "direction_accuracy": (self.direction_correct / self.direction_total) if self.direction_total else None,
            "distance_mae_m": mae,
            "first_confirm_distance_m": self.first_confirm_distance,
            "blur_new_confirmed": self.blur_new_confirmed,
            "mean_infer_fps": self.mean_infer_fps,
            "p95_latency_ms": self.p95_latency_ms,
            "overlay_ok": self.overlay_ok,
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
    mount = sim.mount
    geom = GeometryEngine(mount, cfg.geometry, sim.k)
    pipe = RealtimePipeline(cfg, HeuristicPerceptionEngine(cfg), geom)
    overlay_path = out_dir / "overlay.mp4"
    raw_path = out_dir / "raw.mp4"
    write_preview_video(str(raw_path), sim)
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
    for i in range(frames):
        frame, gt = sim.frame_at(i)
        view = pipe.step(frame, ui_mode=ui_mode)
        vw.write(compose(frame.bgr, view, ui_mode))
        alerts += sum(1 for a in view.alerts if a.fired)
        for tr in view.tracks:
            if tr.lifecycle_state == LifecycleState.CONFIRMED:
                confirmed += 1
            if tr.lifecycle_state == LifecycleState.CANDIDATE:
                candidates += 1
        blur = any(a <= gt["t"] <= b for a, b in sim.sim.blur_windows)
        if blur:
            for tr in view.tracks:
                if tr.lifecycle_state == LifecycleState.CONFIRMED and tr.temporal_confidence < 0.35:
                    blur_new += 1
        for g in gt["objects"]:
            # nearest track by polygon center
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
            if best is None or best_d > 70**2:
                continue
            if best.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}:
                if g["id"] not in first_confirm:
                    first_confirm[g["id"]] = g["distance_m"]
                if best.distance_m is not None and best.distance_valid:
                    dist_err.append(abs(best.distance_m - g["distance_m"]))
                want = _gt_direction(g["x_m"], cfg.geometry.corridor_half_width_m)
                if g["semantic"] == "speed_bump":
                    want = Direction.CENTER_FRONT
                dir_n += 1
                if best.direction in {want, Direction.ACROSS} and want == Direction.CENTER_FRONT:
                    dir_ok += 1
                elif best.direction == want:
                    dir_ok += 1
                prev = last_ids.get(g["id"])
                if prev is not None and prev != best.track_id:
                    switches += 1
                last_ids[g["id"]] = best.track_id
    vw.release()
    metrics = GoldenMetrics(
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
        overlay_ok=overlay_path.exists() and overlay_path.stat().st_size > 1000,
    )
    payload = metrics.to_dict()
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_video_file(path: Path, out_dir: Path, cfg: RparConfig | None = None, max_frames: int = 400) -> dict[str, Any]:
    cfg = cfg or load_config()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    mount = default_mount(w, h)
    k = default_intrinsics(w, h)
    pipe = RealtimePipeline(cfg, HeuristicPerceptionEngine(cfg), GeometryEngine(mount, cfg.geometry, k))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    from rpar.models import FrameMeta, SynchronizedFrame

    i = 0
    t0 = 1_000_000_000_000
    while i < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        ts = t0 + int(i * 1e9 / max(fps, 1))
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
        view = pipe.step(frame, ui_mode=UiMode.RESEARCH)
        vw.write(compose(bgr, view, UiMode.RESEARCH))
        i += 1
    cap.release()
    vw.release()
    return {"frames": i, "out": str(out_dir / "overlay.mp4")}
