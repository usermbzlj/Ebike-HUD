"""Record a complete SessionBundle from the simulator + pipeline (REC-001, AR-009)."""

from __future__ import annotations

from pathlib import Path

import cv2

from rpar.config import RparConfig, load_config
from rpar.enums import RunMode, UiMode
from rpar.events import BufferedFrame, EventClipBuffer, write_event_clip
from rpar.geometry import GeometryEngine
from rpar.overlay import compose
from rpar.perception import PerceptionEngine, oracle_engine_for_sim
from rpar.pipeline import RealtimePipeline
from rpar.session import SessionManifest, SessionWriter, new_session_id
from rpar.simulator import RoadSimulator, SimConfig


def record_simulated_session(
    out_dir: Path,
    *,
    cfg: RparConfig | None = None,
    sim_cfg: SimConfig | None = None,
    engine: PerceptionEngine | None = None,
    ui_mode: UiMode = UiMode.RESEARCH,
) -> Path:
    cfg = cfg or load_config()
    sim = RoadSimulator(sim_cfg or SimConfig(duration_s=3.0, fps=30, blur_windows=[(1.1, 1.4)]))
    eng = engine or oracle_engine_for_sim(sim)
    pipe = RealtimePipeline(cfg, eng, GeometryEngine(sim.mount, cfg.geometry, sim.k))
    sid = new_session_id("SIM")
    root = Path(out_dir) / f"session_{sid}"
    man = SessionManifest(
        schema_version="1.0",
        session_id=sid,
        device={"manufacturer": "SIM", "model": "synthetic", "os_version": "desktop"},
        camera_profile="rear_main_1080p60_sdr",
        mount_profile_id=sim.mount.profile_id,
        calibration_hash="sha256:synthetic",
        model_packages=[cfg.model.package_id],
        start_elapsed_realtime_ns=sim.t0_ns,
        end_elapsed_realtime_ns=None,
        video_segments=0,
        capability_report_hash="sha256:synthetic",
        privacy_mode="LOCAL_ONLY",
        run_mode=RunMode.REALTIME_PERCEPTION_FULL_LOG.value,
        config_snapshot=cfg.snapshot(),
        notes="synthetic session for replay/regression",
    )
    writer = SessionWriter(root, man, sim.mount)
    fps = sim.sim.fps
    writer.open_segment(fps, (sim.sim.width, sim.sim.height), cfg.camera.segment_seconds * fps)
    overlay_path = root / "video" / "overlay_preview.mp4"
    ov = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (sim.sim.width, sim.sim.height))
    last_ns = sim.t0_ns
    imu_hz = 200
    clips = EventClipBuffer(pre_s=2.0, post_s=1.5, fps=float(fps))
    clip_i = 0
    for i in range(sim.n_frames()):
        frame, _gt = sim.frame_at(i)
        view = pipe.step(frame, ui_mode=ui_mode)
        loc = None
        if frame.location:
            loc = {
                "timestamp_ns": frame.location.timestamp_ns,
                "latitude": frame.location.latitude,
                "longitude": frame.location.longitude,
                "altitude": frame.location.altitude,
                "speed_mps": frame.location.speed_mps,
                "bearing_deg": frame.location.bearing_deg,
                "horizontal_accuracy_m": frame.location.horizontal_accuracy_m,
                "speed_accuracy_mps": frame.location.speed_accuracy_mps,
            }
        writer.write_frame(
            frame.bgr,
            frame.meta,
            None,
            loc,
            view.tracks,
            view.alerts,
            {
                "blur": view.blur,
                "glare": view.glare,
                "infer_fps": view.infer_fps,
                "latency_p95_ms": view.latency_p95_ms,
                "status": view.status.value,
                "backend": view.backend.value,
                "degrade_reason": view.quality.degrade_reason if view.quality else None,
                "selected_for_infer": view.quality.selected_for_infer if view.quality else None,
                "selected_age_ms": view.quality.selected_age_ms if view.quality else None,
                "occupancy_occluded_ratio": view.quality.occupancy_occluded_ratio if view.quality else None,
                "sharpness": view.quality.global_quality.sharpness if view.quality else None,
                "inferred": pipe.did_infer,
                "thermal_c": view.thermal_c,
                "thermal_reason": pipe.thermal_reason,
                "skip_far_roi": pipe.skip_far_roi,
                "dropped_infer": pipe.dropped_infer,
                "timestamp_ns": frame.meta.sensor_timestamp_ns,
                "queue_depth": view.queue_depth,
            },
            observations=[o.to_dict() for o in pipe.last_observations] if pipe.did_infer else None,
        )
        vis = compose(frame.bgr, view, ui_mode)
        ov.write(vis)
        done = clips.push(BufferedFrame(frame.meta.sensor_timestamp_ns, i, vis, view.blur, len(view.tracks)))
        for a in view.alerts:
            if a.fired:
                clips.on_alert(a.timestamp_ns, a.to_dict())
        for bundle in done:
            clip_i += 1
            write_event_clip(bundle, root / "events" / "clips" / f"clip_{clip_i:03d}.mp4", fps)
        for ev in pipe.status_events:
            writer.write_event(ev)
        pipe.status_events.clear()
        t0 = i / fps
        t1 = (i + 1) / fps
        n = max(1, int(imu_hz / fps))
        for k in range(n):
            t = t0 + (t1 - t0) * k / n
            ts = sim.t0_ns + int(t * 1e9)
            gyro, accel = sim.motion_at(t)
            writer.write_imu(
                {
                    "timestamp_ns": ts,
                    "sensor_type": "GYRO",
                    "x": gyro[0],
                    "y": gyro[1],
                    "z": gyro[2],
                    "accuracy": 3,
                    "source_rate_hz": imu_hz,
                }
            )
            writer.write_imu(
                {
                    "timestamp_ns": ts,
                    "sensor_type": "ACCEL",
                    "x": accel[0],
                    "y": accel[1],
                    "z": accel[2],
                    "accuracy": 3,
                    "source_rate_hz": imu_hz,
                }
            )
            if k == 0 and frame.pose:
                writer.write_imu(
                    {
                        "timestamp_ns": ts,
                        "sensor_type": "ROTATION_VECTOR",
                        "x": frame.pose.quaternion_xyzw[0],
                        "y": frame.pose.quaternion_xyzw[1],
                        "z": frame.pose.quaternion_xyzw[2],
                        "accuracy": 3,
                        "source_rate_hz": imu_hz,
                    }
                )
        last_ns = frame.meta.sensor_timestamp_ns
    ov.release()
    writer.finalize(last_ns)
    return root
