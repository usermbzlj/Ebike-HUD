"""End-to-end realtime loop: quality → schedule → infer → track → geometry → alerts → AR."""

from __future__ import annotations

from collections import deque

import numpy as np

from rpar.alerts import AlertPolicy, visual_score
from rpar.config import RparConfig
from rpar.enums import (
    Direction,
    InferenceBackend,
    LifecycleState,
    PerceptionStatus,
    RIDING_STATUS_COPY,
    UiMode,
)
from rpar.geometry import GeometryEngine, should_mark_passed
from rpar.models import (
    AlertDecision,
    FrameQualityMap,
    PerceptionView,
    RenderPrimitive,
    SynchronizedFrame,
    TrackedRoadObject,
)
from rpar.perception import PerceptionEngine
from rpar.quality import QualityScheduler, evaluate_frame, perception_status
from rpar.tracking import TrackEngine, TrackInternal
from rpar import SCHEMA_VERSION


def _temporal(tr: TrackInternal, now_ns: int, confirm_s: float) -> float:
    age = (now_ns - tr.created_ns) / 1e9
    stab = min(1.0, tr.hits / 6.0) * min(1.0, age / max(confirm_s, 1e-3))
    miss_pen = 0.85 ** tr.misses
    return float(np.clip(stab * miss_pen, 0.05, 1.0))


def _geom_consistency(tr: TrackInternal, dist_conf: float) -> float:
    if tr.road_xy is None:
        return 0.35
    return float(np.clip(0.4 + 0.6 * dist_conf, 0, 1))


class RealtimePipeline:
    def __init__(
        self,
        cfg: RparConfig,
        engine: PerceptionEngine,
        geometry: GeometryEngine,
        *,
        model_version: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.geometry = geometry
        self.tracker = TrackEngine(cfg.tracking)
        self.alerts = AlertPolicy(cfg.alert)
        self.scheduler = QualityScheduler(cfg.quality)
        self.model_version = model_version or cfg.model.package_id
        self.last_ns: int | None = None
        self.bad_streak_s = 0.0
        self._latencies: deque[float] = deque(maxlen=120)
        self._infer_times: deque[int] = deque(maxlen=40)
        self.status = PerceptionStatus.NORMAL
        self.last_view: PerceptionView | None = None
        self.last_quality: FrameQualityMap | None = None
        self.last_observations: list = []
        self.infer_count = 0
        self.frame_count = 0
        self.dropped_infer = 0
        self.thermal_c: float | None = None
        self._last_infer_ns = 0
        self._infer_period_ns = int(1e9 / max(cfg.runtime.infer_fps, 1.0))
        self._base_infer_period_ns = self._infer_period_ns

    def reset(self) -> None:
        self.tracker.reset()
        self.scheduler = QualityScheduler(self.cfg.quality)
        self.last_ns = None
        self.bad_streak_s = 0.0
        self.status = PerceptionStatus.NORMAL
        self.last_observations = []
        self._infer_period_ns = self._base_infer_period_ns

    def _apply_thermal(self) -> None:
        """NFR-007: drop far-ROI / infer rate before touching preview."""
        if self.thermal_c is not None and self.thermal_c >= 42.0:
            self._infer_period_ns = int(1e9 / max(self.cfg.runtime.thermal_min_infer_fps, 1.0))
            if self.status == PerceptionStatus.NORMAL:
                self.status = PerceptionStatus.THERMAL_THROTTLE
        else:
            self._infer_period_ns = self._base_infer_period_ns

    def set_alerts_enabled(self, enabled: bool) -> None:
        self.alerts.set_enabled(enabled)

    def step(self, frame: SynchronizedFrame, *, ui_mode: UiMode = UiMode.RIDING) -> PerceptionView:
        t0 = frame.meta.sensor_timestamp_ns
        dt = 1.0 / 60.0 if self.last_ns is None else max(1e-3, (t0 - self.last_ns) / 1e9)
        self.last_ns = t0
        self.frame_count += 1
        self._apply_thermal()

        qmap = evaluate_frame(frame.bgr, self.cfg.quality)
        self.scheduler.push(frame, qmap)
        sel_frame, sel_q, fresh, age_ms = self.scheduler.select(t0)
        if sel_q.global_quality.usable:
            self.bad_streak_s = 0.0
        else:
            self.bad_streak_s += dt
        self.status = perception_status(sel_q, self.bad_streak_s)
        if self.thermal_c is not None and self.thermal_c >= 42.0 and self.status == PerceptionStatus.NORMAL:
            self.status = PerceptionStatus.THERMAL_THROTTLE
        self.last_quality = sel_q

        allow_new = fresh and sel_q.global_quality.usable and self.status not in {
            PerceptionStatus.SEVERE_BLUR,
            PerceptionStatus.PERCEPTION_LIMITED,
            PerceptionStatus.LENS_CONTAMINATION,
        }
        do_infer = fresh and (t0 - self._last_infer_ns) >= self._infer_period_ns
        observations = []
        backend = InferenceBackend.HEURISTIC
        infer_ms = 0.0
        dual = True
        if do_infer:
            result = self.engine.infer(sel_frame, sel_q)
            observations = result.observations
            self.last_observations = observations
            backend = result.backend
            infer_ms = result.latency_ms
            dual = result.dual_scale
            self._last_infer_ns = t0
            self.infer_count += 1
            self._infer_times.append(t0)
        else:
            self.dropped_infer += 1

        tracks = self.tracker.update(
            observations,
            t0,
            allow_new_high_conf=allow_new,
            quality_ok=allow_new,
            dt_s=dt,
        )

        speed = frame.speed_mps
        if speed is None and frame.location is not None:
            speed = frame.location.speed_mps

        tracked_objs: list[TrackedRoadObject] = []
        fired: list[AlertDecision] = []
        for tr in tracks:
            dist, dconf, dvalid = self.geometry.distance_for_track(tr, t0)
            prev = tr.distance_hist[-1][1] if tr.distance_hist else None
            if dist is not None:
                tr.distance_hist.append((t0, dist))
            if should_mark_passed(tr, dist, prev, self.cfg.geometry.near_boundary_m):
                self.tracker.mark_passed(tr.track_id, t0, "near_boundary")
                tr.state = LifecycleState.PASSED
            ttc = self.geometry.ttc(tr, speed, t0)
            direction = self.geometry.direction_for(tr)
            relevance = self.geometry.path_relevance(tr)
            temporal = _temporal(tr, t0, self.cfg.tracking.confirm_window_s)
            gcons = _geom_consistency(tr, dconf)
            vis = float(np.clip(tr.quality_at_mask, 0, 1))
            eff = visual_score(tr.model_confidence, vis, temporal, gcons)
            if not dvalid:
                direction = direction if direction != Direction.UNKNOWN else Direction.CENTER_FRONT
            sev_table = {0: 0.05, 1: 0.3, 2: 0.7, 3: 1.0, -1: 0.22}
            risk = eff * sev_table.get(int(tr.severity), 0.22) * relevance
            obj = TrackedRoadObject(
                schema_version=SCHEMA_VERSION,
                track_id=tr.track_id,
                timestamp_ns=t0,
                lifecycle_state=tr.state,
                semantic_type=tr.semantic,
                geometry_type=tr.geometry,
                object_state=tr.object_state,
                severity=tr.severity,
                direction=direction,
                distance_m=dist if dvalid else None,
                distance_confidence=dconf,
                distance_valid=dvalid and self.geometry.valid,
                ttc_s=ttc,
                model_confidence=tr.model_confidence,
                visibility_confidence=vis,
                temporal_confidence=temporal,
                geometry_consistency=gcons,
                effective_confidence=eff,
                path_relevance=relevance,
                risk_score=risk,
                alert_score=0.0,
                polygon=tr.polygon,
                bbox=tr.bbox,
                mask_rle=None,
                source_frame_id=tr.source_frame_id,
                mount_profile_id=self.geometry.mount.profile_id,
                model_version=self.model_version,
                visual_style="dashed" if tr.state == LifecycleState.CANDIDATE else "solid",
                road_xy_m=tuple(tr.road_xy.tolist()) if tr.road_xy is not None else None,
            )
            decision = self.alerts.evaluate(obj, self.status, t0, self.geometry.valid)
            obj.alert_score = decision.alert_score
            if decision.fired:
                tr.state = LifecycleState.ALERTED
                tr.alerted = True
                obj.lifecycle_state = LifecycleState.ALERTED
                fired.append(decision)
            elif decision.reasons:
                # keep non-fire snapshots only in research logs via fired list? store all gated decisions
                if "below_threshold" not in decision.reasons[:1] or ui_mode == UiMode.RESEARCH:
                    pass
            tracked_objs.append(obj)

        tracked_objs.sort(key=lambda o: (o.risk_score, -(o.distance_m or 99)), reverse=True)
        for i, o in enumerate(tracked_objs):
            o.label_rank = i

        primitives = self._primitives(tracked_objs, ui_mode, sel_q)
        e2e_ms = (age_ms if do_infer else 0.0) + infer_ms
        self._latencies.append(e2e_ms)
        p95 = float(np.percentile(self._latencies, 95)) if self._latencies else e2e_ms
        infer_fps = 0.0
        if len(self._infer_times) >= 2:
            span = (self._infer_times[-1] - self._infer_times[0]) / 1e9
            infer_fps = (len(self._infer_times) - 1) / max(span, 1e-3)

        speed_kmh = None if speed is None else float(speed * 3.6)
        view = PerceptionView(
            timestamp_ns=t0,
            status=self.status,
            status_copy=RIDING_STATUS_COPY[self.status],
            tracks=tracked_objs,
            primitives=primitives,
            alerts=fired,
            quality=sel_q,
            speed_kmh=speed_kmh,
            backend=backend,
            model_version=self.model_version,
            infer_fps=infer_fps,
            ar_fps=1.0 / dt,
            latency_p95_ms=p95,
            queue_depth=len(self.scheduler._buf),
            thermal_c=self.thermal_c,
            blur=sel_q.global_quality.motion_blur,
            glare=sel_q.global_quality.glare,
            rec_seconds=self.frame_count / 60.0,
            dual_scale=dual,
        )
        self.last_view = view
        return view

    def _primitives(
        self,
        objs: list[TrackedRoadObject],
        ui_mode: UiMode,
        qmap: FrameQualityMap,
    ) -> list[RenderPrimitive]:
        prims: list[RenderPrimitive] = []
        # occlusion tiles as unknown, not danger-red
        if qmap.occupancy_occluded_ratio > 0.25:
            prims.append(
                RenderPrimitive(
                    track_id=-1,
                    polygon=[],
                    color_rgba=(0.45, 0.5, 0.58, 0.22),
                    dashed=True,
                    thickness=1.0,
                    label=None,
                    label_priority=99,
                    fade=1.0,
                    kind="occlusion",
                )
            )
        corridor = self.geometry.project_corridor_pixels()
        if ui_mode == UiMode.RESEARCH and corridor:
            prims.append(
                RenderPrimitive(
                    track_id=-2,
                    polygon=corridor,
                    color_rgba=(0.2, 0.75, 0.8, 0.18),
                    dashed=True,
                    thickness=2.0,
                    label=None,
                    label_priority=80,
                    fade=1.0,
                    kind="corridor",
                )
            )
        if ui_mode == UiMode.RESEARCH:
            fw, fh = self.cfg.model.input_far
            nw, nh = self.cfg.model.input_near
            prims.append(
                RenderPrimitive(
                    track_id=-3,
                    polygon=[(346, 346), (1574, 346), (1574, 670), (346, 670)],
                    color_rgba=(0.35, 0.9, 0.55, 0.12),
                    dashed=True,
                    thickness=1.0,
                    label=f"far {fw}x{fh}",
                    label_priority=90,
                    fade=0.4,
                    kind="roi",
                )
            )
            prims.append(
                RenderPrimitive(
                    track_id=-4,
                    polygon=[(154, 540), (1766, 540), (1766, 1080), (154, 1080)],
                    color_rgba=(0.9, 0.7, 0.2, 0.1),
                    dashed=True,
                    thickness=1.0,
                    label=f"near {nw}x{nh}",
                    label_priority=91,
                    fade=0.4,
                    kind="roi",
                )
            )
        labeled = 0
        for obj in objs:
            if obj.lifecycle_state in {LifecycleState.EXPIRED}:
                continue
            if obj.lifecycle_state == LifecycleState.PASSED:
                fade = 0.35
            elif obj.lifecycle_state == LifecycleState.CANDIDATE:
                fade = self.cfg.render.candidate_alpha
            else:
                fade = self.cfg.render.confirmed_alpha
            high = obj.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED} and obj.risk_score > 0.4
            if obj.geometry_type.value == "concave":
                color = (0.15, 0.82, 0.78, fade)
            elif obj.geometry_type.value == "rough":
                color = (0.95, 0.78, 0.25, fade)
            else:
                color = (0.55, 0.85, 0.95, fade)
            if high:
                color = (color[0], color[1], color[2], min(1.0, fade + 0.1))
            dashed = obj.lifecycle_state in {LifecycleState.CANDIDATE, LifecycleState.TRACKED}
            label = None
            allow_label = ui_mode == UiMode.RESEARCH or labeled < self.cfg.render.riding_max_labels
            if allow_label and obj.lifecycle_state != LifecycleState.CANDIDATE:
                dist_txt = self.geometry.display_distance(obj.distance_m, obj.distance_valid, obj.distance_confidence)
                if ui_mode == UiMode.RIDING:
                    dir_cn = {
                        "LEFT_FRONT": "左前方",
                        "CENTER_FRONT": "正前方",
                        "RIGHT_FRONT": "右前方",
                        "ACROSS": "正前方",
                    }.get(obj.direction.value, "")
                    label = f"{dir_cn} {dist_txt or ''}".strip()
                else:
                    label = (
                        f"ID {obj.track_id} · {obj.semantic_type.value} · "
                        f"{obj.model_confidence:.2f}"
                        + (f" · {dist_txt}" if dist_txt else "")
                    )
                labeled += 1
            prims.append(
                RenderPrimitive(
                    track_id=obj.track_id,
                    polygon=obj.polygon,
                    color_rgba=color,
                    dashed=dashed,
                    thickness=3.2 if high else 2.0,
                    label=label,
                    label_priority=obj.label_rank or 50,
                    fade=fade,
                    kind="anomaly",
                )
            )
        return prims
