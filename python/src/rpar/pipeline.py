"""End-to-end realtime loop: quality -> schedule -> infer -> track -> geometry -> alerts -> AR."""

from __future__ import annotations

from collections import deque
from time import perf_counter

import cv2
import numpy as np

from rpar import SCHEMA_VERSION
from rpar.alerts import AlertPolicy, severity_weight, visual_score
from rpar.config import RparConfig
from rpar.enums import (
    INFO_LAYER_SEMANTICS,
    RIDING_STATUS_COPY,
    Direction,
    InferenceBackend,
    LifecycleState,
    PerceptionStatus,
    UiMode,
    VisibilityClass,
    bump_kind,
    is_bump_hazard,
)
from rpar.geometry import GeometryEngine, should_mark_passed
from rpar.impact import estimate_impact_score
from rpar.models import (
    AlertDecision,
    FrameQualityMap,
    PerceptionView,
    RenderPrimitive,
    SynchronizedFrame,
    TrackedRoadObject,
)
from rpar.perception import PerceptionEngine
from rpar.quality import ANALYSIS_WIDTH, LensContaminationDetector, QualityScheduler, evaluate_frame, perception_status
from rpar.roi import far_polygon, near_polygon
from rpar.tracking import TrackEngine, TrackInternal
from rpar.transforms import display_compensate


def occlusion_cover_ratio(polys: list[list[tuple[float, float]]], width: int, height: int) -> float:
    """Axis-aligned bbox union proxy for occlusion masks (M2 appendix B)."""
    if not polys or width < 1 or height < 1:
        return 0.0
    area = 0.0
    for poly in polys:
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        area += max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
    return float(min(1.0, area / float(width * height)))


def allow_new_observations(
    status: PerceptionStatus,
    *,
    usable: bool,
    fresh: bool,
    occlusion_ratio: float = 0.0,
    occ_block: float = 0.40,
) -> bool:
    """Do not spawn high-confidence tracks in blur/lens-drop, or when most of the frame is occluded.

    A vehicle occupying ~8-20% of the frame still shows OCCLUDED on the HUD, but potholes
    beside it must be allowed to track. Spec appendix B only blocks instances *under* the mask.
    """
    if not fresh or not usable:
        return False
    if occlusion_ratio >= occ_block:
        return False
    return status not in {
        PerceptionStatus.SEVERE_BLUR,
        PerceptionStatus.PERCEPTION_LIMITED,
        PerceptionStatus.LENS_CONTAMINATION,
    }


def _temporal(tr: TrackInternal, now_ns: int, confirm_s: float) -> float:
    age = (now_ns - tr.created_ns) / 1e9
    stab = min(1.0, tr.hits / 6.0) * min(1.0, age / max(confirm_s, 1e-3))
    miss_pen = 0.85**tr.misses
    return float(np.clip(stab * miss_pen, 0.05, 1.0))


def _geom_consistency(tr: TrackInternal, dist_conf: float) -> float:
    if tr.road_xy is None:
        return 0.35
    return float(np.clip(0.4 + 0.6 * dist_conf, 0, 1))


class LatencyStats:
    """Percentiles over the frames that actually did work (no zero padding)."""

    def __init__(self, maxlen: int = 120) -> None:
        self._infer: deque[float] = deque(maxlen=maxlen)
        self._loop: deque[float] = deque(maxlen=maxlen)

    def add_infer(self, ms: float) -> None:
        self._infer.append(float(ms))

    def add_loop(self, ms: float) -> None:
        self._loop.append(float(ms))

    def infer_p(self, p: float) -> float:
        return float(np.percentile(self._infer, p)) if self._infer else 0.0

    def loop_p(self, p: float) -> float:
        return float(np.percentile(self._loop, p)) if self._loop else 0.0


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
        self.lens = LensContaminationDetector(cfg.quality)
        self.model_version = model_version or cfg.model.package_id
        self.last_ns: int | None = None
        self.first_ns: int | None = None
        self.bad_streak_s = 0.0
        self.latency = LatencyStats()
        self._infer_times: deque[int] = deque(maxlen=40)
        self.status = PerceptionStatus.NORMAL
        self.last_view: PerceptionView | None = None
        self.last_quality: FrameQualityMap | None = None
        self.last_observations: list = []
        self.did_infer = False
        self.infer_count = 0
        self.frame_count = 0
        self.dropped_infer = 0
        self.thermal_c: float | None = None
        self.thermal_reason: str | None = None
        self.skip_far_roi = False
        self.research_heatmap = True
        self.status_events: list[dict] = []
        self._last_status: PerceptionStatus | None = None
        self._last_infer_ns = 0
        self._infer_period_ns = int(1e9 / max(cfg.runtime.infer_fps, 1.0))
        self._base_infer_period_ns = self._infer_period_ns
        self._last_input_sizes: list[tuple[int, int]] = [cfg.model.input_far, cfg.model.input_near]
        self.last_road_polygon: list[tuple[float, float]] = []
        self.last_occluded: list[list[tuple[float, float]]] = []
        self._prev_gray: np.ndarray | None = None
        self._last_tracked: list = []
        self._last_qmap: FrameQualityMap | None = None
        self._last_frame_wh: tuple[int, int] = (1, 1)
        self._last_yaw = 0.0
        self._last_p95 = 0.0
        self._ar_fps_ema: float | None = None

    def view_with_mode(self, view, ui_mode: UiMode):
        from dataclasses import replace

        q = self._last_qmap or view.quality
        if q is None:
            return view
        w, h = self._last_frame_wh
        prims = self._primitives(self._last_tracked, ui_mode, q, w, h, self._last_yaw, self._last_p95)
        return replace(view, primitives=prims)

    def reset(self) -> None:
        self.tracker.reset()
        self.scheduler = QualityScheduler(self.cfg.quality)
        self.lens.reset()
        self.last_ns = None
        self.first_ns = None
        self.bad_streak_s = 0.0
        self.status = PerceptionStatus.NORMAL
        self.last_observations = []
        self.did_infer = False
        self.last_road_polygon = []
        self.last_occluded = []
        self._infer_period_ns = self._base_infer_period_ns
        self.thermal_reason = None
        self.skip_far_roi = False
        self.status_events = []
        self._last_status = None
        self._prev_gray = None
        self._ar_fps_ema = None

    def _apply_thermal(self) -> None:
        """NFR-007: drop far-ROI then infer rate before touching preview."""
        prev = self.thermal_reason
        th = self.cfg.thermal
        t = self.thermal_c
        if t is None or t < th.warn_c:
            self._infer_period_ns = self._base_infer_period_ns
            self.skip_far_roi = False
            self.thermal_reason = None
        elif t >= th.drop_infer_hz_c:
            self.skip_far_roi = True
            self._infer_period_ns = int(1e9 / max(self.cfg.runtime.thermal_min_infer_fps, 1.0))
            self.thermal_reason = "THERMAL_DROP_INFER_HZ"
        elif t >= th.drop_far_roi_c:
            self.skip_far_roi = True
            self._infer_period_ns = self._base_infer_period_ns
            self.thermal_reason = "THERMAL_DROP_FAR_ROI"
        else:
            self.skip_far_roi = False
            slowed = max(self.cfg.runtime.thermal_min_infer_fps, self.cfg.runtime.infer_fps * th.slow_factor)
            self._infer_period_ns = int(1e9 / max(slowed, 1.0))
            self.thermal_reason = "THERMAL_DROP_INFER_HZ"
        if hasattr(self.engine, "skip_far_roi"):
            self.engine.skip_far_roi = self.skip_far_roi
        if prev != self.thermal_reason and self.thermal_reason:
            self.status_events.append(
                {"domain": "NFR", "code": self.thermal_reason, "detail": f"thermal_c={self.thermal_c}"}
            )

    def _note_status(self) -> None:
        if self._last_status == self.status:
            return
        code = {
            PerceptionStatus.PERCEPTION_LIMITED: "PERCEPTION_LIMITED",
            PerceptionStatus.LENS_CONTAMINATION: "LENS_CONTAMINATION",
            PerceptionStatus.OCCLUDED: "OCCLUDED",
            PerceptionStatus.SEVERE_BLUR: "SEVERE_BLUR",
            PerceptionStatus.THERMAL_THROTTLE: self.thermal_reason or "THERMAL_THROTTLE",
        }.get(self.status)
        if code:
            self.status_events.append({"domain": "QUAL", "code": code, "detail": self.status.value})
        self._last_status = self.status

    def set_alerts_enabled(self, enabled: bool) -> None:
        self.alerts.set_enabled(enabled)

    # ------------------------------------------------------------------ main step
    def step(self, frame: SynchronizedFrame, *, ui_mode: UiMode = UiMode.RIDING) -> PerceptionView:
        loop_t0 = perf_counter()
        t0 = frame.meta.sensor_timestamp_ns
        if self.first_ns is None:
            self.first_ns = t0
        dt = 1.0 / 60.0 if self.last_ns is None else max(1e-3, (t0 - self.last_ns) / 1e9)
        self.last_ns = t0
        self.frame_count += 1
        self._apply_thermal()
        qcfg = self.cfg.quality

        qmap = evaluate_frame(
            frame.bgr,
            qcfg,
            headlight_mean=self.geometry.mount.headlight_mean if self.geometry.mount.headlight_valid else None,
        )
        gray_full = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray_full,
            (ANALYSIS_WIDTH, max(1, int(ANALYSIS_WIDTH * gray_full.shape[0] / gray_full.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
        self.lens.update(small, t0)
        if self.lens.flagged:
            qmap.global_quality.visibility_class = VisibilityClass.LENS_DROP
            qmap.global_quality.usable = False
            qmap.global_quality.reason = "lens_drop"
            qmap.degrade_reason = "lens_drop"
        self.scheduler.push(frame, qmap)
        sel_frame, sel_q, fresh, age_ms = self.scheduler.select(t0)
        if sel_q.global_quality.usable:
            self.bad_streak_s = 0.0
        else:
            self.bad_streak_s += dt
        self.status = perception_status(sel_q, self.bad_streak_s, qcfg)
        occ_ratio = occlusion_cover_ratio(self.last_occluded, frame.meta.width, frame.meta.height)
        if occ_ratio >= qcfg.occlusion_status_ratio and self.status == PerceptionStatus.NORMAL:
            self.status = PerceptionStatus.OCCLUDED
        if self.thermal_c is not None and self.thermal_c >= self.cfg.thermal.warn_c and self.status == PerceptionStatus.NORMAL:
            self.status = PerceptionStatus.THERMAL_THROTTLE
        self.last_quality = sel_q

        do_infer = fresh and (t0 - self._last_infer_ns) >= self._infer_period_ns
        observations = []
        backend = InferenceBackend.HEURISTIC
        infer_ms = 0.0
        dual = not self.skip_far_roi
        if do_infer:
            result = self.engine.infer(sel_frame, sel_q)
            observations = result.observations
            self.last_observations = observations
            self.last_road_polygon = list(result.road_polygon)
            self.last_occluded = list(result.occluded_polygons)
            self.did_infer = True
            backend = result.backend
            infer_ms = result.latency_ms
            dual = result.dual_scale and not self.skip_far_roi
            self._last_input_sizes = list(result.input_sizes) or self._last_input_sizes
            self._last_infer_ns = t0
            self.infer_count += 1
            self._infer_times.append(t0)
            self.latency.add_infer(age_ms + infer_ms)
            occ_ratio = occlusion_cover_ratio(self.last_occluded, frame.meta.width, frame.meta.height)
            if occ_ratio >= qcfg.occlusion_status_ratio and self.status == PerceptionStatus.NORMAL:
                self.status = PerceptionStatus.OCCLUDED
            if self.last_occluded:
                from rpar.segengine import gate_observations

                observations = gate_observations(observations, [], self.last_occluded, None)
        else:
            self.dropped_infer += 1
            self.did_infer = False
            observations = list(self.last_observations)
        self._note_status()
        allow_new = allow_new_observations(
            self.status,
            usable=sel_q.global_quality.usable,
            fresh=fresh,
            occlusion_ratio=occ_ratio,
            occ_block=qcfg.occlusion_block_ratio,
        )

        optical_flow: dict[int, tuple[float, float]] = {}
        if self._prev_gray is not None and self._prev_gray.shape == gray_full.shape and self.tracker.tracks:
            tids = [tid for tid, tr in self.tracker.tracks.items() if tr.active]
            if tids:
                pts = np.array(
                    [[[self.tracker.tracks[t].mean[0], self.tracker.tracks[t].mean[1]]] for t in tids],
                    dtype=np.float32,
                )
                nxt, st, _err = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray_full, pts, None, winSize=(21, 21), maxLevel=2
                )
                if nxt is not None and st is not None:
                    for i, tid in enumerate(tids):
                        if int(st[i][0]) == 1:
                            optical_flow[tid] = (
                                float(nxt[i, 0, 0] - pts[i, 0, 0]),
                                float(nxt[i, 0, 1] - pts[i, 0, 1]),
                            )
        self._prev_gray = gray_full

        quality_ok = bool(sel_q.global_quality.usable) and self.status not in {
            PerceptionStatus.SEVERE_BLUR,
            PerceptionStatus.LENS_CONTAMINATION,
            PerceptionStatus.PERCEPTION_LIMITED,
        }
        yaw_rate = float(frame.angular_velocity[2]) if frame.angular_velocity else 0.0
        before = set(self.tracker.tracks.keys())
        tracks = self.tracker.update(
            observations,
            t0,
            allow_new_high_conf=allow_new,
            quality_ok=quality_ok,
            dt_s=dt,
            camera_yaw_rate=yaw_rate,
            frame_w=float(frame.meta.width),
            optical_flow=optical_flow,
            focal_px=float(self.geometry.k.fx),
        )
        for gone in before - set(self.tracker.tracks.keys()):
            self.geometry.forget(gone)

        speed = frame.speed_mps
        if speed is None and frame.location is not None:
            speed = frame.location.speed_mps
        gravity = frame.pose.gravity_xyz if frame.pose is not None else None

        tracked_objs: list[TrackedRoadObject] = []
        fired: list[AlertDecision] = []
        for tr in tracks:
            dist, dconf, dvalid = self.geometry.distance_for_track(tr, t0)
            prev = tr.distance_hist[-1][1] if tr.distance_hist else None
            if dist is not None:
                tr.distance_hist.append((t0, dist))
                del tr.distance_hist[:-12]
            if should_mark_passed(tr, dist, prev, self.cfg.geometry.near_boundary_m):
                self.tracker.mark_passed(tr.track_id, t0, "near_boundary")
            ttc = self.geometry.ttc(tr, speed, t0)
            direction = self.geometry.direction_for(tr)
            relevance = self.geometry.path_relevance(tr)
            temporal = _temporal(tr, t0, self.cfg.tracking.confirm_window_s)
            gcons = _geom_consistency(tr, dconf)
            vis = float(np.clip(tr.quality_at_mask, 0, 1))
            eff = visual_score(tr.model_confidence, vis, temporal, gcons)
            if not dvalid and direction == Direction.UNKNOWN:
                direction = Direction.CENTER_FRONT
            risk = eff * severity_weight(tr.severity, self.cfg.alert.severity_weights) * relevance
            depth_conf = float(dconf) if dvalid else 0.0
            impact = estimate_impact_score(
                frame.linear_accel,
                speed,
                dist if dvalid else None,
                gravity_xyz=gravity,
            )
            show_dist = dvalid and self.geometry.valid
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
                distance_m=dist if show_dist else None,
                distance_confidence=dconf,
                distance_valid=show_dist,
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
                mask_rle=tr.mask_rle,
                source_frame_id=tr.source_frame_id,
                mount_profile_id=self.geometry.mount.profile_id,
                model_version=self.model_version,
                visual_style="dashed" if tr.state == LifecycleState.CANDIDATE else "solid",
                road_xy_m=tuple(tr.road_xy.tolist()) if tr.road_xy is not None else None,
                depth_confidence=depth_conf,
                impact_score=impact,
            )
            decision = self.alerts.evaluate(obj, self.status, t0, self.geometry.valid)
            obj.alert_score = decision.alert_score
            if decision.fired:
                tr.state = LifecycleState.ALERTED
                tr.alerted = True
                obj.lifecycle_state = LifecycleState.ALERTED
                fired.append(decision)
            tracked_objs.append(obj)

        tracked_objs.sort(key=lambda o: (o.risk_score, -(o.distance_m or 99)), reverse=True)
        for i, o in enumerate(tracked_objs):
            o.label_rank = i

        p95 = self.latency.infer_p(95)
        p50 = self.latency.infer_p(50)
        primitives = self._primitives(
            tracked_objs,
            ui_mode,
            sel_q,
            sel_frame.meta.width,
            sel_frame.meta.height,
            yaw_rate=yaw_rate,
            latency_ms=p95,
        )
        self._last_tracked = tracked_objs
        self._last_qmap = sel_q
        self._last_frame_wh = (sel_frame.meta.width, sel_frame.meta.height)
        self._last_yaw = yaw_rate
        self._last_p95 = p95
        infer_fps = 0.0
        if len(self._infer_times) >= 2:
            span = (self._infer_times[-1] - self._infer_times[0]) / 1e9
            infer_fps = (len(self._infer_times) - 1) / max(span, 1e-3)

        inst_fps = 1.0 / dt
        self._ar_fps_ema = inst_fps if self._ar_fps_ema is None else 0.9 * self._ar_fps_ema + 0.1 * inst_fps
        speed_kmh = None if speed is None else float(speed * 3.6)
        self.latency.add_loop((perf_counter() - loop_t0) * 1000.0)
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
            ar_fps=float(self._ar_fps_ema),
            latency_p95_ms=p95,
            queue_depth=len(self.scheduler),
            thermal_c=self.thermal_c,
            blur=sel_q.global_quality.motion_blur,
            glare=sel_q.global_quality.glare,
            rec_seconds=(t0 - self.first_ns) / 1e9,
            dual_scale=dual,
            latency_p50_ms=p50,
            dropped_infer=self.dropped_infer,
            input_far=self._last_input_sizes[0] if self._last_input_sizes else self.cfg.model.input_far,
            input_near=self._last_input_sizes[-1] if self._last_input_sizes else self.cfg.model.input_near,
            road_polygon=self.last_road_polygon,
            occluded_polygons=self.last_occluded,
            loop_p95_ms=self.latency.loop_p(95),
        )
        self.last_view = view
        return view

    # ------------------------------------------------------------------ render primitives
    def _primitives(
        self,
        objs: list[TrackedRoadObject],
        ui_mode: UiMode,
        qmap: FrameQualityMap,
        frame_w: int,
        frame_h: int,
        yaw_rate: float = 0.0,
        latency_ms: float = 0.0,
    ) -> list[RenderPrimitive]:
        prims: list[RenderPrimitive] = []
        focal = float(self.geometry.k.fx)
        if self.last_road_polygon:
            prims.append(
                RenderPrimitive(
                    track_id=-6,
                    polygon=self.last_road_polygon,
                    color_rgba=(0.12, 0.92, 0.38, 0.10 if ui_mode == UiMode.RIDING else 0.22),
                    dashed=False,
                    thickness=2.0,
                    label=None,
                    label_priority=85,
                    fade=0.32,
                    kind="road",
                )
            )
        for i, poly in enumerate(self.last_occluded):
            if len(poly) < 3:
                continue
            prims.append(
                RenderPrimitive(
                    track_id=-7 - i,
                    polygon=poly,
                    color_rgba=(0.55, 0.58, 0.68, 0.42),
                    dashed=False,
                    thickness=2.0,
                    label=None if ui_mode != UiMode.RESEARCH else "vehicle",
                    label_priority=86,
                    fade=0.7,
                    kind="occlusion",
                )
            )
        # occupancy tiles as unknown, not danger-red
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
            sizes = self._last_input_sizes or [self.cfg.model.input_far, self.cfg.model.input_near]
            fw, fh = sizes[0]
            nw, nh = sizes[-1]
            prims.append(
                RenderPrimitive(
                    track_id=-3,
                    polygon=far_polygon(frame_w, frame_h, self.cfg.perception.far_roi),
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
                    polygon=near_polygon(frame_w, frame_h, self.cfg.perception.near_roi),
                    color_rgba=(0.9, 0.7, 0.2, 0.1),
                    dashed=True,
                    thickness=1.0,
                    label=f"near {nw}x{nh}",
                    label_priority=91,
                    fade=0.4,
                    kind="roi",
                )
            )
            if self.research_heatmap:
                for t in qmap.tiles:
                    if t.visibility == VisibilityClass.CLEAR:
                        continue
                    a = 0.16
                    if t.visibility == VisibilityClass.BLUR:
                        col = (0.16, 0.35, 0.82, a)
                    elif t.visibility == VisibilityClass.GLARE:
                        col = (1.0, 0.86, 0.16, a)
                    elif t.visibility == VisibilityClass.UNDEREXPOSED:
                        col = (0.16, 0.31, 0.7, a)
                    elif t.visibility == VisibilityClass.OVEREXPOSED:
                        col = (0.94, 0.94, 0.94, a)
                    else:
                        col = (0.47, 0.47, 0.47, a)
                    prims.append(
                        RenderPrimitive(
                            track_id=-10,
                            polygon=[
                                (float(t.x0), float(t.y0)),
                                (float(t.x1), float(t.y0)),
                                (float(t.x1), float(t.y1)),
                                (float(t.x0), float(t.y1)),
                            ],
                            color_rgba=col,
                            dashed=False,
                            thickness=1.0,
                            label=None,
                            label_priority=95,
                            fade=a,
                            kind="heatmap",
                        )
                    )
        labeled = 0
        stroke = float(self.cfg.render.stroke_scale)
        overlay_a = float(np.clip(self.cfg.render.overlay_alpha, 0.15, 1.0))
        show_info = bool(self.cfg.render.show_info_layer)
        for obj in objs:
            if obj.lifecycle_state in {LifecycleState.EXPIRED, LifecycleState.CANDIDATE}:
                continue
            info = obj.semantic_type in INFO_LAYER_SEMANTICS
            if info and not show_info:
                continue
            bump = is_bump_hazard(obj.semantic_type, obj.geometry_type, obj.object_state)
            # PER-010: TRACKED rough/unknown is not a confirmed instance. Bump hazards
            # may draw dashed so the visual layer stays ahead of voice.
            if (not info) and obj.lifecycle_state == LifecycleState.TRACKED and not bump:
                continue
            if obj.lifecycle_state == LifecycleState.PASSED:
                fade = 0.35
            else:
                fade = self.cfg.render.confirmed_alpha
            fade = float(np.clip(fade * overlay_a, 0.05, 1.0))
            high = (
                (not info)
                and obj.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}
                and obj.risk_score > 0.4
            )
            if info and obj.semantic_type.value == "puddle":
                color = (0.35, 0.55, 0.88, fade)
            elif info:
                color = (0.72, 0.66, 0.42, fade)
            elif bump:
                color = (0.95, 0.28, 0.16, fade)
            elif obj.geometry_type.value == "concave":
                color = (0.15, 0.82, 0.78, fade)
            elif obj.geometry_type.value == "rough":
                color = (0.95, 0.78, 0.25, fade)
            else:
                color = (0.55, 0.85, 0.95, fade)
            if high:
                color = (color[0], color[1], color[2], min(1.0, fade + 0.1))
            dashed = obj.lifecycle_state == LifecycleState.TRACKED
            label = None
            allow_label = ui_mode == UiMode.RESEARCH or labeled < self.cfg.render.riding_max_labels
            if (
                allow_label
                and obj.lifecycle_state in {LifecycleState.CONFIRMED, LifecycleState.ALERTED}
                and not (info and ui_mode == UiMode.RIDING)
            ):
                dist_txt = self.geometry.display_distance(obj.distance_m, obj.distance_valid, obj.distance_confidence)
                if ui_mode == UiMode.RIDING:
                    dir_cn = {
                        "LEFT_FRONT": "左前方",
                        "CENTER_FRONT": "正前方",
                        "RIGHT_FRONT": "右前方",
                        "ACROSS": "正前方",
                    }.get(obj.direction.value, "")
                    if bump:
                        kind = bump_kind(obj.semantic_type, obj.geometry_type, obj.object_state, obj.severity)
                        label = f"{dir_cn}{kind}" + (f" · {dist_txt}" if dist_txt else "")
                    else:
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
                    polygon=display_compensate(obj.polygon, yaw_rate, latency_ms, frame_w, focal),
                    color_rgba=color,
                    dashed=dashed,
                    thickness=(3.2 if high else 2.0) * stroke,
                    label=label,
                    label_priority=obj.label_rank or 50,
                    fade=fade,
                    kind="info" if info else ("bump" if bump else "anomaly"),
                )
            )
        return prims
