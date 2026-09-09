"""Synthetic 1080p60 road world with known distances — used for golden regression."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from rpar.enums import GeometryType, ObjectState, SemanticType, Severity
from rpar.models import FrameMeta, Intrinsics, LocationSample, MountProfile, PoseSample, SynchronizedFrame
from rpar.transforms import default_intrinsics, default_mount, project_vehicle_point


@dataclass
class WorldObject:
    obj_id: str
    semantic: SemanticType
    geometry: GeometryType
    state: ObjectState
    severity: Severity
    y0_m: float
    x_m: float
    length_m: float
    width_m: float
    color: tuple[int, int, int]


@dataclass
class SimConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 60
    speed_mps: float = 10.5  # ~38 km/h
    duration_s: float = 4.0
    night: bool = False
    wet: bool = False
    blur_windows: list[tuple[float, float]] = field(default_factory=list)
    glare_windows: list[tuple[float, float]] = field(default_factory=list)
    occlude_windows: list[tuple[float, float]] = field(default_factory=list)
    rain: bool = False
    vibration: bool = False
    lens_drops: bool = False


def default_world() -> list[WorldObject]:
    return [
        WorldObject("pot_center", SemanticType.POTHOLE, GeometryType.CONCAVE, ObjectState.ABNORMAL, Severity.HEAVY, 28.0, 0.15, 1.1, 0.9, (18, 18, 18)),
        WorldObject("manhole_right", SemanticType.MANHOLE_COVER, GeometryType.CONCAVE, ObjectState.ABNORMAL, Severity.MEDIUM, 22.0, 1.35, 0.8, 0.8, (50, 55, 60)),
        WorldObject("bump", SemanticType.SPEED_BUMP, GeometryType.CONVEX, ObjectState.ABNORMAL, Severity.MEDIUM, 36.0, 0.0, 0.45, 3.4, (30, 30, 30)),
        # fresh asphalt patch: darker than the road but far from a shadowed hole
        WorldObject("patch_flat", SemanticType.REPAIR_PATCH, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 18.0, -1.8, 1.6, 1.1, (84, 84, 90)),
        WorldObject("rough_left", SemanticType.ROUGH_BROKEN, GeometryType.ROUGH, ObjectState.ABNORMAL, Severity.LIGHT, 16.0, -1.1, 2.2, 1.4, (92, 90, 88)),
        WorldObject("joint", SemanticType.ROAD_JOINT, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE, 14.0, 0.0, 0.12, 3.2, (150, 150, 156)),
    ]


class RoadSimulator:
    def __init__(self, sim: SimConfig | None = None, mount: MountProfile | None = None, k: Intrinsics | None = None) -> None:
        self.sim = sim or SimConfig()
        self.mount = mount or default_mount(self.sim.width, self.sim.height)
        self.k = k or default_intrinsics(self.sim.width, self.sim.height)
        self.objects = default_world()
        self.t0_ns = 1_000_000_000_000

    def n_frames(self) -> int:
        return int(self.sim.duration_s * self.sim.fps)

    def ego_y(self, t: float) -> float:
        return self.sim.speed_mps * t

    def frame_at(self, index: int) -> tuple[SynchronizedFrame, dict]:
        t = index / self.sim.fps
        ts = self.t0_ns + int(t * 1e9)
        bgr, gt = self._render(t)
        meta = FrameMeta(
            frame_id=index,
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
            width=self.sim.width,
            height=self.sim.height,
            availability={
                "sensor_timestamp": True,
                "exposure": False,
                "iso": False,
                "focal": False,
                "af": False,
                "ae": False,
                "awb": False,
                "crop": False,
                "ois": False,
                "eis": False,
            },
        )
        gyro, accel = self.motion_at(t)
        pose = PoseSample(ts, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 9.81), 0.99)
        loc = LocationSample(ts, 31.23, 121.47, 8.0, self.sim.speed_mps, 12.0, 4.0, 0.4)
        frame = SynchronizedFrame(
            meta=meta,
            bgr=bgr,
            pose=pose,
            angular_velocity=gyro,
            linear_accel=accel,
            location=loc,
            speed_mps=self.sim.speed_mps,
        )
        return frame, gt

    def motion_at(self, t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        rough = 0.05 * np.sin(t * 28.0)
        if self.sim.vibration:
            rough *= 10.0
        impact = 1.8 if self._in_windows(t, self.sim.blur_windows) else 0.0
        if self.sim.vibration:
            impact += 0.55
        impact += self._geometry_impact_at(t)
        gyro = (float(0.03 * np.sin(t * 17.0)), float(0.04 * np.sin(t * 11.0) + impact * 0.4), float(rough + impact))
        accel = (float(0.15 * np.sin(t * 9.0)), float(self.sim.speed_mps * 0.02), float(9.81 + impact * 4.0 + rough))
        return gyro, accel

    def _geometry_impact_at(self, t: float) -> float:
        """Vertical IMU when the ego is on concave/convex/rough/step. Flat patches stay quiet."""
        y = self.ego_y(t)
        extra = 0.0
        for obj in self.objects:
            rel = obj.y0_m - y
            if rel < -(obj.length_m + 0.4) or rel > 0.4:
                continue
            if obj.geometry in {GeometryType.CONCAVE, GeometryType.CONVEX, GeometryType.ROUGH, GeometryType.STEP}:
                extra = max(extra, 1.35 + 0.4 * max(int(obj.severity), 0))
        return extra

    def classmap_at(self, index: int) -> np.ndarray:
        """Ground-truth dense labels: 0 bg, 1 road, 2 anomaly, 3 occlusion (PER-014)."""
        from rpar.segdecode import CLASS_ANOMALY, CLASS_OCC, CLASS_ROAD

        t = index / self.sim.fps
        w, h = self.sim.width, self.sim.height
        m = np.zeros((h, w), dtype=np.uint8)
        pts = self._road_pixel_pts()
        if len(pts) >= 3:
            cv2.fillConvexPoly(m, np.array(pts, dtype=np.int32), CLASS_ROAD)
        for obj in self.objects:
            poly = self.object_polygon(obj, t)
            if poly and len(poly) >= 3:
                cv2.fillPoly(m, [np.array(poly, dtype=np.int32)], CLASS_ANOMALY)
        if self._in_windows(t, self.sim.occlude_windows):
            cv2.rectangle(m, (int(w * 0.34), int(h * 0.28)), (int(w * 0.72), int(h * 0.70)), CLASS_OCC, -1)
        return m

    def _road_pixel_pts(self) -> list:
        pts = []
        for x, y in [(-4.5, 3.0), (4.5, 3.0), (6.5, 45.0), (-6.5, 45.0)]:
            uv = project_vehicle_point(np.array([x, y, 0.0]), self.mount, self.k)
            if uv is not None:
                pts.append(uv)
        return pts

    def object_polygon(self, obj: WorldObject, t: float) -> list[tuple[float, float]] | None:
        rel_y = obj.y0_m - self.ego_y(t)
        if rel_y < 1.5 or rel_y > 55:
            return None
        if obj.semantic == SemanticType.MANHOLE_COVER:
            center = project_vehicle_point(np.array([obj.x_m, rel_y + obj.length_m / 2, 0.0]), self.mount, self.k)
            edge = project_vehicle_point(
                np.array([obj.x_m + obj.width_m / 2, rel_y + obj.length_m / 2, 0.0]), self.mount, self.k
            )
            if center is None or edge is None:
                return None
            r = float(np.linalg.norm(edge - center))
            return [
                (float(center[0] + r * np.cos(a)), float(center[1] + 0.7 * r * np.sin(a)))
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ]
        pix = []
        xs = [obj.x_m - obj.width_m / 2, obj.x_m + obj.width_m / 2]
        ys = [rel_y, rel_y + obj.length_m]
        for x, y in [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]:
            uv = project_vehicle_point(np.array([x, y, 0.0]), self.mount, self.k)
            if uv is None:
                return None
            pix.append((float(uv[0]), float(uv[1])))
        return pix

    def _in_windows(self, t: float, windows: list[tuple[float, float]]) -> bool:
        return any(a <= t <= b for a, b in windows)

    def _render(self, t: float) -> tuple[np.ndarray, dict]:
        w, h = self.sim.width, self.sim.height
        if self.sim.night:
            bgr = np.full((h, w, 3), 18, dtype=np.uint8)
        else:
            # Daytime verge / off-road fill. Real daytime asphalt sits around 110-145 grey after
            # auto exposure; the old 42 made every "day" frame look like night to the detector.
            bgr = np.full((h, w, 3), 96, dtype=np.uint8)
        # sky
        sky_h = int(h * 0.38)
        if self.sim.night:
            bgr[:sky_h] = (18, 12, 8)
        else:
            grad = np.linspace(210, 160, sky_h, dtype=np.uint8)
            bgr[:sky_h, :, 0] = 40
            bgr[:sky_h, :, 1] = (grad * 0.55).astype(np.uint8)[:, None]
            bgr[:sky_h, :, 2] = grad[:, None]
        ego = self.ego_y(t)
        self._draw_road(bgr, ego)
        gt_objs = []
        for obj in self.objects:
            rel_y = obj.y0_m - ego
            if rel_y < 1.5 or rel_y > 55:
                continue
            poly_px, visible = self._draw_object(bgr, obj, rel_y)
            if visible and poly_px:
                gt_objs.append(
                    {
                        "id": obj.obj_id,
                        "semantic": obj.semantic.value,
                        "geometry": obj.geometry.value,
                        "state": obj.state.value,
                        "severity": int(obj.severity),
                        "distance_m": rel_y,
                        "x_m": obj.x_m,
                        "polygon": poly_px,
                    }
                )
        if self._in_windows(t, self.sim.occlude_windows):
            cv2.rectangle(bgr, (int(w * 0.34), int(h * 0.28)), (int(w * 0.72), int(h * 0.70)), (40, 40, 80), -1)
            cv2.rectangle(bgr, (int(w * 0.38), int(h * 0.22)), (int(w * 0.68), int(h * 0.32)), (200, 40, 30), -1)
        if self._in_windows(t, self.sim.glare_windows):
            overlay = bgr.copy()
            cv2.circle(overlay, (w // 2, int(h * 0.22)), 220, (255, 255, 255), -1)
            bgr = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 20)
        if self.sim.rain:
            rng = np.random.default_rng(int(t * 1000) % 10_000)
            overlay = bgr.copy()
            for _ in range(90):
                x = int(rng.integers(0, w))
                y = int(rng.integers(0, h))
                cv2.line(overlay, (x, y), (x + 2, y + 16), (190, 190, 200), 1)
            bgr = cv2.addWeighted(overlay, 0.28, bgr, 0.72, 0)
            bgr = cv2.GaussianBlur(bgr, (5, 5), 1.1)
        if self.sim.lens_drops:
            n = 12
            r = max(6, int(min(w, h) * 0.028))
            for i in range(n):
                x = int(w * (0.12 + 0.07 * (i % 6)))
                y = int(h * (0.08 + 0.08 * (i // 6)))
                # Real drops are bright refraction with a darker rim — and they stay put.
                cv2.circle(bgr, (x, y), r, (210, 220, 230), -1)
                cv2.circle(bgr, (x, y), max(2, r - 2), (40, 44, 48), 1)
                cv2.circle(bgr, (x - r // 3, y - r // 3), max(1, r // 4), (245, 248, 255), -1)
        if self._in_windows(t, self.sim.blur_windows):
            k = 21
            bgr = cv2.GaussianBlur(bgr, (k, k), 8)
            M = np.float32([[1, 0, 18], [0, 1, 0]])
            shifted = cv2.warpAffine(bgr, M, (w, h))
            bgr = cv2.addWeighted(bgr, 0.45, shifted, 0.55, 0)
        if self.sim.vibration:
            dx = int(5 * np.sin(t * 41.0))
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            bgr = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            bgr = cv2.GaussianBlur(bgr, (9, 9), 2.0)
        gt = {"t": t, "ego_y": ego, "objects": gt_objs, "speed_mps": self.sim.speed_mps}
        return bgr, gt

    def _draw_road(self, bgr: np.ndarray, ego_y: float) -> None:
        w, h = self.sim.width, self.sim.height
        wet = self.sim.wet or self.sim.rain
        asphalt = (118, 118, 124) if not self.sim.night else (36, 36, 38)
        if wet:
            asphalt = (96, 88, 82) if not self.sim.night else (28, 26, 24)
        pts = self._road_pixel_pts()
        if len(pts) >= 3:
            cv2.fillConvexPoly(bgr, np.array(pts, dtype=np.int32), asphalt)
        # lane dashes every 4 m
        phase = ego_y % 8.0
        for s in np.arange(4.0 - phase, 42.0, 8.0):
            for x in (-0.08, 0.08):
                p0 = project_vehicle_point(np.array([x, s, 0.0]), self.mount, self.k)
                p1 = project_vehicle_point(np.array([x, s + 3.0, 0.0]), self.mount, self.k)
                if p0 is not None and p1 is not None:
                    cv2.line(bgr, tuple(p0.astype(int)), tuple(p1.astype(int)), (210, 210, 220), 3, cv2.LINE_AA)
        if wet:
            overlay = bgr.copy()
            for x in (-1.6, 0.0, 1.6):
                p0 = project_vehicle_point(np.array([x, 6.0, 0.0]), self.mount, self.k)
                p1 = project_vehicle_point(np.array([x, 22.0, 0.0]), self.mount, self.k)
                if p0 is not None and p1 is not None:
                    cv2.line(overlay, tuple(p0.astype(int)), tuple(p1.astype(int)), (190, 190, 200), 6, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.22, bgr, 0.78, 0, bgr)
        # headlights cone at night
        if self.sim.night:
            overlay = bgr.copy()
            cone = []
            for x, y in [(-3.2, 4), (3.2, 4), (5.5, 28), (-5.5, 28)]:
                uv = project_vehicle_point(np.array([x, y, 0.0]), self.mount, self.k)
                if uv is not None:
                    cone.append(uv)
            if len(cone) >= 3:
                cv2.fillConvexPoly(overlay, np.array(cone, dtype=np.int32), (70, 70, 40))
                cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0, bgr)
        if wet:
            rng = np.random.default_rng(int(ego_y * 10) % 10_000)
            overlay = bgr.copy()
            for _ in range(6):
                x = int(rng.integers(int(w * 0.25), int(w * 0.75)))
                y = int(rng.integers(int(h * 0.45), int(h * 0.92)))
                cv2.ellipse(overlay, (x, y), (int(w * 0.08), 10), 0, 0, 360, (210, 210, 220), -1)
            cv2.addWeighted(overlay, 0.18, bgr, 0.82, 0, bgr)

    def _draw_object(self, bgr: np.ndarray, obj: WorldObject, rel_y: float) -> tuple[list[tuple[float, float]], bool]:
        xs = [obj.x_m - obj.width_m / 2, obj.x_m + obj.width_m / 2]
        ys = [rel_y, rel_y + obj.length_m]
        corners = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
        pix = []
        for x, y in corners:
            uv = project_vehicle_point(np.array([x, y, 0.0]), self.mount, self.k)
            if uv is None:
                return [], False
            pix.append((float(uv[0]), float(uv[1])))
        pts = np.array(pix, dtype=np.int32)
        if obj.semantic == SemanticType.MANHOLE_COVER:
            center = project_vehicle_point(np.array([obj.x_m, rel_y + obj.length_m / 2, 0.0]), self.mount, self.k)
            edge = project_vehicle_point(np.array([obj.x_m + obj.width_m / 2, rel_y + obj.length_m / 2, 0.0]), self.mount, self.k)
            if center is None or edge is None:
                return [], False
            r = int(np.linalg.norm(edge - center))
            cv2.circle(bgr, tuple(center.astype(int)), max(6, r), obj.color, -1)
            cv2.circle(bgr, tuple(center.astype(int)), max(6, r), (20, 20, 20), 2)
            poly = [
                (float(center[0] + r * np.cos(a)), float(center[1] + 0.7 * r * np.sin(a)))
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ]
            return poly, True
        cv2.fillConvexPoly(bgr, pts, obj.color)
        if obj.semantic == SemanticType.SPEED_BUMP:
            cv2.polylines(bgr, [pts], True, (0, 200, 255), 2)
        if obj.semantic == SemanticType.ROAD_JOINT:
            cv2.polylines(bgr, [pts], True, (160, 160, 165), 2)
        if obj.semantic == SemanticType.ROUGH_BROKEN:
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            noise = np.random.default_rng(int(rel_y * 10)).integers(0, 40, (max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.uint8)
            roi = bgr[y0:y1, x0:x1]
            if roi.size and noise.shape[:2] == roi.shape[:2]:
                bgr[y0:y1, x0:x1] = cv2.add(roi, noise)
        return pix, True


def write_preview_video(path: str, sim: RoadSimulator, max_frames: int | None = None) -> None:
    n = sim.n_frames() if max_frames is None else min(max_frames, sim.n_frames())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, sim.sim.fps, (sim.sim.width, sim.sim.height))
    for i in range(n):
        frame, _ = sim.frame_at(i)
        vw.write(frame.bgr)
    vw.release()
