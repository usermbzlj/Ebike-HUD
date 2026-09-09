"""Device / camera / vehicle / road / gravity transforms (SYNC-005, GEO-006)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from rpar.models import Intrinsics, MountProfile


def deg2rad(d: float) -> float:
    return d * np.pi / 180.0


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_from_vehicle(mount: MountProfile) -> np.ndarray:
    """Rotation taking vehicle coordinates (x right, y forward, z up) into camera (x right, y down, z forward)."""
    pitch = deg2rad(mount.pitch_deg)
    roll = deg2rad(mount.roll_deg)
    yaw = deg2rad(mount.yaw_deg + mount.handlebar_neutral_yaw_deg)
    # vehicle -> camera optical: look along +Y vehicle, then apply mount errors
    basis = np.array(
        [
            [1.0, 0.0, 0.0],  # camera x = vehicle x (right)
            [0.0, 0.0, -1.0],  # camera y = down = -vehicle z
            [0.0, 1.0, 0.0],  # camera z = forward = vehicle y
        ],
        dtype=np.float64,
    )
    return rot_z(roll) @ rot_x(pitch) @ rot_y(yaw) @ basis


def camera_translation_vehicle(mount: MountProfile) -> np.ndarray:
    # vehicle: x right, y forward, z up. Left handlebar is negative x.
    return np.array([mount.lateral_offset_m, 0.0, mount.camera_height_m], dtype=np.float64)


def project_vehicle_point(p_vehicle: np.ndarray, mount: MountProfile, k: Intrinsics) -> np.ndarray | None:
    r = camera_from_vehicle(mount)
    t = -r @ camera_translation_vehicle(mount)
    p_cam = r @ p_vehicle + t
    if p_cam[2] <= 0.15:
        return None
    uv = k.matrix() @ p_cam
    return np.array([uv[0] / uv[2], uv[1] / uv[2]], dtype=np.float64)


def pixel_to_ground(uv: np.ndarray, mount: MountProfile, k: Intrinsics) -> np.ndarray | None:
    """Ray-plane intersection on vehicle Z=0. Result is already vehicle-centerline XY (GEO-006)."""
    x = (float(uv[0]) - k.cx) / k.fx
    y = (float(uv[1]) - k.cy) / k.fy
    ray_cam = np.array([x, y, 1.0], dtype=np.float64)
    n = np.linalg.norm(ray_cam)
    if n < 1e-9:
        return None
    ray_cam = ray_cam / n
    r_v_to_c = camera_from_vehicle(mount)
    t_v = camera_translation_vehicle(mount)
    dir_v = r_v_to_c.T @ ray_cam
    if abs(dir_v[2]) < 1e-8:
        return None
    scale = -t_v[2] / dir_v[2]
    if scale < 0.2:
        return None
    p = t_v + scale * dir_v
    if not np.isfinite(p).all() or p[1] < 0.4 or p[1] > 90:
        return None
    return p[:2]


def ground_to_pixel(xy: np.ndarray, mount: MountProfile, k: Intrinsics) -> np.ndarray | None:
    p = np.array([xy[0], xy[1], 0.0], dtype=np.float64)
    return project_vehicle_point(p, mount, k)


def compensate_left_handlebar(xy_cam_ground: np.ndarray, mount: MountProfile) -> np.ndarray:
    """Shift from camera-centered ground coords into vehicle-centerline road coords (GEO-006)."""
    return np.array([xy_cam_ground[0] - mount.lateral_offset_m, xy_cam_ground[1]], dtype=np.float64)


def rotate_points_about_principal(points: np.ndarray, cx: float, cy: float, roll_rad: float) -> np.ndarray:
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    centered = points - np.array([cx, cy])
    return centered @ r.T + np.array([cx, cy])


def default_intrinsics(width: int = 1920, height: int = 1080, hfov_deg: float = 68.0) -> Intrinsics:
    fx = (width / 2.0) / np.tan(deg2rad(hfov_deg) / 2.0)
    fy = fx
    return Intrinsics(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def chessboard_overlay_error_px(
    mount: MountProfile | None = None,
    k: Intrinsics | None = None,
    n: int = 5,
) -> float:
    """Numerical self-consistency of the ground <-> pixel chain on a grid (CAM-009 desktop half).

    This only checks that projection and back-projection agree; the preview/AR/model
    coordinate agreement on the phone is measured by `CameraTransformChain` on device.
    """
    mount = mount or default_mount()
    k = k or default_intrinsics()
    errs: list[float] = []
    for x in np.linspace(-1.6, 1.6, n):
        for y in np.linspace(5.0, 28.0, n):
            uv = ground_to_pixel(np.array([x, y]), mount, k)
            if uv is None:
                continue
            xy = pixel_to_ground(uv, mount, k)
            if xy is None:
                continue
            uv2 = ground_to_pixel(xy, mount, k)
            if uv2 is None:
                continue
            errs.append(float(np.linalg.norm(uv - uv2)))
    return max(errs) if errs else 999.0


def display_compensate(
    poly: list[tuple[float, float]],
    yaw_rate: float,
    latency_ms: float,
    frame_w: int,
    focal_px: float | None = None,
) -> list[tuple[float, float]]:
    """GEO-007: shift overlay by predicted camera yaw over display latency.

    dx = f * yaw_rate * latency; without intrinsics assume ~85 deg HFOV (f ~ 0.55 w).
    """
    f = float(focal_px) if focal_px and focal_px > 0 else float(frame_w) * 0.55
    dx = float(yaw_rate) * (latency_ms / 1000.0) * f
    if abs(dx) < 0.5:
        return poly
    return [(x + dx, y) for x, y in poly]


def default_mount(width: int = 1920, height: int = 1080) -> MountProfile:
    """Nominal left-handlebar profile. The stored horizon is the one this pose projects to."""
    k = default_intrinsics(width, height)
    mount = MountProfile(
        profile_id="left_handlebar_v1",
        name="Left handlebar landscape 1x main",
        camera_id="rear_main",
        landscape=True,
        camera_height_m=1.12,
        pitch_deg=18.0,
        roll_deg=0.0,
        yaw_deg=0.0,
        lateral_offset_m=-0.32,
        handlebar_neutral_yaw_deg=0.0,
        near_reference_m=3.0,
        horizon_y_px=0.0,
        vehicle_centerline_x_px=float(width) * 0.52,
        valid=True,
    )
    far = project_vehicle_point(np.array([0.0, 80.0, 0.0]), mount, k)
    horizon = float(far[1]) if far is not None else float(k.cy) * 0.42
    return replace(mount, horizon_y_px=horizon)
