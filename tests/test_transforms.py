from __future__ import annotations

import numpy as np

from rpar.models import Intrinsics
from rpar.transforms import (
    compensate_left_handlebar,
    default_intrinsics,
    default_mount,
    ground_to_pixel,
    pixel_to_ground,
    project_vehicle_point,
    rotate_points_about_principal,
)


def test_roundtrip_ground_pixel():
    k = default_intrinsics()
    mount = default_mount()
    uv = ground_to_pixel(np.array([0.0, 12.0]), mount, k)
    assert uv is not None
    xy = pixel_to_ground(uv, mount, k)
    assert xy is not None
    assert abs(xy[1] - 12.0) < 0.8
    assert abs(xy[0] - 0.0) < 0.5


def test_left_handlebar_shifts_centerline():
    mount = default_mount()
    cam_ground = np.array([0.0, 10.0])
    road = compensate_left_handlebar(cam_ground, mount)
    assert road[0] > 0.2  # camera is left of vehicle, so camera-center is right of vehicle-x=0? 
    # lateral_offset_m = -0.32 (camera is to the left). compensate: x_cam - offset = 0 - (-0.32) = +0.32
    assert abs(road[0] - 0.32) < 1e-6


def test_landscape_projection_forward_is_in_image():
    k = default_intrinsics(1920, 1080)
    mount = default_mount(1920, 1080)
    uv = project_vehicle_point(np.array([0.0, 15.0, 0.0]), mount, k)
    assert uv is not None
    assert 0 < uv[0] < 1920
    assert 0 < uv[1] < 1080


def test_roll_rotates_points():
    pts = np.array([[100.0, 50.0], [200.0, 50.0]])
    out = rotate_points_about_principal(pts, 150.0, 50.0, np.pi / 2)
    assert out.shape == pts.shape


def test_chessboard_overlay_error_under_8px():
    from rpar.transforms import chessboard_overlay_error_px

    assert chessboard_overlay_error_px() <= 8.0


def test_intrinsics_matrix_shape():
    k = Intrinsics(1000, 1000, 960, 540, 1920, 1080)
    m = k.matrix()
    assert m.shape == (3, 3)
    assert m[0, 0] == 1000
