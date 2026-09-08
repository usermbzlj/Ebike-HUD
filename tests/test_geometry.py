from __future__ import annotations

import numpy as np

from rpar.config import GeometryConfig
from rpar.enums import Direction
from rpar.geometry import GeometryEngine
from rpar.tracking import TrackInternal
from rpar.transforms import default_intrinsics, default_mount


def test_center_vs_right_direction():
    mount = default_mount()
    geo = GeometryEngine(mount, GeometryConfig(), default_intrinsics())
    tr = TrackInternal(track_id=1, hits=4, misses=0, created_ns=0, last_ns=0, polygon=[(960, 700), (980, 700), (980, 740), (960, 740)])
    tr.road_xy = np.array([0.1, 12.0])
    assert geo.direction_for(tr) == Direction.CENTER_FRONT
    tr.road_xy = np.array([2.2, 12.0])
    assert geo.direction_for(tr) == Direction.RIGHT_FRONT
    tr.road_xy = np.array([-2.2, 12.0])
    assert geo.direction_for(tr) == Direction.LEFT_FRONT


def test_invalid_mount_hides_distance():
    mount = default_mount()
    mount.valid = False
    geo = GeometryEngine(mount, GeometryConfig(), default_intrinsics())
    tr = TrackInternal(track_id=1, hits=4, misses=0, created_ns=0, last_ns=0, polygon=[(960, 800)])
    d, c, v = geo.distance_for_track(tr, 1)
    assert v is False
    assert d is None
    assert geo.display_distance(12.0, False, 0.9) is None


def test_distance_rounding_rules():
    geo = GeometryEngine(default_mount(), GeometryConfig(), default_intrinsics())
    assert geo.display_distance(7.4, True, 0.8) == "7 m"
    txt = geo.display_distance(17.0, True, 0.8)
    assert txt and "约" in txt


def test_known_distance_fit_recovers_pitch():
    from rpar.geometry import fit_mount_from_distance_markers
    from rpar.transforms import project_vehicle_point

    k = default_intrinsics()
    truth = default_mount()
    markers = []
    for d in (5.0, 10.0, 20.0):
        uv = project_vehicle_point(np.array([0.0, d, 0.0]), truth, k)
        assert uv is not None
        markers.append((d, float(uv[1])))
    skewed = default_mount()
    skewed.pitch_deg = 14.0
    skewed.camera_height_m = 1.35
    skewed.valid = False
    fitted = fit_mount_from_distance_markers(skewed, k, markers)
    assert abs(fitted.pitch_deg - truth.pitch_deg) < 1.5
    assert abs(fitted.camera_height_m - truth.camera_height_m) < 0.2
    assert fitted.valid is True


def test_health_check_hides_distance_on_bad_horizon():
    geo = GeometryEngine(default_mount(), GeometryConfig(), default_intrinsics())
    ok = geo.health_check(0.0, 0.0, horizon_y_px=20.0)
    assert ok is False
    assert geo.valid is False


def test_display_compensate_is_identity_without_yaw():
    from rpar.transforms import display_compensate

    poly = [(10.0, 20.0), (30.0, 40.0)]
    assert display_compensate(poly, 0.0, 50.0, 1920) == poly
