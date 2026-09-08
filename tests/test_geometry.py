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
