from __future__ import annotations

import numpy as np

from rpar.models import RenderPrimitive
from rpar.overlay import draw_poly


def test_road_fill_punches_bump_hole():
    img = np.full((80, 80, 3), 12, dtype=np.uint8)
    road = RenderPrimitive(
        track_id=-6,
        polygon=[(2, 2), (77, 2), (77, 77), (2, 77)],
        color_rgba=(0.12, 0.92, 0.38, 0.5),
        dashed=False,
        thickness=2.0,
        label=None,
        label_priority=85,
        fade=1.0,
        kind="road",
    )
    bump = RenderPrimitive(
        track_id=1,
        polygon=[(30, 30), (50, 30), (50, 50), (30, 50)],
        color_rgba=(0.95, 0.28, 0.16, 0.9),
        dashed=False,
        thickness=2.0,
        label=None,
        label_priority=50,
        fade=1.0,
        kind="bump",
    )
    draw_poly(img, road, holes=[bump])
    road_px = img[8, 8]
    hole_px = img[40, 40]
    assert int(road_px[1]) > int(road_px[0])
    assert int(hole_px.sum()) < int(road_px.sum())
