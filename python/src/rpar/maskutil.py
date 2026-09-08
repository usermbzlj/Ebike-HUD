"""Polygon / RLE / bbox helpers used by perception, tracking and AR."""

from __future__ import annotations

import numpy as np


def polygon_bbox(poly: list[tuple[float, float]] | np.ndarray) -> tuple[float, float, float, float]:
    pts = np.asarray(poly, dtype=np.float64)
    if pts.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def polygon_area(poly: list[tuple[float, float]]) -> float:
    pts = np.asarray(poly, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    pts = np.asarray(poly, dtype=np.float64)
    if pts.size == 0:
        return (0.0, 0.0)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def ground_contact(poly: list[tuple[float, float]]) -> tuple[float, float]:
    """Bottom-weighted contact point used for distance / corridor projection."""
    pts = np.asarray(poly, dtype=np.float64)
    if pts.size == 0:
        return (0.0, 0.0)
    y_max = pts[:, 1].max()
    band = pts[pts[:, 1] >= y_max - 6.0]
    if band.size == 0:
        band = pts
    return float(band[:, 0].mean()), float(band[:, 1].mean())


def ellipse_polygon(cx: float, cy: float, rx: float, ry: float, n: int = 18) -> list[tuple[float, float]]:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(cx + rx * np.cos(a)), float(cy + ry * np.sin(a))) for a in ang]


def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def mask_from_polygon(poly: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    import cv2

    img = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(img, [pts], 1)
    return img


def rle_encode(mask: np.ndarray) -> list[int]:
    """COCO-style counts starting with zeros-run (may be length 0)."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    prev = 0
    run = 0
    for v in flat:
        vv = 1 if v else 0
        if vv == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = vv
    counts.append(run)
    return counts


def rle_decode(counts: list[int], width: int, height: int) -> np.ndarray:
    total = width * height
    flat = np.zeros(total, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        if val:
            end = min(total, pos + c)
            flat[pos:end] = 1
        pos += c
        if pos >= total:
            break
        val = 1 - val
    return flat.reshape((height, width), order="F")


def mask_rle_from_polygon(poly: list[tuple[float, float]], max_side: int = 160):
    """Instance RLE over the object bbox (PER-004). Counts are Fortran-order on the bbox grid."""
    from rpar.models import MaskRle

    if len(poly) < 3:
        return None
    x0, y0, x1, y1 = polygon_bbox(poly)
    bw = max(1, int(np.ceil(x1) - np.floor(x0)))
    bh = max(1, int(np.ceil(y1) - np.floor(y0)))
    scale = 1.0
    if max(bw, bh) > max_side:
        scale = max_side / float(max(bw, bh))
        bw = max(1, int(round(bw * scale)))
        bh = max(1, int(round(bh * scale)))
    shifted = [((x - x0) * scale, (y - y0) * scale) for x, y in poly]
    mask = mask_from_polygon(shifted, bw, bh)
    return MaskRle(width=bw, height=bh, counts=rle_encode(mask), encoding="rle_cocoa")


def simplify_polygon(poly: list[tuple[float, float]], epsilon: float = 2.5) -> list[tuple[float, float]]:
    import cv2

    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        return list(poly)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def nms_polygons(
    items: list[tuple[list[tuple[float, float]], float]],
    iou_thr: float = 0.45,
) -> list[int]:
    order = np.argsort([-s for _, s in items])
    keep: list[int] = []
    suppressed = np.zeros(len(items), dtype=bool)
    bboxes = [polygon_bbox(p) for p, _ in items]
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in order:
            if suppressed[j] or j == i:
                continue
            if bbox_iou(bboxes[i], bboxes[j]) >= iou_thr:
                suppressed[j] = True
    return keep
