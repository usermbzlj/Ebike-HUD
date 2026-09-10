"""AR overlay compositor used by desktop replay, golden videos and reports."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from rpar.enums import PerceptionStatus, UiMode
from rpar.models import PerceptionView, RenderPrimitive


PALETTE = {
    "anomaly": (40, 210, 200),
    "occlusion": (140, 150, 160),
    "corridor": (180, 200, 40),
    "status": (240, 240, 240),
    "road": (90, 180, 80),
    "info": (180, 160, 60),
}

_ROAD_HOLE_KINDS = frozenset({"anomaly", "bump"})


def draw_poly(img: np.ndarray, prim: RenderPrimitive, holes: Iterable[RenderPrimitive] | None = None) -> None:
    if len(prim.polygon) < 3:
        return
    pts = np.array(prim.polygon, dtype=np.int32)
    color = tuple(int(np.clip(c * 255, 0, 255)) for c in prim.color_rgba[:3][::-1])
    alpha = float(np.clip(prim.color_rgba[3], 0, 1))
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    fill = 0.58 if prim.kind == "road" else 0.46 if prim.kind == "occlusion" else 0.28
    if prim.kind == "road" and holes:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        for hole in holes:
            if len(hole.polygon) < 3:
                continue
            cv2.fillPoly(mask, [np.array(hole.polygon, dtype=np.int32)], 0)
        mix = cv2.addWeighted(overlay, alpha * fill, img, 1 - alpha * fill, 0)
        img[mask > 0] = mix[mask > 0]
    else:
        cv2.addWeighted(overlay, alpha * fill, img, 1 - alpha * fill, 0, img)
    if prim.dashed:
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            if i % 2 == 0:
                cv2.line(img, tuple(a), tuple(b), color, max(1, int(prim.thickness)), cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], True, color, max(1, int(prim.thickness)), cv2.LINE_AA)


def draw_label(img: np.ndarray, prim: RenderPrimitive) -> None:
    if not prim.label or len(prim.polygon) < 1:
        return
    pts = np.array(prim.polygon, dtype=np.int32)
    x = int(np.clip(pts[:, 0].mean(), 8, img.shape[1] - 160))
    y = int(np.clip(pts[:, 1].min() - 12, 28, img.shape[0] - 8))
    (tw, th), _ = cv2.getTextSize(prim.label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 10, y + 6), (10, 12, 16), -1)
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 10, y + 6), (40, 210, 200), 1)
    cv2.putText(img, prim.label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (236, 244, 248), 1, cv2.LINE_AA)


def compose(bgr: np.ndarray, view: PerceptionView, ui_mode: UiMode = UiMode.RIDING, night: bool = False) -> np.ndarray:
    img = bgr.copy()
    if night:
        img = np.clip(img.astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
    prims = sorted(view.primitives, key=lambda p: p.label_priority, reverse=True)
    holes = [p for p in prims if p.kind in _ROAD_HOLE_KINDS and len(p.polygon) >= 3]
    roads = [p for p in prims if p.kind == "road"]
    others = [p for p in prims if p.kind != "road"]
    for p in roads:
        draw_poly(img, p, holes=holes)
    if others:
        overlay = img.copy()
        for p in others:
            if len(p.polygon) < 3:
                continue
            pts = np.array(p.polygon, dtype=np.int32)
            color = tuple(int(np.clip(c * 255, 0, 255)) for c in p.color_rgba[:3][::-1])
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
        for p in others:
            if len(p.polygon) < 3:
                continue
            pts = np.array(p.polygon, dtype=np.int32)
            color = tuple(int(np.clip(c * 255, 0, 255)) for c in p.color_rgba[:3][::-1])
            if p.dashed:
                for i in range(len(pts)):
                    a = pts[i]
                    b = pts[(i + 1) % len(pts)]
                    if i % 2 == 0:
                        cv2.line(img, tuple(a), tuple(b), color, max(1, int(p.thickness)), cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], True, color, max(1, int(p.thickness)), cv2.LINE_AA)
    for p in prims:
        draw_label(img, p)
    if ui_mode == UiMode.RESEARCH and view.quality is not None:
        _heatmap(img, view)
    _hud(img, view, ui_mode)
    return img


def _heatmap(img: np.ndarray, view: PerceptionView) -> None:
    tiles = getattr(view.quality, "tiles", None) or []
    overlay = img.copy()
    for t in tiles:
        vis = t.visibility.value if hasattr(t.visibility, "value") else str(t.visibility)
        if vis == "clear":
            continue
        color = {
            "blur": (40, 90, 210),
            "glare": (40, 220, 255),
            "underexposed": (180, 80, 40),
            "overexposed": (240, 240, 240),
            "occluded": (140, 140, 150),
            "lens_drop": (90, 90, 200),
        }.get(vis, (120, 120, 120))
        cv2.rectangle(overlay, (int(t.x0), int(t.y0)), (int(t.x1), int(t.y1)), color, -1)
    if tiles:
        cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)


def _hud(img: np.ndarray, view: PerceptionView, ui_mode: UiMode) -> None:
    h, w = img.shape[:2]
    speed = f"{view.speed_kmh:.0f} km/h" if view.speed_kmh is not None else "-- km/h"
    cv2.rectangle(img, (24, 20), (210, 64), (8, 10, 14), -1)
    cv2.rectangle(img, (24, 20), (210, 64), (40, 210, 200), 1)
    cv2.putText(img, speed, (36, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (236, 244, 248), 2, cv2.LINE_AA)
    if ui_mode == UiMode.RESEARCH:
        far = view.input_far
        near = view.input_near
        lines = [
            f"FPS {view.ar_fps:.0f} / AI {view.infer_fps:.0f}",
            f"p50 {view.latency_p50_ms:.0f}  p95 {view.latency_p95_ms:.0f}ms",
            f"q={view.queue_depth} drop={view.dropped_infer} dual={int(view.dual_scale)}",
            f"{view.backend.value} {view.model_version} {far[0]}x{far[1]}/{near[0]}x{near[1]}",
        ]
        cv2.rectangle(img, (24, 78), (520, 186), (8, 10, 14), -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (36, 106 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 214, 220), 1, cv2.LINE_AA)
    bar = img[h - 54 : h, 0:w].copy()
    cv2.rectangle(bar, (0, 0), (w, 54), (8, 10, 14), -1)
    cv2.addWeighted(bar, 0.72, img[h - 54 : h, 0:w], 0.28, 0, img[h - 54 : h, 0:w])
    status = view.status_copy
    rec = f"REC {int(view.rec_seconds // 3600):02d}:{int(view.rec_seconds % 3600 // 60):02d}:{int(view.rec_seconds % 60):02d}"
    cv2.putText(
        img,
        f"{status}  ·  {rec}  ·  {view.model_version}",
        (28, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 230, 236),
        1,
        cv2.LINE_AA,
    )
    if view.status != PerceptionStatus.NORMAL:
        cv2.putText(img, "PERCEPTION DEGRADED", (w - 360, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)
