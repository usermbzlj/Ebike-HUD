"""Open-vocab bump prompts mapped onto spec enums. No torch / ultralytics import."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rpar.enums import GeometryType, ObjectState, SemanticType, Severity, VisibilityClass
from rpar.maskutil import bbox_iou, ellipse_polygon, mask_rle_from_polygon, polygon_bbox, simplify_polygon
from rpar.models import RoadObservation
from rpar.quality import mask_visibility

YOLO_NAMES = ("pothole", "speed_bump", "manhole_cover")
CLASS_TO_ID = {name: i for i, name in enumerate(YOLO_NAMES)}

WORLD_PROMPTS = (
    "pothole",
    "large pothole",
    "speed bump",
    "sunken manhole cover",
    "manhole cover",
)
WORLD_INFER_PROMPTS = WORLD_PROMPTS + ("",)

_SUNKEN_PROMPTS = frozenset({"sunken manhole", "sunken manhole cover", "settled manhole"})

PROMPT_TO_CLASS = {
    "pothole": "pothole",
    "large pothole": "pothole",
    "asphalt pothole": "pothole",
    "road hole": "pothole",
    "asphalt hole": "pothole",
    "speed bump": "speed_bump",
    "speed hump": "speed_bump",
    "rubber speed bump": "speed_bump",
    "speedbreaker": "speed_bump",
    "sunken manhole": "manhole_cover",
    "sunken manhole cover": "manhole_cover",
    "settled manhole": "manhole_cover",
    "manhole cover": "manhole_cover",
    "manhole": "manhole_cover",
}

BUMP_TYPES = {SemanticType.POTHOLE, SemanticType.SPEED_BUMP, SemanticType.MANHOLE_COVER}


def _norm(raw: str) -> str:
    return " ".join(str(raw).strip().lower().replace("_", " ").replace("-", " ").split())


def map_det_name(raw: str) -> str | None:
    key = _norm(raw)
    if key in YOLO_NAMES:
        return key
    return PROMPT_TO_CLASS.get(key)


def clip_class_hint(name: str) -> str | None:
    """Optional filename tag so inbox clips can bias a class. Mixed riding clips stay None."""
    s = Path(name).stem.lower().replace("-", "_").replace(" ", "_")
    if any(k in s for k in ("sunken", "xiachen", "下沉")):
        return "manhole_cover"
    if any(k in s for k in ("manhole", "井盖")):
        return "manhole_cover"
    if any(k in s for k in ("speedbump", "speed_bump", "speed-bump", "hump", "减速带", "减速")):
        return "speed_bump"
    if any(k in s for k in ("pothole", "大坑", "坑洼")):
        return "pothole"
    return None


def is_sunken_prompt(raw: str) -> bool:
    return _norm(raw) in _SUNKEN_PROMPTS


def spec_attrs(yolo_class: str, raw_prompt: str = "", *, sunken: bool = False) -> tuple[SemanticType, GeometryType, ObjectState, Severity]:
    kind = map_det_name(yolo_class) or map_det_name(raw_prompt)
    if kind == "pothole":
        return SemanticType.POTHOLE, GeometryType.CONCAVE, ObjectState.ABNORMAL, Severity.HEAVY
    if kind == "speed_bump":
        return SemanticType.SPEED_BUMP, GeometryType.CONVEX, ObjectState.ABNORMAL, Severity.MEDIUM
    if kind == "manhole_cover":
        if sunken or is_sunken_prompt(raw_prompt):
            return SemanticType.MANHOLE_COVER, GeometryType.CONCAVE, ObjectState.ABNORMAL, Severity.HEAVY
        return SemanticType.MANHOLE_COVER, GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE
    return SemanticType.UNKNOWN_ANOMALY, GeometryType.UNKNOWN, ObjectState.UNKNOWN, Severity.UNKNOWN


def refine_manhole_geometry(
    bgr: np.ndarray, bbox: tuple[float, float, float, float], sunken_delta: float = 12.0
) -> tuple[GeometryType, ObjectState, Severity]:
    """Sunken cover: disk whose interior is darker than the rim. Flat covers stay info-only.

    Shares `manhole_is_sunken` with the heuristic engine so both paths agree. A settled cover
    is MEDIUM, not HEAVY: a brightness test cannot measure depth.
    """
    from rpar.perception import manhole_is_sunken

    if bgr.size == 0:
        return GeometryType.UNKNOWN, ObjectState.UNKNOWN, Severity.UNKNOWN
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    if manhole_is_sunken(gray, bbox, sunken_delta):
        return GeometryType.CONCAVE, ObjectState.ABNORMAL, Severity.MEDIUM
    return GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE


def keep_box(yolo_class: str, bbox: tuple[float, float, float, float], width: int, height: int) -> bool:
    x0, y0, x1, y1 = bbox
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    if bw < 12.0 or bh < 8.0:
        return False
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    # 15–20 m 中大型 hazards sit near the handlebar horizon, not the near wheel.
    if cy < 0.16 * height:
        return False
    if cx < 0.08 * width or cx > 0.92 * width:
        return False
    area = bw * bh
    frac = area / max(1.0, float(width * height))
    if frac > 0.20 or bw > 0.88 * width:
        return False
    aspect = bw / max(bh, 1.0)
    if yolo_class == "pothole":
        return 0.00035 <= frac <= 0.12 and aspect < 3.5
    if yolo_class == "speed_bump":
        return 0.0008 <= frac <= 0.12 and (bw >= 0.10 * width or aspect >= 1.6)
    if yolo_class == "manhole_cover":
        # Handlebar view flattens circular covers into wide ellipses.
        return 0.00035 <= frac <= 0.08 and 0.35 <= aspect <= 4.2
    return False


def nms_xyxy(dets: list[dict], iou_thr: float = 0.50) -> list[dict]:
    ordered = sorted(dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    keep: list[dict] = []
    for d in ordered:
        box = tuple(d["bbox"])
        if all(bbox_iou(box, tuple(k["bbox"])) < iou_thr for k in keep):
            keep.append(d)
    return keep


def xyxy_to_yolo(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    xc = (0.5 * (x0 + x1)) / max(width, 1)
    yc = (0.5 * (y0 + y1)) / max(height, 1)
    bw = (x1 - x0) / max(width, 1)
    bh = (y1 - y0) / max(height, 1)
    return (
        float(np.clip(xc, 0, 1)),
        float(np.clip(yc, 0, 1)),
        float(np.clip(bw, 0, 1)),
        float(np.clip(bh, 0, 1)),
    )


def yolo_line(class_id: int, bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    xc, yc, bw, bh = xyxy_to_yolo(bbox, width, height)
    return f"{int(class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def detection_to_obs(
    frame,
    raw_name: str,
    bbox: tuple[float, float, float, float],
    conf: float,
    bgr: np.ndarray,
    quality=None,
    *,
    force_sunken: bool = False,
) -> RoadObservation | None:
    kind = map_det_name(raw_name)
    if kind is None:
        return None
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    box = (
        float(np.clip(x0, 0, w - 1)),
        float(np.clip(y0, 0, h - 1)),
        float(np.clip(x1, 1, w)),
        float(np.clip(y1, 1, h)),
    )
    if not keep_box(kind, box, w, h):
        return None
    semantic, geometry, state, severity = spec_attrs(kind, raw_name, sunken=force_sunken)
    if kind == "manhole_cover" and not (force_sunken or is_sunken_prompt(raw_name)):
        geometry, state, severity = refine_manhole_geometry(bgr, box)
        semantic = SemanticType.MANHOLE_COVER
    rx = 0.5 * (box[2] - box[0])
    ry = 0.5 * (box[3] - box[1])
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    poly = simplify_polygon(ellipse_polygon(cx, cy, max(6.0, rx), max(6.0, ry)), 3.0)
    vis = mask_visibility(quality, box) if quality is not None else 0.65
    vclass = quality.global_quality.visibility_class if quality is not None else VisibilityClass.CLEAR
    return RoadObservation(
        timestamp_ns=frame.meta.sensor_timestamp_ns,
        source_frame_id=frame.meta.frame_id,
        semantic_type=semantic,
        geometry_type=geometry,
        state=state,
        severity=severity,
        mask_rle=mask_rle_from_polygon(poly),
        polygon=poly,
        bbox=polygon_bbox(poly) if poly else box,
        model_confidence=float(np.clip(conf, 0, 1)),
        quality_at_mask=float(vis),
        visibility=vclass,
        calibrated_confidence=float(np.clip(conf, 0, 1)),
    )
