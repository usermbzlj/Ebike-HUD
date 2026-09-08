"""Runtime dataclasses for frames, observations, tracks and alert decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np

from rpar.enums import (
    Direction,
    GeometryType,
    InferenceBackend,
    LifecycleState,
    ObjectState,
    PerceptionStatus,
    SemanticType,
    SensorType,
    Severity,
    StabilizationMode,
    VisibilityClass,
)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    return value


@dataclass(slots=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    available: bool = True

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(slots=True)
class MountProfile:
    profile_id: str
    name: str
    camera_id: str
    landscape: bool
    camera_height_m: float
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    lateral_offset_m: float
    handlebar_neutral_yaw_deg: float
    near_reference_m: float
    horizon_y_px: float
    vehicle_centerline_x_px: float
    known_distance_5m_px: float | None = None
    known_distance_10m_px: float | None = None
    known_distance_20m_px: float | None = None
    calibration_hash: str = ""
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(slots=True)
class PoseSample:
    timestamp_ns: int
    quaternion_xyzw: tuple[float, float, float, float]
    gravity_xyz: tuple[float, float, float]
    pose_confidence: float


@dataclass(slots=True)
class ImuSample:
    timestamp_ns: int
    sensor_type: SensorType
    x: float
    y: float
    z: float
    accuracy: int
    source_rate_hz: float


@dataclass(slots=True)
class LocationSample:
    timestamp_ns: int
    latitude: float | None
    longitude: float | None
    altitude: float | None
    speed_mps: float | None
    bearing_deg: float | None
    horizontal_accuracy_m: float | None
    speed_accuracy_mps: float | None


@dataclass(slots=True)
class FrameMeta:
    frame_id: int
    sensor_timestamp_ns: int
    image_timestamp_ns: int
    exposure_time_ns: int | None
    iso: int | None
    focal_length_mm: float | None
    focus_distance_diopters: float | None
    af_state: str | None
    ae_state: str | None
    awb_state: str | None
    crop_region: tuple[int, int, int, int] | None
    stabilization_mode: StabilizationMode | None
    width: int
    height: int
    availability: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class FrameQuality:
    sharpness: float
    motion_blur: float
    defocus: float
    underexposure: float
    overexposure: float
    glare: float
    usable: bool
    visibility_class: VisibilityClass
    road_visible_ratio: float
    reason: str = ""


@dataclass(slots=True)
class QualityTile:
    x0: int
    y0: int
    x1: int
    y1: int
    visibility: VisibilityClass
    score: float


@dataclass(slots=True)
class FrameQualityMap:
    global_quality: FrameQuality
    tiles: list[QualityTile]
    occupancy_occluded_ratio: float
    selected_for_infer: bool
    selected_age_ms: float
    degrade_reason: str | None = None


@dataclass(slots=True)
class MaskRle:
    width: int
    height: int
    counts: list[int]
    encoding: str = "rle_cocoa"

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "counts": self.counts, "encoding": self.encoding}


@dataclass(slots=True)
class RoadObservation:
    timestamp_ns: int
    source_frame_id: int
    semantic_type: SemanticType
    geometry_type: GeometryType
    state: ObjectState
    severity: Severity
    mask_rle: MaskRle | None
    polygon: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    model_confidence: float
    quality_at_mask: float
    visibility: VisibilityClass
    calibrated_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = _to_jsonable(self)
        d["semantic_type"] = self.semantic_type.value
        d["geometry_type"] = self.geometry_type.value
        d["state"] = self.state.value
        d["severity"] = int(self.severity)
        d["visibility"] = self.visibility.value
        return d


@dataclass(slots=True)
class PerceptionResult:
    timestamp_ns: int
    source_frame_id: int
    road_polygon: list[tuple[float, float]]
    occluded_polygons: list[list[tuple[float, float]]]
    observations: list[RoadObservation]
    backend: InferenceBackend
    latency_ms: float
    input_sizes: list[tuple[int, int]]
    dual_scale: bool = True


@dataclass(slots=True)
class TrackedRoadObject:
    schema_version: str
    track_id: int
    timestamp_ns: int
    lifecycle_state: LifecycleState
    semantic_type: SemanticType
    geometry_type: GeometryType
    object_state: ObjectState
    severity: Severity
    direction: Direction
    distance_m: float | None
    distance_confidence: float
    distance_valid: bool
    ttc_s: float | None
    model_confidence: float
    visibility_confidence: float
    temporal_confidence: float
    geometry_consistency: float
    effective_confidence: float
    path_relevance: float
    risk_score: float
    alert_score: float
    polygon: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    mask_rle: MaskRle | None
    source_frame_id: int
    mount_profile_id: str
    model_version: str
    visual_style: str
    label_rank: int | None = None
    road_xy_m: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = _to_jsonable(self)
        d["lifecycle_state"] = self.lifecycle_state.value
        d["semantic_type"] = self.semantic_type.value
        d["geometry_type"] = self.geometry_type.value
        d["object_state"] = self.object_state.value
        d["severity"] = int(self.severity)
        d["direction"] = self.direction.value
        return d


@dataclass(slots=True)
class AlertDecision:
    timestamp_ns: int
    track_id: int
    fired: bool
    phrase: str
    direction: Direction
    semantic_type: SemanticType
    alert_score: float
    threshold: float
    reasons: list[str]
    snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "track_id": self.track_id,
            "fired": self.fired,
            "phrase": self.phrase,
            "direction": self.direction.value,
            "semantic_type": self.semantic_type.value,
            "alert_score": self.alert_score,
            "threshold": self.threshold,
            "reasons": self.reasons,
            "snapshot": self.snapshot,
        }


@dataclass(slots=True)
class RenderPrimitive:
    track_id: int
    polygon: list[tuple[float, float]]
    color_rgba: tuple[float, float, float, float]
    dashed: bool
    thickness: float
    label: str | None
    label_priority: int
    fade: float
    kind: str  # anomaly | occlusion | corridor | status


@dataclass(slots=True)
class PerceptionView:
    timestamp_ns: int
    status: PerceptionStatus
    status_copy: str
    tracks: list[TrackedRoadObject]
    primitives: list[RenderPrimitive]
    alerts: list[AlertDecision]
    quality: FrameQualityMap | None
    speed_kmh: float | None
    backend: InferenceBackend
    model_version: str
    infer_fps: float
    ar_fps: float
    latency_p95_ms: float
    queue_depth: int
    thermal_c: float | None
    blur: float
    glare: float
    rec_seconds: float
    dual_scale: bool


@dataclass(slots=True)
class SynchronizedFrame:
    meta: FrameMeta
    bgr: np.ndarray
    pose: PoseSample | None
    angular_velocity: tuple[float, float, float] | None
    linear_accel: tuple[float, float, float] | None
    location: LocationSample | None
    speed_mps: float | None
