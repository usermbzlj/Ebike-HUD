"""Canonical enumerations shared by Android, desktop, models and session bundles."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class SemanticType(StrEnum):
    POTHOLE = "pothole"
    MANHOLE_COVER = "manhole_cover"
    SPEED_BUMP = "speed_bump"
    ROAD_JOINT = "road_joint"
    REPAIR_PATCH = "repair_patch"
    ROUGH_BROKEN = "rough_broken"
    PUDDLE = "puddle"
    GRAVEL = "gravel"
    UNKNOWN_ANOMALY = "unknown_anomaly"


class GeometryType(StrEnum):
    CONCAVE = "concave"
    CONVEX = "convex"
    ROUGH = "rough"
    STEP = "step"
    FLAT = "flat"
    UNKNOWN = "unknown"


class ObjectState(StrEnum):
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    UNKNOWN = "unknown"


class Severity(IntEnum):
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3
    UNKNOWN = -1


class SurfaceCondition(StrEnum):
    DRY = "dry"
    WET = "wet"
    REFLECTIVE = "reflective"
    PUDDLED = "puddled"
    DUSTY = "dusty"
    UNKNOWN = "unknown"


class VisibilityClass(StrEnum):
    CLEAR = "clear"
    BLUR = "blur"
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    GLARE = "glare"
    OCCLUDED = "occluded"
    LENS_DROP = "lens_drop"
    UNKNOWN = "unknown"


class LifecycleState(StrEnum):
    CANDIDATE = "CANDIDATE"
    TRACKED = "TRACKED"
    CONFIRMED = "CONFIRMED"
    ALERTED = "ALERTED"
    PASSED = "PASSED"
    EXPIRED = "EXPIRED"


class Direction(StrEnum):
    LEFT_FRONT = "LEFT_FRONT"
    CENTER_FRONT = "CENTER_FRONT"
    RIGHT_FRONT = "RIGHT_FRONT"
    ACROSS = "ACROSS"
    UNKNOWN = "UNKNOWN"


class PerceptionStatus(StrEnum):
    NORMAL = "NORMAL"
    DEGRADED_VISIBILITY = "DEGRADED_VISIBILITY"
    SEVERE_BLUR = "SEVERE_BLUR"
    OCCLUDED = "OCCLUDED"
    LENS_CONTAMINATION = "LENS_CONTAMINATION"
    THERMAL_THROTTLE = "THERMAL_THROTTLE"
    STORAGE_LOW = "STORAGE_LOW"
    SAFE_MODE = "SAFE_MODE"
    PERCEPTION_LIMITED = "PERCEPTION_LIMITED"


class RunMode(StrEnum):
    CAPTURE_ONLY = "CAPTURE_ONLY"
    REALTIME_PERCEPTION = "REALTIME_PERCEPTION"
    REALTIME_PERCEPTION_FULL_LOG = "REALTIME_PERCEPTION_FULL_LOG"
    SAFE_MODE = "SAFE_MODE"
    REPLAY = "REPLAY"


class UiMode(StrEnum):
    RIDING = "RIDING"
    RESEARCH = "RESEARCH"


class StabilizationMode(StrEnum):
    OFF = "OFF"
    STANDARD = "STANDARD"
    PREVIEW = "PREVIEW"
    UNAVAILABLE = "UNAVAILABLE"


class InferenceBackend(StrEnum):
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    HEURISTIC = "HEURISTIC"
    ORACLE = "ORACLE"


class SensorType(StrEnum):
    GYRO = "GYRO"
    ACCEL = "ACCEL"
    ROTATION_VECTOR = "ROTATION_VECTOR"


class PrivacyMode(StrEnum):
    LOCAL_ONLY = "LOCAL_ONLY"
    SHARE_REDACTED = "SHARE_REDACTED"


INFO_LAYER_SEMANTICS = {SemanticType.PUDDLE, SemanticType.GRAVEL}
LOW_RISK_WHEN_NORMAL = {SemanticType.MANHOLE_COVER, SemanticType.REPAIR_PATCH, SemanticType.ROAD_JOINT}
ALERT_FORBIDDEN_PHRASES = (
    "向左避让",
    "向右避让",
    "向左转向",
    "向右转向",
    "刹车",
    "制动",
    "steer left",
    "steer right",
    "brake now",
)

DIRECTION_TTS = {
    Direction.LEFT_FRONT: "左前方",
    Direction.CENTER_FRONT: "正前方",
    Direction.RIGHT_FRONT: "右前方",
    Direction.ACROSS: "正前方",
    Direction.UNKNOWN: "前方",
}

SEMANTIC_TTS = {
    SemanticType.POTHOLE: "坑洼",
    SemanticType.MANHOLE_COVER: "井盖",
    SemanticType.SPEED_BUMP: "减速带",
    SemanticType.ROAD_JOINT: "接缝",
    SemanticType.REPAIR_PATCH: "修补",
    SemanticType.ROUGH_BROKEN: "粗糙路面",
    SemanticType.PUDDLE: "积水",
    SemanticType.GRAVEL: "散落物",
    SemanticType.UNKNOWN_ANOMALY: "路面异常",
}

RIDING_STATUS_COPY = {
    PerceptionStatus.NORMAL: "感知正常",
    PerceptionStatus.DEGRADED_VISIBILITY: "可见性下降",
    PerceptionStatus.SEVERE_BLUR: "画面模糊，短时续接",
    PerceptionStatus.OCCLUDED: "前方道路被遮挡",
    PerceptionStatus.LENS_CONTAMINATION: "请在安全处清洁镜头",
    PerceptionStatus.THERMAL_THROTTLE: "性能降级",
    PerceptionStatus.STORAGE_LOW: "存储空间不足，即将停止录像",
    PerceptionStatus.SAFE_MODE: "安全模式：仅采集",
    PerceptionStatus.PERCEPTION_LIMITED: "感知受限",
}
