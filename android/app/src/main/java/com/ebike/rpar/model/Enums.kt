package com.ebike.rpar.model

enum class SemanticType(val wire: String) {
    POTHOLE("pothole"),
    MANHOLE_COVER("manhole_cover"),
    SPEED_BUMP("speed_bump"),
    ROAD_JOINT("road_joint"),
    REPAIR_PATCH("repair_patch"),
    ROUGH_BROKEN("rough_broken"),
    PUDDLE("puddle"),
    GRAVEL("gravel"),
    UNKNOWN_ANOMALY("unknown_anomaly");

    companion object {
        fun fromWire(v: String): SemanticType =
            entries.firstOrNull { it.wire == v } ?: UNKNOWN_ANOMALY
    }
}

enum class GeometryType(val wire: String) {
    CONCAVE("concave"),
    CONVEX("convex"),
    ROUGH("rough"),
    STEP("step"),
    FLAT("flat"),
    UNKNOWN("unknown");

    companion object {
        fun fromWire(v: String): GeometryType =
            entries.firstOrNull { it.wire == v } ?: UNKNOWN
    }
}

enum class ObjectState(val wire: String) {
    NORMAL("normal"),
    ABNORMAL("abnormal"),
    UNKNOWN("unknown");
}

enum class Severity(val code: Int) {
    NONE(0),
    LIGHT(1),
    MEDIUM(2),
    HEAVY(3),
    UNKNOWN(-1);

    companion object {
        fun fromCode(v: Int): Severity = entries.firstOrNull { it.code == v } ?: UNKNOWN
    }
}

enum class VisibilityClass(val wire: String) {
    CLEAR("clear"),
    BLUR("blur"),
    UNDEREXPOSED("underexposed"),
    OVEREXPOSED("overexposed"),
    GLARE("glare"),
    OCCLUDED("occluded"),
    LENS_DROP("lens_drop"),
    UNKNOWN("unknown");
}

enum class LifecycleState(val wire: String) {
    CANDIDATE("CANDIDATE"),
    TRACKED("TRACKED"),
    CONFIRMED("CONFIRMED"),
    ALERTED("ALERTED"),
    PASSED("PASSED"),
    EXPIRED("EXPIRED");
}

enum class Direction(val wire: String) {
    LEFT_FRONT("LEFT_FRONT"),
    CENTER_FRONT("CENTER_FRONT"),
    RIGHT_FRONT("RIGHT_FRONT"),
    ACROSS("ACROSS"),
    UNKNOWN("UNKNOWN");
}

enum class PerceptionStatus(val wire: String) {
    NORMAL("NORMAL"),
    DEGRADED_VISIBILITY("DEGRADED_VISIBILITY"),
    SEVERE_BLUR("SEVERE_BLUR"),
    OCCLUDED("OCCLUDED"),
    LENS_CONTAMINATION("LENS_CONTAMINATION"),
    THERMAL_THROTTLE("THERMAL_THROTTLE"),
    STORAGE_LOW("STORAGE_LOW"),
    SAFE_MODE("SAFE_MODE"),
    PERCEPTION_LIMITED("PERCEPTION_LIMITED");
}

enum class RunMode(val wire: String) {
    CAPTURE_ONLY("CAPTURE_ONLY"),
    REALTIME_PERCEPTION("REALTIME_PERCEPTION"),
    REALTIME_PERCEPTION_FULL_LOG("REALTIME_PERCEPTION_FULL_LOG"),
    SAFE_MODE("SAFE_MODE"),
    REPLAY("REPLAY");
}

enum class UiMode(val wire: String) {
    RIDING("RIDING"),
    RESEARCH("RESEARCH");
}

enum class StabilizationMode(val wire: String) {
    OFF("OFF"),
    STANDARD("STANDARD"),
    PREVIEW("PREVIEW"),
    UNAVAILABLE("UNAVAILABLE");
}

enum class InferenceBackend(val wire: String) {
    CPU("CPU"),
    GPU("GPU"),
    NPU("NPU"),
    HEURISTIC("HEURISTIC"),
    ORACLE("ORACLE");
}

enum class SensorType(val wire: String) {
    GYRO("GYRO"),
    ACCEL("ACCEL"),
    ROTATION_VECTOR("ROTATION_VECTOR");
}

enum class PrivacyMode(val wire: String) {
    LOCAL_ONLY("LOCAL_ONLY"),
    SHARE_REDACTED("SHARE_REDACTED");
}

object EnumCopy {
    val INFO_LAYER = setOf(SemanticType.PUDDLE, SemanticType.GRAVEL)
    val LOW_RISK_WHEN_NORMAL = setOf(
        SemanticType.MANHOLE_COVER,
        SemanticType.REPAIR_PATCH,
        SemanticType.ROAD_JOINT,
        SemanticType.ROUGH_BROKEN,
    )
    val ALERT_FORBIDDEN = listOf(
        "向左避让", "向右避让", "向左转向", "向右转向", "刹车", "制动",
        "steer left", "steer right", "brake now",
    )
    val DIRECTION_TTS = mapOf(
        Direction.LEFT_FRONT to "左前方",
        Direction.CENTER_FRONT to "正前方",
        Direction.RIGHT_FRONT to "右前方",
        Direction.ACROSS to "正前方",
        Direction.UNKNOWN to "前方",
    )
    val SEMANTIC_TTS = mapOf(
        SemanticType.POTHOLE to "大坑",
        SemanticType.MANHOLE_COVER to "井盖",
        SemanticType.SPEED_BUMP to "减速带",
        SemanticType.ROAD_JOINT to "接缝",
        SemanticType.REPAIR_PATCH to "修补",
        SemanticType.ROUGH_BROKEN to "粗糙路面",
        SemanticType.PUDDLE to "积水",
        SemanticType.GRAVEL to "散落物",
        SemanticType.UNKNOWN_ANOMALY to "路面异常",
    )
    val RIDING_STATUS = mapOf(
        PerceptionStatus.NORMAL to "感知正常",
        PerceptionStatus.DEGRADED_VISIBILITY to "可见性下降",
        PerceptionStatus.SEVERE_BLUR to "画面模糊，短时续接",
        PerceptionStatus.OCCLUDED to "前方道路被遮挡",
        PerceptionStatus.LENS_CONTAMINATION to "请在安全处清洁镜头",
        PerceptionStatus.THERMAL_THROTTLE to "性能降级",
        PerceptionStatus.STORAGE_LOW to "存储空间不足，即将停止录像",
        PerceptionStatus.SAFE_MODE to "安全模式：仅采集",
        PerceptionStatus.PERCEPTION_LIMITED to "感知受限",
    )

    fun isBumpHazard(semantic: SemanticType, geometry: GeometryType, state: ObjectState): Boolean {
        if (semantic == SemanticType.POTHOLE && geometry == GeometryType.CONCAVE && state != ObjectState.NORMAL) return true
        if (semantic == SemanticType.SPEED_BUMP && state != ObjectState.NORMAL) return true
        if (semantic == SemanticType.MANHOLE_COVER && geometry == GeometryType.CONCAVE && state == ObjectState.ABNORMAL) return true
        return false
    }

    fun bumpKind(semantic: SemanticType, geometry: GeometryType, state: ObjectState): String {
        if (semantic == SemanticType.POTHOLE) return "大坑"
        if (semantic == SemanticType.SPEED_BUMP) return "减速带"
        if (semantic == SemanticType.MANHOLE_COVER && geometry == GeometryType.CONCAVE && state == ObjectState.ABNORMAL) return "下沉井盖"
        return SEMANTIC_TTS[semantic] ?: "路面异常"
    }
}
