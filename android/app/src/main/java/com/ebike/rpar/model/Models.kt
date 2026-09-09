package com.ebike.rpar.model

import android.graphics.Bitmap
import org.json.JSONArray
import org.json.JSONObject

const val SCHEMA_VERSION = "1.0"
const val APP_VERSION = "0.1.0"

data class Intrinsics(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val width: Int,
    val height: Int,
    val available: Boolean = true,
)

data class MountProfile(
    val profileId: String,
    val name: String,
    val cameraId: String,
    val landscape: Boolean,
    val cameraHeightM: Double,
    val pitchDeg: Double,
    val rollDeg: Double,
    val yawDeg: Double,
    val lateralOffsetM: Double,
    val handlebarNeutralYawDeg: Double,
    val nearReferenceM: Double,
    val horizonYPx: Double,
    val vehicleCenterlineXPx: Double,
    val knownDistance5mPx: Double? = null,
    val knownDistance10mPx: Double? = null,
    val knownDistance20mPx: Double? = null,
    val headlightMean: Double? = null,
    val headlightValid: Boolean = false,
    val calibrationHash: String = "",
    val valid: Boolean = true,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("profile_id", profileId)
        .put("name", name)
        .put("camera_id", cameraId)
        .put("landscape", landscape)
        .put("camera_height_m", cameraHeightM)
        .put("pitch_deg", pitchDeg)
        .put("roll_deg", rollDeg)
        .put("yaw_deg", yawDeg)
        .put("lateral_offset_m", lateralOffsetM)
        .put("handlebar_neutral_yaw_deg", handlebarNeutralYawDeg)
        .put("near_reference_m", nearReferenceM)
        .put("horizon_y_px", horizonYPx)
        .put("vehicle_centerline_x_px", vehicleCenterlineXPx)
        .put("known_distance_5m_px", knownDistance5mPx)
        .put("known_distance_10m_px", knownDistance10mPx)
        .put("known_distance_20m_px", knownDistance20mPx)
        .put("headlight_mean", headlightMean)
        .put("headlight_valid", headlightValid)
        .put("calibration_hash", calibrationHash)
        .put("valid", valid)

    companion object {
        fun fromJson(o: JSONObject): MountProfile = MountProfile(
            profileId = o.optString("profile_id"),
            name = o.optString("name"),
            cameraId = o.optString("camera_id", "rear_main"),
            landscape = o.optBoolean("landscape", true),
            cameraHeightM = o.optDouble("camera_height_m", 1.12),
            pitchDeg = o.optDouble("pitch_deg", 18.0),
            rollDeg = o.optDouble("roll_deg", 0.0),
            yawDeg = o.optDouble("yaw_deg", 0.0),
            lateralOffsetM = o.optDouble("lateral_offset_m", -0.32),
            handlebarNeutralYawDeg = o.optDouble("handlebar_neutral_yaw_deg", 0.0),
            nearReferenceM = o.optDouble("near_reference_m", 3.0),
            horizonYPx = o.optDouble("horizon_y_px", 400.0),
            vehicleCenterlineXPx = o.optDouble("vehicle_centerline_x_px", 960.0),
            knownDistance5mPx = o.optDoubleOrNull("known_distance_5m_px"),
            knownDistance10mPx = o.optDoubleOrNull("known_distance_10m_px"),
            knownDistance20mPx = o.optDoubleOrNull("known_distance_20m_px"),
            headlightMean = o.optDoubleOrNull("headlight_mean"),
            headlightValid = o.optBoolean("headlight_valid", false),
            calibrationHash = o.optString("calibration_hash", ""),
            valid = o.optBoolean("valid", true),
        )
    }
}

data class PoseSample(
    val timestampNs: Long,
    val quaternionXyzw: DoubleArray,
    val gravityXyz: DoubleArray,
    val poseConfidence: Double,
)

data class ImuSample(
    val timestampNs: Long,
    val sensorType: SensorType,
    val x: Double,
    val y: Double,
    val z: Double,
    val accuracy: Int,
    val sourceRateHz: Double,
) {
    fun toJson(): JSONObject {
        val o = JSONObject()
        o.put("timestamp_ns", timestampNs)
        o.put("sensor_type", sensorType.wire)
        o.put("x", x)
        o.put("y", y)
        o.put("z", z)
        o.put("accuracy", accuracy)
        o.put("source_rate_hz", sourceRateHz)
        return o
    }
}

data class LocationSample(
    val timestampNs: Long,
    val latitude: Double?,
    val longitude: Double?,
    val altitude: Double?,
    val speedMps: Double?,
    val bearingDeg: Double?,
    val horizontalAccuracyM: Double?,
    val speedAccuracyMps: Double?,
    val interpolated: Boolean = false,
) {
    fun toJson(includePrecise: Boolean): JSONObject {
        val o = JSONObject()
        o.put("timestamp_ns", timestampNs)
        o.put("speed_mps", speedMps)
        o.put("bearing_deg", bearingDeg)
        o.put("horizontal_accuracy_m", horizontalAccuracyM)
        o.put("speed_accuracy_mps", speedAccuracyMps)
        o.put("interpolated", interpolated)
        if (includePrecise) {
            o.put("latitude", latitude)
            o.put("longitude", longitude)
            o.put("altitude", altitude)
        }
        return o
    }
}

data class FrameMeta(
    val frameId: Long,
    val sensorTimestampNs: Long,
    val imageTimestampNs: Long,
    val exposureTimeNs: Long?,
    val iso: Int?,
    val focalLengthMm: Double?,
    val focusDistanceDiopters: Double?,
    val afState: String?,
    val aeState: String?,
    val awbState: String?,
    val cropRegion: IntArray?,
    val stabilizationMode: StabilizationMode?,
    val width: Int,
    val height: Int,
    val availability: Map<String, Boolean> = emptyMap(),
) {
    fun toJson(): JSONObject {
        val avail = JSONObject()
        availability.forEach { (k, v) -> avail.put(k, v) }
        val crop = if (cropRegion == null) JSONObject.NULL else JSONArray().apply {
            cropRegion.forEach { put(it) }
        }
        return JSONObject()
            .put("frame_id", frameId)
            .put("sensor_timestamp_ns", sensorTimestampNs)
            .put("image_timestamp_ns", imageTimestampNs)
            .put("exposure_time_ns", exposureTimeNs)
            .put("iso", iso)
            .put("focal_length_mm", focalLengthMm)
            .put("focus_distance_diopters", focusDistanceDiopters)
            .put("af_state", afState)
            .put("ae_state", aeState)
            .put("awb_state", awbState)
            .put("crop_region", crop)
            .put("stabilization_mode", stabilizationMode?.wire)
            .put("width", width)
            .put("height", height)
            .put("availability", avail)
    }
}

data class FrameQuality(
    val sharpness: Double,
    val motionBlur: Double,
    val defocus: Double,
    val underexposure: Double,
    val overexposure: Double,
    val glare: Double,
    var usable: Boolean,
    var visibilityClass: VisibilityClass,
    val roadVisibleRatio: Double,
    var reason: String = "",
)

data class QualityTile(
    val x0: Int,
    val y0: Int,
    val x1: Int,
    val y1: Int,
    val visibility: VisibilityClass,
    val score: Double,
)

data class FrameQualityMap(
    val globalQuality: FrameQuality,
    val tiles: List<QualityTile>,
    val occupancyOccludedRatio: Double,
    var selectedForInfer: Boolean,
    var selectedAgeMs: Double,
    var degradeReason: String? = null,
)

data class MaskRle(
    val width: Int,
    val height: Int,
    val counts: List<Int>,
    val encoding: String = "rle_cocoa",
) {
    fun toJson(): JSONObject {
        val c = JSONArray()
        counts.forEach { c.put(it) }
        val o = JSONObject()
        o.put("width", width)
        o.put("height", height)
        o.put("counts", c)
        o.put("encoding", encoding)
        return o
    }
}

data class RoadObservation(
    val timestampNs: Long,
    val sourceFrameId: Long,
    val semanticType: SemanticType,
    val geometryType: GeometryType,
    val state: ObjectState,
    val severity: Severity,
    val maskRle: MaskRle?,
    val polygon: List<Pair<Float, Float>>,
    val bbox: FloatArray,
    val modelConfidence: Double,
    val qualityAtMask: Double,
    val visibility: VisibilityClass,
    val calibratedConfidence: Double? = null,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("timestamp_ns", timestampNs)
        .put("source_frame_id", sourceFrameId)
        .put("semantic_type", semanticType.wire)
        .put("geometry_type", geometryType.wire)
        .put("state", state.wire)
        .put("severity", severity.code)
        .put("polygon", polygonToJson(polygon))
        .put("bbox", JSONArray().put(bbox[0]).put(bbox[1]).put(bbox[2]).put(bbox[3]))
        .put("model_confidence", modelConfidence)
        .put("quality_at_mask", qualityAtMask)
        .put("visibility", visibility.wire)
        .put("calibrated_confidence", calibratedConfidence)
        .put("mask_rle", maskRle?.toJson() ?: JSONObject.NULL)
}

data class PerceptionResult(
    val timestampNs: Long,
    val sourceFrameId: Long,
    val roadPolygon: List<Pair<Float, Float>>,
    val occludedPolygons: List<List<Pair<Float, Float>>>,
    val observations: List<RoadObservation>,
    val backend: InferenceBackend,
    val latencyMs: Double,
    val inputSizes: List<IntArray>,
    val dualScale: Boolean = true,
)

data class TrackedRoadObject(
    val schemaVersion: String,
    val trackId: Int,
    val timestampNs: Long,
    var lifecycleState: LifecycleState,
    val semanticType: SemanticType,
    val geometryType: GeometryType,
    val objectState: ObjectState,
    val severity: Severity,
    val direction: Direction,
    val distanceM: Double?,
    val distanceConfidence: Double,
    val distanceValid: Boolean,
    val ttcS: Double?,
    val modelConfidence: Double,
    val visibilityConfidence: Double,
    val temporalConfidence: Double,
    val geometryConsistency: Double,
    val effectiveConfidence: Double,
    val pathRelevance: Double,
    val riskScore: Double,
    var alertScore: Double,
    val polygon: List<Pair<Float, Float>>,
    val bbox: FloatArray,
    val maskRle: MaskRle?,
    val sourceFrameId: Long,
    val mountProfileId: String,
    val modelVersion: String,
    val visualStyle: String,
    var labelRank: Int? = null,
    val roadXyM: Pair<Double, Double>? = null,
    val depthConfidence: Double = 0.0,
    val impactScore: Double? = null,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("schema_version", schemaVersion)
        .put("track_id", trackId)
        .put("timestamp_ns", timestampNs)
        .put("lifecycle_state", lifecycleState.wire)
        .put("semantic_type", semanticType.wire)
        .put("geometry_type", geometryType.wire)
        .put("object_state", objectState.wire)
        .put("severity", severity.code)
        .put("direction", direction.wire)
        .put("distance_m", distanceM)
        .put("distance_confidence", distanceConfidence)
        .put("distance_valid", distanceValid)
        .put("ttc_s", ttcS)
        .put("model_confidence", modelConfidence)
        .put("visibility_confidence", visibilityConfidence)
        .put("temporal_confidence", temporalConfidence)
        .put("geometry_consistency", geometryConsistency)
        .put("effective_confidence", effectiveConfidence)
        .put("path_relevance", pathRelevance)
        .put("risk_score", riskScore)
        .put("alert_score", alertScore)
        .put("polygon", polygonToJson(polygon))
        .put("bbox", JSONArray().put(bbox[0]).put(bbox[1]).put(bbox[2]).put(bbox[3]))
        .put("source_frame_id", sourceFrameId)
        .put("mount_profile_id", mountProfileId)
        .put("model_version", modelVersion)
        .put("visual_style", visualStyle)
        .put("label_rank", labelRank)
        .put("mask_rle", maskRle?.toJson() ?: JSONObject.NULL)
        .put("depth_confidence", depthConfidence)
        .put("impact_score", impactScore)
}

data class AlertDecision(
    val timestampNs: Long,
    val trackId: Int,
    val fired: Boolean,
    val phrase: String,
    val direction: Direction,
    val semanticType: SemanticType,
    val alertScore: Double,
    val threshold: Double,
    val reasons: List<String>,
    val snapshot: JSONObject,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("timestamp_ns", timestampNs)
        .put("track_id", trackId)
        .put("fired", fired)
        .put("phrase", phrase)
        .put("direction", direction.wire)
        .put("semantic_type", semanticType.wire)
        .put("alert_score", alertScore)
        .put("threshold", threshold)
        .put("reasons", JSONArray(reasons))
        .put("snapshot", snapshot)
}

data class RenderPrimitive(
    val trackId: Int,
    val polygon: List<Pair<Float, Float>>,
    val colorRgba: FloatArray,
    val dashed: Boolean,
    val thickness: Float,
    val label: String?,
    val labelPriority: Int,
    val fade: Float,
    val kind: String,
)

data class PerceptionView(
    val timestampNs: Long,
    val status: PerceptionStatus,
    val statusCopy: String,
    val tracks: List<TrackedRoadObject>,
    val primitives: List<RenderPrimitive>,
    val alerts: List<AlertDecision>,
    val quality: FrameQualityMap?,
    val speedKmh: Double?,
    val backend: InferenceBackend,
    val modelVersion: String,
    val inferFps: Double,
    val arFps: Double,
    val latencyP95Ms: Double,
    val queueDepth: Int,
    val thermalC: Double?,
    val blur: Double,
    val glare: Double,
    val recSeconds: Double,
    val dualScale: Boolean,
    val latencyP50Ms: Double = 0.0,
    val droppedInfer: Int = 0,
    val inputFar: IntArray = intArrayOf(768, 384),
    val inputNear: IntArray = intArrayOf(640, 480),
    val roadPolygon: List<Pair<Float, Float>> = emptyList(),
    val occludedPolygons: List<List<Pair<Float, Float>>> = emptyList(),
)

data class YuvImageBuffer(
    val width: Int,
    val height: Int,
    val y: ByteArray,
    val yRowStride: Int,
    val uv: ByteArray? = null,
    val u: ByteArray? = null,
    val v: ByteArray? = null,
    val uRowStride: Int = 0,
    val vRowStride: Int = 0,
    val uPixelStride: Int = 1,
    val vPixelStride: Int = 1,
) {
    fun grayAt(x: Int, yPos: Int): Int {
        val xx = x.coerceIn(0, width - 1)
        val yy = yPos.coerceIn(0, height - 1)
        return y[yy * yRowStride + xx].toInt() and 0xFF
    }

    fun chromaU(x: Int, yPos: Int): Int = chromaAt(u, uRowStride, uPixelStride, x, yPos)

    fun chromaV(x: Int, yPos: Int): Int = chromaAt(v, vRowStride, vPixelStride, x, yPos)

    private fun chromaAt(plane: ByteArray?, rowStride: Int, pixelStride: Int, x: Int, yPos: Int): Int {
        if (plane == null || plane.isEmpty() || rowStride <= 0) return 128
        val cx = (x / 2).coerceAtLeast(0)
        val cy = (yPos / 2).coerceAtLeast(0)
        val idx = cy * rowStride + cx * pixelStride.coerceAtLeast(1)
        if (idx < 0 || idx >= plane.size) return 128
        return plane[idx].toInt() and 0xFF
    }
}

data class SynchronizedFrame(
    val meta: FrameMeta,
    val bitmap: Bitmap?,
    val yuv: YuvImageBuffer?,
    val pose: PoseSample?,
    val angularVelocity: DoubleArray?,
    val linearAccel: DoubleArray?,
    val location: LocationSample?,
    val speedMps: Double?,
)

fun polygonToJson(poly: List<Pair<Float, Float>>): JSONArray {
    val a = JSONArray()
    poly.forEach { (x, y) -> a.put(JSONArray().put(x).put(y)) }
    return a
}

fun JSONObject.optDoubleOrNull(key: String): Double? =
    if (has(key) && !isNull(key)) optDouble(key) else null

fun bboxOf(poly: List<Pair<Float, Float>>): FloatArray {
    if (poly.isEmpty()) return floatArrayOf(0f, 0f, 0f, 0f)
    var x0 = Float.POSITIVE_INFINITY
    var y0 = Float.POSITIVE_INFINITY
    var x1 = Float.NEGATIVE_INFINITY
    var y1 = Float.NEGATIVE_INFINITY
    poly.forEach { (x, y) ->
        x0 = minOf(x0, x); y0 = minOf(y0, y)
        x1 = maxOf(x1, x); y1 = maxOf(y1, y)
    }
    return floatArrayOf(x0, y0, x1, y1)
}

fun bboxIou(a: FloatArray, b: FloatArray): Double {
    val ix0 = maxOf(a[0], b[0]); val iy0 = maxOf(a[1], b[1])
    val ix1 = minOf(a[2], b[2]); val iy1 = minOf(a[3], b[3])
    val iw = maxOf(0f, ix1 - ix0); val ih = maxOf(0f, iy1 - iy0)
    val inter = iw * ih
    if (inter <= 0f) return 0.0
    val areaA = maxOf(0f, a[2] - a[0]) * maxOf(0f, a[3] - a[1])
    val areaB = maxOf(0f, b[2] - b[0]) * maxOf(0f, b[3] - b[1])
    val union = areaA + areaB - inter
    return if (union > 0f) inter / union.toDouble() else 0.0
}

fun polygonCentroid(poly: List<Pair<Float, Float>>): Pair<Float, Float> {
    if (poly.isEmpty()) return 0f to 0f
    return poly.map { it.first }.average().toFloat() to poly.map { it.second }.average().toFloat()
}

fun groundContact(poly: List<Pair<Float, Float>>): Pair<Float, Float> {
    if (poly.isEmpty()) return 0f to 0f
    val yMax = poly.maxOf { it.second }
    val band = poly.filter { it.second >= yMax - 6f }
    val pts = band.ifEmpty { poly }
    return pts.map { it.first }.average().toFloat() to pts.map { it.second }.average().toFloat()
}

fun ellipsePolygon(cx: Float, cy: Float, rx: Float, ry: Float, n: Int = 18): List<Pair<Float, Float>> {
    return (0 until n).map { i ->
        val a = (i * 2.0 * Math.PI / n)
        (cx + rx * Math.cos(a).toFloat()) to (cy + ry * Math.sin(a).toFloat())
    }
}

fun rectPolygon(x0: Float, y0: Float, x1: Float, y1: Float) = listOf(
    x0 to y0, x1 to y0, x1 to y1, x0 to y1,
)

fun nmsPolygons(items: List<Pair<List<Pair<Float, Float>>, Double>>, iouThr: Double = 0.4): List<Int> {
    val order = items.indices.sortedByDescending { items[it].second }
    val suppressed = BooleanArray(items.size)
    val bboxes = items.map { bboxOf(it.first) }
    val keep = mutableListOf<Int>()
    for (i in order) {
        if (suppressed[i]) continue
        keep += i
        for (j in order) {
            if (suppressed[j] || j == i) continue
            if (bboxIou(bboxes[i], bboxes[j]) >= iouThr) suppressed[j] = true
        }
    }
    return keep
}

fun pointInPolygon(x: Float, y: Float, poly: List<Pair<Float, Float>>): Boolean {
    var inside = false
    var j = poly.lastIndex
    for (i in poly.indices) {
        val yi = poly[i].second
        val yj = poly[j].second
        val xi = poly[i].first
        val xj = poly[j].first
        val hit = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-6f) + xi)
        if (hit) inside = !inside
        j = i
    }
    return inside
}

fun rleEncode(mask: BooleanArray): List<Int> {
    val counts = ArrayList<Int>()
    var prev = false
    var run = 0
    for (v in mask) {
        if (v == prev) run++
        else {
            counts += run
            run = 1
            prev = v
        }
    }
    counts += run
    return counts
}

fun rleFromPolygon(poly: List<Pair<Float, Float>>, maxSide: Int = 160): MaskRle? {
    if (poly.size < 3) return null
    val bb = bboxOf(poly)
    var bw = maxOf(1, kotlin.math.ceil(bb[2] - bb[0]).toInt())
    var bh = maxOf(1, kotlin.math.ceil(bb[3] - bb[1]).toInt())
    var scale = 1f
    val longSide = maxOf(bw, bh)
    if (longSide > maxSide) {
        scale = maxSide / longSide.toFloat()
        bw = maxOf(1, kotlin.math.round(bw * scale).toInt())
        bh = maxOf(1, kotlin.math.round(bh * scale).toInt())
    }
    val mask = BooleanArray(bw * bh)
    val x0 = bb[0]
    val y0 = bb[1]
    for (y in 0 until bh) for (x in 0 until bw) {
        val px = x0 + (x + 0.5f) / scale
        val py = y0 + (y + 0.5f) / scale
        mask[x * bh + y] = pointInPolygon(px, py, poly) // Fortran-order like COCO
    }
    return MaskRle(bw, bh, rleEncode(mask))
}

object LocationInterp {
    fun lerp(a: LocationSample, b: LocationSample, tNs: Long): LocationSample {
        val span = maxOf(1L, b.timestampNs - a.timestampNs)
        val u = ((tNs - a.timestampNs).toDouble() / span).coerceIn(0.0, 1.0)
        fun mix(x: Double?, y: Double?): Double? {
            if (x == null) return y
            if (y == null) return x
            return x + (y - x) * u
        }
        var br: Double? = null
        if (a.bearingDeg != null && b.bearingDeg != null) {
            val d = ((b.bearingDeg - a.bearingDeg + 540.0) % 360.0) - 180.0
            br = (a.bearingDeg + d * u) % 360.0
        } else {
            br = a.bearingDeg ?: b.bearingDeg
        }
        val gapS = span / 1e9
        val acc = (mix(a.speedAccuracyMps, b.speedAccuracyMps) ?: 0.4) * (1.0 + maxOf(0.0, gapS * 2.0))
        return LocationSample(
            timestampNs = tNs,
            latitude = mix(a.latitude, b.latitude),
            longitude = mix(a.longitude, b.longitude),
            altitude = mix(a.altitude, b.altitude),
            speedMps = mix(a.speedMps, b.speedMps),
            bearingDeg = br,
            horizontalAccuracyM = mix(a.horizontalAccuracyM, b.horizontalAccuracyM),
            speedAccuracyMps = acc,
            interpolated = true,
        )
    }

    fun at(samples: List<LocationSample>, tNs: Long): LocationSample? {
        if (samples.isEmpty()) return null
        val s = samples.sortedBy { it.timestampNs }
        if (tNs <= s.first().timestampNs) return s.first()
        if (tNs >= s.last().timestampNs) return s.last()
        for (i in 0 until s.lastIndex) {
            if (s[i + 1].timestampNs >= tNs) return lerp(s[i], s[i + 1], tNs)
        }
        return s.last()
    }
}
