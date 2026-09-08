package com.ebike.rpar.config

import android.content.Context
import com.ebike.rpar.model.APP_VERSION
import org.json.JSONObject

data class QualityConfig(
    val laplacianUsable: Double = 18.0,
    val motionBlurBlock: Double = 0.62,
    val glareBlock: Double = 0.45,
    val underexposureBlock: Double = 0.72,
    val overexposureBlock: Double = 0.55,
    val minRoadVisible: Double = 0.18,
    val maxSelectedAgeMs: Double = 100.0,
    val bufferSeconds: Double = 0.5,
    val tileCols: Int = 8,
    val tileRows: Int = 6,
    val lensDropBlobMin: Int = 12,
)

data class TrackingConfig(
    val confirmWindowS: Double = 0.40,
    val minConfirmHits: Int = 3,
    val candidateMaxAgeS: Double = 0.35,
    val lowQualityHoldS: Double = 0.45,
    val passedRemoveS: Double = 0.50,
    val iouMatch: Double = 0.25,
    val centerMatchPx: Double = 80.0,
    val processNoise: Double = 18.0,
    val measNoise: Double = 6.0,
    val unknownAnomalyExtraHits: Int = 2,
)

data class GeometryConfig(
    val corridorHalfWidthM: Double = 0.85,
    val acrossMinWidthM: Double = 1.6,
    val nearBoundaryM: Double = 2.2,
    val farDisplayM: Double = 30.0,
    val distanceMaeNearM: Double = 2.5,
    val distanceMaeFarM: Double = 5.0,
    val minSpeedForTtcMps: Double = 2.0,
    val pitchHealthDeg: Double = 8.0,
    val rollHealthDeg: Double = 10.0,
    val hideDistanceIfInvalid: Boolean = true,
)

data class AlertConfig(
    val enabledDefault: Boolean = true,
    val scoreThreshold: Double = 0.62,
    val globalCooldownS: Double = 2.2,
    val classCooldownS: Double = 4.0,
    val minSeverity: Int = 2,
    val minPathRelevance: Double = 0.45,
    val minVisibility: Double = 0.40,
    val minEffective: Double = 0.48,
    val realertSeverityJump: Int = 1,
    val pauseOnDegraded: Boolean = true,
)

data class RenderConfig(
    val ridingMaxLabels: Int = 5,
    val arTargetFps: Int = 60,
    val arMinFps: Int = 30,
    val candidateAlpha: Float = 0.35f,
    val confirmedAlpha: Float = 0.85f,
    val overlayErrorPx: Double = 8.0,
    val nightBrightness: Double = 0.72,
    val strokeScale: Float = 1f,
    val fontScale: Float = 1f,
    val overlayAlpha: Float = 1f,
    val showInfoLayer: Boolean = true,
)

data class CameraConfig(
    val width: Int = 1920,
    val height: Int = 1080,
    val targetFps: Int = 60,
    val fallbackFps: Int = 30,
    val dynamicRange: String = "SDR",
    val preferPreviewPlusYuvPlusRecord: Boolean = true,
    val segmentSeconds: Int = 300,
    val bitrateMbps: Double = 25.0,
    val lockPhysicalCamera: Boolean = true,
)

data class RuntimeConfig(
    val inferFps: Double = 12.0,
    val qualityFps: Double = 30.0,
    val trackingFps: Double = 30.0,
    val e2eP95Ms: Double = 150.0,
    val thermalMinInferFps: Double = 8.0,
    val memoryTargetMb: Double = 1536.0,
    val startupPreviewS: Double = 5.0,
)

data class ModelPackageRef(
    val packageId: String = "heuristic-cv-0.1.0",
    val engine: String = "heuristic",
    val inputFar: IntArray = intArrayOf(768, 384),
    val inputNear: IntArray = intArrayOf(640, 480),
    val sha256: String = "",
    val compatibleApp: String = ">=0.1.0",
)

data class RparConfig(
    val schemaVersion: String = "1.0",
    val quality: QualityConfig = QualityConfig(),
    val tracking: TrackingConfig = TrackingConfig(),
    val geometry: GeometryConfig = GeometryConfig(),
    val alert: AlertConfig = AlertConfig(),
    val render: RenderConfig = RenderConfig(),
    val camera: CameraConfig = CameraConfig(),
    val runtime: RuntimeConfig = RuntimeConfig(),
    val model: ModelPackageRef = ModelPackageRef(),
    val defaultMountId: String = "left_handlebar_v1",
) {
    fun snapshot(): JSONObject = JSONObject()
        .put("schema_version", schemaVersion)
        .put("default_mount_id", defaultMountId)
        .put("quality", JSONObject()
            .put("laplacian_usable", quality.laplacianUsable)
            .put("motion_blur_block", quality.motionBlurBlock)
            .put("glare_block", quality.glareBlock)
            .put("underexposure_block", quality.underexposureBlock)
            .put("overexposure_block", quality.overexposureBlock)
            .put("min_road_visible", quality.minRoadVisible)
            .put("max_selected_age_ms", quality.maxSelectedAgeMs)
            .put("buffer_seconds", quality.bufferSeconds)
            .put("tile_cols", quality.tileCols)
            .put("tile_rows", quality.tileRows)
            .put("lens_drop_blob_min", quality.lensDropBlobMin))
        .put("tracking", JSONObject()
            .put("confirm_window_s", tracking.confirmWindowS)
            .put("min_confirm_hits", tracking.minConfirmHits)
            .put("candidate_max_age_s", tracking.candidateMaxAgeS)
            .put("low_quality_hold_s", tracking.lowQualityHoldS)
            .put("passed_remove_s", tracking.passedRemoveS)
            .put("iou_match", tracking.iouMatch)
            .put("center_match_px", tracking.centerMatchPx)
            .put("process_noise", tracking.processNoise)
            .put("meas_noise", tracking.measNoise)
            .put("unknown_anomaly_extra_hits", tracking.unknownAnomalyExtraHits))
        .put("geometry", JSONObject()
            .put("corridor_half_width_m", geometry.corridorHalfWidthM)
            .put("across_min_width_m", geometry.acrossMinWidthM)
            .put("near_boundary_m", geometry.nearBoundaryM)
            .put("far_display_m", geometry.farDisplayM)
            .put("min_speed_for_ttc_mps", geometry.minSpeedForTtcMps)
            .put("pitch_health_deg", geometry.pitchHealthDeg)
            .put("roll_health_deg", geometry.rollHealthDeg)
            .put("hide_distance_if_invalid", geometry.hideDistanceIfInvalid))
        .put("alert", JSONObject()
            .put("enabled_default", alert.enabledDefault)
            .put("score_threshold", alert.scoreThreshold)
            .put("global_cooldown_s", alert.globalCooldownS)
            .put("class_cooldown_s", alert.classCooldownS)
            .put("min_severity", alert.minSeverity)
            .put("min_path_relevance", alert.minPathRelevance)
            .put("min_visibility", alert.minVisibility)
            .put("min_effective", alert.minEffective)
            .put("realert_severity_jump", alert.realertSeverityJump)
            .put("pause_on_degraded", alert.pauseOnDegraded))
        .put("render", JSONObject()
            .put("riding_max_labels", render.ridingMaxLabels)
            .put("ar_target_fps", render.arTargetFps)
            .put("ar_min_fps", render.arMinFps)
            .put("candidate_alpha", render.candidateAlpha)
            .put("confirmed_alpha", render.confirmedAlpha)
            .put("overlay_error_px", render.overlayErrorPx)
            .put("night_brightness", render.nightBrightness)
            .put("stroke_scale", render.strokeScale)
            .put("font_scale", render.fontScale)
            .put("overlay_alpha", render.overlayAlpha)
            .put("show_info_layer", render.showInfoLayer))
        .put("camera", JSONObject()
            .put("width", camera.width)
            .put("height", camera.height)
            .put("target_fps", camera.targetFps)
            .put("fallback_fps", camera.fallbackFps)
            .put("dynamic_range", camera.dynamicRange)
            .put("segment_seconds", camera.segmentSeconds)
            .put("bitrate_mbps", camera.bitrateMbps)
            .put("lock_physical_camera", camera.lockPhysicalCamera))
        .put("runtime", JSONObject()
            .put("infer_fps", runtime.inferFps)
            .put("quality_fps", runtime.qualityFps)
            .put("e2e_p95_ms", runtime.e2eP95Ms)
            .put("thermal_min_infer_fps", runtime.thermalMinInferFps))
        .put("model", JSONObject()
            .put("package_id", model.packageId)
            .put("engine", model.engine)
            .put("input_far", org.json.JSONArray().put(model.inputFar[0]).put(model.inputFar[1]))
            .put("input_near", org.json.JSONArray().put(model.inputNear[0]).put(model.inputNear[1]))
            .put("sha256", model.sha256)
            .put("compatible_app", model.compatibleApp))
            .put("app_version", APP_VERSION)

    companion object {
        fun load(context: Context): RparConfig {
            val raw = context.assets.open("rpar.defaults.json").bufferedReader().use { it.readText() }
            return fromJson(JSONObject(raw))
        }

        fun fromJson(root: JSONObject): RparConfig {
            val q = root.optJSONObject("quality") ?: JSONObject()
            val t = root.optJSONObject("tracking") ?: JSONObject()
            val g = root.optJSONObject("geometry") ?: JSONObject()
            val a = root.optJSONObject("alert") ?: JSONObject()
            val r = root.optJSONObject("render") ?: JSONObject()
            val c = root.optJSONObject("camera") ?: JSONObject()
            val rt = root.optJSONObject("runtime") ?: JSONObject()
            val m = root.optJSONObject("model") ?: JSONObject()
            val far = m.optJSONArray("input_far")
            val near = m.optJSONArray("input_near")
            return RparConfig(
                schemaVersion = root.optString("schema_version", "1.0"),
                defaultMountId = root.optString("default_mount_id", "left_handlebar_v1"),
                quality = QualityConfig(
                    laplacianUsable = q.optDouble("laplacian_usable", 18.0),
                    motionBlurBlock = q.optDouble("motion_blur_block", 0.62),
                    glareBlock = q.optDouble("glare_block", 0.45),
                    underexposureBlock = q.optDouble("underexposure_block", 0.72),
                    overexposureBlock = q.optDouble("overexposure_block", 0.55),
                    minRoadVisible = q.optDouble("min_road_visible", 0.18),
                    maxSelectedAgeMs = q.optDouble("max_selected_age_ms", 100.0),
                    bufferSeconds = q.optDouble("buffer_seconds", 0.5),
                    tileCols = q.optInt("tile_cols", 8),
                    tileRows = q.optInt("tile_rows", 6),
                    lensDropBlobMin = q.optInt("lens_drop_blob_min", 12),
                ),
                tracking = TrackingConfig(
                    confirmWindowS = t.optDouble("confirm_window_s", 0.40),
                    minConfirmHits = t.optInt("min_confirm_hits", 3),
                    candidateMaxAgeS = t.optDouble("candidate_max_age_s", 0.35),
                    lowQualityHoldS = t.optDouble("low_quality_hold_s", 0.45),
                    passedRemoveS = t.optDouble("passed_remove_s", 0.50),
                    iouMatch = t.optDouble("iou_match", 0.25),
                    centerMatchPx = t.optDouble("center_match_px", 80.0),
                    processNoise = t.optDouble("process_noise", 18.0),
                    measNoise = t.optDouble("meas_noise", 6.0),
                    unknownAnomalyExtraHits = t.optInt("unknown_anomaly_extra_hits", 2),
                ),
                geometry = GeometryConfig(
                    corridorHalfWidthM = g.optDouble("corridor_half_width_m", 0.85),
                    acrossMinWidthM = g.optDouble("across_min_width_m", 1.6),
                    nearBoundaryM = g.optDouble("near_boundary_m", 2.2),
                    farDisplayM = g.optDouble("far_display_m", 30.0),
                    minSpeedForTtcMps = g.optDouble("min_speed_for_ttc_mps", 2.0),
                    pitchHealthDeg = g.optDouble("pitch_health_deg", 8.0),
                    rollHealthDeg = g.optDouble("roll_health_deg", 10.0),
                    hideDistanceIfInvalid = g.optBoolean("hide_distance_if_invalid", true),
                ),
                alert = AlertConfig(
                    enabledDefault = a.optBoolean("enabled_default", true),
                    scoreThreshold = a.optDouble("score_threshold", 0.62),
                    globalCooldownS = a.optDouble("global_cooldown_s", 2.2),
                    classCooldownS = a.optDouble("class_cooldown_s", 4.0),
                    minSeverity = a.optInt("min_severity", 2),
                    minPathRelevance = a.optDouble("min_path_relevance", 0.45),
                    minVisibility = a.optDouble("min_visibility", 0.40),
                    minEffective = a.optDouble("min_effective", 0.48),
                    realertSeverityJump = a.optInt("realert_severity_jump", 1),
                    pauseOnDegraded = a.optBoolean("pause_on_degraded", true),
                ),
                render = RenderConfig(
                    ridingMaxLabels = r.optInt("riding_max_labels", 5),
                    arTargetFps = r.optInt("ar_target_fps", 60),
                    arMinFps = r.optInt("ar_min_fps", 30),
                    candidateAlpha = r.optDouble("candidate_alpha", 0.35).toFloat(),
                    confirmedAlpha = r.optDouble("confirmed_alpha", 0.85).toFloat(),
                    overlayErrorPx = r.optDouble("overlay_error_px", 8.0),
                    nightBrightness = r.optDouble("night_brightness", 0.72),
                    strokeScale = r.optDouble("stroke_scale", 1.0).toFloat(),
                    fontScale = r.optDouble("font_scale", 1.0).toFloat(),
                    overlayAlpha = r.optDouble("overlay_alpha", 1.0).toFloat(),
                    showInfoLayer = r.optBoolean("show_info_layer", true),
                ),
                camera = CameraConfig(
                    width = c.optInt("width", 1920),
                    height = c.optInt("height", 1080),
                    targetFps = c.optInt("target_fps", 60),
                    fallbackFps = c.optInt("fallback_fps", 30),
                    dynamicRange = c.optString("dynamic_range", "SDR"),
                    preferPreviewPlusYuvPlusRecord = c.optBoolean("prefer_preview_plus_yuv_plus_record", true),
                    segmentSeconds = c.optInt("segment_seconds", 300),
                    bitrateMbps = c.optDouble("bitrate_mbps", 25.0),
                    lockPhysicalCamera = c.optBoolean("lock_physical_camera", true),
                ),
                runtime = RuntimeConfig(
                    inferFps = rt.optDouble("infer_fps", 12.0),
                    qualityFps = rt.optDouble("quality_fps", 30.0),
                    trackingFps = rt.optDouble("tracking_fps", 30.0),
                    e2eP95Ms = rt.optDouble("e2e_p95_ms", 150.0),
                    thermalMinInferFps = rt.optDouble("thermal_min_infer_fps", 8.0),
                    memoryTargetMb = rt.optDouble("memory_target_mb", 1536.0),
                    startupPreviewS = rt.optDouble("startup_preview_s", 5.0),
                ),
                model = ModelPackageRef(
                    packageId = m.optString("package_id", "heuristic-cv-0.1.0"),
                    engine = m.optString("engine", "heuristic"),
                    inputFar = intArrayOf(far?.optInt(0, 768) ?: 768, far?.optInt(1, 384) ?: 384),
                    inputNear = intArrayOf(near?.optInt(0, 640) ?: 640, near?.optInt(1, 480) ?: 480),
                    sha256 = m.optString("sha256", ""),
                    compatibleApp = m.optString("compatible_app", ">=0.1.0"),
                ),
            )
        }
    }
}
