package com.ebike.rpar.camera

import android.graphics.Matrix
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.MeteringRectangle
import android.util.Size
import com.ebike.rpar.model.StabilizationMode

/**
 * Shared transform chain from sensor array → crop → buffer → preview view.
 * Used by AR overlay so geometry stays consistent across preview / YUV / record.
 */
data class CameraTransformChain(
    val cameraId: String,
    val sensorArray: Rect?,
    val cropRegion: Rect?,
    val bufferSize: Size,
    val previewSize: Size,
    val sensorToBuffer: Matrix,
    val bufferToView: Matrix,
) {
    fun sensorToView(): Matrix {
        val out = Matrix(sensorToBuffer)
        out.postConcat(bufferToView)
        return out
    }

    companion object {
        fun identity(w: Int, h: Int) = CameraTransformChain(
            cameraId = "none",
            sensorArray = Rect(0, 0, w, h),
            cropRegion = Rect(0, 0, w, h),
            bufferSize = Size(w, h),
            previewSize = Size(w, h),
            sensorToBuffer = Matrix(),
            bufferToView = Matrix(),
        )
    }
}

data class CaptureAvailability(
    val sensorTimestamp: Boolean,
    val exposure: Boolean,
    val iso: Boolean,
    val focal: Boolean,
    val focusDistance: Boolean,
    val af: Boolean,
    val ae: Boolean,
    val awb: Boolean,
    val crop: Boolean,
    val ois: Boolean,
    val eis: Boolean,
    val previewStab: Boolean,
) {
    fun toMap() = mapOf(
        "sensor_timestamp" to sensorTimestamp,
        "exposure" to exposure,
        "iso" to iso,
        "focal" to focal,
        "focus_distance" to focusDistance,
        "af" to af,
        "ae" to ae,
        "awb" to awb,
        "crop" to crop,
        "ois" to ois,
        "eis" to eis,
        "preview_stab" to previewStab,
    )
}

object CameraMeta {
    fun availability(chars: CameraCharacteristics): CaptureAvailability {
        val keys = chars.availableCaptureRequestKeys.map { it.name }.toSet()
        val resultKeys = chars.availableCaptureResultKeys.map { it.name }.toSet()
        fun has(name: String) = name in resultKeys || name in keys
        return CaptureAvailability(
            sensorTimestamp = has(CaptureResult.SENSOR_TIMESTAMP.name),
            exposure = has(CaptureResult.SENSOR_EXPOSURE_TIME.name),
            iso = has(CaptureResult.SENSOR_SENSITIVITY.name),
            focal = has(CaptureResult.LENS_FOCAL_LENGTH.name),
            focusDistance = has(CaptureResult.LENS_FOCUS_DISTANCE.name),
            af = has(CaptureResult.CONTROL_AF_STATE.name),
            ae = has(CaptureResult.CONTROL_AE_STATE.name),
            awb = has(CaptureResult.CONTROL_AWB_STATE.name),
            crop = has(CaptureResult.SCALER_CROP_REGION.name),
            ois = has(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE.name),
            eis = has(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE.name),
            previewStab = android.os.Build.VERSION.SDK_INT >= 33,
        )
    }

    fun afName(v: Int?) = when (v) {
        CaptureResult.CONTROL_AF_STATE_INACTIVE -> "INACTIVE"
        CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN -> "PASSIVE_SCAN"
        CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED -> "FOCUSED"
        CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN -> "ACTIVE_SCAN"
        CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> "FOCUSED"
        CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED -> "NOT_FOCUSED"
        CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED -> "UNFOCUSED"
        null -> null
        else -> "STATE_$v"
    }

    fun aeName(v: Int?) = when (v) {
        CaptureResult.CONTROL_AE_STATE_INACTIVE -> "INACTIVE"
        CaptureResult.CONTROL_AE_STATE_SEARCHING -> "SEARCHING"
        CaptureResult.CONTROL_AE_STATE_CONVERGED -> "CONVERGED"
        CaptureResult.CONTROL_AE_STATE_LOCKED -> "LOCKED"
        CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED -> "FLASH_REQUIRED"
        null -> null
        else -> "STATE_$v"
    }

    fun awbName(v: Int?) = when (v) {
        CaptureResult.CONTROL_AWB_STATE_INACTIVE -> "INACTIVE"
        CaptureResult.CONTROL_AWB_STATE_SEARCHING -> "SEARCHING"
        CaptureResult.CONTROL_AWB_STATE_CONVERGED -> "CONVERGED"
        CaptureResult.CONTROL_AWB_STATE_LOCKED -> "LOCKED"
        null -> null
        else -> "STATE_$v"
    }
}

data class StreamChoice(
    val size: Size,
    val fps: Int,
    val combo: String,
    val recordOk: Boolean,
    val degradeReason: String?,
)

object SizePicker {
    fun pick(map: android.hardware.camera2.params.StreamConfigurationMap, targetW: Int, targetH: Int, targetFps: Int, fallbackFps: Int): Pair<Size, Int> {
        val yuv = map.getOutputSizes(android.graphics.ImageFormat.YUV_420_888) ?: emptyArray()
        val want = Size(targetW, targetH)
        val has1080 = yuv.any { it.width == targetW && it.height == targetH }
        val size = if (has1080) want else yuv.minByOrNull { kotlin.math.abs(it.width * it.height - targetW * targetH) } ?: Size(1280, 720)
        val fpsRanges = map.getHighSpeedVideoFpsRangesFor(size) ?: emptyArray()
        val normal = map.getHighSpeedVideoSizes()?.contains(size) == true
        val range = map.getOutputMinFrameDuration(android.graphics.ImageFormat.YUV_420_888, size)
        val maxFps = if (range > 0) (1_000_000_000.0 / range).toInt() else 30
        val fps = when {
            maxFps >= targetFps -> targetFps
            maxFps >= fallbackFps -> fallbackFps
            else -> maxOf(15, maxFps)
        }
        return size to fps
    }
}

fun cropFromResult(result: android.hardware.camera2.CaptureResult): IntArray? {
    val r: Rect = result.get(CaptureResult.SCALER_CROP_REGION) ?: return null
    return intArrayOf(r.left, r.top, r.right, r.bottom)
}

fun desiredStabMode(mode: StabilizationMode, chars: CameraCharacteristics): Pair<Int?, Int?> {
    val videoModes = chars.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES) ?: intArrayOf()
    val ois = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION) ?: intArrayOf()
    return when (mode) {
        StabilizationMode.OFF, StabilizationMode.UNAVAILABLE -> {
            val v = if (CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_OFF in videoModes.toList())
                CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_OFF else null
            v to if (CaptureResult.LENS_OPTICAL_STABILIZATION_MODE_OFF in ois.toList())
                CaptureResult.LENS_OPTICAL_STABILIZATION_MODE_OFF else null
        }
        StabilizationMode.STANDARD -> {
            val v = if (CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_ON in videoModes.toList())
                CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_ON else null
            v to null
        }
        StabilizationMode.PREVIEW -> {
            val v = if (CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_ON in videoModes.toList())
                CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_ON else null
            v to if (CaptureResult.LENS_OPTICAL_STABILIZATION_MODE_ON in ois.toList())
                CaptureResult.LENS_OPTICAL_STABILIZATION_MODE_ON else null
        }
    }
}

@Suppress("UNUSED_PARAMETER")
fun unusedMetering(r: MeteringRectangle?) = r
