package com.ebike.rpar.capability

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.media.AudioManager
import android.os.Build
import android.speech.tts.TextToSpeech
import android.util.SizeF
import com.ebike.rpar.model.SCHEMA_VERSION
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

class CapabilityProbe(private val context: Context) {
    fun run(
        concurrentCombo: JSONArray = JSONArray(),
        sensorRates: JSONObject = JSONObject(),
        ttsReady: Boolean = false,
        modelFile: File? = null,
    ): JSONObject {
        val cam = probeCameras()
        val sessionCombos = if (concurrentCombo.length() > 0) concurrentCombo else cam.optJSONArray("concurrent_streams") ?: JSONArray()
        val audio = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val arcore = try {
            Class.forName("com.google.ar.core.ArCoreApk")
            "present_optional"
        } catch (_: Throwable) {
            "unavailable"
        }
        val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        return JSONObject()
            .put("schema_version", SCHEMA_VERSION)
            .put("generated_at", iso.format(Date()))
            .put("source", "android_probe")
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL)
                .put("product", Build.PRODUCT)
                .put("device", Build.DEVICE)
                .put("os_version", Build.VERSION.RELEASE)
                .put("sdk_int", Build.VERSION.SDK_INT))
            .put("camera", cam)
            .put("sensors", sensorRates)
            .put("acceleration", LiteRTBench.run(modelFile))
            .put("arcore", JSONObject()
                .put("available", arcore)
                .put("depth", "unavailable")
                .put("required", false))
            .put("audio", JSONObject()
                .put("tts", ttsReady)
                .put("bluetooth", audio.isBluetoothA2dpOn)
                .put("speakerphone", audio.isSpeakerphoneOn)
                .put("routes", JSONArray().put("speaker").put("headset").put("bluetooth")))
            .put("availability_flags", JSONObject()
                .put("ois", flag(cam))
                .put("preview_stabilization", if (Build.VERSION.SDK_INT >= 33) "available" else "unavailable")
                .put("npu", "unavailable_until_litert_package"))
            .put("concurrent_streams", sessionCombos)
    }

    private fun flag(cam: JSONObject): String {
        val cams = cam.optJSONArray("cameras") ?: return "unavailable"
        for (i in 0 until cams.length()) {
            if (cams.optJSONObject(i)?.optString("ois") == "available") return "available"
        }
        return "unavailable"
    }

    private fun probeCameras(): JSONObject {
        val mgr = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameras = JSONArray()
        var hw = "unavailable"
        try {
            for (id in mgr.cameraIdList) {
                val ch = mgr.getCameraCharacteristics(id)
                val level = ch.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                hw = when (level) {
                    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
                    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
                    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
                    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
                    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
                    else -> "unknown"
                }
                val facing = when (ch.get(CameraCharacteristics.LENS_FACING)) {
                    CameraCharacteristics.LENS_FACING_BACK -> "BACK"
                    CameraCharacteristics.LENS_FACING_FRONT -> "FRONT"
                    else -> "EXTERNAL"
                }
                val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                val outputs = JSONArray()
                if (map != null) {
                    for (fmt in intArrayOf(android.graphics.ImageFormat.YUV_420_888, android.graphics.ImageFormat.PRIVATE, android.graphics.ImageFormat.JPEG)) {
                        val sizes = map.getOutputSizes(fmt) ?: continue
                        val arr = JSONArray()
                        var bestNs = Long.MAX_VALUE
                        sizes.take(12).forEach { sz ->
                            arr.put("${sz.width}x${sz.height}")
                            val d = try {
                                map.getOutputMinFrameDuration(fmt, sz)
                            } catch (_: Throwable) {
                                0L
                            }
                            if (d > 0L && d < bestNs) bestNs = d
                        }
                        sizes.firstOrNull { it.width == 1920 && it.height == 1080 }?.let { hd ->
                            val d = try {
                                map.getOutputMinFrameDuration(fmt, hd)
                            } catch (_: Throwable) {
                                0L
                            }
                            if (d > 0L) bestNs = d
                        }
                        val fpsList = fpsCandidatesFromMinDurationNs(if (bestNs == Long.MAX_VALUE) 0L else bestNs)
                        val fps = JSONArray()
                        for (f in fpsList) fps.put(f)
                        outputs.put(JSONObject().put("format", fmtName(fmt)).put("sizes", arr).put("fps", fps))
                    }
                }
                val ois = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
                val vs = ch.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)
                val af = ch.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
                val intr = ch.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
                val aeRanges = JSONArray()
                ch.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.forEach { r ->
                    val row = JSONObject()
                    row.put("min", r.lower)
                    row.put("max", r.upper)
                    aeRanges.put(row)
                }
                val yuvNs = minDurationNs(map, android.graphics.ImageFormat.YUV_420_888, 1920, 1080)
                val recNs = minDurationNs(map, android.graphics.ImageFormat.PRIVATE, 1920, 1080).let { ns ->
                    if (ns > 0L) ns else minDurationNs(map, android.graphics.ImageFormat.JPEG, 1920, 1080)
                }
                val combos = JSONArray()
                for (c in candidateConcurrentCombos(yuvNs, recNs)) {
                    val row = JSONObject()
                    row.put("combo", c.combo)
                    row.put("yuv", c.yuvSize)
                    row.put("record", c.recordSize)
                    row.put("requested_fps", c.requestedFps)
                    row.put("max_fps_from_duration", c.maxFpsFromDuration)
                    row.put("status", c.status)
                    combos.put(row)
                }
                val camObj = JSONObject()
                camObj.put("id", id)
                camObj.put(
                    "logical",
                    (ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf())
                        .contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA),
                )
                camObj.put("facing", facing)
                camObj.put("outputs", outputs)
                camObj.put("ois", if (ois != null && ois.isNotEmpty()) "available" else "unavailable")
                camObj.put("video_stabilization", if (vs != null && vs.any { it != 0 }) "available" else "unavailable")
                camObj.put("preview_stabilization", if (Build.VERSION.SDK_INT >= 33) "available" else "unavailable")
                camObj.put("dynamic_range", JSONArray().put("SDR"))
                camObj.put("af", if (af != null && af.isNotEmpty()) "available" else "unavailable")
                camObj.put("intrinsics", if (intr != null) "available" else "unavailable")
                camObj.put("sensor_size", ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.let { sz: SizeF -> "${sz.width}x${sz.height}" } ?: "unavailable")
                camObj.put("rolling_shutter", "runtime-detected")
                camObj.put("ae_target_fps_ranges", aeRanges)
                camObj.put("concurrent_streams", combos)
                cameras.put(camObj)
            }
        } catch (t: Throwable) {
            hw = "unavailable:${t.message}"
        }
        val fallbackCombos = JSONArray()
        for (i in 0 until cameras.length()) {
            val cam = cameras.optJSONObject(i) ?: continue
            if (cam.optString("facing") == "BACK") {
                val src = cam.optJSONArray("concurrent_streams") ?: JSONArray()
                for (j in 0 until src.length()) fallbackCombos.put(src.optJSONObject(j))
                break
            }
        }
        return JSONObject()
            .put("hardware_level", hw)
            .put("cameras", cameras)
            .put("concurrent_streams", fallbackCombos)
    }

    private fun minDurationNs(map: android.hardware.camera2.params.StreamConfigurationMap?, fmt: Int, w: Int, h: Int): Long {
        if (map == null) return 0L
        val sz = map.getOutputSizes(fmt)?.firstOrNull { it.width == w && it.height == h } ?: return 0L
        return try {
            map.getOutputMinFrameDuration(fmt, sz)
        } catch (_: Throwable) {
            0L
        }
    }

    private fun fmtName(fmt: Int) = when (fmt) {
        android.graphics.ImageFormat.YUV_420_888 -> "YUV_420_888"
        android.graphics.ImageFormat.PRIVATE -> "PRIVATE"
        android.graphics.ImageFormat.JPEG -> "JPEG"
        else -> fmt.toString()
    }

    fun write(report: JSONObject): File {
        val dir = File(context.filesDir, "capability")
        dir.mkdirs()
        val f = File(dir, "capability_report.json")
        f.writeText(report.toString(2))
        return f
    }

    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { ins ->
            val buf = ByteArray(8192)
            while (true) {
                val n = ins.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}

@Suppress("unused")
private val ttsHint = TextToSpeech.SUCCESS
