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
    ): JSONObject {
        val cam = probeCameras()
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
            .put("acceleration", JSONObject()
                .put("candidates", JSONArray().put("CPU").put("GPU").put("NPU"))
                .put("note", "LiteRT microbench not run in V0.1")
                .put("results", JSONArray()
                    .put(JSONObject().put("backend", "CPU").put("status", "unavailable_until_benchmarked"))
                    .put(JSONObject().put("backend", "GPU").put("status", "unavailable_until_benchmarked"))
                    .put(JSONObject().put("backend", "NPU").put("status", "unavailable_until_benchmarked"))))
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
                .put("npu", "unavailable_until_benchmarked"))
            .put("concurrent_streams", concurrentCombo)
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
                        sizes.take(12).forEach { arr.put("${it.width}x${it.height}") }
                        val fps = JSONArray().put(30).put(60)
                        outputs.put(JSONObject().put("format", fmtName(fmt)).put("sizes", arr).put("fps", fps))
                    }
                }
                val ois = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
                val vs = ch.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)
                val af = ch.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
                val intr = ch.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
                cameras.put(
                    JSONObject()
                        .put("id", id)
                        .put("logical", (ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf())
                            .contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA))
                        .put("facing", facing)
                        .put("outputs", outputs)
                        .put("ois", if (ois != null && ois.isNotEmpty()) "available" else "unavailable")
                        .put("video_stabilization", if (vs != null && vs.any { it != 0 }) "available" else "unavailable")
                        .put("preview_stabilization", if (Build.VERSION.SDK_INT >= 33) "available" else "unavailable")
                        .put("dynamic_range", JSONArray().put("SDR"))
                        .put("af", if (af != null && af.isNotEmpty()) "available" else "unavailable")
                        .put("intrinsics", if (intr != null) "available" else "unavailable")
                        .put("sensor_size", ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.let { sz: SizeF -> "${sz.width}x${sz.height}" } ?: "unavailable")
                        .put("rolling_shutter", "runtime-detected"),
                )
            }
        } catch (t: Throwable) {
            hw = "unavailable:${t.message}"
        }
        return JSONObject()
            .put("hardware_level", hw)
            .put("cameras", cameras)
            .put("concurrent_streams", JSONArray())
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
