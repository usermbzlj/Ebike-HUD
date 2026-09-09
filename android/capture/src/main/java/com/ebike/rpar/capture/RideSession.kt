package com.ebike.rpar.capture

import android.os.Build
import android.os.StatFs
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.ConcurrentHashMap

/** Writes a session bundle the desktop `rpar` tools already know how to ingest. */
class RideSession(val root: File, startNs: Long) {
    private val writers = ConcurrentHashMap<String, BufferedWriter>()
    private val startNs = startNs
    var videoSegments = 0
        private set
    var closed = false
        private set
    var markCount = 0
        private set

    init {
        listOf(
            root,
            File(root, "calibration"),
            File(root, "video"),
            File(root, "camera"),
            File(root, "imu"),
            File(root, "location"),
            File(root, "events"),
            File(root, "thumbs"),
            File(root, "diagnostics"),
        ).forEach { it.mkdirs() }
        File(root, "calibration/mount_profile.json").writeText(
            JSONObject()
                .put("profile_id", "left_handlebar_v1")
                .put("notes", "Ride Capture; mount assumed left handlebar landscape")
                .toString(2),
        )
        writeManifest(null)
    }

    fun noteVideoSegment() {
        videoSegments++
        writeManifest(null)
    }

    fun writeFrame(o: JSONObject) {
        if (closed) return
        append("camera/frame_metadata.jsonl", o)
    }

    fun writeImu(kind: String, o: JSONObject) {
        if (closed) return
        val name = when (kind) {
            "ACCEL" -> "imu/accelerometer.jsonl"
            "ROTATION_VECTOR" -> "imu/rotation_vector.jsonl"
            "GRAVITY" -> "imu/gravity.jsonl"
            else -> "imu/gyro.jsonl"
        }
        append(name, o)
    }

    fun writeLocation(o: JSONObject) {
        if (closed) return
        append("location/location.jsonl", o)
    }

    fun writeRuntime(o: JSONObject) {
        if (closed) return
        append("diagnostics/runtime.jsonl", o)
    }

    fun writeEvent(code: String, detail: String) {
        if (closed) return
        append(
            "diagnostics/events.jsonl",
            JSONObject()
                .put("timestamp_ns", System.nanoTime())
                .put("domain", "CAP")
                .put("code", code)
                .put("detail", detail),
        )
    }

    fun writeMark(timestampNs: Long, note: String) {
        if (closed) return
        markCount++
        append(
            "events/marks.jsonl",
            JSONObject().put("timestamp_ns", timestampNs).put("note", note),
        )
    }

    fun remainingHours(bitrateMbps: Double = 20.0): Double {
        val stat = StatFs(root.absolutePath)
        val bytesPerHour = bitrateMbps * 1e6 / 8.0 * 3600.0
        return maxOf(0.0, stat.availableBytes / bytesPerHour)
    }

    fun lowStorage(): Boolean {
        val stat = StatFs(root.absolutePath)
        return stat.availableBytes < 300L * 1024 * 1024 || remainingHours() < 0.08
    }

    fun finalize(endNs: Long) {
        if (closed) return
        writers.values.forEach { runCatching { it.flush(); it.close() } }
        writers.clear()
        writeManifest(endNs)
        writeChecksums()
        closed = true
    }

    private fun append(rel: String, obj: JSONObject) {
        val w = writers.getOrPut(rel) {
            File(root, rel).parentFile?.mkdirs()
            File(root, rel).bufferedWriter()
        }
        synchronized(w) {
            w.write(obj.toString())
            w.newLine()
        }
    }

    private fun writeManifest(endNs: Long?) {
        val o = JSONObject()
            .put("schema_version", "1.1")
            .put("session_id", root.name.removePrefix("session_"))
            .put(
                "device",
                JSONObject()
                    .put("manufacturer", Build.MANUFACTURER)
                    .put("model", Build.MODEL)
                    .put("os", Build.VERSION.RELEASE),
            )
            .put("camera_profile", "rear_main_landscape_sdr")
            .put("mount_profile_id", "left_handlebar_v1")
            .put("start_elapsed_realtime_ns", startNs)
            .put("end_elapsed_realtime_ns", endNs)
            .put("video_segments", videoSegments)
            .put("privacy_mode", "LOCAL_ONLY")
            .put("run_mode", "CAPTURE_ONLY")
            .put("app", "com.ebike.rpar.capture")
            .put("app_version", BuildConfig.VERSION_NAME)
            .put("notes", "Ride Capture. No perception on device. LOCAL_ONLY.")
        val tmp = File(root, "manifest.json.tmp")
        tmp.writeText(o.toString(2))
        tmp.renameTo(File(root, "manifest.json"))
    }

    private fun writeChecksums() {
        val md = MessageDigest.getInstance("SHA-256")
        val lines = StringBuilder()
        root.walkTopDown()
            .filter { it.isFile && it.name != "checksums.sha256" && !it.name.endsWith(".tmp") }
            .sortedBy { it.relativeTo(root).invariantSeparatorsPath }
            .forEach { f ->
                md.reset()
                f.inputStream().use { ins ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = ins.read(buf)
                        if (n <= 0) break
                        md.update(buf, 0, n)
                    }
                }
                val hex = md.digest().joinToString("") { "%02x".format(it) }
                lines.append(hex).append("  ").append(f.relativeTo(root).invariantSeparatorsPath).append('\n')
            }
        File(root, "checksums.sha256").writeText(lines.toString())
    }

    companion object {
        fun newId(): String {
            val fmt = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).apply {
                timeZone = TimeZone.getTimeZone("UTC")
            }
            return "${fmt.format(Date())}_${Build.MODEL.replace(" ", "")}"
        }

        fun zipSession(src: File, destDir: File): File {
            destDir.mkdirs()
            val zip = File(destDir, "${src.name}.zip")
            java.util.zip.ZipOutputStream(zip.outputStream().buffered()).use { zos ->
                src.walkTopDown().filter { it.isFile }.forEach { f ->
                    val entry = java.util.zip.ZipEntry(f.relativeTo(src).invariantSeparatorsPath)
                    zos.putNextEntry(entry)
                    f.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
            return zip
        }
    }
}
