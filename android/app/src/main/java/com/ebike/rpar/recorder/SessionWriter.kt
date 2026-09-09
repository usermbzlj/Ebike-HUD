package com.ebike.rpar.recorder

import android.os.StatFs
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.diagnostics.DiagnosticEvent
import com.ebike.rpar.model.AlertDecision
import com.ebike.rpar.model.FrameMeta
import com.ebike.rpar.model.ImuSample
import com.ebike.rpar.model.LocationSample
import com.ebike.rpar.model.MountProfile
import com.ebike.rpar.model.PrivacyMode
import com.ebike.rpar.model.RenderPrimitive
import com.ebike.rpar.model.RoadObservation
import com.ebike.rpar.model.RunMode
import com.ebike.rpar.model.SCHEMA_VERSION
import com.ebike.rpar.model.TrackedRoadObject
import com.ebike.rpar.model.polygonToJson
import org.json.JSONArray
import org.json.JSONObject
import android.os.SystemClock
import java.io.BufferedWriter
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

class SessionWriter(
    val root: File,
    private val mount: MountProfile,
    private val cfg: RparConfig,
    private val runMode: RunMode,
    private val device: JSONObject,
    private val capabilityHash: String,
    private val modelPackages: List<String>,
    startNs: Long,
) {
    private val fullLog = runMode == RunMode.REALTIME_PERCEPTION_FULL_LOG || runMode == RunMode.CAPTURE_ONLY
    private val privacy = PrivacyMode.LOCAL_ONLY
    var closed = false
        private set
    private var videoSegments = 0
    private val startNs = startNs
    private val writers = ConcurrentHashMap<String, BufferedWriter>()
    private var hoursCached = 0.0
    private var hoursAt = 0L

    init {
        listOf(
            root, File(root, "calibration"), File(root, "video"), File(root, "camera"),
            File(root, "imu"), File(root, "location"), File(root, "perception"),
            File(root, "events"), File(root, "events/clips"), File(root, "diagnostics"),
        ).forEach { it.mkdirs() }
        File(root, "calibration/mount_profile.json").writeText(mount.toJson().toString(2))
        listOf(
            "camera/frame_metadata.jsonl",
            "imu/gyro.jsonl", "imu/accelerometer.jsonl", "imu/rotation_vector.jsonl",
            "location/location.jsonl",
            "perception/observations.jsonl", "perception/tracks.jsonl",
            "perception/road.jsonl", "perception/occlusion.jsonl",
            "events/alerts.jsonl", "events/marks.jsonl", "diagnostics/runtime.jsonl", "diagnostics/events.jsonl",
        ).forEach { File(root, it).appendText("") }
        writeManifest(null)
    }

    fun noteVideoSegment() {
        videoSegments++
        writeManifest(null)
    }

    fun append(
        meta: FrameMeta,
        imu: List<ImuSample>,
        location: LocationSample?,
        observations: List<RoadObservation>?,
        tracks: List<TrackedRoadObject>?,
        alerts: List<AlertDecision>?,
        diagnostics: List<DiagnosticEvent>,
        runtime: JSONObject?,
    ) {
        if (closed) return
        appendLine("camera/frame_metadata.jsonl", meta.toJson())
        imu.forEach { s ->
            val name = when (s.sensorType.wire) {
                "ACCEL" -> "imu/accelerometer.jsonl"
                "ROTATION_VECTOR" -> "imu/rotation_vector.jsonl"
                else -> "imu/gyro.jsonl"
            }
            appendLine(name, s.toJson())
        }
        if (location != null) appendLine("location/location.jsonl", location.toJson(includePrecise = true))
        if (fullLog) {
            observations?.forEach { appendLine("perception/observations.jsonl", it.toJson()) }
        }
        tracks?.forEach { appendLine("perception/tracks.jsonl", it.toJson()) }
        alerts?.forEach { appendLine("events/alerts.jsonl", it.toJson()) }
        diagnostics.forEach { appendLine("diagnostics/events.jsonl", it.toJson()) }
        if (runtime != null) appendLine("diagnostics/runtime.jsonl", runtime)
    }

    fun writeMasks(timestampNs: Long, road: List<Pair<Float, Float>>, occ: List<List<Pair<Float, Float>>>) {
        if (closed) return
        val roadObj = JSONObject()
        roadObj.put("timestamp_ns", timestampNs)
        roadObj.put("road_polygon", polygonToJson(road))
        appendLine("perception/road.jsonl", roadObj)
        val occObj = JSONObject()
        occObj.put("timestamp_ns", timestampNs)
        val arr = JSONArray()
        occ.forEach { arr.put(polygonToJson(it)) }
        occObj.put("occluded_polygons", arr)
        appendLine("perception/occlusion.jsonl", occObj)
    }

    fun writeMark(timestampNs: Long, note: String) {
        if (closed) return
        File(root, "events").mkdirs()
        appendLine(
            "events/marks.jsonl",
            JSONObject().put("timestamp_ns", timestampNs).put("note", note),
        )
    }

    fun writeImuIntervals(obj: JSONObject) {
        if (closed) return
        File(root, "diagnostics/imu_intervals.json").writeText(obj.toString(2))
    }

    fun writeOverlay(timestampNs: Long, primitives: List<RenderPrimitive>) {
        if (closed) return
        val a = JSONArray()
        primitives.forEach { p ->
            a.put(
                JSONObject()
                    .put("track_id", p.trackId)
                    .put("kind", p.kind)
                    .put("label", p.label)
                    .put("polygon", polygonToJson(p.polygon)),
            )
        }
        appendLine(
            "video/overlay_preview.jsonl",
            JSONObject().put("timestamp_ns", timestampNs).put("n", primitives.size).put("primitives", a),
        )
    }

    fun remainingHours(bitrateMbps: Double = cfg.camera.bitrateMbps): Double {
        val now = SystemClock.elapsedRealtime()
        if (now - hoursAt < 2000 && hoursAt > 0L) return hoursCached
        val stat = StatFs(root.absolutePath)
        val free = stat.availableBytes
        val bytesPerHour = bitrateMbps * 1e6 / 8.0 * 3600.0
        hoursCached = maxOf(0.0, free / bytesPerHour)
        hoursAt = now
        return hoursCached
    }

    fun lowStorage(): Boolean {
        val stat = StatFs(root.absolutePath)
        return stat.availableBytes < 200L * 1024 * 1024 || remainingHours() < 0.1
    }

    fun finalize(endNs: Long) {
        if (closed) return
        writers.values.forEach { runCatching { it.flush(); it.close() } }
        writers.clear()
        writeImpactAlign()
        writeManifest(endNs)
        writeChecksums()
        closed = true
    }

    private fun writeImpactAlign() {
        val trackLines = File(root, "perception/tracks.jsonl")
        val accelFile = File(root, "imu/accelerometer.jsonl")
        val locFile = File(root, "location/location.jsonl")
        val seeds = ArrayList<com.ebike.rpar.model.FutureImpactSeed>()
        if (trackLines.exists()) {
            trackLines.readLines().forEach { line ->
                val s = line.trim()
                if (s.isEmpty()) return@forEach
                val o = JSONObject(s)
                val life = o.optString("lifecycle_state")
                if (life != "CONFIRMED" && life != "ALERTED") return@forEach
                val dist = if (o.has("distance_m") && !o.isNull("distance_m")) o.optDouble("distance_m") else null
                seeds.add(
                    com.ebike.rpar.model.FutureImpactSeed(
                        trackId = o.optInt("track_id"),
                        timestampNs = o.optLong("timestamp_ns"),
                        lifecycle = life,
                        geometryType = o.optString("geometry_type", "unknown"),
                        semanticType = o.optString("semantic_type", "unknown_anomaly"),
                        distanceM = dist,
                        speedMps = null,
                    ),
                )
            }
        }
        val accel = ArrayList<com.ebike.rpar.model.AccelZ>()
        if (accelFile.exists()) {
            accelFile.readLines().forEach { line ->
                val s = line.trim()
                if (s.isEmpty()) return@forEach
                val o = JSONObject(s)
                accel.add(
                    com.ebike.rpar.model.AccelZ(
                        o.optLong("timestamp_ns"),
                        o.optDouble("z"),
                        o.optDouble("x", 0.0),
                        o.optDouble("y", 0.0),
                    ),
                )
            }
        }
        var speed = 10.0
        if (locFile.exists()) {
            val speeds = ArrayList<Double>()
            locFile.readLines().forEach { line ->
                val s = line.trim()
                if (s.isEmpty()) return@forEach
                val o = JSONObject(s)
                if (o.has("speed_mps") && !o.isNull("speed_mps")) speeds.add(o.optDouble("speed_mps"))
            }
            if (speeds.isNotEmpty()) {
                speeds.sort()
                speed = speeds[speeds.size / 2]
            }
        }
        val rows = com.ebike.rpar.model.alignTracksToFutureImpact(seeds, accel, defaultSpeedMps = speed)
        val out = JSONObject()
        out.put("used_for_alert", false)
        out.put("n_aligned", rows.size)
        out.put("horizon_s", 3.0)
        out.put("speed_mps", speed)
        val arr = JSONArray()
        for (r in rows.take(64)) {
            val row = JSONObject()
            row.put("track_id", r.trackId)
            row.put("peak_ms2", r.peakMs2)
            row.put("impact_label", r.impactLabel)
            row.put("used_for_alert", false)
            row.put("geometry_type", r.geometryType)
            row.put("semantic_type", r.semanticType)
            arr.put(row)
        }
        out.put("rows", arr)
        File(root, "perception/impact_align.json").writeText(out.toString(2))
    }

    private fun appendLine(rel: String, obj: JSONObject) {
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
            .put("schema_version", SCHEMA_VERSION)
            .put("session_id", root.name.removePrefix("session_"))
            .put("device", device)
            .put("camera_profile", "rear_main_landscape_sdr")
            .put("mount_profile_id", mount.profileId)
            .put("calibration_hash", mount.calibrationHash)
            .put("model_packages", JSONArray(modelPackages))
            .put("start_elapsed_realtime_ns", startNs)
            .put("end_elapsed_realtime_ns", endNs)
            .put("video_segments", videoSegments)
            .put("capability_report_hash", capabilityHash)
            .put("privacy_mode", privacy.wire)
            .put("run_mode", runMode.wire)
            .put("config_snapshot", cfg.snapshot())
            .put("notes", "LOCAL_ONLY; no upload paths")
        val tmp = File(root, "manifest.json.tmp")
        tmp.writeText(o.toString(2))
        tmp.renameTo(File(root, "manifest.json"))
    }

    private fun writeChecksums() {
        val md = MessageDigest.getInstance("SHA-256")
        val lines = StringBuilder()
        root.walkTopDown().filter { it.isFile && it.name != "checksums.sha256" && !it.name.endsWith(".tmp") }
            .sortedBy { it.relativeTo(root).invariantSeparatorsPath }
            .forEach { f ->
                md.reset()
                f.inputStream().use { ins ->
                    val buf = ByteArray(8192)
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
        fun newSessionId(model: String): String {
            val fmt = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }
            return "${fmt.format(Date())}_${model.replace(" ", "")}"
        }

        fun estimateHoursRemaining(freeBytes: Long, bitrateMbps: Double = 25.0): Double {
            val bytesPerHour = bitrateMbps * 1e6 / 8.0 * 3600.0
            return maxOf(0.0, freeBytes / bytesPerHour)
        }

        val EXPORT_TYPES = listOf(
            "video (H.264 segments)",
            "camera/frame_metadata.jsonl",
            "imu gyro / accelerometer / rotation_vector jsonl",
            "location jsonl (precise coordinates, local only)",
            "perception observations + tracks jsonl",
            "perception/impact_align.json (M5 future IMU, never alerts)",
            "perception/road.jsonl + occlusion.jsonl",
            "events/alerts.jsonl (decision snapshots)",
            "events/clips (JPEG ring around voice alerts)",
            "video/overlay_preview.jsonl (AR primitives)",
            "diagnostics/runtime.jsonl + events.jsonl",
            "calibration/mount_profile.json",
            "manifest.json + checksums.sha256",
        )
    }
}
