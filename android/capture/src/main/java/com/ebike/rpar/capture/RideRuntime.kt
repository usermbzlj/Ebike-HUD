package com.ebike.rpar.capture

import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.os.Build
import android.os.SystemClock
import android.view.TextureView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

data class RideUi(
    val recording: Boolean = false,
    val elapsedS: Long = 0,
    val remainingH: Double = 0.0,
    val gps: Boolean = false,
    val speedKmh: Double? = null,
    val imuHz: Int = 0,
    val battery: Int = -1,
    val charging: Boolean = false,
    val camera: String = "",
    val lastError: String? = null,
    val lastMark: String = "",
    val marks: Int = 0,
    val impactHits: Int = 0,
    val night: Boolean = false,
    val sessions: List<String> = emptyList(),
    val lowStorage: Boolean = false,
    val accepted: Boolean = false,
)

class RideRuntime(private val app: Application) {
    private val prefs = app.getSharedPreferences("ride_capture", 0)
    val sensors = RideSensors(app)
    val camera = RideCamera(app)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var session: RideSession? = null
    private var ticker: Job? = null
    private var recStart = 0L
    private var lastThumb = 0L
    private var lastRuntime = 0L
    private var preview: TextureView? = null

    private val _ui = MutableStateFlow(
        RideUi(accepted = prefs.getBoolean("accepted", false), night = isNightHour(), sessions = listSessions()),
    )
    val ui = _ui.asStateFlow()

    fun accept() {
        prefs.edit().putBoolean("accepted", true).apply()
        _ui.value = _ui.value.copy(accepted = true)
    }

    fun attachPreview(view: TextureView) {
        preview = view
    }

    @Volatile private var starting = false

    fun start(surface: SurfaceTexture) {
        if (_ui.value.recording || starting) return
        starting = true
        sensors.start()
        recStart = SystemClock.elapsedRealtime()
        val id = RideSession.newId()
        val root = File(app.filesDir, "sessions/session_$id")
        val sess = RideSession(root, SystemClock.elapsedRealtimeNanos())
        session = sess
        camera.onFrameMeta = { meta ->
            sess.writeFrame(meta)
            maybeThumb()
        }
        camera.onSegment = { sess.noteVideoSegment() }
        camera.open(surface, File(root, "video"))
        val svc = Intent(app, RideService::class.java)
        if (Build.VERSION.SDK_INT >= 26) app.startForegroundService(svc) else app.startService(svc)
        ticker?.cancel()
        ticker = scope.launch {
            while (isActive) {
                pump()
                delay(200)
            }
        }
        _ui.value = _ui.value.copy(
            recording = true,
            lastError = camera.lastError,
            camera = "${camera.size.width}x${camera.size.height}@${camera.fps}",
            remainingH = sess.remainingHours(),
        )
        sess.writeEvent("start", camera.cameraId ?: "")
        starting = false
    }

    fun stop() {
        starting = false
        ticker?.cancel()
        ticker = null
        camera.close()
        sensors.stop()
        val sess = session
        session = null
        app.stopService(Intent(app, RideService::class.java))
        scope.launch(Dispatchers.IO) {
            sess?.finalize(SystemClock.elapsedRealtimeNanos())
            _ui.value = _ui.value.copy(
                recording = false,
                elapsedS = 0,
                sessions = listSessions(),
            )
        }
    }

    fun mark(note: String = "manual") {
        val ts = SystemClock.elapsedRealtimeNanos()
        session?.writeMark(ts, note)
        _ui.value = _ui.value.copy(lastMark = note, marks = session?.markCount ?: _ui.value.marks)
    }

    fun exportLatest(): File? {
        val latest = File(app.filesDir, "sessions").listFiles()
            ?.filter { it.isDirectory }
            ?.maxByOrNull { it.lastModified() }
            ?: return null
        return RideSession.zipSession(latest, File(app.filesDir, "exports"))
    }

    private fun pump() {
        val sess = session ?: return
        sensors.refreshBattery()
        sensors.drainImu().forEach { row ->
            sess.writeImu(row.optString("sensor_type"), row)
        }
        sensors.drainLocation().forEach { sess.writeLocation(it) }
        if (sensors.lastResidual >= 8.0) {
            sess.writeMark(SystemClock.elapsedRealtimeNanos(), "imu_impact")
            _ui.value = _ui.value.copy(impactHits = _ui.value.impactHits + 1, marks = sess.markCount)
        }
        val now = SystemClock.elapsedRealtime()
        if (now - lastRuntime > 1000) {
            lastRuntime = now
            sess.writeRuntime(
                JSONObject()
                    .put("elapsed_s", (now - recStart) / 1000)
                    .put("battery_pct", sensors.batteryPct)
                    .put("charging", sensors.charging)
                    .put("gps", sensors.hasFix)
                    .put("speed_mps", sensors.speedMps)
                    .put("gyro_hz", sensors.gyroHz)
                    .put("accel_hz", sensors.accelHz)
                    .put("vertical_residual", sensors.lastResidual)
                    .put("camera", camera.cameraId)
                    .put("hfov_deg", camera.hfovDeg)
                    .put("privacy_mode", "LOCAL_ONLY"),
            )
        }
        if (sess.lowStorage()) {
            sess.writeEvent("storage_low", "auto_stop")
            stop()
            _ui.value = _ui.value.copy(lowStorage = true, lastError = "存储不足，已安全停止")
            return
        }
        _ui.value = _ui.value.copy(
            recording = true,
            elapsedS = (now - recStart) / 1000,
            remainingH = sess.remainingHours(),
            gps = sensors.hasFix,
            speedKmh = sensors.speedMps?.times(3.6),
            imuHz = sensors.accelHz.toInt(),
            battery = sensors.batteryPct,
            charging = sensors.charging,
            lastError = camera.lastError ?: _ui.value.lastError,
            marks = sess.markCount,
            night = isNightHour(),
            lowStorage = false,
        )
    }

    private fun maybeThumb() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastThumb < 2500) return
        lastThumb = now
        val view = preview ?: return
        val sess = session ?: return
        view.post {
            val bmp = view.bitmap ?: return@post
            val small = Bitmap.createScaledBitmap(bmp, 320, 180, true)
            val dest = File(sess.root, "thumbs/thumb_%06d.jpg".format(sess.markCount + (now / 2500).toInt()))
            dest.parentFile?.mkdirs()
            FileOutputStream(dest).use { small.compress(Bitmap.CompressFormat.JPEG, 70, it) }
            if (small != bmp) small.recycle()
        }
    }

    private fun listSessions(): List<String> =
        File(app.filesDir, "sessions").listFiles()
            ?.filter { it.isDirectory }
            ?.sortedByDescending { it.lastModified() }
            ?.take(8)
            ?.map { it.name }
            ?: emptyList()

    private fun isNightHour(): Boolean {
        val h = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        return h >= 19 || h < 6
    }
}
