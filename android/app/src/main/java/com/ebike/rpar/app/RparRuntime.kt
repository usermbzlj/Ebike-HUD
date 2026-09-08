package com.ebike.rpar.app

import android.content.Context
import android.content.Intent
import android.graphics.SurfaceTexture
import android.os.Build
import android.os.SystemClock
import android.util.Size
import com.ebike.rpar.alert.VoiceAlerts
import com.ebike.rpar.calibration.CalibrationStore
import com.ebike.rpar.camera.CameraController
import com.ebike.rpar.camera.TestPatternSource
import com.ebike.rpar.capability.CapabilityProbe
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.diagnostics.DiagnosticBus
import com.ebike.rpar.geometry.GeometryEngine
import com.ebike.rpar.geometry.Transforms
import com.ebike.rpar.model.PerceptionView
import com.ebike.rpar.model.PrivacyMode
import com.ebike.rpar.model.RunMode
import com.ebike.rpar.model.StabilizationMode
import com.ebike.rpar.model.UiMode
import com.ebike.rpar.perception.ModelManager
import com.ebike.rpar.recorder.SessionWriter
import com.ebike.rpar.sensor.SensorHub
import com.ebike.rpar.sync.FrameSynchronizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

enum class AppScreen { DISCLAIMER, FIRST_RUN, HUD, SETTINGS, CALIBRATION, CAPABILITY, EXPORT }

data class UiState(
    val screen: AppScreen = AppScreen.DISCLAIMER,
    val uiMode: UiMode = UiMode.RIDING,
    val runMode: RunMode = RunMode.REALTIME_PERCEPTION,
    val view: PerceptionView? = null,
    val previewSize: Size = Size(1920, 1080),
    val touchLocked: Boolean = false,
    val alertsEnabled: Boolean = true,
    val stab: StabilizationMode = StabilizationMode.OFF,
    val statusLine: String = "感知正常 · 声音提醒",
    val remainingHours: Double = 0.0,
    val usingTestPattern: Boolean = true,
    val cameraDegrade: String? = null,
    val emergency: Boolean = false,
    val capabilityJson: String = "",
    val modelId: String = "heuristic-cv-0.1.0",
    val mountName: String = "left_handlebar_v1",
    val firstRunDone: Boolean = false,
    val disclaimerAccepted: Boolean = false,
    val exportTypes: List<String> = SessionWriter.EXPORT_TYPES,
    val lastError: String? = null,
)

class RparRuntime(private val app: android.app.Application) {
    val cfg: RparConfig = RparConfig.load(app)
    val diagnostics = DiagnosticBus()
    val calibration = CalibrationStore(app)
    val sensors = SensorHub(app, diagnostics)
    val sync = FrameSynchronizer(sensors)
    val voice = VoiceAlerts(app)
    val probe = CapabilityProbe(app)
    val models = ModelManager(app, cfg)
    val camera = CameraController(app, cfg.camera, diagnostics)
    val testPattern = TestPatternSource(app)
    private val prefs = app.getSharedPreferences("rpar", Context.MODE_PRIVATE)
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val mutex = Mutex()
    private var writer: SessionWriter? = null
    private var sessionStart = 0L
    private val _ui = MutableStateFlow(loadInitial())
    val ui = _ui.asStateFlow()
    lateinit var pipeline: RealtimePipeline
        private set
    private var recStartElapsed = 0L
    @Volatile var latestBitmap: android.graphics.Bitmap? = null
    val privacy: PrivacyMode = PrivacyMode.LOCAL_ONLY

    init {
        val mount = calibration.loadActive(cfg.camera.width, cfg.camera.height)
        val geom = GeometryEngine(mount, cfg.geometry, Transforms.defaultIntrinsics(cfg.camera.width, cfg.camera.height))
        val loaded = models.load(safeMode = _ui.value.runMode == RunMode.SAFE_MODE)
        pipeline = RealtimePipeline(cfg, loaded.engine, geom, loaded.packageId)
        pipeline.alerts.enabled = _ui.value.alertsEnabled
        pipeline.runMode = _ui.value.runMode
        voice.start()
        testPattern.onFrame = { frame -> ingest(frame, fromCamera = false) }
        camera.onFrame = { frame -> ingest(frame, fromCamera = true) }
        camera.onPreviewSize = { size -> _ui.value = _ui.value.copy(previewSize = size) }
    }

    private fun loadInitial(): UiState {
        val disc = prefs.getBoolean("disclaimer", false)
        val first = prefs.getBoolean("first_run", false)
        val mode = try {
            RunMode.valueOf(prefs.getString("run_mode", RunMode.REALTIME_PERCEPTION.wire)!!)
        } catch (_: Throwable) { RunMode.REALTIME_PERCEPTION }
        val uiMode = try {
            UiMode.valueOf(prefs.getString("ui_mode", UiMode.RIDING.wire)!!)
        } catch (_: Throwable) { UiMode.RIDING }
        val screen = when {
            !disc -> AppScreen.DISCLAIMER
            !first -> AppScreen.FIRST_RUN
            else -> AppScreen.HUD
        }
        return UiState(
            screen = screen,
            uiMode = uiMode,
            runMode = mode,
            alertsEnabled = prefs.getBoolean("alerts", cfg.alert.enabledDefault),
            stab = try { StabilizationMode.valueOf(prefs.getString("stab", "OFF")!!) } catch (_: Throwable) { StabilizationMode.OFF },
            firstRunDone = first,
            disclaimerAccepted = disc,
            usingTestPattern = mode == RunMode.SAFE_MODE || mode == RunMode.REPLAY,
        )
    }

    fun acceptDisclaimer() {
        prefs.edit().putBoolean("disclaimer", true).apply()
        val next = if (_ui.value.firstRunDone) AppScreen.HUD else AppScreen.FIRST_RUN
        _ui.value = _ui.value.copy(disclaimerAccepted = true, screen = next)
    }

    fun completeFirstRun() {
        prefs.edit().putBoolean("first_run", true).apply()
        _ui.value = _ui.value.copy(firstRunDone = true, screen = AppScreen.HUD)
    }

    fun navigate(screen: AppScreen) {
        _ui.value = _ui.value.copy(screen = screen)
    }

    fun setUiMode(mode: UiMode) {
        prefs.edit().putString("ui_mode", mode.wire).apply()
        _ui.value = _ui.value.copy(uiMode = mode)
    }

    fun setRunMode(mode: RunMode) {
        prefs.edit().putString("run_mode", mode.wire).apply()
        pipeline.runMode = mode
        if (mode == RunMode.SAFE_MODE) {
            val loaded = models.load(safeMode = true)
            pipeline.engine = loaded.engine
            pipeline.modelVersion = loaded.packageId
        } else {
            val loaded = models.load(safeMode = false)
            pipeline.engine = loaded.engine
            pipeline.modelVersion = loaded.packageId
        }
        _ui.value = _ui.value.copy(runMode = mode, modelId = pipeline.modelVersion)
    }

    fun setAlerts(enabled: Boolean) {
        prefs.edit().putBoolean("alerts", enabled).apply()
        pipeline.alerts.enabled = enabled
        _ui.value = _ui.value.copy(alertsEnabled = enabled)
    }

    fun setStab(mode: StabilizationMode) {
        prefs.edit().putString("stab", mode.wire).apply()
        camera.recreateForStabilization(mode)
        _ui.value = _ui.value.copy(stab = mode)
    }

    fun setTouchLock(locked: Boolean) {
        _ui.value = _ui.value.copy(touchLocked = locked)
    }

    fun emergencyStop() {
        _ui.value = _ui.value.copy(emergency = true, touchLocked = false)
        stopCapture()
        diagnostics.event("APP", "emergency_stop", "user")
    }

    fun startCapture(surface: SurfaceTexture?, preferCamera: Boolean) {
        val useCam = preferCamera &&
            _ui.value.runMode != RunMode.REPLAY &&
            _ui.value.runMode != RunMode.SAFE_MODE
        sessionStart = SystemClock.elapsedRealtimeNanos()
        recStartElapsed = SystemClock.elapsedRealtime()
        val mount = calibration.loadActive(cfg.camera.width, cfg.camera.height)
        pipeline.geometry.setMount(mount, Transforms.defaultIntrinsics(cfg.camera.width, cfg.camera.height))
        pipeline.reset()
        pipeline.runMode = _ui.value.runMode
        val id = SessionWriter.newSessionId(Build.MODEL)
        val root = File(app.filesDir, "sessions/session_$id")
        val capFile = File(app.filesDir, "capability/capability_report.json")
        val capHash = if (capFile.exists()) probe.sha256(capFile) else ""
        writer = SessionWriter(
            root = root,
            mount = mount,
            cfg = cfg,
            runMode = _ui.value.runMode,
            device = JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("os", Build.VERSION.RELEASE),
            capabilityHash = capHash,
            modelPackages = listOf(pipeline.modelVersion),
            startNs = sessionStart,
        )
        sensors.start()
        if (useCam) {
            val recDir = if (_ui.value.runMode != RunMode.SAFE_MODE) File(root, "video") else null
            camera.stabMode = _ui.value.stab
            camera.open(surface, recDir, enableRecord = recDir != null)
            _ui.value = _ui.value.copy(
                usingTestPattern = false,
                cameraDegrade = camera.lastDegrade,
            )
            if (!camera.running.get()) {
                diagnostics.event("CAM", "fallback_test_pattern", camera.lastDegrade ?: "open_failed")
                testPattern.start(scope)
                _ui.value = _ui.value.copy(usingTestPattern = true)
            }
        } else {
            testPattern.start(scope)
            _ui.value = _ui.value.copy(usingTestPattern = true)
        }
        val svc = Intent(app, CaptureService::class.java)
        if (Build.VERSION.SDK_INT >= 26) app.startForegroundService(svc) else app.startService(svc)
    }

    fun stopCapture() {
        testPattern.stop()
        camera.close()
        sensors.stop()
        writer?.finalize(SystemClock.elapsedRealtimeNanos())
        writer = null
        app.stopService(Intent(app, CaptureService::class.java))
    }

    fun onServiceStarted() {}
    fun onServiceStopped() {}

    fun reloadModel() {
        val loaded = models.load(safeMode = _ui.value.runMode == RunMode.SAFE_MODE)
        pipeline.engine = loaded.engine
        pipeline.modelVersion = loaded.packageId
        _ui.value = _ui.value.copy(modelId = loaded.packageId, lastError = loaded.error)
    }

    fun runCapability(): String {
        val combo = JSONArray().put(
            JSONObject()
                .put("combo", camera.lastChoice?.combo ?: "unprobed")
                .put("status", camera.lastChoice?.degradeReason ?: camera.lastChoice?.combo ?: "ok")
                .put("actual_fps", camera.actualFps),
        )
        val sensorsJson = JSONObject()
            .put("gyro", JSONObject().put("actual_hz", sensors.gyroHz).put("status", "probed"))
            .put("accel", JSONObject().put("actual_hz", sensors.accelHz).put("status", "probed"))
            .put("rotation_vector", JSONObject().put("actual_hz", sensors.rvHz).put("status", "probed"))
            .put("high_sampling_rate_permission", "declared")
            .put("intervals", sensors.intervalJson())
        val report = probe.run(combo, sensorsJson, voice.ready)
        val f = probe.write(report)
        val text = f.readText()
        _ui.value = _ui.value.copy(capabilityJson = text)
        return text
    }

    fun exportSessionZip(redacted: Boolean = false): File? {
        val sessions = File(app.filesDir, "sessions")
        val latest = sessions.listFiles()?.maxByOrNull { it.lastModified() } ?: return null
        val outDir = File(app.filesDir, "exports").also { it.mkdirs() }
        val packed = if (redacted) {
            val red = File(outDir, "${latest.name}_share")
            com.ebike.rpar.privacy.ShareRedactor.export(latest, red)
            red
        } else {
            latest
        }
        val zip = File(outDir, packed.name + ".zip")
        ZipOutputStream(zip.outputStream()).use { zos ->
            packed.walkTopDown().filter { it.isFile }.forEach { f ->
                zos.putNextEntry(ZipEntry(f.relativeTo(packed).invariantSeparatorsPath))
                f.inputStream().use { it.copyTo(zos) }
                zos.closeEntry()
            }
        }
        return zip
    }

    fun setAfLock(lock: Boolean) {
        camera.lockFarFocus = lock
        camera.recreateForStabilization(_ui.value.stab)
    }

    fun setExposureCap(on: Boolean) {
        camera.exposureCapNs = if (on) 8_000_000L else null
        camera.recreateForStabilization(_ui.value.stab)
    }

    fun setToneMode(on: Boolean) {
        voice.toneMode = on
    }

    private fun ingest(raw: com.ebike.rpar.model.SynchronizedFrame, fromCamera: Boolean) {
        scope.launch {
            mutex.withLock {
                if (_ui.value.emergency && fromCamera) return@withLock
                val frame = sync.attach(raw)
                latestBitmap = frame.bitmap
                pipeline.thermalC = sensors.thermalC
                pipeline.recSeconds = (SystemClock.elapsedRealtime() - recStartElapsed) / 1000.0
                val view = pipeline.step(frame, _ui.value.uiMode)
                view.alerts.filter { it.fired }.forEach { voice.speak(it.phrase) }
                val w = writer
                if (w != null) {
                    if (w.lowStorage()) {
                        pipeline.status = com.ebike.rpar.model.PerceptionStatus.STORAGE_LOW
                        diagnostics.event("REC", "storage_low", "stop")
                        stopCapture()
                        return@withLock
                    }
                    val imu = listOfNotNull(
                        sensors.gyro.peekLast(),
                        sensors.accel.peekLast(),
                        sensors.rv.peekLast(),
                    )
                    w.append(
                        meta = frame.meta,
                        imu = imu,
                        location = frame.location,
                        observations = null,
                        tracks = if (_ui.value.runMode == RunMode.REALTIME_PERCEPTION_FULL_LOG) view.tracks else emptyList(),
                        alerts = view.alerts,
                        diagnostics = diagnostics.drain(),
                        runtime = JSONObject()
                            .put("infer_fps", view.inferFps)
                            .put("ar_fps", view.arFps)
                            .put("latency_p95_ms", view.latencyP95Ms)
                            .put("thermal_c", sensors.thermalC)
                            .put("battery_pct", sensors.batteryPct)
                            .put("charging", sensors.charging)
                            .put("using_test_pattern", _ui.value.usingTestPattern)
                            .put("privacy_mode", privacy.wire),
                    )
                }
                val sound = if (_ui.value.alertsEnabled) "声音提醒" else "静音"
                _ui.value = _ui.value.copy(
                    view = view,
                    statusLine = "${view.statusCopy} · $sound",
                    remainingHours = writer?.remainingHours() ?: 0.0,
                    mountName = pipeline.geometry.mount.profileId,
                    cameraDegrade = camera.lastDegrade,
                )
            }
        }
    }
}
