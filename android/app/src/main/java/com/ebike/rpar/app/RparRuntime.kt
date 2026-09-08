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
import com.ebike.rpar.recorder.EventClipBuffer
import com.ebike.rpar.recorder.SegmentRecovery
import com.ebike.rpar.recorder.SessionWriter
import com.ebike.rpar.recorder.SplitZip
import com.ebike.rpar.sensor.SensorHub
import com.ebike.rpar.sensor.severeImpact
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
    val emergencyReason: String = "",
    val capabilityJson: String = "",
    val modelId: String = "heuristic-cv-0.1.0",
    val mountName: String = "left_handlebar_v1",
    val firstRunDone: Boolean = false,
    val disclaimerAccepted: Boolean = false,
    val exportTypes: List<String> = SessionWriter.EXPORT_TYPES,
    val lastError: String? = null,
    val storageLight: Boolean = false,
    val nightPalette: Boolean = false,
    val researchHeatmap: Boolean = true,
    val strokeScale: Float = 1f,
    val fontScale: Float = 1f,
    val overlayAlpha: Float = 1f,
    val showInfoLayer: Boolean = true,
    val dampingArm: String? = null,
    val dampingNote: String = "",
    val paused: Boolean = false,
    val lastMark: String = "",
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
    private var clips: EventClipBuffer? = null
    private var sessionStart = 0L
    private val _ui = MutableStateFlow(loadInitial())
    val ui = _ui.asStateFlow()
    lateinit var pipeline: RealtimePipeline
        private set
    private var recStartElapsed = 0L
    private var dampingBlur = ArrayList<Double>()
    private var dampingGyro = ArrayList<Double>()
    @Volatile var latestBitmap: android.graphics.Bitmap? = null
    val privacy: PrivacyMode = PrivacyMode.LOCAL_ONLY

    init {
        val mount = calibration.loadActive(cfg.camera.width, cfg.camera.height)
        val geom = GeometryEngine(mount, cfg.geometry, Transforms.defaultIntrinsics(cfg.camera.width, cfg.camera.height))
        val loaded = models.load(safeMode = _ui.value.runMode == RunMode.SAFE_MODE)
        pipeline = RealtimePipeline(cfg, loaded.engine, geom, loaded.packageId)
        pipeline.alerts.enabled = _ui.value.alertsEnabled
        pipeline.runMode = _ui.value.runMode
        pipeline.nightPalette = _ui.value.nightPalette
        pipeline.researchHeatmap = _ui.value.researchHeatmap
        pipeline.strokeScale = _ui.value.strokeScale
        pipeline.fontScale = _ui.value.fontScale
        pipeline.overlayAlpha = _ui.value.overlayAlpha
        pipeline.showInfoLayer = _ui.value.showInfoLayer
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
            touchLocked = screen == AppScreen.HUD && uiMode == UiMode.RIDING,
            nightPalette = prefs.getBoolean("night_palette", false),
            researchHeatmap = prefs.getBoolean("research_heatmap", true),
            strokeScale = prefs.getFloat("stroke_scale", cfg.render.strokeScale),
            fontScale = prefs.getFloat("font_scale", cfg.render.fontScale),
            overlayAlpha = prefs.getFloat("overlay_alpha", cfg.render.overlayAlpha),
            showInfoLayer = prefs.getBoolean("show_info_layer", cfg.render.showInfoLayer),
        )
    }

    fun acceptDisclaimer() {
        prefs.edit().putBoolean("disclaimer", true).apply()
        val next = if (_ui.value.firstRunDone) AppScreen.HUD else AppScreen.FIRST_RUN
        _ui.value = _ui.value.copy(disclaimerAccepted = true, screen = next)
    }

    fun completeFirstRun() {
        prefs.edit().putBoolean("first_run", true).apply()
        _ui.value = _ui.value.copy(
            firstRunDone = true,
            screen = AppScreen.HUD,
            touchLocked = _ui.value.uiMode == UiMode.RIDING,
        )
    }

    fun navigate(screen: AppScreen) {
        val lock = screen == AppScreen.HUD && _ui.value.uiMode == UiMode.RIDING
        _ui.value = _ui.value.copy(screen = screen, touchLocked = if (lock) true else _ui.value.touchLocked)
    }

    fun setUiMode(mode: UiMode) {
        prefs.edit().putString("ui_mode", mode.wire).apply()
        val lock = mode == UiMode.RIDING && _ui.value.screen == AppScreen.HUD
        _ui.value = _ui.value.copy(uiMode = mode, touchLocked = lock)
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

    fun emergencyStop(source: String = "user") {
        val line = if (source == "imu_crash") "疑似碰撞/剧烈振动，已停止采集，停车后确认" else "紧急停止"
        clips?.onAlert(-1, "emergency:$source")
        _ui.value = _ui.value.copy(emergency = true, touchLocked = false, statusLine = line, emergencyReason = source)
        stopCapture()
        diagnostics.event("APP", "emergency_stop", source)
    }

    fun startCapture(surface: SurfaceTexture?, preferCamera: Boolean) {
        if (_ui.value.emergency) {
            diagnostics.event("APP", "emergency_cleared", "resume")
            _ui.value = _ui.value.copy(emergency = false, emergencyReason = "")
        }
        val useCam = preferCamera &&
            _ui.value.runMode != RunMode.REPLAY &&
            _ui.value.runMode != RunMode.SAFE_MODE
        val hours = remainingHoursNow()
        val light = hours < 0.10
        sessionStart = SystemClock.elapsedRealtimeNanos()
        recStartElapsed = SystemClock.elapsedRealtime()
        val mount = calibration.loadActive(cfg.camera.width, cfg.camera.height)
        pipeline.geometry.setMount(mount, Transforms.defaultIntrinsics(cfg.camera.width, cfg.camera.height))
        if (listOf(mount.knownDistance5mPx, mount.knownDistance10mPx, mount.knownDistance20mPx).count { it != null } >= 2) {
            pipeline.geometry.applyKnownDistanceMarkers()
        }
        if (mount.valid) {
            val ok = pipeline.geometry.healthCheck(0.0, 0.0, mount.horizonYPx)
            if (!ok) diagnostics.event("CAL", "install_health_fail", pipeline.geometry.invalidReason ?: "horizon")
        } else {
            pipeline.geometry.valid = false
            diagnostics.event("CAL", "no_mount_profile", "hide_distance")
        }
        pipeline.reset()
        pipeline.runMode = _ui.value.runMode
        pipeline.nightPalette = _ui.value.nightPalette
        pipeline.researchHeatmap = _ui.value.researchHeatmap
        pipeline.strokeScale = _ui.value.strokeScale
        pipeline.fontScale = _ui.value.fontScale
        pipeline.overlayAlpha = _ui.value.overlayAlpha
        pipeline.showInfoLayer = _ui.value.showInfoLayer
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
        clips = EventClipBuffer(File(root, "events/clips"))
        sensors.start()
        SegmentRecovery.recoverPartFiles(File(app.filesDir, "sessions"))
        camera.onSegmentFinalized = { writer?.noteVideoSegment() }
        if (useCam) {
            val recDir = if (_ui.value.runMode != RunMode.SAFE_MODE && !light) File(root, "video") else null
            camera.stabMode = _ui.value.stab
            camera.open(surface, recDir, enableRecord = recDir != null)
            _ui.value = _ui.value.copy(
                usingTestPattern = false,
                cameraDegrade = camera.lastDegrade,
                remainingHours = hours,
                storageLight = light,
                touchLocked = _ui.value.uiMode == UiMode.RIDING,
            )
            if (!camera.running.get()) {
                diagnostics.event("CAM", "fallback_test_pattern", camera.lastDegrade ?: "open_failed")
                testPattern.start(scope)
                _ui.value = _ui.value.copy(usingTestPattern = true)
            }
        } else {
            testPattern.start(scope)
            _ui.value = _ui.value.copy(usingTestPattern = true, remainingHours = hours, storageLight = light)
        }
        val svc = Intent(app, CaptureService::class.java)
        if (Build.VERSION.SDK_INT >= 26) app.startForegroundService(svc) else app.startService(svc)
    }

    fun togglePause() {
        _ui.value = _ui.value.copy(paused = !_ui.value.paused)
    }

    fun screenshot(): java.io.File? {
        val bmp = latestBitmap ?: return null
        val dir = File(app.filesDir, "screenshots").also { it.mkdirs() }
        val f = File(dir, "frame_${SystemClock.elapsedRealtime()}.jpg")
        f.outputStream().use { bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, it) }
        return f
    }

    fun markEvent(note: String = "manual") {
        val ts = _ui.value.view?.timestampNs ?: SystemClock.elapsedRealtimeNanos()
        writer?.writeMark(ts, note)
        _ui.value = _ui.value.copy(lastMark = note)
    }

    fun stopCapture() {
        testPattern.stop()
        camera.close()
        sensors.stop()
        writer?.writeImuIntervals(sensors.intervalJson())
        writer?.finalize(SystemClock.elapsedRealtimeNanos())
        writer = null
        clips = null
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

    fun rollbackModel() {
        val loaded = models.rollback()
        pipeline.engine = loaded.engine
        pipeline.modelVersion = loaded.packageId
        _ui.value = _ui.value.copy(modelId = loaded.packageId, lastError = loaded.error ?: "rolled_back")
    }

    fun setNightPalette(on: Boolean) {
        prefs.edit().putBoolean("night_palette", on).apply()
        pipeline.nightPalette = on
        _ui.value = _ui.value.copy(nightPalette = on)
    }

    fun setResearchHeatmap(on: Boolean) {
        prefs.edit().putBoolean("research_heatmap", on).apply()
        pipeline.researchHeatmap = on
        _ui.value = _ui.value.copy(researchHeatmap = on)
    }

    fun setStrokeScale(v: Float) {
        prefs.edit().putFloat("stroke_scale", v).apply()
        pipeline.strokeScale = v
        _ui.value = _ui.value.copy(strokeScale = v)
    }

    fun setFontScale(v: Float) {
        prefs.edit().putFloat("font_scale", v).apply()
        pipeline.fontScale = v
        _ui.value = _ui.value.copy(fontScale = v)
    }

    fun setOverlayAlpha(v: Float) {
        prefs.edit().putFloat("overlay_alpha", v).apply()
        pipeline.overlayAlpha = v
        _ui.value = _ui.value.copy(overlayAlpha = v)
    }

    fun setShowInfoLayer(on: Boolean) {
        prefs.edit().putBoolean("show_info_layer", on).apply()
        pipeline.showInfoLayer = on
        _ui.value = _ui.value.copy(showInfoLayer = on)
    }

    fun sampleHeadlightField(): Double? {
        val bmp = latestBitmap ?: return null
        val gray = com.ebike.rpar.quality.GrayImage.fromBitmap(bmp, 480)
        val mean = com.ebike.rpar.quality.fitHeadlightMean(gray)
        val cur = calibration.loadActive(cfg.camera.width, cfg.camera.height)
        val saved = calibration.save(cur.copy(headlightMean = mean, headlightValid = true))
        pipeline.geometry.setMount(saved)
        return mean
    }

    fun startDampingSample(arm: String) {
        dampingBlur.clear()
        dampingGyro.clear()
        _ui.value = _ui.value.copy(dampingArm = arm, dampingNote = "采样 $arm")
    }

    fun stopDampingSample(): String {
        val arm = _ui.value.dampingArm ?: "A"
        val blurShare = if (dampingBlur.isEmpty()) 0.0 else dampingBlur.count { it > 0.5 } / dampingBlur.size.toDouble()
        val gyroPeak = dampingGyro.maxOrNull() ?: 0.0
        val o = JSONObject()
            .put("arm", arm)
            .put("n", dampingBlur.size)
            .put("blur_share", blurShare)
            .put("gyro_peak", gyroPeak)
            .put("method", "auto_stats")
        File(app.filesDir, "damping_ab_$arm.json").writeText(o.toString(2))
        _ui.value = _ui.value.copy(dampingArm = null, dampingNote = "$arm blur_share=${"%.2f".format(blurShare)} gyro_peak=${"%.2f".format(gyroPeak)}")
        return o.toString()
    }

    private fun remainingHoursNow(): Double {
        val stat = android.os.StatFs(app.filesDir.absolutePath)
        return SessionWriter.estimateHoursRemaining(stat.availableBytes, cfg.camera.bitrateMbps)
    }

    fun runCapability(): String {
        val combo = JSONArray()
        val row = JSONObject()
        val choice = camera.lastChoice
        row.put("combo", choice?.combo ?: "unprobed")
        row.put("status", choice?.degradeReason ?: if (choice != null) "ok" else "unprobed")
        row.put("requested_fps", camera.actualFps)
        row.put("actual_fps", camera.actualFps)
        row.put("measured_yuv_fps", camera.measuredFps)
        row.put("measured_n", camera.measuredFpsN)
        row.put("size", "${camera.size.width}x${camera.size.height}")
        row.put("record_ok", choice?.recordOk ?: false)
        combo.put(row)
        val sensorsJson = JSONObject()
            .put("gyro", JSONObject().put("actual_hz", sensors.gyroHz).put("status", "probed"))
            .put("accel", JSONObject().put("actual_hz", sensors.accelHz).put("status", "probed"))
            .put("rotation_vector", JSONObject().put("actual_hz", sensors.rvHz).put("status", "probed"))
            .put("high_sampling_rate_permission", "declared")
            .put("intervals", sensors.intervalJson())
        val report = probe.run(
            combo,
            sensorsJson,
            voice.ready,
            File(app.filesDir, "models/${pipeline.modelVersion}/model.tflite").takeIf { it.exists() },
        )
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
        val parts = SplitZip.export(packed, File(outDir, packed.name), 512L * 1024 * 1024)
        return parts.firstOrNull()
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
                if (_ui.value.paused) return@withLock
                val frame = sync.attach(raw)
                val acc = frame.linearAccel
                val gyro = frame.angularVelocity
                val gmag = if (gyro != null && gyro.size >= 3) {
                    kotlin.math.sqrt(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2])
                } else {
                    0.0
                }
                if (
                    fromCamera &&
                    !_ui.value.usingTestPattern &&
                    acc != null &&
                    acc.size >= 3 &&
                    severeImpact(acc[0], acc[1], acc[2], gmag)
                ) {
                    emergencyStop("imu_crash")
                    return@withLock
                }
                latestBitmap = frame.bitmap
                pipeline.thermalC = sensors.thermalC
                pipeline.recSeconds = (SystemClock.elapsedRealtime() - recStartElapsed) / 1000.0
                val view = pipeline.step(frame, _ui.value.uiMode)
                pipeline.lastStatusEvent?.let {
                    diagnostics.event("QUAL", it, pipeline.status.wire)
                    pipeline.lastStatusEvent = null
                }
                pipeline.lastThermalEvent?.let {
                    diagnostics.event("NFR", it, "thermal_c=${sensors.thermalC}")
                    pipeline.lastThermalEvent = null
                }
                clips?.push(frame.meta.sensorTimestampNs, frame.bitmap)
                view.alerts.filter { it.fired }.forEach {
                    voice.speak(it.phrase)
                    clips?.onAlert(it.trackId, it.phrase)
                }
                if (_ui.value.dampingArm != null) {
                    dampingBlur += view.blur
                    val g = sensors.gyro.peekLast()
                    if (g != null) dampingGyro += kotlin.math.abs(g.x) + kotlin.math.abs(g.y) + kotlin.math.abs(g.z)
                }
                val w = writer
                if (w != null) {
                    if (w.lowStorage()) {
                        pipeline.status = com.ebike.rpar.model.PerceptionStatus.STORAGE_LOW
                        diagnostics.event("REC", "storage_low", "stop")
                        stopCapture()
                        return@withLock
                    }
                    val imu = sensors.drainPending()
                    w.append(
                        meta = frame.meta,
                        imu = imu,
                        location = frame.location,
                        observations = if (_ui.value.runMode == RunMode.REALTIME_PERCEPTION_FULL_LOG) pipeline.lastObservations else null,
                        tracks = view.tracks,
                        alerts = view.alerts,
                        diagnostics = diagnostics.drain(),
                        runtime = JSONObject()
                            .put("infer_fps", view.inferFps)
                            .put("ar_fps", view.arFps)
                            .put("latency_p95_ms", view.latencyP95Ms)
                            .put("latency_p50_ms", view.latencyP50Ms)
                            .put("thermal_c", sensors.thermalC)
                            .put("thermal_reason", pipeline.thermalReason)
                            .put("skip_far_roi", pipeline.skipFarRoi)
                            .put("inferred", pipeline.didInfer)
                            .put("degrade_reason", view.quality?.degradeReason)
                            .put("selected_for_infer", view.quality?.selectedForInfer)
                            .put("selected_age_ms", view.quality?.selectedAgeMs)
                            .put("occupancy_occluded_ratio", view.quality?.occupancyOccludedRatio)
                            .put("sharpness", view.quality?.globalQuality?.sharpness)
                            .put("blur", view.blur)
                            .put("glare", view.glare)
                            .put("battery_pct", sensors.batteryPct)
                            .put("charging", sensors.charging)
                            .put("using_test_pattern", _ui.value.usingTestPattern)
                            .put("privacy_mode", privacy.wire)
                            .put("status", view.status.wire)
                            .put("memory_mb", (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024.0 * 1024.0))
                            .put("dropped_infer", pipeline.droppedInfer)
                            .put("backend", view.backend.wire)
                            .put("queue_depth", view.queueDepth)
                            .put("location_hz", sensors.locationHz)
                            .put("timestamp_ns", frame.meta.sensorTimestampNs)
                            .put("location_interpolated", frame.location?.interpolated == true),
                    )
                    if (_ui.value.runMode == RunMode.REALTIME_PERCEPTION_FULL_LOG) {
                        w.writeMasks(frame.meta.sensorTimestampNs, pipeline.lastRoadPolygon, pipeline.lastOccPolygons)
                    }
                    w.writeOverlay(frame.meta.sensorTimestampNs, view.primitives)
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
