package com.ebike.rpar.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import com.ebike.rpar.config.CameraConfig
import com.ebike.rpar.diagnostics.DiagnosticBus
import com.ebike.rpar.model.FrameMeta
import com.ebike.rpar.model.StabilizationMode
import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.model.YuvImageBuffer
import com.ebike.rpar.recorder.SegmentRecorder
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class CameraController(
    private val context: Context,
    private val cfg: CameraConfig,
    private val diagnostics: DiagnosticBus,
) {
    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var yuvReader: ImageReader? = null
    private var previewSurface: Surface? = null
    private var recorder: SegmentRecorder? = null
    var cameraId: String? = null
        private set
    var transform: CameraTransformChain = CameraTransformChain.identity(cfg.width, cfg.height)
        private set
    var lastChoice: StreamChoice? = null
        private set
    var availability: CaptureAvailability? = null
        private set
    var stabMode: StabilizationMode = StabilizationMode.OFF
    var lockFarFocus: Boolean = false
    var exposureCapNs: Long? = null
    var onFrame: ((SynchronizedFrame) -> Unit)? = null
    var onPreviewSize: ((Size) -> Unit)? = null
    var onSegmentFinalized: (() -> Unit)? = null
    val running = AtomicBoolean(false)
    private val frameId = AtomicLong(0)
    private var lockedPhysical: String? = null
    private var chars: CameraCharacteristics? = null
    var lastDegrade: String? = null
        private set
    var actualFps: Int = cfg.fallbackFps
        private set
    var size: Size = Size(cfg.width, cfg.height)
        private set
    var measuredFps: Double = 0.0
        private set
    var measuredFpsN: Int = 0
        private set
    private var lastYuvTs = 0L
    private val yuvGapsMs = ArrayList<Double>(128)

    fun open(surfaceTexture: SurfaceTexture?, recordDir: File?, enableRecord: Boolean) {
        close()
        val th = HandlerThread("rpar-camera2").also { it.start() }
        thread = th
        handler = Handler(th.looper)
        val id = pickRearMain()
        if (id == null) {
            lastDegrade = "no_rear_camera"
            diagnostics.event("CAM", "no_rear_camera", "opening skipped")
            return
        }
        cameraId = id
        val ch = manager.getCameraCharacteristics(id)
        chars = ch
        availability = CameraMeta.availability(ch)
        val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        if (map == null) {
            lastDegrade = "no_stream_map"
            return
        }
        val picked = SizePicker.pick(map, cfg.width, cfg.height, cfg.targetFps, cfg.fallbackFps)
        size = picked.first
        actualFps = picked.second
        surfaceTexture?.setDefaultBufferSize(size.width, size.height)
        val preview = if (surfaceTexture != null) Surface(surfaceTexture) else null
        previewSurface = preview
        yuvReader = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 4).also { reader ->
            reader.setOnImageAvailableListener({ r ->
                val img = r.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    handleYuv(img)
                } finally {
                    img.close()
                }
            }, handler)
        }
        if (enableRecord && recordDir != null) {
            recorder = SegmentRecorder(context, recordDir, size.width, size.height, actualFps, cfg.bitrateMbps, cfg.segmentSeconds, diagnostics).also { rec ->
                rec.onFinalized = { onSegmentFinalized?.invoke() }
                rec.onRotateRequested = { handler?.post { rotateSegment() } }
            }
        }
        @SuppressLint("MissingPermission")
        try {
            manager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(d: CameraDevice) {
                    device = d
                    running.set(true)
                    createSession(enableRecord)
                }
                override fun onDisconnected(d: CameraDevice) {
                    diagnostics.event("CAM", "disconnected", id)
                    close()
                }
                override fun onError(d: CameraDevice, error: Int) {
                    diagnostics.event("CAM", "error", "code=$error")
                    close()
                }
            }, handler)
        } catch (t: Throwable) {
            lastDegrade = "open_failed:${t.message}"
            diagnostics.event("CAM", "open_failed", t.message ?: "")
        }
    }

    fun recreateForStabilization(mode: StabilizationMode) {
        stabMode = mode
        if (device != null && running.get()) createSession(recorder != null)
    }

    private fun rotateSegment() {
        try { session?.stopRepeating() } catch (_: Throwable) {}
        recorder?.stop()
        if (running.get() && device != null) createSession(true)
    }

    private fun createSession(wantRecord: Boolean) {
        val d = device ?: return
        try { session?.stopRepeating() } catch (_: Throwable) {}
        try { session?.close() } catch (_: Throwable) {}
        session = null
        val surfaces = ArrayList<Surface>()
        previewSurface?.let { surfaces += it }
        yuvReader?.surface?.let { surfaces += it }
        var recordOk = false
        var combo = streamComboLabel(false)
        var reason: String? = null
        val recSurface = if (wantRecord) recorder?.recordSurface() ?: recorder?.prepareSurface() else null
        if (wantRecord && recSurface != null && cfg.preferPreviewPlusYuvPlusRecord) {
            try {
                val all = ArrayList(surfaces).also { it += recSurface }
                d.createCaptureSession(all, sessionCb(all, true), handler)
                recordOk = true
                combo = streamComboLabel(true)
                lastChoice = StreamChoice(size, actualFps, combo, true, null)
                lastDegrade = null
                return
            } catch (t: Throwable) {
                reason = "concurrent_record_failed:${t.message}"
                diagnostics.event("CAM", "combo_degrade", reason)
                recorder?.releaseSurface()
            }
        }
        try {
            d.createCaptureSession(surfaces, sessionCb(surfaces, false), handler)
            combo = streamComboLabel(false)
            lastChoice = StreamChoice(size, actualFps, combo, recordOk, reason)
            lastDegrade = reason
        } catch (t: Throwable) {
            lastDegrade = "session_failed:${t.message}"
            diagnostics.event("CAM", "session_failed", t.message ?: "")
        }
        onPreviewSize?.invoke(size)
        transform = CameraTransformChain(
            cameraId = cameraId ?: "",
            sensorArray = chars?.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE),
            cropRegion = chars?.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE),
            bufferSize = size,
            previewSize = size,
            sensorToBuffer = android.graphics.Matrix(),
            bufferToView = android.graphics.Matrix(),
        )
    }

    private fun sessionCb(surfaces: List<Surface>, recording: Boolean) = object : CameraCaptureSession.StateCallback() {
        override fun onConfigured(s: CameraCaptureSession) {
            session = s
            if (recording) recorder?.start()
            startRepeating()
        }
        override fun onConfigureFailed(s: CameraCaptureSession) {
            diagnostics.event("CAM", "configure_failed", surfaces.size.toString())
            lastDegrade = "configure_failed"
        }
    }

    private fun startRepeating() {
        val d = device ?: return
        val s = session ?: return
        val preview = previewSurface
        val yuv = yuvReader?.surface ?: return
        val b = d.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        preview?.let { b.addTarget(it) }
        b.addTarget(yuv)
        recorder?.recordSurface()?.let { b.addTarget(it) }
        b.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
        val ranges = chars?.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: emptyArray()
        val fpsRange = ranges.firstOrNull { it.upper >= actualFps && it.lower <= actualFps }
            ?: ranges.maxByOrNull { it.upper }
            ?: Range(cfg.fallbackFps, cfg.fallbackFps)
        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)
        b.set(CaptureRequest.CONTROL_AE_LOCK, false)
        b.set(CaptureRequest.CONTROL_AWB_LOCK, false)
        if (lockFarFocus) {
            b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            b.set(CaptureRequest.LENS_FOCUS_DISTANCE, 0.0f)
        } else {
            b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        }
        val cap = exposureCapNs
        if (cap != null) {
            b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, cap)
            b.set(CaptureRequest.SENSOR_SENSITIVITY, 100)
            diagnostics.event("CAM", "exposure_cap", "requested_ns=$cap")
        }
        chars?.let { ch ->
            val (video, ois) = desiredStabMode(stabMode, ch)
            if (video != null) b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, video)
            if (ois != null) b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, ois)
            if (Build.VERSION.SDK_INT >= 33 && stabMode == StabilizationMode.PREVIEW) {
                try {
                    @Suppress("UNCHECKED_CAST")
                    val key = CaptureRequest::class.java
                        .getField("CONTROL_PREVIEW_STABILIZATION_MODE")
                        .get(null) as CaptureRequest.Key<Int>
                    val on = CaptureRequest::class.java
                        .getField("CONTROL_PREVIEW_STABILIZATION_MODE_ON")
                        .getInt(null)
                    b.set(key, on)
                } catch (_: Throwable) { }
            }
        }
        if (cfg.lockPhysicalCamera && Build.VERSION.SDK_INT >= 30) {
            b.set(CaptureRequest.CONTROL_ZOOM_RATIO, 1.0f)
        }
        s.setRepeatingRequest(b.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
                resultRing.pushResult(ts, result).forEach { code ->
                    diagnostics.event("SYNC", code, "ts=$ts")
                }
            }
        }, handler)
    }

    private val resultRing = CaptureResultRing()

    private fun handleYuv(image: Image) {
        val y = copyY(image)
        val imageTs = image.timestamp
        val hit = resultRing.matchImage(imageTs)
        hit.codes.forEach { code ->
            diagnostics.event("SYNC", code, "image_ts=$imageTs")
        }
        val result = hit.payload as? TotalCaptureResult
        val ts = result?.get(CaptureResult.SENSOR_TIMESTAMP)
        val exposure = result?.get(CaptureResult.SENSOR_EXPOSURE_TIME)
        val requested = exposureCapNs
        if (requested != null && exposure != null && exposure != requested) {
            diagnostics.event("CAM", "exposure_mismatch", "requested=$requested actual=$exposure")
        }
        val iso = result?.get(CaptureResult.SENSOR_SENSITIVITY)
        val focal = result?.get(CaptureResult.LENS_FOCAL_LENGTH)
        val focus = result?.get(CaptureResult.LENS_FOCUS_DISTANCE)
        val af = result?.get(CaptureResult.CONTROL_AF_STATE)
        val ae = result?.get(CaptureResult.CONTROL_AE_STATE)
        val awb = result?.get(CaptureResult.CONTROL_AWB_STATE)
        val crop = if (result != null) cropFromResult(result) else null
        val sensorTs = ts ?: imageTs
        noteYuvInterval(sensorTs)
        val meta = FrameMeta(
            frameId = frameId.incrementAndGet(),
            sensorTimestampNs = sensorTs,
            imageTimestampNs = imageTs,
            exposureTimeNs = exposure,
            iso = iso,
            focalLengthMm = focal?.toDouble(),
            focusDistanceDiopters = focus?.toDouble(),
            afState = CameraMeta.afName(af),
            aeState = CameraMeta.aeName(ae),
            awbState = CameraMeta.awbName(awb),
            cropRegion = crop,
            stabilizationMode = stabMode,
            width = image.width,
            height = image.height,
            availability = availabilityFor(result, imageTs),
        )
        onFrame?.invoke(
            SynchronizedFrame(
                meta = meta,
                bitmap = null,
                yuv = y,
                pose = null,
                angularVelocity = null,
                linearAccel = null,
                location = null,
                speedMps = null,
            ),
        )
    }

    private fun noteYuvInterval(sensorTs: Long) {
        if (lastYuvTs > 0L && sensorTs > lastYuvTs) {
            val gap = (sensorTs - lastYuvTs) / 1e6
            if (gap in 1.0..200.0) {
                yuvGapsMs += gap
                if (yuvGapsMs.size > 240) yuvGapsMs.removeAt(0)
                val fps = com.ebike.rpar.capability.fpsFromIntervalMs(yuvGapsMs)
                if (fps != null) measuredFps = fps
                measuredFpsN = yuvGapsMs.size
            }
        }
        lastYuvTs = sensorTs
    }

    private fun availabilityFor(result: TotalCaptureResult?, imageTs: Long): Map<String, Boolean> {
        val map = HashMap<String, Boolean>()
        map["sensor_timestamp"] = result?.get(CaptureResult.SENSOR_TIMESTAMP) != null || imageTs > 0L
        map["exposure"] = result?.get(CaptureResult.SENSOR_EXPOSURE_TIME) != null
        map["iso"] = result?.get(CaptureResult.SENSOR_SENSITIVITY) != null
        map["focal"] = result?.get(CaptureResult.LENS_FOCAL_LENGTH) != null
        map["focus_distance"] = result?.get(CaptureResult.LENS_FOCUS_DISTANCE) != null
        map["af"] = result?.get(CaptureResult.CONTROL_AF_STATE) != null
        map["ae"] = result?.get(CaptureResult.CONTROL_AE_STATE) != null
        map["awb"] = result?.get(CaptureResult.CONTROL_AWB_STATE) != null
        map["crop"] = result?.get(CaptureResult.SCALER_CROP_REGION) != null
        map["ois"] = result?.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE) != null
        map["eis"] = result?.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE) != null
        return map
    }

    private fun copyY(image: Image): YuvImageBuffer {
        val plane = image.planes[0]
        val buf = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val h = image.height
        val arr = ByteArray(rowStride * h)
        buf.rewind()
        val n = minOf(arr.size, buf.remaining())
        buf.get(arr, 0, n)
        return YuvImageBuffer(image.width, h, arr, rowStride)
    }

    private fun streamComboLabel(includeRecord: Boolean): String {
        val streams = when {
            includeRecord -> "preview+yuv+record"
            previewSurface != null -> "preview+yuv"
            else -> "yuv"
        }
        val res = if (size.width == 1920 && size.height == 1080) "1080p$actualFps" else "${size.width}x${size.height}@$actualFps"
        return "$streams@$res"
    }

    fun pickRearMain(): String? {
        val ids = manager.cameraIdList
        val backs = ids.map { it to manager.getCameraCharacteristics(it) }
            .filter { it.second.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK }
        if (backs.isEmpty()) return null
        val physical = backs.filter { (it.second.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf())
            .none { cap -> cap == CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA } }
        val pool = physical.ifEmpty { backs }
        val chosen = pool.minByOrNull { (_, ch) ->
            val focals = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS) ?: floatArrayOf(4f)
            focals.minOrNull() ?: 4f
        } ?: pool.first()
        lockedPhysical = chosen.first
        return chosen.first
    }

    fun close() {
        running.set(false)
        try { session?.close() } catch (_: Throwable) {}
        session = null
        try { device?.close() } catch (_: Throwable) {}
        device = null
        try { yuvReader?.close() } catch (_: Throwable) {}
        yuvReader = null
        previewSurface?.release()
        previewSurface = null
        recorder?.stop()
        recorder = null
        lastYuvTs = 0L
        yuvGapsMs.clear()
        measuredFps = 0.0
        measuredFpsN = 0
        thread?.quitSafely()
        thread = null
        handler = null
    }

    companion object {
        private const val TAG = "CameraController"
        init { Log.d(TAG, "camera2") }
    }
}
