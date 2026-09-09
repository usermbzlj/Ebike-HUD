package com.ebike.rpar.capture

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Size
import android.view.Surface
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.atan

class RideCamera(
    private val context: Context,
    private val preferredHfovDeg: Double = 75.0,
    private val targetW: Int = 1920,
    private val targetH: Int = 1080,
    private val targetFps: Int = 30,
    private val bitrateMbps: Double = 20.0,
    private val segmentSeconds: Int = 300,
) {
    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var previewSurface: Surface? = null
    private var recorder: MediaRecorder? = null
    private var recordSurface: Surface? = null
    private var videoDir: File? = null
    private var segmentIndex = 0
    private var partFile: File? = null
    var size: Size = Size(targetW, targetH)
        private set
    var fps: Int = targetFps
        private set
    var cameraId: String? = null
        private set
    var hfovDeg: Double? = null
        private set
    var lastError: String? = null
        private set
    val running = AtomicBoolean(false)
    private val frameId = AtomicLong(0)
    var onFrameMeta: ((JSONObject) -> Unit)? = null
    var onSegment: (() -> Unit)? = null
    private var chars: CameraCharacteristics? = null

    fun open(surfaceTexture: SurfaceTexture, recordDir: File) {
        close()
        val th = HandlerThread("ride-camera2").also { it.start() }
        thread = th
        handler = Handler(th.looper)
        videoDir = recordDir.also { it.mkdirs() }
        val id = pickRearMain()
        if (id == null) {
            lastError = "no_rear_camera"
            return
        }
        cameraId = id
        val ch = manager.getCameraCharacteristics(id)
        chars = ch
        hfovDeg = estimateHfov(ch)
        val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: run {
            lastError = "no_stream_map"
            return
        }
        val sizes = map.getOutputSizes(SurfaceTexture::class.java) ?: emptyArray()
        size = sizes.minByOrNull { abs(it.width * it.height - targetW * targetH) } ?: Size(targetW, targetH)
        surfaceTexture.setDefaultBufferSize(size.width, size.height)
        previewSurface = Surface(surfaceTexture)
        @SuppressLint("MissingPermission")
        try {
            manager.openCamera(
                id,
                object : CameraDevice.StateCallback() {
                    override fun onOpened(d: CameraDevice) {
                        device = d
                        running.set(true)
                        createSession()
                    }
                    override fun onDisconnected(d: CameraDevice) { close() }
                    override fun onError(d: CameraDevice, error: Int) {
                        lastError = "camera_error_$error"
                        close()
                    }
                },
                handler,
            )
        } catch (t: Throwable) {
            lastError = t.message
        }
    }

    private fun createSession() {
        val d = device ?: return
        try { session?.stopRepeating() } catch (_: Throwable) {}
        try { session?.close() } catch (_: Throwable) {}
        session = null
        prepareRecorder()
        val surfaces = ArrayList<Surface>()
        previewSurface?.let { surfaces += it }
        recordSurface?.let { surfaces += it }
        if (surfaces.isEmpty()) return
        try {
            d.createCaptureSession(
                surfaces,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) {
                        session = s
                        try { recorder?.start() } catch (t: Throwable) { lastError = t.message }
                        startRepeating()
                    }
                    override fun onConfigureFailed(s: CameraCaptureSession) {
                        lastError = "configure_failed"
                    }
                },
                handler,
            )
        } catch (t: Throwable) {
            lastError = t.message
        }
    }

    private fun prepareRecorder() {
        val dir = videoDir ?: return
        try { recorder?.release() } catch (_: Throwable) {}
        recordSurface = null
        val rec = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(context) else MediaRecorder()
        rec.setVideoSource(MediaRecorder.VideoSource.SURFACE)
        rec.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        rec.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
        rec.setVideoSize(size.width, size.height)
        rec.setVideoFrameRate(fps.coerceIn(15, 60))
        rec.setVideoEncodingBitRate((bitrateMbps * 1_000_000).toInt().coerceAtLeast(4_000_000))
        rec.setMaxDuration(segmentSeconds * 1000)
        val part = File(dir, "segment_%03d.part.mp4".format(segmentIndex))
        partFile = part
        rec.setOutputFile(part.absolutePath)
        rec.setOnInfoListener { _, what, _ ->
            if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED) {
                handler?.post { rotate() }
            }
        }
        rec.prepare()
        recordSurface = rec.surface
        recorder = rec
    }

    private fun rotate() {
        finalizeSegment()
        segmentIndex++
        if (running.get() && device != null) createSession()
    }

    private fun finalizeSegment() {
        try { session?.stopRepeating() } catch (_: Throwable) {}
        try { recorder?.stop() } catch (_: Throwable) {}
        try { recorder?.release() } catch (_: Throwable) {}
        recorder = null
        recordSurface = null
        val part = partFile
        if (part != null && part.exists() && part.length() > 1024) {
            val dest = File(part.parentFile, "segment_%03d.mp4".format(segmentIndex))
            part.renameTo(dest)
            onSegment?.invoke()
        }
        partFile = null
    }

    private fun startRepeating() {
        val d = device ?: return
        val s = session ?: return
        val b = d.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        previewSurface?.let { b.addTarget(it) }
        recordSurface?.let { b.addTarget(it) }
        b.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        val ranges = chars?.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: emptyArray()
        val range = ranges.firstOrNull { it.upper >= fps && it.lower <= fps }
            ?: ranges.maxByOrNull { it.upper }
            ?: Range(fps, fps)
        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range)
        s.setRepeatingRequest(
            b.build(),
            object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    session: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult,
                ) {
                    val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
                    onFrameMeta?.invoke(
                        JSONObject()
                            .put("frame_id", frameId.incrementAndGet())
                            .put("sensor_timestamp_ns", ts)
                            .put("image_timestamp_ns", ts)
                            .put("exposure_time_ns", result.get(CaptureResult.SENSOR_EXPOSURE_TIME))
                            .put("iso", result.get(CaptureResult.SENSOR_SENSITIVITY))
                            .put("focal_length_mm", result.get(CaptureResult.LENS_FOCAL_LENGTH)?.toDouble())
                            .put("af_state", result.get(CaptureResult.CONTROL_AF_STATE))
                            .put("ae_state", result.get(CaptureResult.CONTROL_AE_STATE))
                            .put("width", size.width)
                            .put("height", size.height),
                    )
                }
            },
            handler,
        )
    }

    fun pickRearMain(): String? {
        val backs = manager.cameraIdList.map { it to manager.getCameraCharacteristics(it) }
            .filter { it.second.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK }
        if (backs.isEmpty()) return null
        val physical = backs.filter { (_, ch) ->
            (ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf())
                .none { it == CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA }
        }
        val pool = physical.ifEmpty { backs }
        return pool.minByOrNull { abs(estimateHfov(it.second) - preferredHfovDeg) }?.first
    }

    fun close() {
        running.set(false)
        try { session?.close() } catch (_: Throwable) {}
        session = null
        finalizeSegment()
        try { device?.close() } catch (_: Throwable) {}
        device = null
        previewSurface?.release()
        previewSurface = null
        thread?.quitSafely()
        thread = null
        handler = null
    }

    companion object {
        fun estimateHfov(ch: CameraCharacteristics): Double {
            val focals = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS) ?: floatArrayOf(4.5f)
            val sensor = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val f = focals.maxOrNull()?.toDouble() ?: 4.5
            val w = sensor?.width?.toDouble() ?: 6.4
            return Math.toDegrees(2.0 * atan(w / (2.0 * f)))
        }
    }
}
