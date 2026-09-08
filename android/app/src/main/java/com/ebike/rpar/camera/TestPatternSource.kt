package com.ebike.rpar.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.os.SystemClock
import com.ebike.rpar.model.FrameMeta
import com.ebike.rpar.model.LocationSample
import com.ebike.rpar.model.PoseSample
import com.ebike.rpar.model.StabilizationMode
import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.model.YuvImageBuffer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.cos
import kotlin.math.sin

/**
 * Generated road scene so the app boots on emulator / REPLAY / SAFE_MODE without a camera.
 */
class TestPatternSource(
    @Suppress("UNUSED_PARAMETER") context: Context,
    private val width: Int = 1280,
    private val height: Int = 720,
    private val fps: Int = 30,
    private val speedMps: Double = 10.5,
) {
    private val asphalt = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    private val frameBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    private var job: Job? = null
    var onFrame: ((SynchronizedFrame) -> Unit)? = null
    private var frameId = 0L
    private val t0 = SystemClock.elapsedRealtimeNanos()

    init {
        paintBase()
    }

    private fun paintBase() {
        val c = Canvas(asphalt)
        c.drawColor(Color.rgb(28, 48, 72))
        val sky = Paint().apply { color = Color.rgb(18, 36, 58) }
        c.drawRect(0f, 0f, width.toFloat(), height * 0.38f, sky)
        val road = Paint().apply { color = Color.rgb(58, 58, 62); isAntiAlias = true }
        val p = Path().apply {
            moveTo(width * 0.08f, height.toFloat())
            lineTo(width * 0.92f, height.toFloat())
            lineTo(width * 0.62f, height * 0.40f)
            lineTo(width * 0.38f, height * 0.40f)
            close()
        }
        c.drawPath(p, road)
        val dash = Paint().apply { color = Color.rgb(200, 200, 180); strokeWidth = 4f }
        for (i in 0..8) {
            val t = 0.42f + i * 0.06f
            val x = width * 0.5f
            c.drawLine(x, height * t, x, height * (t + 0.03f), dash)
        }
    }

    fun start(scope: CoroutineScope) {
        stop()
        job = scope.launch(Dispatchers.Default) {
            val period = 1000L / fps
            while (isActive) {
                val now = SystemClock.elapsedRealtimeNanos()
                val t = (now - t0) / 1e9
                val bmp = render(t)
                val y = yFromBitmap(bmp)
                val meta = FrameMeta(
                    frameId = frameId++,
                    sensorTimestampNs = now,
                    imageTimestampNs = now,
                    exposureTimeNs = null,
                    iso = null,
                    focalLengthMm = null,
                    focusDistanceDiopters = null,
                    afState = null,
                    aeState = null,
                    awbState = null,
                    cropRegion = null,
                    stabilizationMode = StabilizationMode.OFF,
                    width = width,
                    height = height,
                    availability = mapOf(
                        "sensor_timestamp" to true,
                        "exposure" to false,
                        "iso" to false,
                        "focal" to false,
                        "af" to false,
                        "ae" to false,
                        "awb" to false,
                        "crop" to false,
                        "ois" to false,
                        "eis" to false,
                    ),
                )
                val frame = SynchronizedFrame(
                    meta = meta,
                    bitmap = bmp,
                    yuv = y,
                    pose = PoseSample(now, doubleArrayOf(0.0, 0.0, 0.0, 1.0), doubleArrayOf(0.0, 0.0, 9.8), 0.9),
                    angularVelocity = doubleArrayOf(0.0, 0.0, 0.0),
                    linearAccel = doubleArrayOf(0.0, 0.0, 9.8),
                    location = LocationSample(now, null, null, null, speedMps, 0.0, 8.0, 0.4),
                    speedMps = speedMps,
                )
                onFrame?.invoke(frame)
                delay(period)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }

    private fun render(t: Double): Bitmap {
        val c = Canvas(frameBmp)
        c.drawBitmap(asphalt, 0f, 0f, null)
        val ego = speedMps * t
        drawPothole(c, ego, 28.0, 0.15, Color.rgb(16, 16, 16))
        drawCircle(c, ego, 22.0, 1.2, Color.rgb(70, 72, 78))
        drawBump(c, ego, 36.0)
        return frameBmp
    }

    private fun projectY(distM: Double): Float {
        val near = 2.0
        val far = 40.0
        val u = ((distM - near) / (far - near)).coerceIn(0.0, 1.0)
        return (height * (0.92 - u * 0.50)).toFloat()
    }

    private fun projectX(xM: Double, distM: Double): Float {
        val scale = (1.2 / maxOf(distM, 2.0))
        return (width * (0.5 + xM * scale * 0.35)).toFloat()
    }

    private fun drawPothole(c: Canvas, ego: Double, y0: Double, xM: Double, color: Int) {
        val d = y0 - (ego % 80.0)
        if (d < 3 || d > 38) return
        val y = projectY(d)
        val x = projectX(xM, d)
        val s = (80f * (12.0 / d)).toFloat()
        val p = Paint().apply { this.color = color; isAntiAlias = true }
        c.drawOval(x - s, y - s * 0.45f, x + s, y + s * 0.45f, p)
    }

    private fun drawCircle(c: Canvas, ego: Double, y0: Double, xM: Double, color: Int) {
        val d = y0 - (ego % 70.0)
        if (d < 3 || d > 38) return
        val y = projectY(d)
        val x = projectX(xM, d)
        val s = (55f * (10.0 / d)).toFloat()
        val p = Paint().apply { this.color = color; style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true }
        c.drawCircle(x, y, s, p)
    }

    private fun drawBump(c: Canvas, ego: Double, y0: Double) {
        val d = y0 - (ego % 90.0)
        if (d < 4 || d > 38) return
        val y = projectY(d)
        val x0 = projectX(-1.6, d)
        val x1 = projectX(1.6, d)
        val p = Paint().apply { color = Color.rgb(36, 36, 36); strokeWidth = 10f }
        c.drawLine(x0, y, x1, y, p)
        val wobble = (4 * sin(tPhase(ego))).toFloat()
        c.drawLine(x0, y + wobble, x1, y + wobble, p)
    }

    private fun tPhase(ego: Double) = ego * 0.3

    private fun yFromBitmap(bmp: Bitmap): YuvImageBuffer {
        val w = bmp.width; val h = bmp.height
        val argb = IntArray(w * h)
        bmp.getPixels(argb, 0, w, 0, 0, w, h)
        val y = ByteArray(w * h)
        for (i in argb.indices) {
            val c = argb[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            y[i] = ((r * 299 + g * 587 + b * 114) / 1000).toByte()
        }
        return YuvImageBuffer(w, h, y, w)
    }

    @Suppress("unused")
    private fun wobble(a: Double) = cos(a)
}
