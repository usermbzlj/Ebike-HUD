package com.ebike.rpar.perception

import android.graphics.Color
import android.util.Log
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.model.FrameQualityMap
import com.ebike.rpar.model.InferenceBackend
import com.ebike.rpar.model.PerceptionResult
import com.ebike.rpar.model.SynchronizedFrame
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * LiteRT sidecar (M3 / PER-014). Observations stay on [HeuristicEngine] until a mapped
 * RoadObservation graph exists; a failed Interpreter must never blank the HUD.
 */
class HybridEngine(
    private val primary: PerceptionEngine,
    private val litert: LiteRtEngine?,
) : PerceptionEngine {
    override fun capability(): Map<String, Any> {
        val cap = HashMap(primary.capability())
        cap["litert_sidecar"] = litert != null
        if (litert != null) cap.putAll(litert.capability())
        return cap
    }

    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult {
        val h = primary.infer(frame, quality)
        val t = litert ?: return h
        return try {
            val probe = t.infer(frame, quality)
            if (probe.observations.isNotEmpty()) {
                probe.copy(latencyMs = maxOf(h.latencyMs, probe.latencyMs))
            } else {
                val score = t.lastScore
                val obs = if (score == null) h.observations else h.observations.map { o ->
                    val prev = o.calibratedConfidence ?: o.modelConfidence
                    o.copy(calibratedConfidence = (0.55 * prev + 0.45 * score).coerceIn(0.0, 1.0))
                }
                h.copy(
                    observations = obs,
                    roadPolygon = probe.roadPolygon.ifEmpty { h.roadPolygon },
                    occludedPolygons = probe.occludedPolygons.ifEmpty { h.occludedPolygons },
                    latencyMs = maxOf(h.latencyMs, probe.latencyMs),
                )
            }
        } catch (_: Throwable) {
            h
        }
    }

    override fun close() {
        primary.close()
        litert?.close()
    }
}

class LiteRtEngine(
    private val cfg: RparConfig,
    private val interpreter: Interpreter,
    private val modelFile: File,
) : PerceptionEngine {
    @Volatile var lastScore: Double? = null
        private set

    override fun capability(): Map<String, Any> = mapOf(
        "backend" to InferenceBackend.CPU.wire,
        "runtime" to "tflite",
        "model" to modelFile.name,
        "dual_scale" to true,
        "replaceable" to true,
        "outputs" to "RoadObservation",
        "input" to "far_near_features_12_or_nhwc_classmap",
    )

    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult {
        val far = cfg.model.inputFar
        val near = cfg.model.inputNear
        val t0 = System.nanoTime()
        lastScore = null
        var decoded: PerceptionResult? = null
        try {
            if (interpreter.inputTensorCount < 1 || interpreter.outputTensorCount < 1) {
                throw IllegalStateException("no tensors")
            }
            val in0 = interpreter.getInputTensor(0)
            val nIn = in0.shape().fold(1) { a, b -> a * b }
            if (nIn == 12) {
                val feats = DualScaleFeatures.fromFrame(frame)
                val buf = ByteBuffer.allocateDirect(in0.numBytes().coerceAtLeast(48)).order(ByteOrder.nativeOrder())
                feats.forEach { buf.putFloat(it) }
                buf.rewind()
                val out0 = interpreter.getOutputTensor(0)
                val outBuf = ByteBuffer.allocateDirect(out0.numBytes().coerceAtLeast(4)).order(ByteOrder.nativeOrder())
                interpreter.run(buf, outBuf)
                outBuf.rewind()
                val logit = outBuf.float.toDouble()
                lastScore = logit.coerceIn(0.0, 1.0)
                Log.d(TAG, "LiteRT sidecar score=$lastScore")
            } else {
                decoded = runClassmap(frame, quality, in0, t0)
                if (decoded == null) {
                    Log.w(TAG, "LiteRT input elems=$nIn expected 12-D sidecar or NHWC classmap; skip")
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "LiteRT infer skipped: ${t.message}")
            lastScore = null
            decoded = null
        }
        val latency = (System.nanoTime() - t0) / 1e6
        return decoded?.copy(latencyMs = latency) ?: PerceptionResult(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            roadPolygon = emptyList(),
            occludedPolygons = emptyList(),
            observations = emptyList(),
            backend = InferenceBackend.CPU,
            latencyMs = latency,
            inputSizes = listOf(far, near),
            dualScale = true,
        )
    }

    private fun runClassmap(
        frame: SynchronizedFrame,
        quality: FrameQualityMap?,
        in0: Tensor,
        t0: Long,
    ): PerceptionResult? {
        val inShape = in0.shape()
        if (inShape.size != 4) return null
        val ih = inShape[1]
        val iw = inShape[2]
        val ic = inShape[3]
        if (ih < 8 || iw < 8 || ic !in 1..4) return null
        val bmp = frame.bitmap ?: return null
        val scaled = android.graphics.Bitmap.createScaledBitmap(bmp, iw, ih, true)
        val pixels = IntArray(iw * ih)
        scaled.getPixels(pixels, 0, iw, 0, 0, iw, ih)
        val inBuf = ByteBuffer.allocateDirect(in0.numBytes()).order(ByteOrder.nativeOrder())
        for (p in pixels) {
            val r = Color.red(p) / 255f
            val g = Color.green(p) / 255f
            val b = Color.blue(p) / 255f
            when (ic) {
                1 -> inBuf.putFloat((0.299f * r + 0.587f * g + 0.114f * b))
                3 -> {
                    inBuf.putFloat(r); inBuf.putFloat(g); inBuf.putFloat(b)
                }
                else -> {
                    inBuf.putFloat(r); inBuf.putFloat(g); inBuf.putFloat(b); inBuf.putFloat(1f)
                }
            }
        }
        inBuf.rewind()
        val out0 = interpreter.getOutputTensor(0)
        val hw = parseHwc(out0.shape()) ?: return null
        val (oh, ow, oc) = hw
        val outBuf = ByteBuffer.allocateDirect(out0.numBytes().coerceAtLeast(oh * ow * oc * 4)).order(ByteOrder.nativeOrder())
        interpreter.run(inBuf, outBuf)
        outBuf.rewind()
        val fb = outBuf.asFloatBuffer()
        val floats = FloatArray(fb.remaining())
        fb.get(floats)
        val labels = if (oc <= 1) {
            IntArray(ow * oh) { i -> floats[i].toInt() }
        } else {
            ClassmapDecoder.argmaxNhwc(floats, oh, ow, oc)
        }
        val sx = frame.meta.width / ow.toFloat()
        val sy = frame.meta.height / oh.toFloat()
        return ClassmapDecoder.decode(
            labels, ow, oh, frame, quality, sx, sy, (System.nanoTime() - t0) / 1e6,
        )
    }

    override fun close() {
        try { interpreter.close() } catch (_: Throwable) {}
    }

    companion object {
        private const val TAG = "LiteRtEngine"

        private fun parseHwc(shape: IntArray): Triple<Int, Int, Int>? = when (shape.size) {
            4 -> Triple(shape[1], shape[2], shape[3])
            3 -> if (shape[0] == 1) Triple(shape[1], shape[2], 1) else Triple(shape[0], shape[1], shape[2])
            2 -> Triple(shape[0], shape[1], 1)
            else -> null
        }

        fun tryLoad(cfg: RparConfig, file: File): LiteRtEngine? {
            if (!file.exists() || file.length() < 64) return null
            return try {
                val opts = Interpreter.Options().apply { setNumThreads(2) }
                val interp = Interpreter(file, opts)
                if (interp.inputTensorCount < 1) {
                    interp.close()
                    return null
                }
                LiteRtEngine(cfg, interp, file)
            } catch (t: Throwable) {
                Log.i(TAG, "not a runnable LiteRT pack (${file.name}): ${t.message}")
                null
            }
        }
    }
}
