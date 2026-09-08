package com.ebike.rpar.perception

import android.util.Log
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.model.FrameQualityMap
import com.ebike.rpar.model.InferenceBackend
import com.ebike.rpar.model.PerceptionResult
import com.ebike.rpar.model.SynchronizedFrame
import org.tensorflow.lite.Interpreter
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
                h.copy(observations = obs, latencyMs = maxOf(h.latencyMs, probe.latencyMs))
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
        "input" to "far_near_features_12",
    )

    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult {
        val far = cfg.model.inputFar
        val near = cfg.model.inputNear
        val t0 = System.nanoTime()
        lastScore = null
        try {
            if (interpreter.inputTensorCount < 1 || interpreter.outputTensorCount < 1) {
                throw IllegalStateException("no tensors")
            }
            val in0 = interpreter.getInputTensor(0)
            val nIn = in0.shape().fold(1) { a, b -> a * b }
            if (nIn != 12) {
                Log.w(TAG, "LiteRT input elems=$nIn expected 12; skip sidecar")
            } else {
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
                Log.d(TAG, "LiteRT run ok score=$lastScore")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "LiteRT infer skipped: ${t.message}")
            lastScore = null
        }
        return PerceptionResult(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            roadPolygon = emptyList(),
            occludedPolygons = emptyList(),
            observations = emptyList(),
            backend = InferenceBackend.CPU,
            latencyMs = (System.nanoTime() - t0) / 1e6,
            inputSizes = listOf(far, near),
            dualScale = true,
        )
    }

    override fun close() {
        try { interpreter.close() } catch (_: Throwable) {}
    }

    companion object {
        private const val TAG = "LiteRtEngine"

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
