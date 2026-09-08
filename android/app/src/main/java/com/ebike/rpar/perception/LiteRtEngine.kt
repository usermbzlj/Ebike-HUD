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
            if (probe.observations.isNotEmpty()) probe else h.copy(latencyMs = maxOf(h.latencyMs, probe.latencyMs))
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
    override fun capability(): Map<String, Any> = mapOf(
        "backend" to InferenceBackend.CPU.wire,
        "runtime" to "tflite",
        "model" to modelFile.name,
        "dual_scale" to true,
        "replaceable" to true,
        "outputs" to "RoadObservation",
    )

    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult {
        val far = cfg.model.inputFar
        val near = cfg.model.inputNear
        val t0 = System.nanoTime()
        try {
            if (interpreter.inputTensorCount < 1 || interpreter.outputTensorCount < 1) {
                throw IllegalStateException("no tensors")
            }
            val in0 = interpreter.getInputTensor(0)
            val shape = in0.shape()
            val bytes = in0.numBytes().coerceAtLeast(4)
            val buf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
            buf.rewind()
            val out0 = interpreter.getOutputTensor(0)
            val outBuf = ByteBuffer.allocateDirect(out0.numBytes().coerceAtLeast(4)).order(ByteOrder.nativeOrder())
            interpreter.run(buf, outBuf)
            Log.d(TAG, "LiteRT run ok in=${shape.contentToString()}")
        } catch (t: Throwable) {
            Log.w(TAG, "LiteRT infer skipped: ${t.message}")
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
