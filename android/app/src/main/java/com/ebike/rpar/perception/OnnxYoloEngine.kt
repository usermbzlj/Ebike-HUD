package com.ebike.rpar.perception

import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.ebike.rpar.config.RparConfig
import com.ebike.rpar.model.FrameQualityMap
import com.ebike.rpar.model.InferenceBackend
import com.ebike.rpar.model.PerceptionResult
import com.ebike.rpar.model.RoadObservation
import com.ebike.rpar.model.SynchronizedFrame
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * YOLO-World detect graph via ONNX Runtime (Windows cannot export Ultralytics LiteRT).
 * Same [YoloDetect] decode as the LiteRT path. Failed session → exception, HybridEngine keeps heuristic.
 */
class OnnxYoloEngine(
    private val session: OrtSession,
    private val env: OrtEnvironment,
    private val modelFile: File,
    private val names: List<String>,
    private val imgsz: Int,
    private val inputName: String,
    private val nhwc: Boolean,
    private val confThr: Float = 0.08f,
) : PerceptionEngine {
    override fun capability(): Map<String, Any> = mapOf(
        "backend" to InferenceBackend.CPU.wire,
        "runtime" to "onnxruntime",
        "model" to modelFile.name,
        "dual_scale" to false,
        "replaceable" to true,
        "outputs" to "RoadObservation",
        "replaces_bump" to true,
        "input" to "letterbox_rgb_yolo_nchw",
        "imgsz" to imgsz,
    )

    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult {
        val t0 = System.nanoTime()
        val nPix = imgsz * imgsz * 3
        val rgb = FloatArray(nPix)
        val meta = YoloDetect.fillLetterboxRgb(frame, imgsz, nhwc, rgb)
        val direct = ByteBuffer.allocateDirect(nPix * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        direct.put(rgb)
        direct.rewind()
        val shape = if (nhwc) longArrayOf(1, imgsz.toLong(), imgsz.toLong(), 3)
        else longArrayOf(1, 3, imgsz.toLong(), imgsz.toLong())
        val tensor = OnnxTensor.createTensor(env, direct, shape)
        val outs = session.run(mapOf(inputName to tensor))
        tensor.close()
        val first = outs.get(0)
        val floats: FloatArray
        val outShape: IntArray
        try {
            val ot = first as OnnxTensor
            val info = ot.info as TensorInfo
            outShape = info.shape.map { it.toInt().coerceAtLeast(1) }.toIntArray()
            val buf = ot.floatBuffer
            floats = FloatArray(buf.remaining())
            buf.get(floats)
        } finally {
            outs.close()
        }
        val dets = YoloDetect.decode(floats, outShape, names, confThr)
        val w = frame.meta.width
        val h = frame.meta.height
        val obs = ArrayList<RoadObservation>()
        for (d in dets) {
            val box = YoloDetect.xyxyToOrig(d.x0, d.y0, d.x1, d.y1, meta, w, h)
            val item = YoloDetect.toObservation(frame, d, box, quality) ?: continue
            obs.add(item)
        }
        return PerceptionResult(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            roadPolygon = emptyList(),
            occludedPolygons = emptyList(),
            observations = obs,
            backend = InferenceBackend.CPU,
            latencyMs = (System.nanoTime() - t0) / 1e6,
            inputSizes = listOf(intArrayOf(imgsz, imgsz)),
            dualScale = false,
        )
    }

    override fun close() {
        try { session.close() } catch (_: Throwable) {}
    }

    companion object {
        private const val TAG = "OnnxYoloEngine"

        fun tryLoad(@Suppress("UNUSED_PARAMETER") cfg: RparConfig, file: File): OnnxYoloEngine? {
            if (!file.exists() || file.length() < 1_000_000L) return null
            return try {
                val env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions()
                opts.setIntraOpNumThreads(2)
                val session = env.createSession(file.absolutePath, opts)
                val inputName = session.inputNames.first()
                val tInfo = (session.inputInfo[inputName]?.info as? TensorInfo)
                val shape = tInfo?.shape ?: longArrayOf(1, 3, 320, 320)
                val nhwc = shape.size == 4 && shape[3] == 3L
                val imgsz = when {
                    nhwc -> shape[1].toInt().coerceAtLeast(shape[2].toInt())
                    shape.size == 4 && shape[1] == 3L -> shape[2].toInt().coerceAtLeast(shape[3].toInt())
                    else -> 320
                }
                val names = YoloDetect.readNames(File(file.parentFile, "labels.json"))
                val conf = 0.08f
                OnnxYoloEngine(session, env, file, names, imgsz.coerceAtLeast(160), inputName, nhwc, conf)
            } catch (t: Throwable) {
                Log.i(TAG, "onnx bump graph skipped (${file.name}): ${t.message}")
                null
            }
        }
    }
}
