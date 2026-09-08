package com.ebike.rpar.perception

import android.graphics.Bitmap
import com.ebike.rpar.model.FrameQualityMap
import com.ebike.rpar.model.InferenceBackend
import com.ebike.rpar.model.PerceptionResult
import com.ebike.rpar.model.SynchronizedFrame
import org.json.JSONObject
import java.io.File

/** Sidecar from `seg_weights.json` (YOLOPv2 teacher distill). Empty output must not blank the HUD. */
class JsonClassmapEngine(
    private val weights: Array<DoubleArray>,
    private val source: File,
) : PerceptionEngine {
    override fun capability(): Map<String, Any> = mapOf(
        "backend" to "dual_scale_classmap",
        "runtime" to "json-lstsq",
        "model" to source.name,
        "dual_scale" to true,
        "replaceable" to true,
        "outputs" to "RoadObservation",
        "tasks" to listOf("drivable_area", "vehicle_occlusion"),
        "not" to listOf("pothole", "manhole", "speed_bump"),
        "tensors_to_ui" to false,
    )

    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult {
        val bmp = frame.bitmap
        val fw = frame.meta.width.coerceAtLeast(1)
        val fh = frame.meta.height.coerceAtLeast(1)
        if (bmp == null) {
            return emptyResult(frame, fw, fh, 0.0)
        }
        val t0 = System.nanoTime()
        val labels = stitch(bmp, fw, fh)
        val decoded = ClassmapDecoder.decode(labels, fw, fh, frame, quality, 1f, 1f, 0.0)
        return decoded.copy(
            backend = InferenceBackend.CPU,
            latencyMs = (System.nanoTime() - t0) / 1e6,
            inputSizes = listOf(
                intArrayOf(DualScaleClassmap.SMALL_W, DualScaleClassmap.SMALL_H),
                intArrayOf(DualScaleClassmap.SMALL_W, DualScaleClassmap.SMALL_H),
            ),
            dualScale = true,
        )
    }

    private fun stitch(bmp: Bitmap, fw: Int, fh: Int): IntArray {
        val canvas = IntArray(fw * fh)
        val rois = listOf(DualScaleRoi.farPx(fw, fh), DualScaleRoi.nearPx(fw, fh))
        for (roi in rois) {
            val rw = (roi.x1 - roi.x0).coerceAtLeast(1)
            val rh = (roi.y1 - roi.y0).coerceAtLeast(1)
            val crop = Bitmap.createBitmap(bmp, roi.x0, roi.y0, rw, rh)
            val small = Bitmap.createScaledBitmap(crop, DualScaleClassmap.SMALL_W, DualScaleClassmap.SMALL_H, true)
            val pixels = IntArray(DualScaleClassmap.SMALL_W * DualScaleClassmap.SMALL_H)
            small.getPixels(pixels, 0, DualScaleClassmap.SMALL_W, 0, 0, DualScaleClassmap.SMALL_W, DualScaleClassmap.SMALL_H)
            val labels = DualScaleClassmap.classifyRgb(
                pixels, DualScaleClassmap.SMALL_W, DualScaleClassmap.SMALL_H, weights,
            )
            ClassmapDecoder.pasteRoi(
                canvas, fw, fh, labels,
                DualScaleClassmap.SMALL_W, DualScaleClassmap.SMALL_H,
                roi.x0, roi.y0, roi.x1, roi.y1,
            )
        }
        return canvas
    }

    private fun emptyResult(frame: SynchronizedFrame, fw: Int, fh: Int, latency: Double) = PerceptionResult(
        timestampNs = frame.meta.sensorTimestampNs,
        sourceFrameId = frame.meta.frameId,
        roadPolygon = emptyList(),
        occludedPolygons = emptyList(),
        observations = emptyList(),
        backend = InferenceBackend.CPU,
        latencyMs = latency,
        inputSizes = listOf(intArrayOf(fw, fh)),
        dualScale = true,
    )

    companion object {
        fun tryLoad(file: File): JsonClassmapEngine? {
            if (!file.exists() || file.length() < 32) return null
            return try {
                val root = JSONObject(file.readText())
                val arr = root.optJSONArray("weights") ?: return null
                if (arr.length() < 2) return null
                val weights = Array(arr.length()) { i ->
                    val row = arr.getJSONArray(i)
                    DoubleArray(row.length()) { j -> row.getDouble(j) }
                }
                JsonClassmapEngine(weights, file)
            } catch (_: Throwable) {
                null
            }
        }
    }
}
