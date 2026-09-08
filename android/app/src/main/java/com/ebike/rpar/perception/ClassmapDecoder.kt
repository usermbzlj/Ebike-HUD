package com.ebike.rpar.perception

import com.ebike.rpar.model.FrameQualityMap
import com.ebike.rpar.model.GeometryType
import com.ebike.rpar.model.InferenceBackend
import com.ebike.rpar.model.ObjectState
import com.ebike.rpar.model.PerceptionResult
import com.ebike.rpar.model.RoadObservation
import com.ebike.rpar.model.SemanticType
import com.ebike.rpar.model.Severity
import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.model.VisibilityClass
import com.ebike.rpar.model.bboxOf
import com.ebike.rpar.model.rectPolygon
import com.ebike.rpar.model.rleFromPolygon
import com.ebike.rpar.quality.Blob
import com.ebike.rpar.quality.connectedComponents
import com.ebike.rpar.quality.maskVisibility

/** PER-014: dense class map → RoadObservation. Tensors never leave this helper. */
object ClassmapDecoder {
    const val CLASS_BG = 0
    const val CLASS_ROAD = 1
    const val CLASS_ANOMALY = 2
    const val CLASS_OCC = 3

    fun decode(
        labels: IntArray,
        w: Int,
        h: Int,
        frame: SynchronizedFrame,
        quality: FrameQualityMap?,
        sx: Float = 1f,
        sy: Float = 1f,
        latencyMs: Double = 0.0,
    ): PerceptionResult {
        val minArea = (0.002 * w * h).toInt().coerceAtLeast(8)
        val road = blobsFor(labels, w, h, CLASS_ROAD, minArea)
        val occ = blobsFor(labels, w, h, CLASS_OCC, minArea)
        val anom = blobsFor(labels, w, h, CLASS_ANOMALY, minArea)
        val roadPoly = road.maxByOrNull { it.area }?.let { scaleRect(it, sx, sy) } ?: emptyList()
        val occPolys = occ.map { scaleRect(it, sx, sy) }
        val vis = quality?.globalQuality?.visibilityClass ?: VisibilityClass.UNKNOWN
        val obs = anom.map { b ->
            val poly = scaleRect(b, sx, sy)
            val bbox = bboxOf(poly)
            val qv = if (quality != null) maskVisibility(quality, bbox) else 0.6
            RoadObservation(
                timestampNs = frame.meta.sensorTimestampNs,
                sourceFrameId = frame.meta.frameId,
                semanticType = SemanticType.UNKNOWN_ANOMALY,
                geometryType = GeometryType.UNKNOWN,
                state = ObjectState.UNKNOWN,
                severity = Severity.LIGHT,
                maskRle = rleFromPolygon(poly),
                polygon = poly,
                bbox = bbox,
                modelConfidence = 0.7,
                qualityAtMask = qv.coerceIn(0.0, 1.0),
                visibility = vis,
                calibratedConfidence = 0.64,
            )
        }
        return PerceptionResult(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            roadPolygon = roadPoly,
            occludedPolygons = occPolys,
            observations = obs,
            backend = InferenceBackend.CPU,
            latencyMs = latencyMs,
            inputSizes = listOf(intArrayOf(w, h)),
            dualScale = true,
        )
    }

    fun argmaxNhwc(buf: FloatArray, h: Int, w: Int, c: Int): IntArray {
        val out = IntArray(w * h)
        var i = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                var best = 0
                var bestV = Float.NEGATIVE_INFINITY
                val base = (y * w + x) * c
                for (k in 0 until c) {
                    val v = buf[base + k]
                    if (v > bestV) {
                        bestV = v
                        best = k
                    }
                }
                out[i++] = best
            }
        }
        return out
    }

    private fun blobsFor(labels: IntArray, w: Int, h: Int, cls: Int, minArea: Int): List<Blob> {
        val mask = BooleanArray(w * h) { labels[it] == cls }
        return connectedComponents(mask, w, h, minArea)
    }

    private fun scaleRect(b: Blob, sx: Float, sy: Float) =
        rectPolygon(b.x0 * sx, b.y0 * sy, b.x1 * sx, b.y1 * sy)
}
