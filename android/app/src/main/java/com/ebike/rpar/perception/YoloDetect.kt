package com.ebike.rpar.perception

import android.graphics.Color
import com.ebike.rpar.model.FrameQualityMap
import com.ebike.rpar.model.GeometryType
import com.ebike.rpar.model.ObjectState
import com.ebike.rpar.model.RoadObservation
import com.ebike.rpar.model.SemanticType
import com.ebike.rpar.model.Severity
import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.model.VisibilityClass
import com.ebike.rpar.model.YuvImageBuffer
import com.ebike.rpar.model.bboxIou
import com.ebike.rpar.model.bboxOf
import com.ebike.rpar.model.ellipsePolygon
import com.ebike.rpar.model.rleFromPolygon
import com.ebike.rpar.quality.maskVisibility
import org.json.JSONObject
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class YoloDet(
    val name: String,
    val cls: Int,
    val conf: Float,
    val x0: Float,
    val y0: Float,
    val x1: Float,
    val y1: Float,
)

data class LetterboxMeta(
    val gain: Float,
    val padX: Float,
    val padY: Float,
    val imgsz: Int,
)

/**
 * Ultralytics YOLO detect decode + handlebar keep-box, matching `rpar.ml.yolo_decode`
 * / `bump_prompts`. Tensors never leave this helper (PER-014).
 */
object YoloDetect {
    val DEFAULT_NAMES = listOf(
        "pothole",
        "large pothole",
        "speed bump",
        "sunken manhole cover",
        "manhole cover",
        "",
    )

    private val promptToClass = mapOf(
        "pothole" to "pothole",
        "large pothole" to "pothole",
        "asphalt pothole" to "pothole",
        "road hole" to "pothole",
        "asphalt hole" to "pothole",
        "speed bump" to "speed_bump",
        "speed_bump" to "speed_bump",
        "speed hump" to "speed_bump",
        "rubber speed bump" to "speed_bump",
        "speedbreaker" to "speed_bump",
        "sunken manhole" to "manhole_cover",
        "sunken manhole cover" to "manhole_cover",
        "settled manhole" to "manhole_cover",
        "manhole cover" to "manhole_cover",
        "manhole_cover" to "manhole_cover",
        "manhole" to "manhole_cover",
    )
    private val sunkenPrompts = setOf("sunken manhole", "sunken manhole cover", "settled manhole")
    val bumpTypes = setOf(SemanticType.POTHOLE, SemanticType.SPEED_BUMP, SemanticType.MANHOLE_COVER)

    fun letterboxMeta(height: Int, width: Int, imgsz: Int): LetterboxMeta {
        val h = max(height, 1)
        val w = max(width, 1)
        val size = max(imgsz, 1)
        val gain = min(size.toFloat() / h, size.toFloat() / w)
        val newW = (w * gain).roundToInt()
        val newH = (h * gain).roundToInt()
        return LetterboxMeta(gain, (size - newW) / 2f, (size - newH) / 2f, size)
    }

    fun xyxyToOrig(x0: Float, y0: Float, x1: Float, y1: Float, meta: LetterboxMeta, width: Int, height: Int): FloatArray {
        val g = if (meta.gain == 0f) 1f else meta.gain
        val a0 = (min(x0, x1) - meta.padX) / g
        val b0 = (min(y0, y1) - meta.padY) / g
        val a1 = (max(x0, x1) - meta.padX) / g
        val b1 = (max(y0, y1) - meta.padY) / g
        val w = max(width, 1)
        val h = max(height, 1)
        return floatArrayOf(
            a0.coerceIn(0f, (w - 1).toFloat()),
            b0.coerceIn(0f, (h - 1).toFloat()),
            a1.coerceIn(1f, w.toFloat()),
            b1.coerceIn(1f, h.toFloat()),
        )
    }

    fun isYoloDetectShape(outShape: IntArray): Boolean {
        val dims = outShape.filter { it > 0 }
        if (dims.size == 3) {
            val n = dims[1]
            val m = dims[2]
            if (m in 6..32 && n in 1..4000) return true
            if (n in 6..32 && m >= 80) return true
        }
        if (dims.size == 2) {
            val n = dims[0]
            val m = dims[1]
            if (m in 6..32 && n in 1..4000) return true
            if (n in 6..32 && m >= 80) return true
        }
        return false
    }

    fun decode(output: FloatArray, outShape: IntArray, names: List<String>, confThr: Float = 0.08f, iouThr: Float = 0.50f): List<YoloDet> {
        val pred = squeezeTo2d(output, outShape) ?: return emptyList()
        val rows = pred.size
        val cols = pred[0].size
        val nc = names.size.coerceAtLeast(1)
        val dets = ArrayList<YoloDet>()
        val nmsLike = (cols == 6 && rows <= 4000) || (rows == 6 && cols > 6)
        if (nmsLike) {
            val table = if (cols == 6) pred else transpose(pred)
            for (row in table) {
                if (row.size < 6) continue
                val conf = row[4]
                if (conf < confThr) continue
                val cls = row[5].roundToInt()
                if (cls < 0 || cls >= names.size) continue
                val name = names[cls]
                if (name.isBlank()) continue
                dets += YoloDet(name, cls, conf, row[0], row[1], row[2], row[3])
            }
        } else {
            val table = when {
                cols == 4 + nc -> pred
                rows == 4 + nc -> transpose(pred)
                cols > rows -> transpose(pred)
                else -> pred
            }
            val ch = table.firstOrNull()?.size ?: return emptyList()
            if (ch < 5) return emptyList()
            for (row in table) {
                var best = 0
                var bestS = row[4]
                for (c in 5 until row.size) {
                    if (row[c] > bestS) {
                        bestS = row[c]
                        best = c - 4
                    }
                }
                if (bestS < confThr) continue
                if (best < 0 || best >= names.size) continue
                val name = names[best]
                if (name.isBlank()) continue
                val xc = row[0]
                val yc = row[1]
                val bw = row[2]
                val bh = row[3]
                dets += YoloDet(name, best, bestS, xc - bw * 0.5f, yc - bh * 0.5f, xc + bw * 0.5f, yc + bh * 0.5f)
            }
        }
        return nms(dets, iouThr)
    }

    fun mapDetName(raw: String): String? {
        val key = raw.trim().lowercase().replace('_', ' ').replace('-', ' ').split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
        if (key == "pothole" || key == "speed bump" || key == "manhole cover") {
            return when (key) {
                "pothole" -> "pothole"
                "speed bump" -> "speed_bump"
                else -> "manhole_cover"
            }
        }
        return promptToClass[key]
    }

    fun keepBox(kind: String, box: FloatArray, width: Int, height: Int): Boolean {
        val bw = max(0f, box[2] - box[0])
        val bh = max(0f, box[3] - box[1])
        if (bw < 12f || bh < 8f) return false
        val cy = 0.5f * (box[1] + box[3])
        val cx = 0.5f * (box[0] + box[2])
        if (cy < 0.16f * height) return false
        if (cx < 0.08f * width || cx > 0.92f * width) return false
        val frac = (bw * bh) / max(1f, width.toFloat() * height)
        if (frac > 0.20f || bw > 0.88f * width) return false
        val aspect = bw / max(bh, 1f)
        return when (kind) {
            "pothole" -> frac in 0.00035f..0.12f && aspect < 3.5f
            "speed_bump" -> frac in 0.0008f..0.12f && (bw >= 0.10f * width || aspect >= 1.6f)
            "manhole_cover" -> frac in 0.00035f..0.08f && aspect in 0.35f..4.2f
            else -> false
        }
    }

    fun toObservation(
        frame: SynchronizedFrame,
        det: YoloDet,
        box: FloatArray,
        quality: FrameQualityMap?,
    ): RoadObservation? {
        val kind = mapDetName(det.name) ?: return null
        val w = frame.meta.width
        val h = frame.meta.height
        if (!keepBox(kind, box, w, h)) return null
        val sunken = sunkenPrompts.contains(norm(det.name))
        var geometry: GeometryType
        var state: ObjectState
        var severity: Severity
        val semantic: SemanticType
        when (kind) {
            "pothole" -> {
                semantic = SemanticType.POTHOLE
                geometry = GeometryType.CONCAVE
                state = ObjectState.ABNORMAL
                severity = Severity.HEAVY
            }
            "speed_bump" -> {
                semantic = SemanticType.SPEED_BUMP
                geometry = GeometryType.CONVEX
                state = ObjectState.ABNORMAL
                severity = Severity.MEDIUM
            }
            else -> {
                semantic = SemanticType.MANHOLE_COVER
                if (sunken) {
                    geometry = GeometryType.CONCAVE
                    state = ObjectState.ABNORMAL
                    severity = Severity.HEAVY
                } else {
                    val refined = refineManhole(frame, box)
                    geometry = refined.first
                    state = refined.second
                    severity = refined.third
                }
            }
        }
        val rx = 0.5f * (box[2] - box[0])
        val ry = 0.5f * (box[3] - box[1])
        val cx = 0.5f * (box[0] + box[2])
        val cy = 0.5f * (box[1] + box[3])
        val poly = ellipsePolygon(cx, cy, max(6f, rx), max(6f, ry))
        val vis = if (quality != null) maskVisibility(quality, box) else 0.65
        val vclass = quality?.globalQuality?.visibilityClass ?: VisibilityClass.CLEAR
        return RoadObservation(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            semanticType = semantic,
            geometryType = geometry,
            state = state,
            severity = severity,
            maskRle = rleFromPolygon(poly),
            polygon = poly,
            bbox = if (poly.isNotEmpty()) bboxOf(poly) else box,
            modelConfidence = det.conf.toDouble().coerceIn(0.0, 1.0),
            qualityAtMask = vis,
            visibility = vclass,
            calibratedConfidence = det.conf.toDouble().coerceIn(0.0, 1.0),
        )
    }

    fun readNames(labels: File?): List<String> {
        if (labels == null || !labels.exists()) return DEFAULT_NAMES
        return try {
            val obj = JSONObject(labels.readText())
            val namesObj = obj.optJSONObject("names")
            if (namesObj != null) {
                val keys = ArrayList<String>()
                val it = namesObj.keys()
                while (it.hasNext()) keys.add(it.next())
                keys.sortBy { it.toIntOrNull() ?: Int.MAX_VALUE }
                val list = keys.map { namesObj.optString(it, "") }
                if (list.isNotEmpty()) list else DEFAULT_NAMES
            } else {
                val arr = obj.optJSONArray("names")
                if (arr != null && arr.length() > 0) {
                    (0 until arr.length()).map { arr.optString(it, "") }
                } else DEFAULT_NAMES
            }
        } catch (_: Throwable) {
            DEFAULT_NAMES
        }
    }

    fun readConf(labels: File?, default: Float = 0.08f): Float {
        if (labels == null || !labels.exists()) return default
        return try {
            val v = JSONObject(labels.readText()).optDouble("conf", default.toDouble())
            v.toFloat().coerceIn(0.01f, 0.9f)
        } catch (_: Throwable) {
            default
        }
    }

    fun fillLetterboxRgb(
        frame: SynchronizedFrame,
        imgsz: Int,
        nhwc: Boolean,
        dst: FloatArray,
        mean114: Boolean = true,
    ): LetterboxMeta {
        val w = frame.meta.width.coerceAtLeast(1)
        val h = frame.meta.height.coerceAtLeast(1)
        val meta = letterboxMeta(h, w, imgsz)
        val pad = if (mean114) 114f / 255f else 0.45f
        java.util.Arrays.fill(dst, pad)
        val newW = (w * meta.gain).roundToInt().coerceAtLeast(1)
        val newH = (h * meta.gain).roundToInt().coerceAtLeast(1)
        val x0 = meta.padX.roundToInt()
        val y0 = meta.padY.roundToInt()
        for (yy in 0 until newH) {
            val sy = ((yy + 0.5f) / meta.gain).toInt().coerceIn(0, h - 1)
            for (xx in 0 until newW) {
                val sx = ((xx + 0.5f) / meta.gain).toInt().coerceIn(0, w - 1)
                val rgb = sampleRgb(frame, sx, sy)
                val dx = (x0 + xx).coerceIn(0, imgsz - 1)
                val dy = (y0 + yy).coerceIn(0, imgsz - 1)
                if (nhwc) {
                    val i = ((dy * imgsz) + dx) * 3
                    dst[i] = rgb[0]
                    dst[i + 1] = rgb[1]
                    dst[i + 2] = rgb[2]
                } else {
                    val pix = dy * imgsz + dx
                    dst[pix] = rgb[0]
                    dst[imgsz * imgsz + pix] = rgb[1]
                    dst[2 * imgsz * imgsz + pix] = rgb[2]
                }
            }
        }
        return meta
    }

    fun sampleRgb(frame: SynchronizedFrame, x: Int, y: Int): FloatArray {
        val bmp = frame.bitmap
        if (bmp != null) {
            val c = bmp.getPixel(x.coerceIn(0, bmp.width - 1), y.coerceIn(0, bmp.height - 1))
            return floatArrayOf(Color.red(c) / 255f, Color.green(c) / 255f, Color.blue(c) / 255f)
        }
        val yuv = frame.yuv ?: return floatArrayOf(0.45f, 0.45f, 0.45f)
        return yuvRgb(yuv, x, y)
    }

    fun yuvRgb(yuv: YuvImageBuffer, x: Int, y: Int): FloatArray {
        val yy = yuv.grayAt(x, y)
        if (yuv.u == null || yuv.v == null) {
            val g = yy / 255f
            return floatArrayOf(g, g, g)
        }
        val u = yuv.chromaU(x, y) - 128
        val v = yuv.chromaV(x, y) - 128
        val r = (yy + 1.402f * v).coerceIn(0f, 255f) / 255f
        val g = (yy - 0.344136f * u - 0.714136f * v).coerceIn(0f, 255f) / 255f
        val b = (yy + 1.772f * u).coerceIn(0f, 255f) / 255f
        return floatArrayOf(r, g, b)
    }

    private fun refineManhole(frame: SynchronizedFrame, box: FloatArray): Triple<GeometryType, ObjectState, Severity> {
        val x0 = box[0].toInt().coerceIn(0, frame.meta.width - 1)
        val y0 = box[1].toInt().coerceIn(0, frame.meta.height - 1)
        val x1 = box[2].toInt().coerceIn(x0 + 1, frame.meta.width)
        val y1 = box[3].toInt().coerceIn(y0 + 1, frame.meta.height)
        val ww = x1 - x0
        val hh = y1 - y0
        if (ww < 4 || hh < 4) return Triple(GeometryType.UNKNOWN, ObjectState.UNKNOWN, Severity.UNKNOWN)
        val cy = hh / 2.0
        val cx = ww / 2.0
        val innerR = max(2.0, 0.28 * min(hh, ww))
        var innerS = 0.0
        var innerN = 0
        var ringS = 0.0
        var ringN = 0
        val innerR2 = innerR * innerR
        val ringR2 = (innerR * 2.05) * (innerR * 2.05)
        for (y in 0 until hh) {
            val dy = y - cy
            for (x in 0 until ww) {
                val dx = x - cx
                val d2 = dx * dx + dy * dy
                val g = grayAt(frame, x0 + x, y0 + y).toDouble()
                when {
                    d2 <= innerR2 -> {
                        innerS += g
                        innerN++
                    }
                    d2 <= ringR2 -> {
                        ringS += g
                        ringN++
                    }
                }
            }
        }
        if (innerN == 0 || ringN == 0) return Triple(GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE)
        val di = innerS / innerN
        val dr = ringS / ringN
        return if (di < dr - 10.0) {
            Triple(GeometryType.CONCAVE, ObjectState.ABNORMAL, Severity.HEAVY)
        } else {
            Triple(GeometryType.FLAT, ObjectState.NORMAL, Severity.NONE)
        }
    }

    private fun grayAt(frame: SynchronizedFrame, x: Int, y: Int): Int {
        val yuv = frame.yuv
        if (yuv != null) return yuv.grayAt(x, y)
        val bmp = frame.bitmap ?: return 80
        val c = bmp.getPixel(x.coerceIn(0, bmp.width - 1), y.coerceIn(0, bmp.height - 1))
        return (Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000
    }

    private fun nms(dets: List<YoloDet>, iouThr: Float): List<YoloDet> {
        val ordered = dets.sortedByDescending { it.conf }
        val keep = ArrayList<YoloDet>()
        for (d in ordered) {
            val box = floatArrayOf(d.x0, d.y0, d.x1, d.y1)
            if (keep.all { bboxIou(box, floatArrayOf(it.x0, it.y0, it.x1, it.y1)) < iouThr }) {
                keep += d
            }
        }
        return keep
    }

    private fun squeezeTo2d(output: FloatArray, outShape: IntArray): Array<FloatArray>? {
        val shape = outShape.filter { it > 0 }
        if (shape.isEmpty() || output.isEmpty()) return null
        val dims = if (shape.size >= 3 && shape[0] == 1) shape.drop(1) else shape
        if (dims.size == 1) {
            return arrayOf(output.copyOf())
        }
        val rows = dims[0]
        val cols = if (dims.size >= 2) dims[1] else output.size / max(rows, 1)
        if (rows <= 0 || cols <= 0 || rows * cols > output.size) return null
        return Array(rows) { r -> FloatArray(cols) { c -> output[r * cols + c] } }
    }

    private fun transpose(m: Array<FloatArray>): Array<FloatArray> {
        val rows = m.size
        val cols = m[0].size
        return Array(cols) { c -> FloatArray(rows) { r -> m[r][c] } }
    }

    private fun norm(raw: String): String =
        raw.trim().lowercase().replace('_', ' ').replace('-', ' ').split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
}
