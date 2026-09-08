package com.ebike.rpar.perception

import com.ebike.rpar.config.RparConfig
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
import com.ebike.rpar.model.ellipsePolygon
import com.ebike.rpar.model.nmsPolygons
import com.ebike.rpar.model.rectPolygon
import com.ebike.rpar.model.rleFromPolygon
import com.ebike.rpar.quality.GrayImage
import com.ebike.rpar.quality.connectedComponents
import com.ebike.rpar.quality.inRoadTrapezoid
import com.ebike.rpar.quality.maskVisibility
import com.ebike.rpar.quality.morphologyOpen
import kotlin.math.abs
import kotlin.system.measureNanoTime

interface PerceptionEngine {
    suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap? = null): PerceptionResult
    fun capability(): Map<String, Any>
    fun close() {}
}

class NoOpEngine : PerceptionEngine {
    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult =
        PerceptionResult(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            roadPolygon = emptyList(),
            occludedPolygons = emptyList(),
            observations = emptyList(),
            backend = InferenceBackend.HEURISTIC,
            latencyMs = 0.0,
            inputSizes = listOf(intArrayOf(768, 384), intArrayOf(640, 480)),
            dualScale = true,
        )

    override fun capability() = mapOf("backend" to "HEURISTIC", "disabled" to true)
}

class HeuristicEngine(private val cfg: RparConfig) : PerceptionEngine {
    val modelVersion: String = cfg.model.packageId
    var skipFarRoi: Boolean = false

    override fun capability(): Map<String, Any> = mapOf(
        "backend" to InferenceBackend.HEURISTIC.wire,
        "dual_scale" to true,
        "input_far" to cfg.model.inputFar.toList(),
        "input_near" to cfg.model.inputNear.toList(),
        "replaceable" to true,
        "outputs" to "RoadObservation",
    )

    override suspend fun infer(frame: SynchronizedFrame, quality: FrameQualityMap?): PerceptionResult {
        val fw = frame.meta.width.coerceAtLeast(1)
        val fh = frame.meta.height.coerceAtLeast(1)
        val farW = cfg.model.inputFar[0]; val farH = cfg.model.inputFar[1]
        val nearW = cfg.model.inputNear[0]; val nearH = cfg.model.inputNear[1]
        var observations: List<RoadObservation>
        var roadPoly: List<Pair<Float, Float>>
        var occ: List<List<Pair<Float, Float>>>
        val dtNs = measureNanoTime {
            val full = GrayImage.fromFrameFull(frame.yuv, frame.bitmap, minOf(fw, 960), minOf(fh, 540))
            val sx = fw / full.w.toFloat(); val sy = fh / full.h.toFloat()
            val farBox = DualScaleRoi.farPx(full.w, full.h)
            val nearBox = DualScaleRoi.nearPx(full.w, full.h)
            val far = full.crop(farBox.x0, farBox.y0, farBox.x1, farBox.y1).resize(farW, farH)
            val near = full.crop(nearBox.x0, nearBox.y0, nearBox.x1, nearBox.y1).resize(nearW, nearH)
            val road = roadMask(full)
            occ = occlusionPolygons(full, sx, sy)
            val farObs = if (skipFarRoi) emptyList() else detectOnRoi(
                frame, far, quality, farBox.x0, farBox.y0, full.w, full.h, sx, sy, occ,
            )
            val nearObs = detectOnRoi(frame, near, quality, nearBox.x0, nearBox.y0, full.w, full.h, sx, sy, occ)
            val fullObs = detectBlobs(frame, full, road, quality, occ, sx, sy) +
                detectCircles(frame, full, road, quality, sx, sy) +
                detectBumps(frame, full, road, quality, sx, sy) +
                detectRough(frame, full, road, quality, sx, sy)
            val all = farObs + nearObs + fullObs
            val keep = nmsPolygons(all.map { it.polygon to it.modelConfidence }, 0.4)
            observations = keep.map { all[it] }
            roadPoly = maskToPoly(road, full.w, full.h, sx, sy)
        }
        return PerceptionResult(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            roadPolygon = roadPoly,
            occludedPolygons = occ,
            observations = observations,
            backend = InferenceBackend.HEURISTIC,
            latencyMs = dtNs / 1e6,
            inputSizes = listOf(intArrayOf(farW, farH), intArrayOf(nearW, nearH)),
            dualScale = !skipFarRoi,
        )
    }

    private fun detectOnRoi(
        frame: SynchronizedFrame,
        roi: GrayImage,
        quality: FrameQualityMap?,
        ox: Int,
        oy: Int,
        fullW: Int,
        fullH: Int,
        sx: Float,
        sy: Float,
        occ: List<List<Pair<Float, Float>>>,
    ): List<RoadObservation> {
        val road = BooleanArray(roi.w * roi.h) { i ->
            val x = i % roi.w; val y = i / roi.w
            inRoadTrapezoid(x, y, roi.w, roi.h) && roi.px[i] in 25..210
        }
        val mappedOcc = occ.map { poly ->
            poly.map { (x, y) -> ((x / sx) - ox).toFloat() to ((y / sy) - oy).toFloat() }
        }
        fun lift(obs: RoadObservation): RoadObservation {
            val poly = obs.polygon.map { (x, y) -> ((x + ox) * sx) to ((y + oy) * sy) }
            return obs.copy(polygon = poly, bbox = bboxOf(poly))
        }
        return (detectBlobs(frame, roi, road, quality, mappedOcc, 1f, 1f) +
            detectCircles(frame, roi, road, quality, 1f, 1f) +
            detectBumps(frame, roi, road, quality, 1f, 1f)).map(::lift)
    }

    private fun roadMask(g: GrayImage): BooleanArray {
        val m = BooleanArray(g.w * g.h)
        for (y in 0 until g.h) for (x in 0 until g.w) {
            val v = g.at(x, y)
            m[y * g.w + x] = inRoadTrapezoid(x, y, g.w, g.h) && v in 40..200
        }
        return morphologyOpen(m, g.w, g.h, 3)
    }

    private fun occlusionPolygons(g: GrayImage, sx: Float, sy: Float): List<List<Pair<Float, Float>>> {
        val y0 = (g.h * 0.22).toInt(); val y1 = (g.h * 0.62).toInt()
        val mask = BooleanArray(g.w * g.h)
        for (y in y0 until y1) for (x in 1 until g.w - 1) {
            val v = g.at(x, y)
            val contrast = abs(g.at(x, y) - g.at(x + 1, y))
            mask[y * g.w + x] = v in 60..200 && contrast < 8
        }
        val blobs = connectedComponents(morphologyOpen(mask, g.w, g.h, 5), g.w, g.h, (0.012 * g.w * g.h).toInt())
        return blobs.filter { (it.y1 - it.y0) > 20 && (it.x1 - it.x0) > 20 }.map { b ->
            rectPolygon(b.x0 * sx, b.y0 * sy, b.x1 * sx, b.y1 * sy)
        }
    }

    private fun detectBlobs(
        frame: SynchronizedFrame,
        g: GrayImage,
        road: BooleanArray,
        quality: FrameQualityMap?,
        occ: List<List<Pair<Float, Float>>>,
        sx: Float,
        sy: Float,
    ): List<RoadObservation> {
        val blur = g.boxBlur(5)
        var roadMean = 80.0; var n = 0
        for (i in road.indices) if (road[i]) { roadMean += g.px[i]; n++ }
        if (n > 0) roadMean /= n
        val thr = maxOf(20, (roadMean * 0.55).toInt())
        val dark = BooleanArray(g.w * g.h)
        for (i in dark.indices) dark[i] = road[i] && blur.px[i] < thr
        val opened = morphologyOpen(dark, g.w, g.h, 3)
        val blobs = connectedComponents(opened, g.w, g.h, 80)
        val out = ArrayList<RoadObservation>()
        for (b in blobs) {
            val area = b.area
            if (area > 0.08 * g.w * g.h) continue
            val ar = (b.x1 - b.x0 + 1).toDouble() / maxOf(1, b.y1 - b.y0)
            if (ar > 4.5 || ar < 0.25) continue
            val bbox = floatArrayOf(b.x0 * sx, b.y0 * sy, b.x1 * sx, b.y1 * sy)
            if (insideOcc(bbox, occ)) continue
            val poly = ellipsePolygon(b.cx * sx, b.cy * sy, (b.x1 - b.x0) * 0.5f * sx, (b.y1 - b.y0) * 0.5f * sy)
            val (sem, geo, sev, conf) = if (b.circularity > 0.62) {
                QuadT(SemanticType.POTHOLE, GeometryType.CONCAVE, Severity.MEDIUM, 0.72 + 0.2 * b.circularity)
            } else {
                QuadT(SemanticType.UNKNOWN_ANOMALY, GeometryType.CONCAVE, Severity.LIGHT, 0.55)
            }
            out += makeObs(frame, poly, sem, geo, ObjectState.ABNORMAL, sev, conf, quality)
        }
        return out
    }

    private fun detectCircles(
        frame: SynchronizedFrame,
        g: GrayImage,
        road: BooleanArray,
        quality: FrameQualityMap?,
        sx: Float,
        sy: Float,
    ): List<RoadObservation> {
        val blur = g.boxBlur(5)
        val dark = BooleanArray(g.w * g.h)
        for (i in dark.indices) dark[i] = road[i] && blur.px[i] < 110
        val blobs = connectedComponents(morphologyOpen(dark, g.w, g.h, 3), g.w, g.h, 40)
        val out = ArrayList<RoadObservation>()
        for (b in blobs) {
            if (b.cy < g.h * 0.38f) continue
            if (b.circularity < 0.55) continue
            val r = maxOf(b.x1 - b.x0, b.y1 - b.y0) / 2f
            if (r < 6 || r > 80) continue
            var stdAcc = 0.0; var n = 0; var mean = 0.0
            for (y in b.y0..b.y1) for (x in b.x0..b.x1) {
                val v = g.at(x, y).toDouble(); mean += v; stdAcc += v * v; n++
            }
            if (n == 0) continue
            mean /= n
            val std = kotlin.math.sqrt(maxOf(0.0, stdAcc / n - mean * mean))
            val poly = ellipsePolygon(b.cx * sx, b.cy * sy, r * 1.05f * sx, r * 0.75f * sy)
            val abnormal = std > 18
            out += makeObs(
                frame, poly,
                SemanticType.MANHOLE_COVER,
                if (abnormal) GeometryType.CONCAVE else GeometryType.FLAT,
                if (abnormal) ObjectState.ABNORMAL else ObjectState.NORMAL,
                if (abnormal) Severity.MEDIUM else Severity.NONE,
                (0.55 + std / 80.0).coerceIn(0.0, 0.93),
                quality,
            )
        }
        return out
    }

    private fun detectBumps(
        frame: SynchronizedFrame,
        g: GrayImage,
        road: BooleanArray,
        quality: FrameQualityMap?,
        sx: Float,
        sy: Float,
    ): List<RoadObservation> {
        val energy = DoubleArray(g.h)
        for (y in 0 until g.h) {
            var s = 0.0; var n = 0
            for (x in 1 until g.w - 1) if (road[y * g.w + x]) {
                s += abs(g.at(x, y) - g.at(x + 1, y)); n++
            }
            energy[y] = if (n == 0) 0.0 else s / n
        }
        val out = ArrayList<RoadObservation>()
        var y = (g.h * 0.48).toInt()
        while (y < g.h - 4) {
            if (energy[y] > 12) {
                var y1 = y
                while (y1 < g.h - 1 && energy[y1] > 8) y1++
                if (y1 - y >= 2) {
                    var x0 = g.w; var x1 = 0
                    for (yy in y..y1) for (x in 0 until g.w) if (road[yy * g.w + x]) {
                        x0 = minOf(x0, x); x1 = maxOf(x1, x)
                    }
                    if (x1 - x0 > g.w * 0.22) {
                        val poly = rectPolygon(x0 * sx, y * sy, x1 * sx, (y1 + 4) * sy)
                        out += makeObs(
                            frame, poly, SemanticType.SPEED_BUMP, GeometryType.CONVEX,
                            ObjectState.ABNORMAL, Severity.MEDIUM, 0.7, quality,
                        )
                    }
                }
                y = y1 + 4
            } else y++
        }
        return out
    }

    private fun detectRough(
        frame: SynchronizedFrame,
        g: GrayImage,
        road: BooleanArray,
        quality: FrameQualityMap?,
        sx: Float,
        sy: Float,
    ): List<RoadObservation> {
        val lap = DoubleArray(g.w * g.h)
        for (y in 1 until g.h - 1) for (x in 1 until g.w - 1) {
            val v = (g.at(x, y - 1) + g.at(x - 1, y) + g.at(x + 1, y) + g.at(x, y + 1) - 4 * g.at(x, y)).toDouble()
            lap[y * g.w + x] = v * v
        }
        val roadVals = lap.filterIndexed { i, _ -> road[i] }.sorted()
        if (roadVals.isEmpty()) return emptyList()
        val thr = roadVals[(roadVals.size * 0.92).toInt().coerceIn(0, roadVals.lastIndex)]
        val hot = BooleanArray(g.w * g.h) { i -> road[i] && lap[i] > thr }
        val blobs = connectedComponents(morphologyOpen(hot, g.w, g.h, 5), g.w, g.h, 80)
        return blobs.filter { it.area in 80..(0.12 * g.w * g.h).toInt() }.map { b ->
            val poly = rectPolygon(b.x0 * sx, b.y0 * sy, b.x1 * sx, b.y1 * sy)
            makeObs(frame, poly, SemanticType.ROUGH_BROKEN, GeometryType.ROUGH, ObjectState.ABNORMAL, Severity.LIGHT, 0.58, quality)
        }
    }

    private fun insideOcc(bbox: FloatArray, occ: List<List<Pair<Float, Float>>>): Boolean {
        val cx = 0.5f * (bbox[0] + bbox[2]); val cy = 0.5f * (bbox[1] + bbox[3])
        return occ.any { poly -> pointInPoly(cx, cy, poly) }
    }

    private fun pointInPoly(x: Float, y: Float, poly: List<Pair<Float, Float>>): Boolean {
        var inside = false
        var j = poly.lastIndex
        for (i in poly.indices) {
            val xi = poly[i].first; val yi = poly[i].second
            val xj = poly[j].first; val yj = poly[j].second
            if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi + 1e-6f) + xi) inside = !inside
            j = i
        }
        return inside
    }

    private fun maskToPoly(mask: BooleanArray, w: Int, h: Int, sx: Float, sy: Float): List<Pair<Float, Float>> {
        val blobs = connectedComponents(mask, w, h, 40)
        val b = blobs.maxByOrNull { it.area } ?: return emptyList()
        return rectPolygon(b.x0 * sx, b.y0 * sy, b.x1 * sx, b.y1 * sy)
    }

    private fun makeObs(
        frame: SynchronizedFrame,
        poly: List<Pair<Float, Float>>,
        sem: SemanticType,
        geo: GeometryType,
        state: ObjectState,
        sev: Severity,
        conf: Double,
        quality: FrameQualityMap?,
    ): RoadObservation {
        val bbox = bboxOf(poly)
        val vis = if (quality != null) maskVisibility(quality, bbox) else 0.6
        return RoadObservation(
            timestampNs = frame.meta.sensorTimestampNs,
            sourceFrameId = frame.meta.frameId,
            semanticType = sem,
            geometryType = geo,
            state = state,
            severity = sev,
            maskRle = rleFromPolygon(poly),
            polygon = poly,
            bbox = bbox,
            modelConfidence = conf.coerceIn(0.0, 1.0),
            qualityAtMask = vis.coerceIn(0.0, 1.0),
            visibility = quality?.globalQuality?.visibilityClass ?: VisibilityClass.UNKNOWN,
            calibratedConfidence = (conf * 0.92).coerceIn(0.0, 1.0),
        )
    }

    private data class QuadT(val a: SemanticType, val b: GeometryType, val c: Severity, val d: Double)
}
