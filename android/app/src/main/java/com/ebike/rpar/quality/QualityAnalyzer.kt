package com.ebike.rpar.quality

import com.ebike.rpar.config.QualityConfig
import com.ebike.rpar.model.FrameQuality
import com.ebike.rpar.model.FrameQualityMap
import com.ebike.rpar.model.PerceptionStatus
import com.ebike.rpar.model.QualityTile
import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.model.VisibilityClass
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max

fun motionBlurScore(gray: GrayImage): Double {
    val (mx, my, mag) = gray.sobelMagMean()
    if (mag < 1e-3) return 1.0
    val anisotropy = abs(mx - my) / (mag + 1e-6)
    val lap = gray.laplacianVar()
    val blur = ((18.0 - lap) / 18.0).coerceIn(0.0, 1.0)
    return (0.55 * blur + 0.45 * (anisotropy * 1.8).coerceIn(0.0, 1.0)).coerceIn(0.0, 1.0)
}

fun exposureScores(gray: GrayImage): Pair<Double, Double> {
    val hist = DoubleArray(32)
    gray.px.forEach { hist[(it * 32 / 256).coerceIn(0, 31)] += 1.0 }
    val n = gray.px.size.toDouble().coerceAtLeast(1.0)
    for (i in hist.indices) hist[i] /= n
    val mean = gray.mean() / 255.0
    val under = (hist[0] + hist[1] + hist[2]) * 1.4 + (0.22 - mean) * 2.0
    val over = (hist[30] + hist[31]) * 1.6 + (mean - 0.78) * 2.2
    return under.coerceIn(0.0, 1.0) to over.coerceIn(0.0, 1.0)
}

fun glareScore(gray: GrayImage): Double {
    val h = gray.h
    var bright = 0
    var upper = 0
    var nUpper = 0
    for (y in 0 until h) for (x in 0 until gray.w) {
        val v = gray.at(x, y)
        if (v > 245) {
            bright++
            if (y < h / 2) upper++
        }
        if (y < h / 2) nUpper++
    }
    val u = if (nUpper == 0) 0.0 else upper / nUpper.toDouble()
    val g = bright / gray.px.size.toDouble()
    return (u * 3.5 + g * 1.5).coerceIn(0.0, 1.0)
}

fun defocusScore(gray: GrayImage, lap: Double): Double {
    var edges = 0
    for (y in 1 until gray.h - 1) for (x in 1 until gray.w - 1) {
        if (abs(gray.at(x, y) - gray.at(x + 1, y)) > 18 || abs(gray.at(x, y) - gray.at(x, y + 1)) > 18) edges++
    }
    val dens = edges / gray.px.size.toDouble()
    return if (lap < 8 && dens < 0.04) ((10.0 - lap) / 10.0).coerceIn(0.0, 1.0)
    else ((14.0 - lap) / 28.0).coerceIn(0.0, 1.0) * (1.0 - (dens * 6).coerceIn(0.0, 0.7))
}

fun inRoadTrapezoid(x: Int, y: Int, w: Int, h: Int): Boolean {
    if (y < (h * 0.42).toInt()) return false
    val t = (y - h * 0.42) / (h * 0.58)
    val left = w * (0.38 - t * 0.30)
    val right = w * (0.62 + t * 0.30)
    return x.toDouble() in left..right
}

fun classify(q: FrameQuality, roadVisible: Double, occluded: Double, cfg: QualityConfig): VisibilityClass {
    if (occluded > 0.45) return VisibilityClass.OCCLUDED
    if (q.glare >= cfg.glareBlock) return VisibilityClass.GLARE
    if (q.motionBlur >= cfg.motionBlurBlock || q.sharpness * 80 < cfg.laplacianUsable) return VisibilityClass.BLUR
    if (q.underexposure >= cfg.underexposureBlock) return VisibilityClass.UNDEREXPOSED
    if (q.overexposure >= cfg.overexposureBlock) return VisibilityClass.OVEREXPOSED
    if (roadVisible < cfg.minRoadVisible) return VisibilityClass.UNKNOWN
    return VisibilityClass.CLEAR
}

fun evaluateFrame(gray: GrayImage, fullW: Int, fullH: Int, cfg: QualityConfig): FrameQualityMap {
    val lap = gray.laplacianVar()
    val motion = motionBlurScore(gray)
    val (under, over) = exposureScores(gray)
    val glare = glareScore(gray)
    val defocus = defocusScore(gray, lap)
    var roadN = 0; var roadVis = 0
    for (y in 0 until gray.h) for (x in 0 until gray.w) {
        if (inRoadTrapezoid(x, y, gray.w, gray.h)) {
            roadN++; if (gray.at(x, y) > 18) roadVis++
        }
    }
    val roadVisible = if (roadN == 0) 0.0 else roadVis / roadN.toDouble()
    var edge = 0; var midN = 0
    val y0 = (gray.h * 0.28).toInt(); val y1 = (gray.h * 0.62).toInt()
    for (y in y0 until y1) for (x in 1 until gray.w - 1) {
        midN++
        if (abs(gray.at(x, y) - gray.at(x + 1, y)) > 20) edge++
    }
    val occluded = if (midN == 0) 0.0 else (0.15 + (0.08 - edge / midN.toDouble()) * 4.0).coerceIn(0.0, 1.0)
    val usable = lap >= cfg.laplacianUsable && motion < cfg.motionBlurBlock && glare < cfg.glareBlock &&
        under < cfg.underexposureBlock && over < cfg.overexposureBlock && roadVisible >= cfg.minRoadVisible
    val gq = FrameQuality(
        sharpness = (lap / 80.0).coerceIn(0.0, 1.0),
        motionBlur = motion,
        defocus = defocus,
        underexposure = under,
        overexposure = over,
        glare = glare,
        usable = usable,
        visibilityClass = VisibilityClass.CLEAR,
        roadVisibleRatio = roadVisible,
        reason = if (usable) "" else "low_quality",
    )
    gq.visibilityClass = classify(gq, roadVisible, occluded, cfg)
    val tiles = ArrayList<QualityTile>()
    val tr = cfg.tileRows; val tc = cfg.tileCols
    for (r in 0 until tr) for (c in 0 until tc) {
        val ty0 = r * gray.h / tr; val ty1 = (r + 1) * gray.h / tr
        val tx0 = c * gray.w / tc; val tx1 = (c + 1) * gray.w / tc
        val patch = gray.crop(tx0, ty0, tx1, ty1)
        val plap = patch.laplacianVar()
        val pmean = patch.mean() / 255.0
        var vis = VisibilityClass.CLEAR
        var score = (plap / 60.0).coerceIn(0.0, 1.0)
        if (pmean < 0.12) { vis = VisibilityClass.UNDEREXPOSED; score = minOf(score, 0.25) }
        else if (pmean > 0.9) { vis = VisibilityClass.OVEREXPOSED; score = minOf(score, 0.25) }
        else if (plap < 8) { vis = VisibilityClass.BLUR; score = minOf(score, 0.3) }
        tiles += QualityTile(
            x0 = tx0 * fullW / gray.w, y0 = ty0 * fullH / gray.h,
            x1 = tx1 * fullW / gray.w, y1 = ty1 * fullH / gray.h,
            visibility = vis, score = score,
        )
    }
    val topH = max(1, gray.h / 3)
    val dropMask = BooleanArray(gray.w * topH)
    for (y in 0 until topH) for (x in 0 until gray.w) dropMask[y * gray.w + x] = gray.at(x, y) < 40
    val drops = connectedComponents(dropMask, gray.w, topH, 8).count { b ->
        val ww = b.x1 - b.x0 + 1; val hh = (b.y1 - b.y0 + 1).coerceAtLeast(1)
        b.area in 8..400 && ww.toDouble() / hh in 0.6..1.6
    }
    if (drops >= cfg.lensDropBlobMin) {
        gq.visibilityClass = VisibilityClass.LENS_DROP
        gq.usable = false
        gq.reason = "lens_drop"
    }
    return FrameQualityMap(
        globalQuality = gq,
        tiles = tiles,
        occupancyOccludedRatio = occluded,
        selectedForInfer = gq.usable,
        selectedAgeMs = 0.0,
        degradeReason = if (gq.usable) null else gq.reason.ifEmpty { gq.visibilityClass.wire },
    )
}

fun maskVisibility(quality: FrameQualityMap, bbox: FloatArray): Double {
    val scores = quality.tiles.mapNotNull { t ->
        val ix0 = maxOf(bbox[0], t.x0.toFloat()); val iy0 = maxOf(bbox[1], t.y0.toFloat())
        val ix1 = minOf(bbox[2], t.x1.toFloat()); val iy1 = minOf(bbox[3], t.y1.toFloat())
        if (ix1 > ix0 && iy1 > iy0) t.score else null
    }
    if (scores.isEmpty()) {
        return quality.globalQuality.sharpness * (1.0 - quality.globalQuality.motionBlur)
    }
    return scores.average()
}

fun perceptionStatus(qmap: FrameQualityMap, consecutiveBadS: Double): PerceptionStatus {
    val g = qmap.globalQuality
    if (g.visibilityClass == VisibilityClass.LENS_DROP) return PerceptionStatus.LENS_CONTAMINATION
    if (g.visibilityClass == VisibilityClass.OCCLUDED || qmap.occupancyOccludedRatio > 0.5) return PerceptionStatus.OCCLUDED
    if (consecutiveBadS > 0.45) return PerceptionStatus.PERCEPTION_LIMITED
    if (!g.usable && g.visibilityClass == VisibilityClass.BLUR) return PerceptionStatus.SEVERE_BLUR
    if (g.visibilityClass == VisibilityClass.GLARE ||
        g.visibilityClass == VisibilityClass.UNDEREXPOSED ||
        g.visibilityClass == VisibilityClass.OVEREXPOSED
    ) return PerceptionStatus.DEGRADED_VISIBILITY
    if (!g.usable) return PerceptionStatus.DEGRADED_VISIBILITY
    return PerceptionStatus.NORMAL
}

class QualityScheduler(private val cfg: QualityConfig) {
    private val buf = ArrayDeque<Pair<SynchronizedFrame, FrameQualityMap>>()

    fun push(frame: SynchronizedFrame, qmap: FrameQualityMap) {
        buf.addLast(frame to qmap)
        val now = frame.meta.sensorTimestampNs
        val horizon = (cfg.bufferSeconds * 1e9).toLong()
        while (buf.isNotEmpty() && now - buf.first().first.meta.sensorTimestampNs > horizon) buf.removeFirst()
    }

    fun select(nowNs: Long): Quad {
        if (buf.isEmpty()) error("empty quality buffer")
        val latest = buf.last()
        val maxAge = (cfg.maxSelectedAgeMs * 1e6).toLong()
        for (item in buf.reversed()) {
            val age = nowNs - item.first.meta.sensorTimestampNs
            if (age > maxAge) break
            if (item.second.globalQuality.usable) {
                item.second.selectedForInfer = true
                item.second.selectedAgeMs = age / 1e6
                item.second.degradeReason = null
                return Quad(item.first, item.second, true, age / 1e6)
            }
        }
        val age = nowNs - latest.first.meta.sensorTimestampNs
        latest.second.selectedForInfer = false
        latest.second.selectedAgeMs = age / 1e6
        latest.second.degradeReason = "no_fresh_usable_frame"
        return Quad(latest.first, latest.second, false, age / 1e6)
    }

    val size: Int get() = buf.size
}

data class Quad(
    val frame: SynchronizedFrame,
    val q: FrameQualityMap,
    val fresh: Boolean,
    val ageMs: Double,
)
