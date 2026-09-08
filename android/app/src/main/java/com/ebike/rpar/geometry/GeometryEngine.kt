package com.ebike.rpar.geometry

import com.ebike.rpar.config.GeometryConfig
import com.ebike.rpar.model.Direction
import com.ebike.rpar.model.Intrinsics
import com.ebike.rpar.model.MountProfile
import com.ebike.rpar.model.groundContact
import com.ebike.rpar.tracking.TrackInternal
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot

class GeometryEngine(
    var mount: MountProfile,
    val cfg: GeometryConfig,
    var k: Intrinsics = Transforms.defaultIntrinsics(),
) {
    var valid: Boolean = mount.valid
    var invalidReason: String? = if (mount.valid) null else "no_mount_profile"
    private val distHist = HashMap<Int, ArrayDeque<Pair<Long, Double>>>()
    private val ttcHist = HashMap<Int, ArrayDeque<Double>>()

    fun setMount(mount: MountProfile, intrinsics: Intrinsics? = null) {
        this.mount = mount
        if (intrinsics != null) k = intrinsics
        valid = mount.valid
        invalidReason = if (mount.valid) null else "no_mount_profile"
    }

    fun healthCheck(pitchErrDeg: Double = 0.0, rollErrDeg: Double = 0.0): Boolean {
        val ok = abs(pitchErrDeg) <= cfg.pitchHealthDeg && abs(rollErrDeg) <= cfg.rollHealthDeg
        if (!ok) {
            valid = false
            invalidReason = "install_health_fail"
        } else if (mount.valid) {
            valid = true
            invalidReason = null
        }
        return valid
    }

    fun contactToRoad(uv: Pair<Float, Float>): DoubleArray? {
        val xy = Transforms.pixelToGround(doubleArrayOf(uv.first.toDouble(), uv.second.toDouble()), mount, k)
            ?: return null
        if (!xy.all { it.isFinite() } || xy[1] < 0.4 || xy[1] > 80) return null
        return xy
    }

    fun distanceForTrack(tr: TrackInternal, nowNs: Long): Triple<Double?, Double, Boolean> {
        if (!valid) return Triple(null, 0.0, false)
        val contact = if (tr.polygon.isNotEmpty()) groundContact(tr.polygon) else tr.mean[0].toFloat() to tr.mean[1].toFloat()
        val road = contactToRoad(contact) ?: return Triple(null, 0.0, false)
        tr.roadXy = road
        val dist = maxOf(0.5, road[1])
        distHist.getOrPut(tr.trackId) { ArrayDeque() }.apply {
            addLast(nowNs to dist)
            while (size > 12) removeFirst()
        }
        var conf = when {
            dist > 30 -> 0.45
            dist > 20 -> 0.62
            else -> 0.82
        }
        if (tr.qualityAtMask < 0.4) conf *= 0.7
        return Triple(dist, conf.coerceIn(0.0, 1.0), true)
    }

    fun ttc(tr: TrackInternal, speedMps: Double?, nowNs: Long): Double? {
        val hist = distHist[tr.trackId] ?: return null
        if (hist.size < 3) return null
        if (speedMps == null || speedMps < cfg.minSpeedForTtcMps) return null
        val t0 = hist.first().first
        val ts = hist.map { (it.first - t0) / 1e9 }
        val ds = hist.map { it.second }
        if (ts.last() < 0.08) return null
        var sumT = 0.0; var sumD = 0.0; var sumTT = 0.0; var sumTD = 0.0
        ts.indices.forEach { i ->
            sumT += ts[i]; sumD += ds[i]; sumTT += ts[i] * ts[i]; sumTD += ts[i] * ds[i]
        }
        val n = ts.size.toDouble()
        val denom = n * sumTT - sumT * sumT
        val slope = if (abs(denom) < 1e-9) 0.0 else (n * sumTD - sumT * sumD) / denom
        val closing = -slope
        if (closing < 0.4) return null
        val fused = 0.55 * closing + 0.45 * speedMps
        val raw = ds.last() / maxOf(fused, 0.2)
        ttcHist.getOrPut(tr.trackId) { ArrayDeque() }.apply {
            addLast(raw)
            while (size > 8) removeFirst()
        }
        val vals = ttcHist[tr.trackId]!!.sorted()
        return vals[vals.size / 2]
    }

    fun directionFor(tr: TrackInternal): Direction {
        val road = tr.roadXy ?: return Direction.UNKNOWN
        val x = road[0]
        var widthM = 0.0
        if (tr.polygon.isNotEmpty() && valid) {
            val xs = tr.polygon.mapNotNull { contactToRoad(it)?.get(0) }
            if (xs.isNotEmpty()) widthM = xs.max() - xs.min()
        }
        val hw = cfg.corridorHalfWidthM
        if (widthM >= cfg.acrossMinWidthM && minOf(abs(x), hw) < hw) return Direction.ACROSS
        if (abs(x) <= hw * 0.72) return Direction.CENTER_FRONT
        return if (x < 0) Direction.LEFT_FRONT else Direction.RIGHT_FRONT
    }

    fun pathRelevance(tr: TrackInternal): Double {
        val road = tr.roadXy ?: return 0.2
        val x = road[0]; val y = road[1]
        val hw = cfg.corridorHalfWidthM
        val lat = exp(-0.5 * (x / (hw * 1.15)).let { it * it })
        val along = ((40.0 - y) / 40.0).coerceIn(0.0, 1.0)
        return (0.75 * lat + 0.25 * along).coerceIn(0.0, 1.0)
    }

    fun displayDistance(dist: Double?, validFlag: Boolean, confidence: Double): String? {
        if (!validFlag || dist == null || !valid) return null
        return when {
            dist < 10 -> "${dist.toInt()} m"
            dist <= 30 -> "约 ${((dist / 2.0).toInt() * 2)} m"
            confidence < 0.5 -> "远处"
            else -> "约 ${dist.toInt()} m"
        }
    }

    fun projectCorridorPixels(): List<Pair<Float, Float>> {
        val hw = cfg.corridorHalfWidthM
        val pts = listOf(
            doubleArrayOf(-hw, 2.0, 0.0),
            doubleArrayOf(-hw, 28.0, 0.0),
            doubleArrayOf(hw, 28.0, 0.0),
            doubleArrayOf(hw, 2.0, 0.0),
        )
        val pix = ArrayList<Pair<Float, Float>>(4)
        for (p in pts) {
            val uv = Transforms.projectVehiclePoint(p, mount, k) ?: return emptyList()
            pix += uv[0].toFloat() to uv[1].toFloat()
        }
        return pix
    }

    companion object {
        fun shouldMarkPassed(tr: TrackInternal, dist: Double?, prevDist: Double?, nearM: Double): Boolean {
            if (dist == null) return false
            if (dist < nearM && tr.mean[3] > 40) return true
            if (prevDist != null && dist > prevDist + 4.0 && dist < 8.0) return true
            if (tr.roadXy != null && tr.roadXy!![1] < 1.6) return true
            return false
        }
    }
}
