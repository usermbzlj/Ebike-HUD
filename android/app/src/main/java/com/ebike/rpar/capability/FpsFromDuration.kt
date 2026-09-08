package com.ebike.rpar.capability

/** CAP-001/003: fps from StreamConfigurationMap min frame duration (ns). */

fun maxFpsFromMinDurationNs(minDurationNs: Long): Int {
    if (minDurationNs <= 0L) return 30
    val hz = 1_000_000_000.0 / minDurationNs.toDouble()
    return kotlin.math.round(hz).toInt().coerceAtLeast(1)
}

fun fpsCandidatesFromMinDurationNs(minDurationNs: Long): List<Int> {
    if (minDurationNs <= 0L) return listOf(30)
    val hz = 1_000_000_000.0 / minDurationNs.toDouble()
    val maxFps = kotlin.math.round(hz).toInt().coerceAtLeast(1)
    val out = ArrayList<Int>(4)
    if (hz >= 29.5) out.add(30)
    if (hz >= 59.0) out.add(60)
    if (maxFps != 30 && maxFps != 60) out.add(maxFps)
    if (out.isEmpty()) out.add(maxFps)
    return out
}

fun pickRequestedFps(maxFps: Int, targetFps: Int, fallbackFps: Int): Int {
    return when {
        maxFps >= targetFps -> targetFps
        maxFps >= fallbackFps -> fallbackFps
        else -> maxOf(15, maxFps)
    }
}

fun pickRequestedFpsFromMinDurationNs(minDurationNs: Long, targetFps: Int, fallbackFps: Int): Int {
    val candidates = fpsCandidatesFromMinDurationNs(minDurationNs)
    if (targetFps in candidates) return targetFps
    if (fallbackFps in candidates) return fallbackFps
    return (candidates.maxOrNull() ?: 15).coerceAtLeast(15)
}

data class StreamComboCandidate(
    val combo: String,
    val yuvSize: String,
    val recordSize: String,
    val requestedFps: Int,
    val maxFpsFromDuration: Int,
    val status: String,
)

fun candidateConcurrentCombos(yuvMinNs: Long, recordMinNs: Long): List<StreamComboCandidate> {
    val limiting = when {
        yuvMinNs <= 0L && recordMinNs <= 0L -> 0L
        yuvMinNs <= 0L -> recordMinNs
        recordMinNs <= 0L -> yuvMinNs
        else -> maxOf(yuvMinNs, recordMinNs)
    }
    val maxFps = maxFpsFromMinDurationNs(limiting)
    fun row(target: Int, fallback: Int): StreamComboCandidate {
        val req = pickRequestedFpsFromMinDurationNs(limiting, target, fallback)
        val status = when {
            req >= target -> "candidate"
            req >= fallback -> "fps_degraded"
            else -> "fps_limited"
        }
        return StreamComboCandidate(
            combo = "preview+yuv+record@1080p$req",
            yuvSize = "1920x1080",
            recordSize = "1920x1080",
            requestedFps = req,
            maxFpsFromDuration = maxFps,
            status = status,
        )
    }
    val hi = row(60, 30)
    val lo = row(30, 15)
    return if (hi.requestedFps == lo.requestedFps) listOf(hi) else listOf(hi, lo)
}

fun fpsFromIntervalMs(gapsMs: List<Double>): Double? {
    if (gapsMs.size < 5) return null
    val sorted = gapsMs.sorted()
    val med = sorted[sorted.size / 2]
    if (med <= 0.2) return null
    return 1000.0 / med
}
