package com.ebike.rpar.capability

/** CAP-001/003: fps list from StreamConfigurationMap min frame duration (ns). */
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
