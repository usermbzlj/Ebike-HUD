package com.ebike.rpar.quality

import com.ebike.rpar.model.PerceptionStatus

fun occlusionCoverRatio(polys: List<List<Pair<Float, Float>>>, width: Int, height: Int): Double {
    if (polys.isEmpty() || width < 1 || height < 1) return 0.0
    var area = 0.0
    for (poly in polys) {
        if (poly.size < 3) continue
        var minX = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in poly) {
            if (p.first < minX) minX = p.first
            if (p.first > maxX) maxX = p.first
            if (p.second < minY) minY = p.second
            if (p.second > maxY) maxY = p.second
        }
        area += (maxX - minX).coerceAtLeast(0f) * (maxY - minY).coerceAtLeast(0f)
    }
    return (area / (width.toDouble() * height)).coerceAtMost(1.0)
}

fun allowNewObservations(
    status: PerceptionStatus,
    usable: Boolean,
    fresh: Boolean,
    occlusionRatio: Double = 0.0,
    occBlock: Double = 0.40,
): Boolean {
    if (!fresh || !usable) return false
    if (occlusionRatio >= occBlock) return false
    return status != PerceptionStatus.SEVERE_BLUR &&
        status != PerceptionStatus.PERCEPTION_LIMITED &&
        status != PerceptionStatus.LENS_CONTAMINATION
}
