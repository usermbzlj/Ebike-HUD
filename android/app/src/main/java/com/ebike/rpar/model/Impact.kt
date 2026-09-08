package com.ebike.rpar.model

/** M5: same-frame spike plus visual→future IMU alignment. Never used for alerts. */

data class FutureImpactSeed(
    val trackId: Int,
    val timestampNs: Long,
    val lifecycle: String,
    val geometryType: String,
    val semanticType: String,
    val distanceM: Double?,
    val speedMps: Double?,
)

data class AccelZ(
    val timestampNs: Long,
    val z: Double,
)

data class FutureImpactRow(
    val trackId: Int,
    val timestampNs: Long,
    val peakMs2: Double,
    val rmsMs2: Double,
    val nImu: Int,
    val impactLabel: String,
    val usedForAlert: Boolean = false,
    val geometryType: String,
    val semanticType: String,
    val expectedPassS: Double?,
)

fun estimateImpactScore(
    linearAccel: DoubleArray?,
    speedMps: Double?,
    distanceM: Double?,
    gravity: Double = 9.81,
    minSpeedMps: Double = 2.0,
    nearM: Double = 6.0,
    spikeG: Double = 3.5,
): Double? {
    if (linearAccel == null || linearAccel.size < 3 || speedMps == null || speedMps < minSpeedMps) return null
    if (distanceM == null || distanceM > nearM) return null
    val az = kotlin.math.abs(linearAccel[2] - gravity)
    if (az < spikeG) return null
    return (az / 12.0).coerceAtMost(1.0)
}

fun impactLabel(peakMs2: Double, speedMps: Double): String {
    val scale = maxOf(speedMps / 10.0, 0.35)
    val adj = peakMs2 / scale
    return when {
        adj < 2.0 -> "none"
        adj < 4.0 -> "weak"
        adj < 7.0 -> "medium"
        else -> "strong"
    }
}

fun alignTracksToFutureImpact(
    tracks: List<FutureImpactSeed>,
    accel: List<AccelZ>,
    horizonS: Double = 3.0,
    gravity: Double = 9.81,
    minSpeedMps: Double = 2.0,
    defaultSpeedMps: Double = 10.0,
): List<FutureImpactRow> {
    val first = LinkedHashMap<Int, FutureImpactSeed>()
    for (t in tracks) {
        if (t.lifecycle != "CONFIRMED" && t.lifecycle != "ALERTED") continue
        if (!first.containsKey(t.trackId)) first[t.trackId] = t
    }
    val sorted = accel.sortedBy { it.timestampNs }
    val out = ArrayList<FutureImpactRow>(first.size)
    for (t in first.values) {
        val raw = t.speedMps
        val spd = if (raw != null && raw >= minSpeedMps) raw else defaultSpeedMps
        if (spd < minSpeedMps) continue
        val tPass = if (t.distanceM != null && spd > 0.0) t.distanceM / spd else null
        val startNs: Long
        val endNs: Long
        if (tPass != null && tPass >= 0.0 && tPass <= horizonS + 0.75) {
            val half = 0.40
            startNs = t.timestampNs + ((maxOf(0.0, tPass - half)) * 1_000_000_000.0).toLong()
            endNs = t.timestampNs + ((tPass + half) * 1_000_000_000.0).toLong()
        } else {
            startNs = t.timestampNs
            endNs = t.timestampNs + (horizonS * 1_000_000_000.0).toLong()
        }
        val zs = ArrayList<Double>()
        for (s in sorted) {
            if (s.timestampNs in startNs..endNs) {
                zs.add(kotlin.math.abs(s.z - gravity))
            }
        }
        var peak = 0.0
        var acc = 0.0
        for (z in zs) {
            if (z > peak) peak = z
            acc += z * z
        }
        val rms = if (zs.isEmpty()) 0.0 else kotlin.math.sqrt(acc / zs.size)
        out.add(
            FutureImpactRow(
                trackId = t.trackId,
                timestampNs = t.timestampNs,
                peakMs2 = peak,
                rmsMs2 = rms,
                nImu = zs.size,
                impactLabel = impactLabel(peak, spd),
                usedForAlert = false,
                geometryType = t.geometryType,
                semanticType = t.semanticType,
                expectedPassS = tPass,
            ),
        )
    }
    return out
}
