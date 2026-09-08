package com.ebike.rpar.model

/** M5 reserved interface: IMU spike near an object. Never used for alerts. */
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
