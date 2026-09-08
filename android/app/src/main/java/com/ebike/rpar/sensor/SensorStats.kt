package com.ebike.rpar.sensor

/** CAP-004: 60 s rate window + percentiles from IMU gap lists. */

fun percentileMs(sorted: List<Double>, p: Double): Double {
    if (sorted.isEmpty()) return 0.0
    val i = ((p / 100.0) * (sorted.size - 1)).toInt().coerceIn(0, sorted.lastIndex)
    return sorted[i]
}

fun pushWindow(values: MutableList<Double>, sample: Double, max: Int) {
    values += sample
    while (values.size > max) values.removeAt(0)
}
