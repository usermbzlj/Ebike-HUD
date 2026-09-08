package com.ebike.rpar.sensor

/** CAM-012: stop capture on crash / violent vibration, not a typical pothole (~1–2 g). */

fun accelMagnitude(ax: Double, ay: Double, az: Double): Double =
    kotlin.math.sqrt(ax * ax + ay * ay + az * az)

fun linearAccelResidual(ax: Double, ay: Double, az: Double, gravity: Double = 9.81): Double =
    kotlin.math.abs(accelMagnitude(ax, ay, az) - gravity)

fun gyroMagnitude(gx: Double, gy: Double, gz: Double): Double =
    kotlin.math.sqrt(gx * gx + gy * gy + gz * gz)

fun severeImpact(
    ax: Double,
    ay: Double,
    az: Double,
    gyroRadS: Double = 0.0,
    residualThreshold: Double = 40.0,
    gyroThreshold: Double = 15.0,
): Boolean {
    return linearAccelResidual(ax, ay, az) >= residualThreshold || kotlin.math.abs(gyroRadS) >= gyroThreshold
}
