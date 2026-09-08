package com.ebike.rpar.sync

import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.sensor.SensorHub

class FrameSynchronizer(private val sensors: SensorHub) {
    fun attach(frame: SynchronizedFrame): SynchronizedFrame {
        val t = frame.meta.sensorTimestampNs
        val pose = sensors.interpolatePose(tNs = t)
        val gyro = sensors.interpolateVec(sensors.gyro, t)
        val accel = sensors.interpolateVec(sensors.accel, t)
        val loc = sensors.lastLocation
        val speed = frame.speedMps ?: loc?.speedMps
        return frame.copy(
            pose = pose ?: frame.pose,
            angularVelocity = gyro ?: frame.angularVelocity,
            linearAccel = accel ?: frame.linearAccel,
            location = loc ?: frame.location,
            speedMps = speed,
        )
    }
}
