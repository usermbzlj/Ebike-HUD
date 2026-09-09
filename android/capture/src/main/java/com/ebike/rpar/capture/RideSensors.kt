package com.ebike.rpar.capture

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.BatteryManager
import android.os.Bundle
import org.json.JSONObject
import java.util.concurrent.ConcurrentLinkedDeque
import kotlin.math.abs
import kotlin.math.sqrt

class RideSensors(private val context: Context) : SensorEventListener, LocationListener {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val pending = ConcurrentLinkedDeque<JSONObject>()
    private val locPending = ConcurrentLinkedDeque<JSONObject>()

    @Volatile var gyroHz = 0.0
    @Volatile var accelHz = 0.0
    @Volatile var rvHz = 0.0
    @Volatile var hasFix = false
    @Volatile var speedMps: Double? = null
    @Volatile var batteryPct = -1
    @Volatile var charging = false
    @Volatile var gravity = floatArrayOf(0f, 0f, 9.81f)
    @Volatile var lastResidual = 0.0

    private var gCount = 0
    private var aCount = 0
    private var rCount = 0
    private var windowNs = 0L

    fun start() {
        listOf(
            Sensor.TYPE_GYROSCOPE,
            Sensor.TYPE_ACCELEROMETER,
            Sensor.TYPE_ROTATION_VECTOR,
            Sensor.TYPE_GRAVITY,
        ).forEach { type ->
            sm.getDefaultSensor(type)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        }
        @SuppressLint("MissingPermission")
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 200L, 0.5f, this)
        } catch (_: Throwable) {
        }
        refreshBattery()
    }

    fun stop() {
        sm.unregisterListener(this)
        try {
            lm.removeUpdates(this)
        } catch (_: Throwable) {
        }
    }

    fun drainImu(): List<JSONObject> {
        val out = ArrayList<JSONObject>(pending.size)
        while (true) {
            val s = pending.poll() ?: break
            out += s
        }
        return out
    }

    fun drainLocation(): List<JSONObject> {
        val out = ArrayList<JSONObject>()
        while (true) {
            val s = locPending.poll() ?: break
            out += s
        }
        return out
    }

    override fun onSensorChanged(event: SensorEvent) {
        val ts = event.timestamp
        when (event.sensor.type) {
            Sensor.TYPE_GRAVITY -> {
                gravity = floatArrayOf(event.values[0], event.values[1], event.values[2])
                pending += JSONObject()
                    .put("timestamp_ns", ts)
                    .put("sensor_type", "GRAVITY")
                    .put("x", event.values[0].toDouble())
                    .put("y", event.values[1].toDouble())
                    .put("z", event.values[2].toDouble())
            }
            Sensor.TYPE_ACCELEROMETER -> {
                aCount++
                val ax = event.values[0]
                val ay = event.values[1]
                val az = event.values[2]
                val g = gravity
                val gn = sqrt((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).toDouble()).coerceAtLeast(1.0)
                val along = (ax * g[0] + ay * g[1] + az * g[2]) / gn
                lastResidual = abs(along - gn)
                pending += JSONObject()
                    .put("timestamp_ns", ts)
                    .put("sensor_type", "ACCEL")
                    .put("x", ax.toDouble())
                    .put("y", ay.toDouble())
                    .put("z", az.toDouble())
                    .put("vertical_residual", lastResidual)
                tickRates(ts)
            }
            Sensor.TYPE_GYROSCOPE -> {
                gCount++
                pending += JSONObject()
                    .put("timestamp_ns", ts)
                    .put("sensor_type", "GYRO")
                    .put("x", event.values[0].toDouble())
                    .put("y", event.values[1].toDouble())
                    .put("z", event.values[2].toDouble())
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                rCount++
                pending += JSONObject()
                    .put("timestamp_ns", ts)
                    .put("sensor_type", "ROTATION_VECTOR")
                    .put("x", event.values[0].toDouble())
                    .put("y", event.values[1].toDouble())
                    .put("z", event.values[2].toDouble())
            }
        }
        if (pending.size > 8_000) pending.poll()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        hasFix = location.accuracy < 40f
        speedMps = if (location.hasSpeed()) location.speed.toDouble() else null
        locPending += JSONObject()
            .put("timestamp_ns", location.elapsedRealtimeNanos)
            .put("latitude", location.latitude)
            .put("longitude", location.longitude)
            .put("altitude", if (location.hasAltitude()) location.altitude else JSONObject.NULL)
            .put("speed_mps", speedMps)
            .put("bearing_deg", if (location.hasBearing()) location.bearing.toDouble() else JSONObject.NULL)
            .put("horizontal_accuracy_m", location.accuracy.toDouble())
            .put("interpolated", false)
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}

    fun refreshBattery() {
        val i = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        if (i != null) {
            val level = i.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = i.getIntExtra(BatteryManager.EXTRA_SCALE, 100).coerceAtLeast(1)
            batteryPct = (level * 100) / scale
            val st = i.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            charging = st == BatteryManager.BATTERY_STATUS_CHARGING || st == BatteryManager.BATTERY_STATUS_FULL
        }
    }

    private fun tickRates(now: Long) {
        if (windowNs == 0L) windowNs = now
        val dt = (now - windowNs) / 1e9
        if (dt >= 1.0) {
            gyroHz = gCount / dt
            accelHz = aCount / dt
            rvHz = rCount / dt
            gCount = 0
            aCount = 0
            rCount = 0
            windowNs = now
        }
    }
}
