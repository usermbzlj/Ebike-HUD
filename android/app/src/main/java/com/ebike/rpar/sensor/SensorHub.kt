package com.ebike.rpar.sensor

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
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import com.ebike.rpar.diagnostics.DiagnosticBus
import com.ebike.rpar.model.ImuSample
import com.ebike.rpar.model.LocationSample
import com.ebike.rpar.model.PoseSample
import com.ebike.rpar.model.SensorType
import java.util.concurrent.ConcurrentLinkedDeque
import kotlin.math.abs

class SensorHub(
    private val context: Context,
    private val diagnostics: DiagnosticBus,
) : SensorEventListener, LocationListener {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    val gyro = ConcurrentLinkedDeque<ImuSample>()
    val accel = ConcurrentLinkedDeque<ImuSample>()
    val rv = ConcurrentLinkedDeque<ImuSample>()
    val poses = ConcurrentLinkedDeque<PoseSample>()
    private val pending = ConcurrentLinkedDeque<ImuSample>()
    @Volatile var lastLocation: LocationSample? = null
    @Volatile var batteryPct: Int = -1
    @Volatile var charging: Boolean = false
    @Volatile var thermalC: Double? = null
    @Volatile var thermalStatus: Int = 0
    private var lastGyroNs = 0L
    private var lastAccelNs = 0L
    private var lastRvNs = 0L
    var gyroHz: Double = 0.0; var accelHz: Double = 0.0; var rvHz: Double = 0.0
    private var gyroCount = 0; private var accelCount = 0; private var rvCount = 0
    private var rateWindowNs = 0L
    private val gyroGapsMs = ArrayList<Double>(512)
    private val accelGapsMs = ArrayList<Double>(512)
    private val rvGapsMs = ArrayList<Double>(512)

    fun intervalJson(): org.json.JSONObject {
        val g = gyroGapsMs.sorted()
        val a = accelGapsMs.sorted()
        val r = rvGapsMs.sorted()
        fun pct(s: List<Double>, p: Double): Double {
            if (s.isEmpty()) return 0.0
            val i = ((p / 100.0) * (s.size - 1)).toInt().coerceIn(0, s.lastIndex)
            return s[i]
        }
        return org.json.JSONObject()
            .put("gyro_hz", gyroHz)
            .put("accel_hz", accelHz)
            .put("rv_hz", rvHz)
            .put("gyro_gap_p5_ms", pct(g, 5.0))
            .put("gyro_gap_p50_ms", pct(g, 50.0))
            .put("gyro_gap_p95_ms", pct(g, 95.0))
            .put("accel_gap_p5_ms", pct(a, 5.0))
            .put("accel_gap_p50_ms", pct(a, 50.0))
            .put("accel_gap_p95_ms", pct(a, 95.0))
            .put("rv_gap_p5_ms", pct(r, 5.0))
            .put("rv_gap_p50_ms", pct(r, 50.0))
            .put("rv_gap_p95_ms", pct(r, 95.0))
            .put("n_gyro", g.size)
            .put("n_accel", a.size)
            .put("n_rv", r.size)
    }

    fun start() {
        val delay = SensorManager.SENSOR_DELAY_FASTEST
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(this, it, delay) }
        sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let { sm.registerListener(this, it, delay) }
        sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.let { sm.registerListener(this, it, delay) }
        try {
            if (lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 200L, 0f, this)
            } else if (lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                lm.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 400L, 0f, this)
            }
        } catch (_: SecurityException) {
            diagnostics.event("SEN", "location_denied", "speed unavailable")
        }
        rateWindowNs = SystemClock.elapsedRealtimeNanos()
    }

    fun stop() {
        sm.unregisterListener(this)
        try { lm.removeUpdates(this) } catch (_: Throwable) {}
    }

    override fun onSensorChanged(event: SensorEvent) {
        val ts = event.timestamp
        detectStall(event.sensor.type, ts)
        unifyAndPush(event, ts)
        val now = SystemClock.elapsedRealtimeNanos()
        if (now - rateWindowNs > 1_000_000_000L) {
            val dt = (now - rateWindowNs) / 1e9
            gyroHz = gyroCount / dt; accelHz = accelCount / dt; rvHz = rvCount / dt
            gyroCount = 0; accelCount = 0; rvCount = 0; rateWindowNs = now
        }
        refreshPower()
    }

    private fun unifyAndPush(event: SensorEvent, ts: Long) {
        val v = event.values
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                gyroCount++
                if (lastGyroNs != 0L) {
                    gyroGapsMs += (ts - lastGyroNs) / 1e6
                    if (gyroGapsMs.size > 4000) gyroGapsMs.removeAt(0)
                }
                lastGyroNs = ts
                push(gyro, ImuSample(ts, SensorType.GYRO, v[0].toDouble(), v[1].toDouble(), v[2].toDouble(), event.accuracy, gyroHz))
            }
            Sensor.TYPE_ACCELEROMETER -> {
                accelCount++
                if (lastAccelNs != 0L) {
                    accelGapsMs += (ts - lastAccelNs) / 1e6
                    if (accelGapsMs.size > 4000) accelGapsMs.removeAt(0)
                }
                lastAccelNs = ts
                push(accel, ImuSample(ts, SensorType.ACCEL, v[0].toDouble(), v[1].toDouble(), v[2].toDouble(), event.accuracy, accelHz))
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                rvCount++
                if (lastRvNs != 0L) {
                    rvGapsMs += (ts - lastRvNs) / 1e6
                    if (rvGapsMs.size > 4000) rvGapsMs.removeAt(0)
                }
                lastRvNs = ts
                val fq = FloatArray(4)
                SensorManager.getQuaternionFromVector(fq, event.values)
                val q = doubleArrayOf(fq[1].toDouble(), fq[2].toDouble(), fq[3].toDouble(), fq[0].toDouble())
                push(rv, ImuSample(ts, SensorType.ROTATION_VECTOR, v[0].toDouble(), v[1].toDouble(), v.getOrElse(2) { 0f }.toDouble(), event.accuracy, rvHz))
                poses.addLast(PoseSample(ts, q, doubleArrayOf(0.0, 0.0, 9.81), event.accuracy / 3.0))
                while (poses.size > 400) poses.pollFirst()
            }
        }
    }

    private fun push(q: ConcurrentLinkedDeque<ImuSample>, s: ImuSample) {
        q.addLast(s)
        pending.addLast(s)
        while (q.size > 800) q.pollFirst()
        while (pending.size > 4000) pending.pollFirst()
    }

    fun drainPending(): List<ImuSample> {
        val out = ArrayList<ImuSample>(64)
        while (true) {
            val s = pending.pollFirst() ?: break
            out += s
        }
        return out
    }

    private fun detectStall(type: Int, ts: Long) {
        val expected = 1_000_000_000L / 200
        val last = when (type) {
            Sensor.TYPE_GYROSCOPE -> lastGyroNs
            Sensor.TYPE_ACCELEROMETER -> lastAccelNs
            else -> lastRvNs
        }
        if (last != 0L && ts - last > expected * 4) {
            diagnostics.event("SYNC", "SENSOR_STALL", "type=$type gap_ms=${(ts - last) / 1e6}")
            diagnostics.lastStallNs = ts
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        lastLocation = LocationSample(
            timestampNs = SystemClock.elapsedRealtimeNanos(),
            latitude = location.latitude,
            longitude = location.longitude,
            altitude = if (location.hasAltitude()) location.altitude else null,
            speedMps = if (location.hasSpeed()) location.speed.toDouble() else null,
            bearingDeg = if (location.hasBearing()) location.bearing.toDouble() else null,
            horizontalAccuracyM = if (location.hasAccuracy()) location.accuracy.toDouble() else null,
            speedAccuracyMps = if (Build.VERSION.SDK_INT >= 26 && location.hasSpeedAccuracy()) location.speedAccuracyMetersPerSecond.toDouble() else null,
        )
    }

    private var lastPowerNs = 0L
    private fun refreshPower() {
        val now = SystemClock.elapsedRealtimeNanos()
        if (now - lastPowerNs < 500_000_000L) return
        lastPowerNs = now
        val ifilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val bat = context.registerReceiver(null, ifilter)
        if (bat != null) {
            val level = bat.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = bat.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
            batteryPct = if (scale > 0) level * 100 / scale else level
            val status = bat.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            charging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL
            val temp = bat.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
            thermalC = temp / 10.0
        }
        if (Build.VERSION.SDK_INT >= 29) {
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            thermalStatus = pm.currentThermalStatus
        }
    }

    fun interpolatePose(tNs: Long): PoseSample? {
        val list = poses.toList()
        if (list.isEmpty()) return null
        if (tNs <= list.first().timestampNs) return list.first()
        if (tNs >= list.last().timestampNs) return list.last()
        var i = 0
        while (i < list.lastIndex && list[i + 1].timestampNs < tNs) i++
        val a = list[i]; val b = list[i + 1]
        val u = (tNs - a.timestampNs).toDouble() / maxOf(1.0, (b.timestampNs - a.timestampNs).toDouble())
        val q = com.ebike.rpar.geometry.Transforms.slerp(a.quaternionXyzw, b.quaternionXyzw, u)
        val g = DoubleArray(3) { a.gravityXyz[it] * (1 - u) + b.gravityXyz[it] * u }
        return PoseSample(tNs, q, g, minOf(a.poseConfidence, b.poseConfidence))
    }

    fun interpolateVec(q: ConcurrentLinkedDeque<ImuSample>, tNs: Long): DoubleArray? {
        val list = q.toList()
        if (list.isEmpty()) return null
        if (tNs <= list.first().timestampNs) return doubleArrayOf(list.first().x, list.first().y, list.first().z)
        if (tNs >= list.last().timestampNs) return doubleArrayOf(list.last().x, list.last().y, list.last().z)
        var i = 0
        while (i < list.lastIndex && list[i + 1].timestampNs < tNs) i++
        val a = list[i]; val b = list[i + 1]
        val u = (tNs - a.timestampNs).toDouble() / maxOf(1.0, (b.timestampNs - a.timestampNs).toDouble())
        return doubleArrayOf(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u)
    }
}

@Suppress("unused")
private fun absUnused(v: Double) = abs(v)
