package com.ebike.rpar.geometry

import com.ebike.rpar.model.Intrinsics
import com.ebike.rpar.model.MountProfile
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Coordinate frames (SYNC-005 / GEO-006):
 * - device: Android sensor frame
 * - camera: x right, y down, z forward (optical)
 * - vehicle: x right, y forward, z up (centerline)
 * - road: vehicle XY on Z=0, with left-handlebar lateral offset applied
 */
object Transforms {
    fun deg2rad(d: Double) = d * Math.PI / 180.0

    fun rotX(a: Double): DoubleArray {
        val c = cos(a); val s = sin(a)
        return doubleArrayOf(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
    }

    fun rotY(a: Double): DoubleArray {
        val c = cos(a); val s = sin(a)
        return doubleArrayOf(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    }

    fun rotZ(a: Double): DoubleArray {
        val c = cos(a); val s = sin(a)
        return doubleArrayOf(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    }

    fun mul3(a: DoubleArray, b: DoubleArray): DoubleArray {
        val o = DoubleArray(9)
        for (r in 0..2) for (c in 0..2) {
            o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]
        }
        return o
    }

    fun mulVec(m: DoubleArray, v: DoubleArray): DoubleArray = doubleArrayOf(
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )

    fun transpose(m: DoubleArray) = doubleArrayOf(
        m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8],
    )

    fun quatToRot(q: DoubleArray): DoubleArray {
        var x = q[0]; var y = q[1]; var z = q[2]; var w = q[3]
        val n = sqrt(x * x + y * y + z * z + w * w).coerceAtLeast(1e-9)
        x /= n; y /= n; z /= n; w /= n
        return doubleArrayOf(
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        )
    }

    fun cameraFromVehicle(mount: MountProfile): DoubleArray {
        val pitch = deg2rad(mount.pitchDeg)
        val roll = deg2rad(mount.rollDeg)
        val yaw = deg2rad(mount.yawDeg + mount.handlebarNeutralYawDeg)
        val basis = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        )
        return mul3(rotZ(roll), mul3(rotX(pitch), mul3(rotY(yaw), basis)))
    }

    fun cameraTranslationVehicle(mount: MountProfile): DoubleArray =
        doubleArrayOf(mount.lateralOffsetM, 0.0, mount.cameraHeightM)

    fun projectVehiclePoint(pVehicle: DoubleArray, mount: MountProfile, k: Intrinsics): DoubleArray? {
        val r = cameraFromVehicle(mount)
        val tCam = mulVec(r, cameraTranslationVehicle(mount)).map { -it }.toDoubleArray()
        val pCam = DoubleArray(3) { i ->
            r[i * 3] * pVehicle[0] + r[i * 3 + 1] * pVehicle[1] + r[i * 3 + 2] * pVehicle[2] + tCam[i]
        }
        if (pCam[2] <= 0.15) return null
        val u = k.fx * pCam[0] / pCam[2] + k.cx
        val v = k.fy * pCam[1] / pCam[2] + k.cy
        return doubleArrayOf(u, v)
    }

    fun pixelToGround(uv: DoubleArray, mount: MountProfile, k: Intrinsics): DoubleArray? {
        val x = (uv[0] - k.cx) / k.fx
        val y = (uv[1] - k.cy) / k.fy
        var ray = doubleArrayOf(x, y, 1.0)
        val n = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2])
        if (n < 1e-9) return null
        ray = doubleArrayOf(ray[0] / n, ray[1] / n, ray[2] / n)
        val rVc = cameraFromVehicle(mount)
        val tV = cameraTranslationVehicle(mount)
        val dirV = mulVec(transpose(rVc), ray)
        if (kotlin.math.abs(dirV[2]) < 1e-8) return null
        val scale = -tV[2] / dirV[2]
        if (scale < 0.2) return null
        val p = doubleArrayOf(tV[0] + scale * dirV[0], tV[1] + scale * dirV[1], tV[2] + scale * dirV[2])
        if (!p.all { it.isFinite() } || p[1] < 0.4 || p[1] > 90) return null
        return doubleArrayOf(p[0], p[1])
    }

    fun compensateLeftHandlebar(xyCamGround: DoubleArray, mount: MountProfile): DoubleArray =
        doubleArrayOf(xyCamGround[0] - mount.lateralOffsetM, xyCamGround[1])

    fun overlayMaxErrorPx(mount: MountProfile, k: Intrinsics, n: Int = 5): Double {
        var maxErr = 0.0
        val xs = DoubleArray(n) { i -> -1.6 + 3.2 * i / (n - 1).coerceAtLeast(1) }
        val ys = DoubleArray(n) { i -> 5.0 + 23.0 * i / (n - 1).coerceAtLeast(1) }
        for (x in xs) for (y in ys) {
            val uv = projectVehiclePoint(doubleArrayOf(x, y, 0.0), mount, k) ?: continue
            val xy = pixelToGround(uv, mount, k) ?: continue
            val uv2 = projectVehiclePoint(doubleArrayOf(xy[0], xy[1], 0.0), mount, k) ?: continue
            val dx = uv[0] - uv2[0]
            val dy = uv[1] - uv2[1]
            val e = sqrt(dx * dx + dy * dy)
            if (e > maxErr) maxErr = e
        }
        return maxErr
    }

    fun projectedHorizonY(mount: MountProfile, k: Intrinsics): Double? {
        val uv = projectVehiclePoint(doubleArrayOf(0.0, 80.0, 0.0), mount, k) ?: return null
        return uv[1]
    }

    fun fitMountFromDistanceMarkers(
        mount: MountProfile,
        k: Intrinsics,
        markers: List<Pair<Double, Double>>,
    ): MountProfile {
        if (markers.size < 2) return mount
        var bestPitch = mount.pitchDeg
        var bestHeight = mount.cameraHeightM
        var bestErr = Double.POSITIVE_INFINITY
        var pitch = mount.pitchDeg - 8.0
        while (pitch <= mount.pitchDeg + 8.0 + 1e-6) {
            var height = maxOf(0.55, mount.cameraHeightM - 0.45)
            val hMax = mount.cameraHeightM + 0.45
            while (height <= hMax + 1e-6) {
                val trial = mount.copy(pitchDeg = pitch, cameraHeightM = height)
                var err = 0.0
                var ok = true
                for ((distM, yPx) in markers) {
                    val uv = projectVehiclePoint(doubleArrayOf(0.0, distM, 0.0), trial, k)
                    if (uv == null) {
                        ok = false
                        break
                    }
                    val d = uv[1] - yPx
                    err += d * d
                }
                if (ok && err < bestErr) {
                    bestErr = err
                    bestPitch = pitch
                    bestHeight = height
                }
                height += 0.05
            }
            pitch += 0.5
        }
        return mount.copy(pitchDeg = bestPitch, cameraHeightM = bestHeight, valid = true)
    }

    fun defaultIntrinsics(width: Int = 1920, height: Int = 1080, hfovDeg: Double = 68.0): Intrinsics {
        val fx = (width / 2.0) / tan(deg2rad(hfovDeg) / 2.0)
        return Intrinsics(fx, fx, width / 2.0, height / 2.0, width, height, true)
    }

    fun defaultMount(width: Int = 1920, height: Int = 1080): MountProfile {
        val k = defaultIntrinsics(width, height)
        return MountProfile(
            profileId = "left_handlebar_v1",
            name = "Left handlebar landscape 1x main",
            cameraId = "rear_main",
            landscape = true,
            cameraHeightM = 1.12,
            pitchDeg = 18.0,
            rollDeg = 0.0,
            yawDeg = 0.0,
            lateralOffsetM = -0.32,
            handlebarNeutralYawDeg = 0.0,
            nearReferenceM = 3.0,
            horizonYPx = k.cy * 0.42,
            vehicleCenterlineXPx = width * 0.52,
            valid = true,
        )
    }

    fun slerp(q0: DoubleArray, q1in: DoubleArray, u: Double): DoubleArray {
        var q1 = q1in.copyOf()
        var dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
        if (dot < 0) {
            q1 = q1.map { -it }.toDoubleArray(); dot = -dot
        }
        if (dot > 0.9995) {
            val o = DoubleArray(4) { q0[it] + u * (q1[it] - q0[it]) }
            val n = sqrt(o.sumOf { it * it }).coerceAtLeast(1e-9)
            return o.map { it / n }.toDoubleArray()
        }
        val theta = kotlin.math.acos(dot.coerceIn(-1.0, 1.0))
        val so = sin(theta)
        val a = sin((1 - u) * theta) / so
        val b = sin(u * theta) / so
        val o = DoubleArray(4) { a * q0[it] + b * q1[it] }
        val n = sqrt(o.sumOf { it * it }).coerceAtLeast(1e-9)
        return o.map { it / n }.toDoubleArray()
    }
}
