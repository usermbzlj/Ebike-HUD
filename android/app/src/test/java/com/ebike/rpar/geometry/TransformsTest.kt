package com.ebike.rpar.geometry

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class TransformsTest {
    @Test
    fun leftHandlebarIsNotScreenThirds() {
        val mount = Transforms.defaultMount()
        assertTrue(mount.lateralOffsetM < 0.0)
        val k = Transforms.defaultIntrinsics()
        val uv = Transforms.projectVehiclePoint(doubleArrayOf(0.0, 12.0, 0.0), mount, k)
        assertNotNull(uv)
        val back = Transforms.pixelToGround(uv!!, mount, k)
        assertNotNull(back)
        assertTrue(abs(back!![1] - 12.0) < 1.5)
    }

    @Test
    fun overlayChainErrorUnder8px() {
        val mount = Transforms.defaultMount()
        val k = Transforms.defaultIntrinsics()
        val err = Transforms.overlayMaxErrorPx(mount, k, 4)
        assertTrue(err <= 8.0)
    }

    @Test
    fun knownDistanceMarkersRecoverPitch() {
        val k = Transforms.defaultIntrinsics()
        val truth = Transforms.defaultMount()
        val y5 = Transforms.projectVehiclePoint(doubleArrayOf(0.0, 5.0, 0.0), truth, k)!![1]
        val y10 = Transforms.projectVehiclePoint(doubleArrayOf(0.0, 10.0, 0.0), truth, k)!![1]
        val y20 = Transforms.projectVehiclePoint(doubleArrayOf(0.0, 20.0, 0.0), truth, k)!![1]
        val skewed = truth.copy(pitchDeg = 14.0, cameraHeightM = 1.35, valid = false)
        val fitted = Transforms.fitMountFromDistanceMarkers(skewed, k, listOf(5.0 to y5, 10.0 to y10, 20.0 to y20))
        assertTrue(abs(fitted.pitchDeg - truth.pitchDeg) < 1.5)
        assertTrue(abs(fitted.cameraHeightM - truth.cameraHeightM) < 0.2)
        assertTrue(fitted.valid)
    }

    @Test
    fun displayCompensateShiftsOnYawRate() {
        val poly = listOf(100f to 200f, 140f to 200f)
        val still = Transforms.displayCompensate(poly, 0.0, 40.0, 1920)
        assertTrue(still == poly)
        val moved = Transforms.displayCompensate(poly, 2.0, 40.0, 1920)
        assertTrue(moved[0].first > poly[0].first)
        assertTrue(moved[0].second == poly[0].second)
    }
}
