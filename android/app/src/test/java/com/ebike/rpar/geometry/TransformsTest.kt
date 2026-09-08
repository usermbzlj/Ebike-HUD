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
}
