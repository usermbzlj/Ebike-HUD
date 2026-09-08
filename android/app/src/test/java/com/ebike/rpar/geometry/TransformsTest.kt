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
    fun compensateDoesNotDoubleApply() {
        val mount = Transforms.defaultMount()
        val xy = doubleArrayOf(0.4, 10.0)
        val once = Transforms.compensateLeftHandlebar(xy, mount)
        assertTrue(abs(once[0] - (0.4 - mount.lateralOffsetM)) < 1e-9)
    }
}
