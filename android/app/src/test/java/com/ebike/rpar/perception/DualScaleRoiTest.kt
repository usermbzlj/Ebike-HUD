package com.ebike.rpar.perception

import com.ebike.rpar.quality.GrayImage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class DualScaleRoiTest {
    @Test
    fun farPolygonScalesWithFrame() {
        val small = DualScaleRoi.farPolygon(640, 360)
        val large = DualScaleRoi.farPolygon(1920, 1080)
        assertTrue(abs(small[0].first - large[0].first) > 1f)
        assertEquals(1920 * 0.18f, large[0].first, 0.01f)
        assertEquals(1080 * 0.32f, large[0].second, 0.01f)
        val near = DualScaleRoi.nearPolygon(640, 360)
        assertEquals(640 * 0.08f, near[0].first, 0.01f)
        assertEquals(360f, near[2].second, 0.01f)
    }
}

class DualScaleFeaturesTest {
    @Test
    fun constantPatchMatchesPythonNorm() {
        val g = GrayImage(48, 24, IntArray(48 * 24) { 128 })
        val f = DualScaleFeatures.patchFeatures(g)
        assertEquals(6, f.size)
        assertEquals(128.0 / 255.0, f[0].toDouble(), 1e-6)
        assertEquals(0.0, f[1].toDouble(), 1e-6)
        assertEquals(0.0, f[2].toDouble(), 1e-6)
        assertEquals(0.0, f[3].toDouble(), 1e-6)
    }
}
