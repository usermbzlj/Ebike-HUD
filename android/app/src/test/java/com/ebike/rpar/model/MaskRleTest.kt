package com.ebike.rpar.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class MaskRleTest {
    @Test
    fun polygonProducesNonEmptyRle() {
        val poly = rectPolygon(10f, 20f, 40f, 50f)
        val rle = rleFromPolygon(poly)
        assertNotNull(rle)
        assertTrue(rle!!.width >= 1)
        assertTrue(rle.height >= 1)
        assertTrue(rle.counts.sum() > 0)
        assertTrue(rle.counts.any { it > 0 })
    }
}

class LocationInterpTest {
    @Test
    fun midpointLatitude() {
        val a = LocationSample(0, 31.0, 121.0, 8.0, 10.0, 0.0, 4.0, 0.4)
        val b = LocationSample(1_000_000_000L, 32.0, 122.0, 10.0, 20.0, 20.0, 4.0, 0.4)
        val mid = LocationInterp.at(listOf(a, b), 500_000_000L)
        assertNotNull(mid)
        assertTrue(mid!!.interpolated)
        assertEquals(31.5, mid.latitude!!, 1e-9)
        assertTrue(abs((mid.speedMps ?: 0.0) - 15.0) < 1e-9)
    }
}
