package com.ebike.rpar.sensor

import org.junit.Assert.assertEquals
import org.junit.Test

class SensorStatsTest {
    @Test
    fun percentileP50OnOddList() {
        assertEquals(5.0, percentileMs(listOf(1.0, 5.0, 9.0), 50.0), 1e-9)
    }

    @Test
    fun emptyPercentileIsZero() {
        assertEquals(0.0, percentileMs(emptyList(), 95.0), 1e-9)
    }

    @Test
    fun windowKeepsLastSixty() {
        val w = ArrayList<Double>()
        for (i in 1..65) pushWindow(w, i.toDouble(), 60)
        assertEquals(60, w.size)
        assertEquals(6.0, w.first(), 1e-9)
        assertEquals(65.0, w.last(), 1e-9)
    }
}
