package com.ebike.rpar.model

import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ImpactTest {
    @Test
    fun reservedInterfaceOnlyFiresOnNearSpike() {
        assertNull(estimateImpactScore(null, 10.0, 3.0))
        assertNull(estimateImpactScore(doubleArrayOf(0.0, 0.0, 9.81), 10.0, 3.0))
        assertNull(estimateImpactScore(doubleArrayOf(0.0, 0.0, 20.0), 10.0, 20.0))
        val s = estimateImpactScore(doubleArrayOf(0.0, 0.0, 9.81 + 8.0), 10.0, 3.0)
        assertTrue(s != null && s > 0.0 && s <= 1.0)
    }
}
