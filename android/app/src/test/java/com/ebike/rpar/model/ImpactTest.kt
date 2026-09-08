package com.ebike.rpar.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test
    fun futureAlignScoresPassWindowAndNeverAlerts() {
        val tracks = listOf(
            FutureImpactSeed(1, 0L, "CONFIRMED", "convex", "speed_bump", 10.0, 10.0),
            FutureImpactSeed(2, 0L, "CONFIRMED", "flat", "repair_patch", 25.0, 10.0),
        )
        val accel = (0..80).map { i ->
            val t = i * 50_000_000L
            val z = if (i in 18..22) 9.81 + 8.0 else 9.81
            AccelZ(t, z)
        }
        val rows = alignTracksToFutureImpact(tracks, accel)
        assertEquals(2, rows.size)
        assertTrue(rows.all { !it.usedForAlert })
        val bump = rows.first { it.trackId == 1 }
        val patch = rows.first { it.trackId == 2 }
        assertTrue(bump.peakMs2 > 5.0)
        assertTrue(bump.impactLabel != "none")
        assertTrue(patch.peakMs2 < bump.peakMs2)
    }
}
