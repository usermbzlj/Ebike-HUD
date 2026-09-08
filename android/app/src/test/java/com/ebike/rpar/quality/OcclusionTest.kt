package com.ebike.rpar.quality

import com.ebike.rpar.model.PerceptionStatus
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OcclusionTest {
    @Test
    fun coverRatioFromBboxAndBlocksNewTracks() {
        val poly = listOf(
            listOf(0f to 0f, 100f to 0f, 100f to 50f, 0f to 50f),
        )
        val r = occlusionCoverRatio(poly, 100, 100)
        assertTrue(r >= 0.49 && r <= 0.51)
        assertFalse(allowNewObservations(PerceptionStatus.OCCLUDED, usable = true, fresh = true, occlusionRatio = 0.0))
        assertFalse(allowNewObservations(PerceptionStatus.NORMAL, usable = true, fresh = true, occlusionRatio = 0.2))
        assertTrue(allowNewObservations(PerceptionStatus.NORMAL, usable = true, fresh = true, occlusionRatio = 0.01))
    }
}
