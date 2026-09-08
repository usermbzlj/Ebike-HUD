package com.ebike.rpar.capability

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FpsFromDurationTest {
    @Test
    fun sixtyFpsFrom16msMinDuration() {
        val fps = fpsCandidatesFromMinDurationNs(16_666_667L)
        assertTrue(fps.contains(30))
        assertTrue(fps.contains(60))
    }

    @Test
    fun unknownDurationFallsBackTo30() {
        assertEquals(listOf(30), fpsCandidatesFromMinDurationNs(0L))
    }
}
