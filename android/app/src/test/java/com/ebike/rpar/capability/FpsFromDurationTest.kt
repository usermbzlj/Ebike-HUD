package com.ebike.rpar.capability

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

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

    @Test
    fun truncatingToIntWouldDropPkc110SixtyFps() {
        val ns = 16_666_667L
        val truncated = (1_000_000_000.0 / ns).toInt()
        assertEquals(59, truncated)
        assertEquals(60, maxFpsFromMinDurationNs(ns))
        assertEquals(60, pickRequestedFpsFromMinDurationNs(ns, 60, 30))
    }

    @Test
    fun pickFallsBackWhenDurationCannotSupportSixty() {
        assertEquals(30, pickRequestedFpsFromMinDurationNs(33_333_333L, 60, 30))
    }

    @Test
    fun concurrentCombosInclude1080p60WhenDurationAllows() {
        val rows = candidateConcurrentCombos(16_666_667L, 16_666_667L)
        assertEquals(2, rows.size)
        assertEquals("preview+yuv+record@1080p60", rows[0].combo)
        assertEquals(60, rows[0].requestedFps)
        assertEquals("candidate", rows[0].status)
        assertEquals(30, rows[1].requestedFps)
    }

    @Test
    fun measuredFpsFromSixteenMsGaps() {
        val gaps = List(12) { 16.67 }
        val fps = fpsFromIntervalMs(gaps)
        assertNotNull(fps)
        assertTrue(abs(fps!! - 60.0) < 0.5)
    }

    @Test
    fun measuredFpsNeedsEnoughSamples() {
        assertEquals(null, fpsFromIntervalMs(listOf(16.67, 16.67)))
    }
}
