package com.ebike.rpar.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureResultRingTest {
    @Test
    fun reorderUnmatchedAndExactMatch() {
        val ring = CaptureResultRing(maxDtNs = 2L)
        ring.pushResult(10L, "a")
        ring.pushResult(20L, "b")
        ring.pushResult(30L, "c")
        val m10 = ring.matchImage(10L)
        assertTrue(m10.matched)
        assertEquals("a", m10.payload)
        val m20 = ring.matchImage(20L)
        assertTrue(m20.matched)
        val m15 = ring.matchImage(15L)
        assertTrue(m15.codes.contains("REORDER"))
        assertTrue(m15.codes.contains("UNMATCHED_FRAME"))
        assertFalse(m15.matched)
        val m30 = ring.matchImage(30L)
        assertTrue(m30.matched)
        assertEquals("c", m30.payload)
    }

    @Test
    fun nearestWithinWindowAndDuplicate() {
        val ring = CaptureResultRing(maxDtNs = 2_000_000L)
        ring.pushResult(1_000_000L, "r")
        val hit = ring.matchImage(1_000_500L)
        assertTrue(hit.matched)
        val dup = ring.matchImage(1_000_500L)
        assertTrue(dup.codes.contains("DUPLICATE"))
        assertTrue(dup.codes.contains("UNMATCHED_FRAME"))
    }

    @Test
    fun orphanWhenCapacityExceeded() {
        val ring = CaptureResultRing(maxDtNs = 2L, capacity = 2)
        ring.pushResult(1L, "a")
        ring.pushResult(2L, "b")
        val orphans = ring.pushResult(3L, "c")
        assertTrue(orphans.contains("ORPHAN_RESULT"))
    }
}
