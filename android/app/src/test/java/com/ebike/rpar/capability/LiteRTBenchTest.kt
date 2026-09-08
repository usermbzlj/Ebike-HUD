package com.ebike.rpar.capability

import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRTBenchTest {
    @Test
    fun percentileMatchesSortedIndex() {
        val s = listOf(1.0, 2.0, 3.0, 4.0, 100.0)
        assertTrue(LiteRTBench.percentile(s, 50.0) == 3.0)
        assertTrue(LiteRTBench.percentile(s, 95.0) >= 4.0)
    }

    @Test
    fun gemmWindowRecordsItersAndMemory() {
        val w = LiteRTBench.measureGemmWindow(80L, n = 24)
        assertTrue(w.nIters >= 1)
        assertTrue(w.p95Ms >= w.p50Ms)
        assertTrue(w.durationS > 0.0)
        assertTrue(w.memoryEndMb >= 0.0)
        assertTrue(w.durationS < 600.0)
    }
}
