package com.ebike.rpar.recorder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class SegmentRecoveryTest {
    @Test
    fun recoversPartAsMarkedFile() {
        val dir = File(System.getProperty("java.io.tmpdir"), "rpar-part-${System.nanoTime()}")
        dir.mkdirs()
        try {
            val part = File(dir, "segment_000.part.mp4")
            part.writeText("truncated")
            val out = SegmentRecovery.recoverPartFiles(dir)
            val recovered = File(dir, "segment_000.mp4.recovered")
            assertTrue(recovered.exists())
            assertEquals("truncated", recovered.readText())
            assertFalse(part.exists())
            assertEquals(1, out.size)
            assertEquals(1, SegmentRecovery.nextSegmentIndex(dir))
        } finally {
            dir.deleteRecursively()
        }
    }
}
