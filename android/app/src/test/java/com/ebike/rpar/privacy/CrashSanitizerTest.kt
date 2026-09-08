package com.ebike.rpar.privacy

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CrashSanitizerTest {
    @Test
    fun stripsGpsAndPaths() {
        val raw = """{"latitude": 31.2304, "path": "C:\\Users\\a\\ride.mp4", "code": "STALL"}"""
        val cleaned = CrashSanitizer.sanitizeText(raw)
        assertFalse(cleaned.contains("31.2304"))
        assertFalse(cleaned.contains(".mp4"))
        assertTrue(cleaned.contains("[path]") || cleaned.contains("""latitude": null"""))
    }
}
