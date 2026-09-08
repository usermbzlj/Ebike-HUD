package com.ebike.rpar.recorder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.zip.ZipFile

class SplitZipTest {
    @Test
    fun splitsWhenOverMaxBytes() {
        val src = File.createTempFile("rpar_sess", "d").apply {
            delete()
            mkdirs()
        }
        File(src, "a.bin").writeBytes(ByteArray(1200) { 1 })
        File(src, "b.bin").writeBytes(ByteArray(1200) { 2 })
        val dest = File(src.parentFile, "pack")
        val parts = SplitZip.export(src, dest, maxBytes = 1500)
        assertTrue(parts.size >= 2)
        assertTrue(parts.all { it.name.contains(".part") && it.exists() })
        ZipFile(parts[0]).use { z ->
            assertEquals(1, z.size())
        }
        src.deleteRecursively()
        parts.forEach { it.delete() }
    }
}
