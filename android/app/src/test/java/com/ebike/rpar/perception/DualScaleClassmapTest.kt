package com.ebike.rpar.perception

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DualScaleClassmapTest {
    @Test
    fun redPixelPicksFirstClass() {
        val weights = arrayOf(
            doubleArrayOf(4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 4.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        val red = (0xFF shl 16) or (0xFF shl 24)
        val green = (0xFF shl 8) or (0xFF shl 24)
        val labels = DualScaleClassmap.classifyRgb(intArrayOf(red, green), 2, 1, weights)
        assertEquals(0, labels[0])
        assertEquals(1, labels[1])
    }

    @Test
    fun resizeNearestKeepsBlock() {
        val src = intArrayOf(7, 7, 0, 0)
        val dst = DualScaleClassmap.resizeNearest(src, 2, 2, 4, 4)
        assertTrue(dst[0] == 7)
        assertTrue(dst[dst.lastIndex] == 0)
    }
}
