package com.ebike.rpar.perception

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloDetectTest {
    @Test
    fun letterbox1080pHasSidePadsOnHeight() {
        val meta = YoloDetect.letterboxMeta(1080, 1920, 320)
        assertEquals(320f / 1920f, meta.gain, 1e-5f)
        assertEquals(0f, meta.padX, 1e-3f)
        assertTrue(meta.padY >= 69f)
    }

    @Test
    fun decodeNmsKeepsSunkenCoverAndDropsOverlap() {
        val names = listOf("pothole", "speed bump", "sunken manhole cover")
        val out = FloatArray(8 * 6)
        // row 0
        out[0] = 100f; out[1] = 120f; out[2] = 180f; out[3] = 200f; out[4] = 0.91f; out[5] = 2f
        // overlapping weaker
        out[6] = 102f; out[7] = 122f; out[8] = 178f; out[9] = 198f; out[10] = 0.40f; out[11] = 2f
        val dets = YoloDetect.decode(out, intArrayOf(1, 8, 6), names, 0.08f)
        assertEquals(1, dets.size)
        assertEquals("sunken manhole cover", dets[0].name)
        assertEquals("manhole_cover", YoloDetect.mapDetName(dets[0].name))
    }

    @Test
    fun decodeRawChannelsFirst() {
        val names = listOf("pothole", "speed bump", "sunken manhole cover")
        val n = 16
        val ch = 7
        val out = FloatArray(ch * n)
        val i = 3
        out[0 * n + i] = 160f
        out[1 * n + i] = 200f
        out[2 * n + i] = 80f
        out[3 * n + i] = 40f
        out[6 * n + i] = 0.88f
        val dets = YoloDetect.decode(out, intArrayOf(1, ch, n), names, 0.08f)
        assertEquals(1, dets.size)
        assertEquals("sunken manhole cover", dets[0].name)
        assertEquals(120f, dets[0].x0, 1e-3f)
        assertEquals(220f, dets[0].y1, 1e-3f)
    }

    @Test
    fun keepBoxRejectsFullFramePothole() {
        assertFalse(YoloDetect.keepBox("pothole", floatArrayOf(0f, 120f, 959f, 534f), 960, 540))
        assertTrue(YoloDetect.keepBox("pothole", floatArrayOf(200f, 180f, 380f, 300f), 640, 360))
        assertTrue(YoloDetect.keepBox("manhole_cover", floatArrayOf(300f, 220f, 420f, 300f), 960, 540))
    }

    @Test
    fun isYoloDetectShapeAcceptsNmsAndRaw() {
        assertTrue(YoloDetect.isYoloDetectShape(intArrayOf(1, 300, 6)))
        assertTrue(YoloDetect.isYoloDetectShape(intArrayOf(1, 7, 8400)))
        assertFalse(YoloDetect.isYoloDetectShape(intArrayOf(1, 48, 96, 6)))
    }
}
