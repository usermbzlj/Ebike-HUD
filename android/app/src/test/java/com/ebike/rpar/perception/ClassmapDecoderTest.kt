package com.ebike.rpar.perception

import com.ebike.rpar.model.FrameMeta
import com.ebike.rpar.model.StabilizationMode
import com.ebike.rpar.model.SynchronizedFrame
import org.junit.Assert.assertTrue
import org.junit.Test

class ClassmapDecoderTest {
    @Test
    fun roadAndAnomalyBecomeObservations() {
        val w = 32
        val h = 24
        val labels = IntArray(w * h)
        for (y in 12 until h) for (x in 4 until 28) labels[y * w + x] = ClassmapDecoder.CLASS_ROAD
        for (y in 14 until 20) for (x in 10 until 18) labels[y * w + x] = ClassmapDecoder.CLASS_ANOMALY
        val meta = FrameMeta(
            0, 1_000L, 1_000L, null, null, null, null, null, null, null, null,
            StabilizationMode.OFF, 32, 24,
        )
        val frame = SynchronizedFrame(meta, null, null, null, null, null, null, null)
        val result = ClassmapDecoder.decode(labels, w, h, frame, null)
        assertTrue(result.roadPolygon.size >= 3)
        assertTrue(result.observations.isNotEmpty())
        assertTrue(result.observations.all { it.polygon.size >= 3 })
    }

    @Test
    fun argmaxPicksHighestChannel() {
        val h = 1
        val w = 2
        val c = 4
        val buf = floatArrayOf(
            0f, 3f, 1f, 0f,
            9f, 0f, 0f, 0f,
        )
        val labels = ClassmapDecoder.argmaxNhwc(buf, h, w, c)
        assertTrue(labels[0] == 1)
        assertTrue(labels[1] == 0)
    }

    @Test
    fun pasteRoiWritesFarThenNear() {
        val dw = 10
        val dh = 10
        val dst = IntArray(dw * dh)
        val src = IntArray(4) { ClassmapDecoder.CLASS_ROAD }
        ClassmapDecoder.pasteRoi(dst, dw, dh, src, 2, 2, 1, 1, 5, 5)
        assertTrue(dst[1 * dw + 1] == ClassmapDecoder.CLASS_ROAD)
        assertTrue(dst[0] == 0)
    }
}
