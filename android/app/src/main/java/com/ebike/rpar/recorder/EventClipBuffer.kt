package com.ebike.rpar.recorder

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.ArrayDeque

/** REC-005: keep ~2 s of JPEG thumbnails and dump a clip around each voice alert. */
class EventClipBuffer(
    private val dir: File,
    private val preFrames: Int = 50,
) {
    private val ring = ArrayDeque<Pair<Long, ByteArray>>(preFrames)
    private var clipIndex = 0

    fun push(timestampNs: Long, bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled) return
        val scaled = if (bitmap.width <= 320) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, 320, (320f * bitmap.height / bitmap.width).toInt().coerceAtLeast(1), true)
        }
        val bos = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 70, bos)
        if (scaled !== bitmap) scaled.recycle()
        ring.addLast(timestampNs to bos.toByteArray())
        while (ring.size > preFrames) ring.removeFirst()
    }

    fun onAlert(trackId: Int, phrase: String) {
        dir.mkdirs()
        clipIndex++
        val out = File(dir, "clip_%03d".format(clipIndex)).also { it.mkdirs() }
        File(out, "meta.txt").writeText("track=$trackId\nphrase=$phrase\nn=${ring.size}\n")
        ring.forEachIndexed { i, (ts, bytes) ->
            File(out, "%04d_%d.jpg".format(i, ts)).writeBytes(bytes)
        }
    }
}
