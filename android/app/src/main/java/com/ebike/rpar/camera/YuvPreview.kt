package com.ebike.rpar.camera

import android.graphics.Bitmap
import com.ebike.rpar.model.YuvImageBuffer

/** Downscale the Y plane to a grayscale bitmap. Cheap enough for 60 fps clips / quality. */
fun yuvToPreviewBitmap(yuv: YuvImageBuffer, maxWidth: Int = 480): Bitmap {
    val scale = maxOf(1, yuv.width / maxWidth)
    val w = (yuv.width / scale).coerceAtLeast(1)
    val h = (yuv.height / scale).coerceAtLeast(1)
    val pixels = IntArray(w * h)
    val y = yuv.y
    val stride = yuv.yRowStride
    var i = 0
    for (row in 0 until h) {
        val rowOff = (row * scale) * stride
        for (col in 0 until w) {
            val lum = y.getOrElse(rowOff + col * scale) { 0 }.toInt() and 0xFF
            pixels[i++] = (0xFF shl 24) or (lum shl 16) or (lum shl 8) or lum
        }
    }
    return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
}
