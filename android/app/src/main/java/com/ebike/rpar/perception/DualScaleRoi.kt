package com.ebike.rpar.perception

import com.ebike.rpar.quality.GrayImage

data class NormRect(val x0: Double, val y0: Double, val x1: Double, val y1: Double)

data class RoiPx(val x0: Int, val y0: Int, val x1: Int, val y1: Int)

/**
 * Far/near road crops in the unified image frame (not hardcoded 1920×1080 rectangles).
 */
object DualScaleRoi {
    val FAR = NormRect(0.18, 0.32, 0.82, 0.62)
    val NEAR = NormRect(0.08, 0.50, 0.92, 1.00)

    fun cropPx(w: Int, h: Int, box: NormRect): RoiPx {
        val x0 = (w * box.x0).toInt().coerceIn(0, (w - 1).coerceAtLeast(0))
        val y0 = (h * box.y0).toInt().coerceIn(0, (h - 1).coerceAtLeast(0))
        val x1 = (w * box.x1).toInt().coerceIn(x0 + 1, w.coerceAtLeast(1))
        val y1 = (h * box.y1).toInt().coerceIn(y0 + 1, h.coerceAtLeast(1))
        return RoiPx(x0, y0, x1, y1)
    }

    fun farPx(w: Int, h: Int) = cropPx(w, h, FAR)
    fun nearPx(w: Int, h: Int) = cropPx(w, h, NEAR)

    fun polygon(w: Int, h: Int, box: NormRect): List<Pair<Float, Float>> {
        val x0 = (w * box.x0).toFloat()
        val y0 = (h * box.y0).toFloat()
        val x1 = (w * box.x1).toFloat()
        val y1 = (h * box.y1).toFloat()
        return listOf(x0 to y0, x1 to y0, x1 to y1, x0 to y1)
    }

    fun farPolygon(w: Int, h: Int) = polygon(w, h, FAR)
    fun nearPolygon(w: Int, h: Int) = polygon(w, h, NEAR)

    fun crop(g: GrayImage, box: NormRect): GrayImage {
        val r = cropPx(g.w, g.h, box)
        return g.crop(r.x0, r.y0, r.x1, r.y1)
    }
}
