package com.ebike.rpar.perception

/**
 * Numpy twin of [python DualScaleSegEngine]: 1×1 linear head on RGB+xy, far then near ROI.
 * Tensors never leave this helper (PER-014). Not a pothole detector.
 */
object DualScaleClassmap {
    const val SMALL_W = 96
    const val SMALL_H = 48

    fun classifyRgb(
        pixels: IntArray,
        w: Int,
        h: Int,
        weights: Array<DoubleArray>,
    ): IntArray {
        val out = IntArray(w * h)
        if (w <= 0 || h <= 0 || weights.isEmpty()) return out
        val wf = w.toDouble()
        val hf = h.toDouble()
        val nClass = weights.size
        val dim = weights[0].size
        for (y in 0 until h) {
            val ys = y / hf
            for (x in 0 until w) {
                val p = pixels[y * w + x]
                val r = ((p ushr 16) and 0xFF) / 255.0
                val g = ((p ushr 8) and 0xFF) / 255.0
                val b = (p and 0xFF) / 255.0
                val xs = x / wf
                var best = 0
                var bestV = Double.NEGATIVE_INFINITY
                for (k in 0 until nClass) {
                    val row = weights[k]
                    var v = 0.0
                    if (dim > 0) v += row[0] * r
                    if (dim > 1) v += row[1] * g
                    if (dim > 2) v += row[2] * b
                    if (dim > 3) v += row[3] * xs
                    if (dim > 4) v += row[4] * ys
                    if (dim > 5) v += row[5]
                    if (v > bestV) {
                        bestV = v
                        best = k
                    }
                }
                out[y * w + x] = best
            }
        }
        return out
    }

    fun resizeNearest(src: IntArray, sw: Int, sh: Int, dw: Int, dh: Int): IntArray {
        val dst = IntArray(dw * dh)
        if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return dst
        for (y in 0 until dh) {
            val sy = (y * sh / dh).coerceIn(0, sh - 1)
            for (x in 0 until dw) {
                val sx = (x * sw / dw).coerceIn(0, sw - 1)
                dst[y * dw + x] = src[sy * sw + sx]
            }
        }
        return dst
    }
}
