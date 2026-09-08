package com.ebike.rpar.quality

import android.graphics.Bitmap
import com.ebike.rpar.model.YuvImageBuffer
import kotlin.math.abs
import kotlin.math.sqrt

class GrayImage(val w: Int, val h: Int, val px: IntArray) {
    fun at(x: Int, y: Int): Int {
        val xx = x.coerceIn(0, w - 1)
        val yy = y.coerceIn(0, h - 1)
        return px[yy * w + xx]
    }

    fun crop(x0: Int, y0: Int, x1: Int, y1: Int): GrayImage {
        val ww = (x1 - x0).coerceAtLeast(1)
        val hh = (y1 - y0).coerceAtLeast(1)
        val out = IntArray(ww * hh)
        for (y in 0 until hh) for (x in 0 until ww) {
            out[y * ww + x] = at(x0 + x, y0 + y)
        }
        return GrayImage(ww, hh, out)
    }

    fun resize(nw: Int, nh: Int): GrayImage {
        val out = IntArray(nw * nh)
        for (y in 0 until nh) for (x in 0 until nw) {
            val sx = x * w / nw
            val sy = y * h / nh
            out[y * nw + x] = at(sx, sy)
        }
        return GrayImage(nw, nh, out)
    }

    fun mean(): Double = px.average()

    fun std(): Double {
        if (px.isEmpty()) return 0.0
        val m = mean()
        var s = 0.0
        for (v in px) {
            val d = v - m
            s += d * d
        }
        return sqrt(s / px.size)
    }

    fun absDiffMean(horizontal: Boolean): Double {
        var s = 0.0
        var n = 0
        if (horizontal) {
            for (y in 0 until h) for (x in 0 until w - 1) {
                s += abs(at(x + 1, y) - at(x, y))
                n++
            }
        } else {
            for (y in 0 until h - 1) for (x in 0 until w) {
                s += abs(at(x, y + 1) - at(x, y))
                n++
            }
        }
        return if (n == 0) 0.0 else s / n
    }

    fun laplacianVar(): Double {
        if (w < 3 || h < 3) return 0.0
        var sum = 0.0
        var sum2 = 0.0
        var n = 0
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val v = (at(x, y - 1) + at(x - 1, y) + at(x + 1, y) + at(x, y + 1) - 4 * at(x, y)).toDouble()
            sum += v; sum2 += v * v; n++
        }
        if (n == 0) return 0.0
        val mu = sum / n
        return sum2 / n - mu * mu
    }

    fun sobelMagMean(): Triple<Double, Double, Double> {
        var sx = 0.0; var sy = 0.0; var mag = 0.0; var n = 0
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val gx = (-at(x - 1, y - 1) + at(x + 1, y - 1) + -2 * at(x - 1, y) + 2 * at(x + 1, y) +
                -at(x - 1, y + 1) + at(x + 1, y + 1)).toDouble()
            val gy = (-at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1) +
                at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)).toDouble()
            sx += abs(gx); sy += abs(gy); mag += sqrt(gx * gx + gy * gy); n++
        }
        if (n == 0) return Triple(0.0, 0.0, 0.0)
        return Triple(sx / n, sy / n, mag / n)
    }

    fun boxBlur(k: Int): GrayImage {
        val r = k / 2
        val out = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0; var n = 0
            for (dy in -r..r) for (dx in -r..r) {
                s += at(x + dx, y + dy); n++
            }
            out[y * w + x] = s / n
        }
        return GrayImage(w, h, out)
    }

    companion object {
        fun fromYuv(yuv: YuvImageBuffer, maxW: Int = 480): GrayImage {
            val scale = if (yuv.width > maxW) yuv.width / maxW.toDouble() else 1.0
            val nw = (yuv.width / scale).toInt().coerceAtLeast(8)
            val nh = (yuv.height / scale).toInt().coerceAtLeast(8)
            val px = IntArray(nw * nh)
            for (y in 0 until nh) for (x in 0 until nw) {
                val sx = (x * yuv.width / nw).coerceIn(0, yuv.width - 1)
                val sy = (y * yuv.height / nh).coerceIn(0, yuv.height - 1)
                px[y * nw + x] = yuv.grayAt(sx, sy)
            }
            return GrayImage(nw, nh, px)
        }

        fun fromBitmap(bmp: Bitmap, maxW: Int = 480): GrayImage {
            val scale = if (bmp.width > maxW) bmp.width / maxW.toDouble() else 1.0
            val nw = (bmp.width / scale).toInt().coerceAtLeast(8)
            val nh = (bmp.height / scale).toInt().coerceAtLeast(8)
            val scaled = Bitmap.createScaledBitmap(bmp, nw, nh, true)
            val px = IntArray(nw * nh)
            val argb = IntArray(nw * nh)
            scaled.getPixels(argb, 0, nw, 0, 0, nw, nh)
            for (i in argb.indices) {
                val c = argb[i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                px[i] = (r * 299 + g * 587 + b * 114) / 1000
            }
            if (scaled !== bmp) scaled.recycle()
            return GrayImage(nw, nh, px)
        }

        fun fromFrameFull(yuv: YuvImageBuffer?, bitmap: Bitmap?, targetW: Int, targetH: Int): GrayImage {
            if (yuv != null) {
                val px = IntArray(targetW * targetH)
                for (y in 0 until targetH) for (x in 0 until targetW) {
                    val sx = x * yuv.width / targetW
                    val sy = y * yuv.height / targetH
                    px[y * targetW + x] = yuv.grayAt(sx, sy)
                }
                return GrayImage(targetW, targetH, px)
            }
            if (bitmap != null) {
                val scaled = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
                val argb = IntArray(targetW * targetH)
                scaled.getPixels(argb, 0, targetW, 0, 0, targetW, targetH)
                val px = IntArray(targetW * targetH)
                for (i in argb.indices) {
                    val c = argb[i]
                    px[i] = (((c shr 16) and 0xFF) * 299 + ((c shr 8) and 0xFF) * 587 + (c and 0xFF) * 114) / 1000
                }
                if (scaled !== bitmap) scaled.recycle()
                return GrayImage(targetW, targetH, px)
            }
            return GrayImage(targetW, targetH, IntArray(targetW * targetH) { 80 })
        }
    }
}

data class Blob(
    val x0: Int, val y0: Int, val x1: Int, val y1: Int,
    val area: Int, val cx: Float, val cy: Float, val circularity: Double,
)

fun connectedComponents(mask: BooleanArray, w: Int, h: Int, minArea: Int = 20): List<Blob> {
    val seen = BooleanArray(w * h)
    val out = ArrayList<Blob>()
    val qx = IntArray(w * h)
    val qy = IntArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        val i = y * w + x
        if (!mask[i] || seen[i]) continue
        var qh = 0; var qt = 0
        qx[qt] = x; qy[qt] = y; qt++
        seen[i] = true
        var minx = x; var maxx = x; var miny = y; var maxy = y
        var area = 0; var sx = 0; var sy = 0
        var peri = 0
        while (qh < qt) {
            val cx = qx[qh]; val cy = qy[qh]; qh++
            area++; sx += cx; sy += cy
            minx = minOf(minx, cx); maxx = maxOf(maxx, cx)
            miny = minOf(miny, cy); maxy = maxOf(maxy, cy)
            val dirs = intArrayOf(-1, 0, 1, 0, 0, -1, 0, 1)
            var border = false
            for (d in 0 until 4) {
                val nx = cx + dirs[d * 2]; val ny = cy + dirs[d * 2 + 1]
                if (nx !in 0 until w || ny !in 0 until h || !mask[ny * w + nx]) {
                    border = true
                    continue
                }
                val ni = ny * w + nx
                if (!seen[ni]) {
                    seen[ni] = true
                    qx[qt] = nx; qy[qt] = ny; qt++
                }
            }
            if (border) peri++
        }
        if (area < minArea) continue
        val circ = 4.0 * Math.PI * area / maxOf((peri * 4.0) * (peri * 4.0) / 16.0, 1e-3)
        out += Blob(minx, miny, maxx, maxy, area, sx / area.toFloat(), sy / area.toFloat(), circ.coerceIn(0.0, 1.5))
    }
    return out
}

fun morphologyOpen(src: BooleanArray, w: Int, h: Int, k: Int = 3): BooleanArray {
    val r = k / 2
    val er = BooleanArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        var ok = true
        loop@ for (dy in -r..r) for (dx in -r..r) {
            val xx = x + dx; val yy = y + dy
            if (xx !in 0 until w || yy !in 0 until h || !src[yy * w + xx]) {
                ok = false; break@loop
            }
        }
        er[y * w + x] = ok
    }
    val out = BooleanArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        var ok = false
        loop@ for (dy in -r..r) for (dx in -r..r) {
            val xx = (x + dx).coerceIn(0, w - 1)
            val yy = (y + dy).coerceIn(0, h - 1)
            if (er[yy * w + xx]) { ok = true; break@loop }
        }
        out[y * w + x] = ok
    }
    return out
}

fun morphologyDilate(src: BooleanArray, w: Int, h: Int, k: Int = 5): BooleanArray {
    val r = k / 2
    val out = BooleanArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        var hit = false
        loop@ for (dy in -r..r) for (dx in -r..r) {
            val xx = (x + dx).coerceIn(0, w - 1)
            val yy = (y + dy).coerceIn(0, h - 1)
            if (src[yy * w + xx]) { hit = true; break@loop }
        }
        out[y * w + x] = hit
    }
    return out
}
