package com.ebike.rpar.perception

import com.ebike.rpar.model.SynchronizedFrame
import com.ebike.rpar.quality.GrayImage

/** 12-D far/near features matching `rpar.ml.synth_train.patch_features`. */
object DualScaleFeatures {
    fun patchFeatures(g: GrayImage): FloatArray {
        val mean = (g.mean() / 255.0).toFloat()
        val std = (g.std() / 255.0).toFloat()
        val gx = (g.absDiffMean(horizontal = true) / 255.0).toFloat()
        val gy = (g.absDiffMean(horizontal = false) / 255.0).toFloat()
        return floatArrayOf(mean, std, gx, gy, mean * std, gx + gy)
    }

    fun fromFrame(frame: SynchronizedFrame, maxW: Int = 320, maxH: Int = 180): FloatArray {
        val fw = frame.meta.width.coerceAtLeast(8)
        val fh = frame.meta.height.coerceAtLeast(8)
        val tw = minOf(fw, maxW)
        val th = minOf(fh, maxH)
        val full = GrayImage.fromFrameFull(frame.yuv, frame.bitmap, tw, th)
        val far = DualScaleRoi.crop(full, DualScaleRoi.FAR).resize(48, 24)
        val near = DualScaleRoi.crop(full, DualScaleRoi.NEAR).resize(40, 32)
        return patchFeatures(far) + patchFeatures(near)
    }
}
