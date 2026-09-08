package com.ebike.rpar.alert

import kotlin.math.sin
import kotlin.math.sqrt

/** ALT-009: left/center/right as stereo pan + pulse rhythm, not speech. */
object DirectionTone {
    fun pcmStereo(dir: Int, sampleRate: Int = 22050, durationMs: Int = 280): ShortArray {
        val n = (sampleRate * durationMs / 1000).coerceAtLeast(64)
        val freq = when (dir) {
            0 -> 660.0
            2 -> 990.0
            else -> 820.0
        }
        val pulses = when (dir) {
            0 -> 2
            2 -> 3
            else -> 1
        }
        val leftGain = when (dir) {
            0 -> 1.0
            2 -> 0.12
            else -> 0.85
        }
        val rightGain = when (dir) {
            0 -> 0.12
            2 -> 1.0
            else -> 0.85
        }
        val out = ShortArray(n * 2)
        val period = (n / pulses).coerceAtLeast(1)
        for (i in 0 until n) {
            val pulseIdx = i / period
            val local = i % period
            val burst = (period * 0.42).toInt().coerceAtLeast(8)
            val inBurst = pulseIdx < pulses && local < burst
            val env = if (!inBurst) {
                0.0
            } else {
                val u = local.toDouble() / burst
                if (u < 0.08 || u > 0.92) 0.32 else 1.0
            }
            val s = sin(2 * Math.PI * freq * i / sampleRate) * 11000.0 * env
            out[i * 2] = (s * leftGain).toInt().coerceIn(-32767, 32767).toShort()
            out[i * 2 + 1] = (s * rightGain).toInt().coerceIn(-32767, 32767).toShort()
        }
        return out
    }

    fun channelRms(pcm: ShortArray, channel: Int): Double {
        var acc = 0.0
        var n = 0
        var i = channel
        while (i < pcm.size) {
            val v = pcm[i].toDouble()
            acc += v * v
            n++
            i += 2
        }
        return if (n == 0) 0.0 else sqrt(acc / n)
    }
}
