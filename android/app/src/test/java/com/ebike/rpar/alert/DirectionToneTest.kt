package com.ebike.rpar.alert

import org.junit.Assert.assertTrue
import org.junit.Test

class DirectionToneTest {
    @Test
    fun leftIsLouderOnLeftChannel() {
        val pcm = DirectionTone.pcmStereo(0)
        val l = DirectionTone.channelRms(pcm, 0)
        val r = DirectionTone.channelRms(pcm, 1)
        assertTrue(l > r * 2)
    }

    @Test
    fun rightIsLouderOnRightChannel() {
        val pcm = DirectionTone.pcmStereo(2)
        val l = DirectionTone.channelRms(pcm, 0)
        val r = DirectionTone.channelRms(pcm, 1)
        assertTrue(r > l * 2)
    }
}
