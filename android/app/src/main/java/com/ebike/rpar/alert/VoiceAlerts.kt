package com.ebike.rpar.alert

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.speech.tts.TextToSpeech
import java.util.Locale
import kotlin.math.sin

class VoiceAlerts(private val context: Context) {
    private val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var tts: TextToSpeech? = null
    var ready = false
        private set
    var useBluetooth = true

    fun start() {
        tts = TextToSpeech(context) { status ->
            ready = status == TextToSpeech.SUCCESS
            if (ready) {
                tts?.language = Locale.CHINA
                tts?.setSpeechRate(1.05f)
            }
        }
    }

    fun stop() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        ready = false
    }

    fun speak(phrase: String) {
        if (phrase.isBlank()) return
        route()
        tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "rpar-alert")
    }

    fun playTestTone() {
        route()
        val sr = 22050
        val n = sr / 2
        val buf = ShortArray(n)
        for (i in buf.indices) {
            buf[i] = (sin(2 * Math.PI * 880 * i / sr) * 12000).toInt().toShort()
        }
        val track = AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(sr)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
            buf.size * 2,
            AudioTrack.MODE_STATIC,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        track.write(buf, 0, buf.size)
        track.play()
        android.os.Handler(context.mainLooper).postDelayed({
            try { track.stop(); track.release() } catch (_: Throwable) {}
        }, 600)
    }

    private fun route() {
        if (useBluetooth && am.isBluetoothA2dpOn) {
            am.mode = AudioManager.MODE_NORMAL
            am.isSpeakerphoneOn = false
        } else {
            am.isSpeakerphoneOn = true
        }
    }
}
