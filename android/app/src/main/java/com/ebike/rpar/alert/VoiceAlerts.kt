package com.ebike.rpar.alert

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.speech.tts.TextToSpeech
import java.util.Locale

class VoiceAlerts(private val context: Context) {
    private val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var tts: TextToSpeech? = null
    private var focusRequest: AudioFocusRequest? = null
    var ready = false
        private set
    var useBluetooth = true
    var toneMode = false

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
        abandonFocus()
        tts?.stop()
        tts?.shutdown()
        tts = null
        ready = false
    }

    fun speak(phrase: String) {
        if (phrase.isBlank()) return
        requestFocus()
        if (toneMode) {
            val dir = when {
                phrase.startsWith("左") -> 0
                phrase.startsWith("右") -> 2
                else -> 1
            }
            playDirectionTone(dir)
            return
        }
        route()
        tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "rpar-alert")
    }

    fun playTestTone() = playDirectionTone(1)

    fun playDirectionTone(dir: Int) {
        requestFocus()
        route()
        val sr = 22050
        val buf = DirectionTone.pcmStereo(dir, sr)
        val track = AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build(),
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(sr)
                .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                .build(),
            buf.size * 2,
            AudioTrack.MODE_STATIC,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        track.write(buf, 0, buf.size)
        track.play()
        android.os.Handler(context.mainLooper).postDelayed({
            try {
                track.stop()
                track.release()
            } catch (_: Throwable) {
            }
            abandonFocus()
        }, 400)
    }

    private fun requestFocus() {
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            .setAudioAttributes(attrs)
            .setAcceptsDelayedFocusGain(false)
            .build()
        focusRequest = req
        am.requestAudioFocus(req)
    }

    private fun abandonFocus() {
        val req = focusRequest ?: return
        try {
            am.abandonAudioFocusRequest(req)
        } catch (_: Throwable) {
        }
        focusRequest = null
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
