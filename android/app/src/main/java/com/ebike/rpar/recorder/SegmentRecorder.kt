package com.ebike.rpar.recorder

import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import android.view.Surface
import com.ebike.rpar.diagnostics.DiagnosticBus
import java.io.File

class SegmentRecorder(
    private val context: Context,
    private val videoDir: File,
    private val width: Int,
    private val height: Int,
    private val fps: Int,
    private val bitrateMbps: Double,
    private val segmentSeconds: Int,
    private val diagnostics: DiagnosticBus,
) {
    private var recorder: MediaRecorder? = null
    private var surface: Surface? = null
    private var index = 0
    private var partFile: File? = null
    private var finalFile: File? = null
    var onRotateRequested: (() -> Unit)? = null
    var onFinalized: (() -> Unit)? = null

    fun prepareSurface(): Surface? {
        videoDir.mkdirs()
        SegmentRecovery.recoverPartFiles(videoDir)
        index = SegmentRecovery.nextSegmentIndex(videoDir)
        return try {
            val rec = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(context) else MediaRecorder()
            rec.setVideoSource(MediaRecorder.VideoSource.SURFACE)
            rec.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            rec.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            rec.setVideoSize(width, height)
            rec.setVideoFrameRate(fps.coerceIn(15, 60))
            rec.setVideoEncodingBitRate((bitrateMbps * 1_000_000).toInt().coerceAtLeast(2_000_000))
            rec.setMaxDuration(segmentSeconds * 1000)
            val part = File(videoDir, "segment_%03d.part.mp4".format(index))
            val fin = File(videoDir, "segment_%03d.mp4".format(index))
            partFile = part
            finalFile = fin
            rec.setOutputFile(part.absolutePath)
            rec.setOnInfoListener { _, what, _ ->
                if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED) {
                    onRotateRequested?.invoke()
                }
            }
            rec.setOnErrorListener { _, w, extra ->
                diagnostics.event("REC", "recorder_error", "what=$w extra=$extra")
            }
            rec.prepare()
            surface = rec.surface
            recorder = rec
            surface
        } catch (t: Throwable) {
            diagnostics.event("REC", "prepare_failed", t.message ?: "")
            null
        }
    }

    fun recordSurface(): Surface? = surface

    fun start() {
        try {
            recorder?.start()
        } catch (t: Throwable) {
            diagnostics.event("REC", "start_failed", t.message ?: "")
        }
    }

    fun rotate() {
        onRotateRequested?.invoke()
    }

    fun stop() {
        try {
            recorder?.stop()
        } catch (_: Throwable) { }
        try {
            recorder?.release()
        } catch (_: Throwable) { }
        recorder = null
        surface = null
        val part = partFile
        val fin = finalFile
        if (part != null && fin != null && part.exists()) {
            if (fin.exists()) fin.delete()
            val ok = part.renameTo(fin)
            if (!ok) {
                part.copyTo(fin, overwrite = true)
                part.delete()
            }
            diagnostics.event("REC", "segment_finalized", fin.name)
            onFinalized?.invoke()
        }
        partFile = null
        finalFile = null
    }

    fun releaseSurface() = stop()
}
