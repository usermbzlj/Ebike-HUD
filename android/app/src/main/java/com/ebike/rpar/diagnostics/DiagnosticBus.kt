package com.ebike.rpar.diagnostics

import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.util.concurrent.ConcurrentLinkedQueue

data class DiagnosticEvent(
    val timestampNs: Long,
    val domain: String,
    val code: String,
    val detail: String,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("timestamp_ns", timestampNs)
        .put("domain", domain)
        .put("code", code)
        .put("detail", detail.take(400))
}

class DiagnosticBus {
    val events = ConcurrentLinkedQueue<DiagnosticEvent>()
    @Volatile var lastStallNs: Long = 0

    fun event(domain: String, code: String, detail: String) {
        val e = DiagnosticEvent(SystemClock.elapsedRealtimeNanos(), domain, code, detail)
        events.add(e)
        while (events.size > 400) events.poll()
        Log.i("RPAR-$domain", "$code $detail")
    }

    fun drain(): List<DiagnosticEvent> {
        val out = ArrayList<DiagnosticEvent>()
        while (true) {
            val e = events.poll() ?: break
            out += e
        }
        return out
    }
}
