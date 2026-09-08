package com.ebike.rpar.app

import android.app.Application
import android.os.Process
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter

class RparApplication : Application() {
    lateinit var runtime: RparRuntime
        private set

    override fun onCreate() {
        super.onCreate()
        installCrashHandler()
        runtime = RparRuntime(this)
    }

    private fun installCrashHandler() {
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            try {
                val dir = File(filesDir, "diagnostics")
                dir.mkdirs()
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                val redacted = sw.toString()
                    .replace(Regex("""(?i)(lat(itude)?|lon(gitude)?)[^\n]*"""), "$1=<redacted>")
                    .replace(Regex("""\b[-+]?\d{1,3}\.\d{4,}\b"""), "<redacted-coord>")
                File(dir, "crash-${System.currentTimeMillis()}.txt").writeText(
                    "thread=${t.name} pid=${Process.myPid()}\n" +
                        "note=crash log excludes video frames and precise lat/lon\n" +
                        redacted.take(8000),
                )
            } catch (_: Throwable) {
            }
            prev?.uncaughtException(t, e) ?: run {
                Log.e("RPAR", "crash", e)
                Process.killProcess(Process.myPid())
            }
        }
    }
}
