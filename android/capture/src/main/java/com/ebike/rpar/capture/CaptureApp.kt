package com.ebike.rpar.capture

import android.app.Application
import android.os.Process
import java.io.File

class CaptureApp : Application() {
    lateinit var runtime: RideRuntime
        private set

    override fun onCreate() {
        super.onCreate()
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            try {
                File(filesDir, "diagnostics").mkdirs()
                File(filesDir, "diagnostics/crash-${System.currentTimeMillis()}.txt")
                    .writeText("thread=${t.name} pid=${Process.myPid()}\n${e.stackTraceToString().take(6000)}")
            } catch (_: Throwable) {
            }
            prev?.uncaughtException(t, e)
        }
        runtime = RideRuntime(this)
    }
}
