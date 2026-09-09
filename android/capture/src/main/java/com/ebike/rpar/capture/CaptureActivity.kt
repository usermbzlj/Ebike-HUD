package com.ebike.rpar.capture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.TextureView
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider

class CaptureActivity : ComponentActivity() {
    private lateinit var runtime: RideRuntime
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (isNight()) {
            val lp = window.attributes
            lp.screenBrightness = 0.18f
            window.attributes = lp
        }
        runtime = (application as CaptureApp).runtime
        setContent {
            val ui by runtime.ui.collectAsState()
            RideCaptureScreen(
                ui = ui,
                modifier = Modifier.fillMaxSize().background(RideColors.bg),
                onAccept = { runtime.accept() },
                onRequestPerms = { requestPerms() },
                onReady = { tv -> bindPreview(tv) },
                onToggle = { tv ->
                    if (runtime.ui.value.recording) runtime.stop()
                    else tv.surfaceTexture?.let { runtime.start(it) }
                },
                onMark = { runtime.mark("tap") },
                onShare = { shareLatest() },
            )
        }
    }

    private fun bindPreview(tv: TextureView) {
        runtime.attachPreview(tv)
        tv.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: android.graphics.SurfaceTexture, w: Int, h: Int) {
                if (hasCamera() && runtime.ui.value.accepted && !runtime.ui.value.recording) {
                    runtime.start(st)
                }
            }
            override fun onSurfaceTextureSizeChanged(st: android.graphics.SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(st: android.graphics.SurfaceTexture): Boolean = true
            override fun onSurfaceTextureUpdated(st: android.graphics.SurfaceTexture) {}
        }
        if (tv.isAvailable && hasCamera() && runtime.ui.value.accepted && !runtime.ui.value.recording) {
            tv.surfaceTexture?.let { runtime.start(it) }
        }
    }

    fun requestPerms() {
        val list = mutableListOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION)
        if (Build.VERSION.SDK_INT >= 33) list += Manifest.permission.POST_NOTIFICATIONS
        permissionLauncher.launch(list.toTypedArray())
    }

    fun hasCamera(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun shareLatest() {
        val zip = runtime.exportLatest() ?: return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", zip)
        startActivity(
            Intent.createChooser(
                Intent(Intent.ACTION_SEND).apply {
                    type = "application/zip"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                },
                "导出采集包",
            ),
        )
    }

    override fun onDestroy() {
        if (isFinishing && runtime.ui.value.recording) runtime.stop()
        super.onDestroy()
    }

    private fun isNight(): Boolean {
        val h = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        return h >= 19 || h < 6
    }
}
