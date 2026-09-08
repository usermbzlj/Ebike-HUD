package com.ebike.rpar.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.os.Build
import android.os.Bundle
import android.view.TextureView
import android.view.WindowManager
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.ebike.rpar.app.AppScreen
import com.ebike.rpar.app.RparApplication
import com.ebike.rpar.app.RparRuntime
import com.ebike.rpar.model.RunMode
import java.io.File

class MainActivity : ComponentActivity() {
    private lateinit var runtime: RparRuntime
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        runtime = (application as RparApplication).runtime
        setContent {
            RparTheme {
                val ui by runtime.ui.collectAsState()
                Box(
                    Modifier
                        .fillMaxSize()
                        .background(HudBg)
                        .pointerInput(ui.touchLocked, ui.screen) {
                            if (ui.touchLocked && ui.screen == AppScreen.HUD) {
                                detectTapGestures(
                                    onLongPress = { runtime.setTouchLock(false) },
                                )
                            }
                        },
                ) {
                    when (ui.screen) {
                        AppScreen.DISCLAIMER -> DisclaimerScreen { runtime.acceptDisclaimer() }
                        AppScreen.FIRST_RUN -> FirstRunScreen(
                            json = ui.capabilityJson,
                            onProbe = { runtime.runCapability() },
                            onShare = { shareCapability() },
                            onContinue = { runtime.completeFirstRun() },
                            onRequestPerms = { requestPerms() },
                        )
                        AppScreen.HUD -> HudScreen(
                            runtime = runtime,
                            onOpenSettings = { runtime.navigate(AppScreen.SETTINGS) },
                        )
                        AppScreen.SETTINGS -> SettingsScreen(runtime)
                        AppScreen.CALIBRATION -> CalibrationScreen(runtime)
                        AppScreen.CAPABILITY -> CapabilityScreen(
                            json = ui.capabilityJson,
                            onProbe = { runtime.runCapability() },
                            onShare = { shareCapability() },
                            onBack = { runtime.navigate(AppScreen.SETTINGS) },
                        )
                        AppScreen.EXPORT -> ExportScreen(
                            types = ui.exportTypes,
                            hours = ui.remainingHours,
                            onExport = { shareExport() },
                            onBack = { runtime.navigate(AppScreen.SETTINGS) },
                        )
                    }
                }
            }
        }
    }

    fun requestPerms() {
        val list = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
        )
        if (Build.VERSION.SDK_INT >= 33) list += Manifest.permission.POST_NOTIFICATIONS
        permissionLauncher.launch(list.toTypedArray())
    }

    fun hasCamera(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun shareCapability() {
        val f = java.io.File(filesDir, "capability/capability_report.json")
        if (!f.exists()) runtime.runCapability()
        val uri = FileProvider.getUriForFile(this, "$packageName.files", File(filesDir, "capability/capability_report.json"))
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Capability report"))
    }

    private fun shareExport() {
        val zip = runtime.exportSessionZip() ?: return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", zip)
        startActivity(
            Intent.createChooser(
                Intent(Intent.ACTION_SEND).apply {
                    type = "application/zip"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                },
                "Export session",
            ),
        )
    }

    override fun onDestroy() {
        if (isFinishing) runtime.stopCapture()
        super.onDestroy()
    }
}

fun bindTexture(textureView: TextureView, runtime: RparRuntime, preferCamera: Boolean) {
    textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(st: SurfaceTexture, width: Int, height: Int) {
            runtime.startCapture(st, preferCamera)
        }
        override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
            runtime.stopCapture()
            return true
        }
        override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
    }
}

fun bindPattern(view: ImageView, runtime: RparRuntime) {
    view.scaleType = ImageView.ScaleType.CENTER_CROP
    view.post(object : Runnable {
        override fun run() {
            runtime.latestBitmap?.let { view.setImageBitmap(it) }
            view.postDelayed(this, 33)
        }
    })
}

fun shouldUseCamera(runtime: RparRuntime, hasCameraPerm: Boolean): Boolean {
    val mode = runtime.ui.value.runMode
    return hasCameraPerm && mode != RunMode.REPLAY && mode != RunMode.SAFE_MODE
}
