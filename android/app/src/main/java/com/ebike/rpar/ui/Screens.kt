package com.ebike.rpar.ui

import android.view.TextureView
import android.widget.ImageView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.ebike.rpar.app.AppScreen
import com.ebike.rpar.app.RparRuntime
import com.ebike.rpar.ar.ArGlView
import com.ebike.rpar.model.RunMode
import com.ebike.rpar.model.StabilizationMode
import com.ebike.rpar.model.UiMode

@Composable
fun DisclaimerScreen(onAccept: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("Road Perception AR", color = HudAccent, fontSize = 28.sp)
        Spacer(Modifier.height(12.dp))
        Text(
            "实验性功能，不能替代骑行者观察，不输出控制指令。本系统不提供转向或制动建议，请始终以自身观察为准。",
            color = HudText,
            fontSize = 16.sp,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Experimental assist only. Does not replace rider observation. No control commands.",
            color = HudMuted,
            fontSize = 14.sp,
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = onAccept, colors = ButtonDefaults.buttonColors(containerColor = HudAccent)) {
            Text("我已了解，继续", color = HudBg)
        }
    }
}

@Composable
fun FirstRunScreen(
    json: String,
    onProbe: () -> String,
    onShare: () -> Unit,
    onContinue: () -> Unit,
    onRequestPerms: () -> Unit,
) {
    Column(Modifier.fillMaxSize().padding(20.dp).verticalScroll(rememberScrollState())) {
        Text("能力探测 (CAP-001)", color = HudAccent, fontSize = 22.sp)
        Text("首次启动：探测 Camera2 / 传感器 / TTS。结果仅保存在本机。", color = HudMuted, fontSize = 14.sp)
        Spacer(Modifier.height(12.dp))
        Row {
            Button(onClick = onRequestPerms) { Text("授予权限") }
            Spacer(Modifier.width(8.dp))
            Button(onClick = { onProbe() }) { Text("运行探测") }
            Spacer(Modifier.width(8.dp))
            Button(onClick = onShare) { Text("导出 JSON") }
        }
        Spacer(Modifier.height(12.dp))
        Text(json.ifBlank { "尚未探测" }, color = HudText, fontSize = 11.sp)
        Spacer(Modifier.height(16.dp))
        Button(onClick = onContinue, colors = ButtonDefaults.buttonColors(containerColor = HudAccent)) {
            Text("进入骑行界面", color = HudBg)
        }
    }
}

@Composable
fun HudScreen(runtime: RparRuntime, onOpenSettings: () -> Unit) {
    val ui by runtime.ui.collectAsState()
    val ctx = LocalContext.current
    val act = ctx as MainActivity
    val useCam = shouldUseCamera(runtime, act.hasCamera())
    Box(Modifier.fillMaxSize()) {
        if (useCam) {
            AndroidView(factory = { TextureView(it).also { tv -> bindTexture(tv, runtime, true) } }, modifier = Modifier.fillMaxSize())
        } else {
            DisposableEffect(Unit) {
                runtime.startCapture(null, false)
                onDispose { runtime.stopCapture() }
            }
            AndroidView(factory = { ImageView(it).also { iv -> bindPattern(iv, runtime) } }, modifier = Modifier.fillMaxSize())
        }
        AndroidView(
            factory = { ArGlView(it) },
            update = { gl ->
                val v = ui.view
                if (v != null) gl.setPrimitives(
                    v.primitives, ui.previewSize.width, ui.previewSize.height,
                    ui.nightPalette, runtime.cfg.render.nightBrightness.toFloat(),
                    ui.overlayAlpha, ui.strokeScale,
                )
            },
            modifier = Modifier.fillMaxSize(),
        )
        Column(Modifier.fillMaxSize()) {
            Row(Modifier.fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                Box(
                    Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(HudPanel)
                        .border(1.dp, HudAccent, RoundedCornerShape(8.dp))
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                ) {
                    val spd = ui.view?.speedKmh
                    Text(if (spd != null) "%.0f km/h".format(spd) else "-- km/h", color = HudText, fontSize = 22.sp)
                }
                if (ui.uiMode == UiMode.RESEARCH) {
                    Column(Modifier.clip(RoundedCornerShape(8.dp)).background(HudPanel).padding(10.dp)) {
                        val v = ui.view
                        Text("FPS ${v?.arFps?.toInt() ?: 0} / AI ${v?.inferFps?.toInt() ?: 0}", color = HudText, fontSize = 12.sp)
                        Text("p50 ${v?.latencyP50Ms?.toInt() ?: 0} / p95 ${v?.latencyP95Ms?.toInt() ?: 0} ms", color = HudText, fontSize = 12.sp)
                        Text("q=${v?.queueDepth ?: 0} drop=${v?.droppedInfer ?: 0} dual=${if (v?.dualScale == true) "on" else "off"}", color = HudText, fontSize = 12.sp)
                        Text("${v?.backend?.wire ?: "-"} ${v?.modelVersion ?: ""}", color = HudText, fontSize = 11.sp)
                        Text("in ${(v?.inputFar?.joinToString("x") ?: "-")} / ${(v?.inputNear?.joinToString("x") ?: "-")}", color = HudMuted, fontSize = 11.sp)
                        Text("Blur ${"%.2f".format(v?.blur ?: 0.0)}  Glare ${"%.2f".format(v?.glare ?: 0.0)}", color = HudText, fontSize = 12.sp)
                        Text("Temp ${v?.thermalC?.let { "%.0f°C".format(it) } ?: "--"}", color = HudText, fontSize = 12.sp)
                        Text("Tracks ${v?.tracks?.joinToString { it.trackId.toString() } ?: "-"}", color = HudMuted, fontSize = 11.sp)
                        Row {
                            TextButton(onClick = { runtime.togglePause() }) { Text(if (ui.paused) "继续" else "暂停", color = HudAccent, fontSize = 12.sp) }
                            TextButton(onClick = { runtime.screenshot() }) { Text("截图", color = HudAccent, fontSize = 12.sp) }
                            TextButton(onClick = { runtime.markEvent() }) { Text("标记", color = HudAccent, fontSize = 12.sp) }
                        }
                        if (ui.lastMark.isNotBlank()) Text("标记 ${ui.lastMark}", color = HudMuted, fontSize = 11.sp)
                    }
                }
                if (!ui.touchLocked) {
                    TextButton(onClick = onOpenSettings) { Text("设置", color = HudAccent) }
                }
            }
            Box(Modifier.fillMaxSize()) {
                val labels = ui.view?.primitives?.filter { it.label != null }?.take(if (ui.uiMode == UiMode.RIDING) 5 else 12).orEmpty()
                Column(Modifier.align(Alignment.CenterEnd).padding(16.dp)) {
                    labels.forEach { p ->
                        Text(
                            p.label ?: "",
                            color = HudText,
                            fontSize = (14 * ui.fontScale).sp,
                            modifier = Modifier.padding(4.dp).background(HudPanel).padding(6.dp),
                        )
                    }
                }
            }
        }
        Row(
            Modifier.align(Alignment.BottomStart).fillMaxWidth().height(48.dp).background(HudPanel).padding(horizontal = 20.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(ui.statusLine, color = HudText, fontSize = 14.sp)
            Spacer(Modifier.weight(1f))
            val rec = ui.view?.recSeconds ?: 0.0
            val hh = (rec / 3600).toInt(); val mm = ((rec % 3600) / 60).toInt(); val ss = (rec % 60).toInt()
            Text("REC %02d:%02d:%02d".format(hh, mm, ss), color = HudAccent, fontSize = 13.sp)
            Spacer(Modifier.width(12.dp))
            Text("%.1fh".format(ui.remainingHours), color = HudMuted, fontSize = 12.sp)
            Spacer(Modifier.width(12.dp))
            Text(ui.modelId, color = HudMuted, fontSize = 12.sp)
        }
        if (ui.touchLocked) {
            Box(Modifier.fillMaxSize().clickable(enabled = false) {})
            Text(
                "触摸已锁定 · 长按解锁",
                color = HudMuted,
                fontSize = 12.sp,
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 56.dp),
            )
        }
        if (ui.emergency) {
            Text(
                if (ui.emergencyReason == "imu_crash") "疑似碰撞/剧烈振动，已停止采集，停车后确认" else "紧急停止",
                color = HudAccent,
                fontSize = 16.sp,
                modifier = Modifier.align(Alignment.TopCenter).padding(top = 48.dp),
            )
        }
        if (ui.storageLight) {
            Text("存储不足，无视频轻量记录", color = HudMuted, fontSize = 12.sp, modifier = Modifier.align(Alignment.TopCenter).padding(top = 8.dp))
        }
        Button(
            onClick = { runtime.emergencyStop() },
            modifier = Modifier.align(Alignment.TopEnd).padding(top = 16.dp, end = 12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = HudPanel),
        ) { Text("紧急停止", color = HudText) }
    }
}

@Composable
fun SettingsScreen(runtime: RparRuntime) {
    val ui by runtime.ui.collectAsState()
    Column(Modifier.fillMaxSize().padding(20.dp).verticalScroll(rememberScrollState())) {
        Text("设置", color = HudAccent, fontSize = 22.sp)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("声音提醒", color = HudText); Spacer(Modifier.weight(1f))
            Switch(checked = ui.alertsEnabled, onCheckedChange = { runtime.setAlerts(it) })
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("骑行触摸锁", color = HudText); Spacer(Modifier.weight(1f))
            Switch(checked = ui.touchLocked, onCheckedChange = { runtime.setTouchLock(it) })
        }
        Text("界面", color = HudMuted)
        Row {
            Chip("骑行 HUD", ui.uiMode == UiMode.RIDING) { runtime.setUiMode(UiMode.RIDING) }
            Chip("研究", ui.uiMode == UiMode.RESEARCH) { runtime.setUiMode(UiMode.RESEARCH) }
        }
        Text("防抖", color = HudMuted)
        Row {
            Chip("关", ui.stab == StabilizationMode.OFF) { runtime.setStab(StabilizationMode.OFF) }
            Chip("标准", ui.stab == StabilizationMode.STANDARD) { runtime.setStab(StabilizationMode.STANDARD) }
            Chip("预览", ui.stab == StabilizationMode.PREVIEW) { runtime.setStab(StabilizationMode.PREVIEW) }
        }
        Text("运行模式", color = HudMuted)
        Column {
            Chip("CAPTURE_ONLY", ui.runMode == RunMode.CAPTURE_ONLY) { runtime.setRunMode(RunMode.CAPTURE_ONLY) }
            Chip("REALTIME", ui.runMode == RunMode.REALTIME_PERCEPTION) { runtime.setRunMode(RunMode.REALTIME_PERCEPTION) }
            Chip("FULL_LOG", ui.runMode == RunMode.REALTIME_PERCEPTION_FULL_LOG) { runtime.setRunMode(RunMode.REALTIME_PERCEPTION_FULL_LOG) }
            Chip("SAFE_MODE", ui.runMode == RunMode.SAFE_MODE) { runtime.setRunMode(RunMode.SAFE_MODE) }
            Chip("REPLAY", ui.runMode == RunMode.REPLAY) { runtime.setRunMode(RunMode.REPLAY) }
        }
        Spacer(Modifier.height(8.dp))
        Button(onClick = { runtime.reloadModel() }) { Text("扫描/加载模型包 (${ui.modelId})") }
        Button(onClick = { runtime.rollbackModel() }) { Text("回滚上一可用模型") }
        Button(onClick = { runtime.voice.playTestTone() }) { Text("测试提示音（扬声器/蓝牙）") }
        var tone by remember { mutableStateOf(runtime.voice.toneMode) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("音调模式（左/正/右）", color = HudText); Spacer(Modifier.weight(1f))
            Switch(checked = tone, onCheckedChange = { tone = it; runtime.setToneMode(it) })
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("夜间配色", color = HudText); Spacer(Modifier.weight(1f))
            Switch(checked = ui.nightPalette, onCheckedChange = { runtime.setNightPalette(it) })
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("研究热图", color = HudText); Spacer(Modifier.weight(1f))
            Switch(checked = ui.researchHeatmap, onCheckedChange = { runtime.setResearchHeatmap(it) })
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("积水/散落物信息层", color = HudText); Spacer(Modifier.weight(1f))
            Switch(checked = ui.showInfoLayer, onCheckedChange = { runtime.setShowInfoLayer(it) })
        }
        Text("轮廓粗细 (AR-010)", color = HudMuted)
        Row {
            Chip("细", ui.strokeScale < 0.85f) { runtime.setStrokeScale(0.7f) }
            Chip("标准", ui.strokeScale in 0.85f..1.15f) { runtime.setStrokeScale(1f) }
            Chip("粗", ui.strokeScale > 1.15f) { runtime.setStrokeScale(1.45f) }
        }
        Text("字体大小", color = HudMuted)
        Row {
            Chip("小", ui.fontScale < 0.9f) { runtime.setFontScale(0.8f) }
            Chip("中", ui.fontScale in 0.9f..1.15f) { runtime.setFontScale(1f) }
            Chip("大", ui.fontScale > 1.15f) { runtime.setFontScale(1.35f) }
        }
        Text("叠加透明度", color = HudMuted)
        Row {
            Chip("低", ui.overlayAlpha < 0.7f) { runtime.setOverlayAlpha(0.55f) }
            Chip("中", ui.overlayAlpha in 0.7f..0.9f) { runtime.setOverlayAlpha(0.82f) }
            Chip("高", ui.overlayAlpha > 0.9f) { runtime.setOverlayAlpha(1f) }
        }
        Text("减震 A/B（CAL-006）", color = HudMuted)
        Row {
            Button(onClick = { runtime.startDampingSample("A") }) { Text("采样 A") }
            Spacer(Modifier.width(8.dp))
            Button(onClick = { runtime.startDampingSample("B") }) { Text("采样 B") }
            Spacer(Modifier.width(8.dp))
            Button(onClick = { runtime.stopDampingSample() }) { Text("结束采样") }
        }
        if (ui.dampingNote.isNotBlank()) Text(ui.dampingNote, color = HudMuted, fontSize = 12.sp)
        Text("相机实验", color = HudMuted)
        var afLock by remember { mutableStateOf(runtime.camera.lockFarFocus) }
        var expCap by remember { mutableStateOf(runtime.camera.exposureCapNs != null) }
        Row {
            Chip("连续对焦", !afLock) { afLock = false; runtime.setAfLock(false) }
            Chip("锁定远焦", afLock) { afLock = true; runtime.setAfLock(true) }
            Chip("曝光上限 8ms", expCap) { expCap = !expCap; runtime.setExposureCap(expCap) }
        }
        Button(onClick = { runtime.navigate(AppScreen.CALIBRATION) }) { Text("标定向导") }
        Button(onClick = { runtime.navigate(AppScreen.CAPABILITY) }) { Text("能力报告") }
        Button(onClick = { runtime.navigate(AppScreen.EXPORT) }) { Text("导出会话") }
        Button(onClick = { runtime.navigate(AppScreen.HUD) }, colors = ButtonDefaults.buttonColors(containerColor = HudAccent)) {
            Text("返回 HUD", color = HudBg)
        }
        Text("隐私：LOCAL_ONLY，无上传路径。麦克风权限非必需。", color = HudMuted, fontSize = 12.sp)
        ui.cameraDegrade?.let { Text("相机降级：$it", color = HudMuted, fontSize = 12.sp) }
        ui.lastError?.let { Text("模型：$it", color = HudMuted, fontSize = 12.sp) }
    }
}

@Composable
private fun Chip(label: String, selected: Boolean, onClick: () -> Unit) {
    Text(
        label,
        color = if (selected) HudBg else HudText,
        fontSize = 13.sp,
        modifier = Modifier
            .padding(4.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(if (selected) HudAccent else HudPanel)
            .clickable { onClick() }
            .padding(horizontal = 10.dp, vertical = 6.dp),
    )
}

@Composable
fun CapabilityScreen(
    json: String,
    benchRunning: Boolean = false,
    onProbe: () -> String,
    onSustained: () -> Unit = {},
    onShare: () -> Unit,
    onBack: () -> Unit,
) {
    Column(Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState())) {
        Text("能力报告", color = HudAccent, fontSize = 22.sp)
        Text("短探测写入 Camera2/IMU/LiteRT 快照。10 分钟基准在后台跑，不阻塞预览。", color = HudMuted, fontSize = 12.sp)
        Row {
            Button(onClick = { onProbe() }, enabled = !benchRunning) { Text("重新探测") }
            Spacer(Modifier.width(8.dp))
            Button(onClick = onSustained, enabled = !benchRunning) {
                Text(if (benchRunning) "基准进行中…" else "CAP-005 10 分钟基准")
            }
        }
        Row {
            Button(onClick = onShare, enabled = !benchRunning) { Text("分享 JSON") }
            Spacer(Modifier.width(8.dp))
            Button(onClick = onBack) { Text("返回") }
        }
        Spacer(Modifier.height(8.dp))
        Text(json, color = HudText, fontSize = 11.sp)
    }
}

@Composable
fun ExportScreen(types: List<String>, hours: Double, onExport: () -> Unit, onShareRedacted: () -> Unit, onBack: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(20.dp)) {
        Text("导出确认 (SEC-003)", color = HudAccent, fontSize = 22.sp)
        Text("本次导出会包含以下本地数据类型。隐私模式 LOCAL_ONLY，不会上传。", color = HudMuted)
        Spacer(Modifier.height(8.dp))
        types.forEach { Text("• $it", color = HudText, fontSize = 14.sp) }
        Spacer(Modifier.height(12.dp))
        Text("估计剩余录像约 %.1f 小时（按 25 Mbps）".format(hours), color = HudMuted)
        Spacer(Modifier.height(16.dp))
        Button(onClick = onExport) { Text("打包并分享 ZIP（原始）") }
        Spacer(Modifier.height(8.dp))
        Button(onClick = onShareRedacted) { Text("分享版 ZIP（GPS 降精度）") }
        Text("人脸/车牌像素模糊请用桌面 rpar share。", color = HudMuted, fontSize = 12.sp)
        TextButton(onClick = onBack) { Text("取消") }
    }
}
