package com.ebike.rpar.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.ebike.rpar.app.AppScreen
import com.ebike.rpar.app.RparRuntime
import com.ebike.rpar.calibration.WizardStep
import com.ebike.rpar.geometry.Transforms

@Composable
fun CalibrationScreen(runtime: RparRuntime) {
    var step by remember { mutableStateOf(WizardStep.UPRIGHT) }
    var name by remember { mutableStateOf("left_handlebar_v1") }
    var horizonY by remember { mutableFloatStateOf(420f) }
    var centerX by remember { mutableFloatStateOf(960f) }
    var nearM by remember { mutableStateOf("3.0") }
    var d5 by remember { mutableStateOf("") }
    var d10 by remember { mutableStateOf("") }
    var d20 by remember { mutableStateOf("") }
    var height by remember { mutableStateOf("1.12") }
    var pitch by remember { mutableStateOf("18") }
    var roll by remember { mutableStateOf("0") }
    var yaw by remember { mutableStateOf("0") }
    var healthMsg by remember { mutableStateOf("") }
    var viewW by remember { mutableIntStateOf(1920) }
    var viewH by remember { mutableIntStateOf(1080) }

    Column(Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState())) {
        Text("标定向导 (CAL)", color = HudAccent, fontSize = 22.sp)
        Text("步骤：${step.name}", color = HudMuted)
        Spacer(Modifier.height(8.dp))
        when (step) {
            WizardStep.UPRIGHT -> {
                Text("将车身竖直停稳，确认手机横向锁死。根据重力估计横滚。", color = HudText)
                Button(onClick = { step = WizardStep.HANDLEBAR_CENTER }) { Text("下一步：车把中心") }
            }
            WizardStep.HANDLEBAR_CENTER -> {
                Text("左把安装：默认横向偏移 -0.32 m（相对车辆中线）。", color = HudText)
                Button(onClick = { step = WizardStep.HORIZON }) { Text("下一步：地平线") }
            }
            WizardStep.HORIZON, WizardStep.CENTERLINE -> {
                Text("点选画面：地平线（水平线）与车辆中线。", color = HudText)
                androidx.compose.foundation.layout.Box(
                    Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                        .background(HudPanel)
                        .onSizeChanged { viewW = it.width; viewH = it.height }
                        .pointerInput(step) {
                            detectTapGestures { off ->
                                if (step == WizardStep.HORIZON) horizonY = off.y * 1080f / viewH
                                else centerX = off.x * 1920f / viewW
                            }
                        },
                )
                Text("horizonY=$horizonY  centerX=$centerX", color = HudMuted)
                Button(onClick = {
                    step = if (step == WizardStep.HORIZON) WizardStep.CENTERLINE else WizardStep.NEAR_REF
                }) { Text("下一步") }
            }
            WizardStep.NEAR_REF -> {
                OutlinedTextField(nearM, { nearM = it }, label = { Text("近处参考距离 m") })
                Button(onClick = { step = WizardStep.KNOWN_DISTANCE }) { Text("下一步：已知距离") }
            }
            WizardStep.KNOWN_DISTANCE -> {
                Text("已知距离场（像素，可选）5 / 10 / 20 m", color = HudText)
                OutlinedTextField(d5, { d5 = it }, label = { Text("5m 像素 y") })
                OutlinedTextField(d10, { d10 = it }, label = { Text("10m 像素 y") })
                OutlinedTextField(d20, { d20 = it }, label = { Text("20m 像素 y") })
                Button(onClick = { step = WizardStep.HEALTH }) { Text("健康检查") }
            }
            WizardStep.HEALTH, WizardStep.DONE -> {
                OutlinedTextField(name, { name = it }, label = { Text("配置名称") })
                OutlinedTextField(height, { height = it }, label = { Text("相机高度 m") })
                OutlinedTextField(pitch, { pitch = it }, label = { Text("俯仰 deg") })
                OutlinedTextField(roll, { roll = it }, label = { Text("横滚 deg") })
                OutlinedTextField(yaw, { yaw = it }, label = { Text("偏航 deg") })
                Row {
                    Button(onClick = {
                        val pErr = roll.toDoubleOrNull() ?: 0.0
                        val rErr = pitch.toDoubleOrNull()?.let { kotlin.math.abs(it - 18.0) } ?: 0.0
                        val (ok, why) = runtime.calibration.health(
                            Transforms.defaultMount(), pErr, rErr,
                            runtime.cfg.geometry.pitchHealthDeg, runtime.cfg.geometry.rollHealthDeg,
                        )
                        healthMsg = if (ok) "健康检查通过" else "失败：$why（将隐藏精确距离，保留左/中/右）"
                        val saved = runtime.calibration.applyWizard(
                            runtime.pipeline.geometry.mount,
                            horizonY, centerX,
                            nearM.toDoubleOrNull() ?: 3.0,
                            d5.toDoubleOrNull(), d10.toDoubleOrNull(), d20.toDoubleOrNull(),
                            pitch.toDoubleOrNull() ?: 18.0,
                            roll.toDoubleOrNull() ?: 0.0,
                            yaw.toDoubleOrNull() ?: 0.0,
                            height.toDoubleOrNull() ?: 1.12,
                            name,
                        )
                        val hOk = ok
                        runtime.pipeline.geometry.setMount(saved.copy(valid = hOk))
                        runtime.pipeline.geometry.healthCheck(pErr, rErr)
                        step = WizardStep.DONE
                    }) { Text("保存配置") }
                }
                Text(healthMsg, color = HudText)
                runtime.calibration.list().forEach { p ->
                    Text("${p.profileId}  hash=${p.calibrationHash} valid=${p.valid}", color = HudMuted, fontSize = 12.sp)
                }
            }
        }
        Spacer(Modifier.height(16.dp))
        Button(onClick = { runtime.navigate(AppScreen.SETTINGS) }, colors = ButtonDefaults.buttonColors(containerColor = HudAccent)) {
            Text("完成", color = HudBg)
        }
    }
}
