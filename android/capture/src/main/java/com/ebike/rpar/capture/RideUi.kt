package com.ebike.rpar.capture

import android.view.TextureView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView

object RideColors {
    val bg = Color(0xFF07080A)
    val panel = Color(0xCC0B1014)
    val text = Color(0xFFE8F2EC)
    val muted = Color(0xFF8AA094)
    val accent = Color(0xFF3DDC97)
    val rec = Color(0xFFE24B4B)
}

@Composable
fun RideCaptureScreen(
    ui: RideUi,
    modifier: Modifier = Modifier,
    onAccept: () -> Unit,
    onRequestPerms: () -> Unit,
    onReady: (TextureView) -> Unit,
    onToggle: (TextureView) -> Unit,
    onMark: () -> Unit,
    onShare: () -> Unit,
) {
    if (!ui.accepted) {
        Column(modifier.padding(28.dp), verticalArrangement = Arrangement.Center) {
            Text("骑行采集", color = RideColors.accent, fontSize = 30.sp)
            Spacer(Modifier.height(10.dp))
            Text("装在车把上，出门按一下就开始录。只写本机：视频 + IMU + GPS + 你点的标记。", color = RideColors.text, fontSize = 16.sp)
            Spacer(Modifier.height(8.dp))
            Text("不能替代你自己看路。颠簸不会停录，只会打一个标记。数据不上传。", color = RideColors.muted, fontSize = 14.sp)
            Spacer(Modifier.height(20.dp))
            Row {
                Button(onClick = onRequestPerms, colors = ButtonDefaults.buttonColors(containerColor = RideColors.panel)) {
                    Text("授予相机和定位", color = RideColors.text)
                }
                Spacer(Modifier.width(12.dp))
                Button(onClick = onAccept, colors = ButtonDefaults.buttonColors(containerColor = RideColors.accent)) {
                    Text("开始骑", color = RideColors.bg)
                }
            }
        }
        return
    }
    var preview by remember { mutableStateOf<TextureView?>(null) }
    Box(modifier) {
        AndroidView(
            factory = { ctx -> TextureView(ctx).also { preview = it; onReady(it) } },
            modifier = Modifier.fillMaxSize(),
        )
        Column(Modifier.fillMaxSize().padding(16.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Pill(if (ui.recording) recClock(ui.elapsedS) else "待机", rec = ui.recording)
                Pill(ui.speedKmh?.let { "%.0f km/h".format(it) } ?: "-- km/h")
                Pill(if (ui.gps) "GPS" else "无定位")
                Pill("约 %.1f h".format(ui.remainingH))
                Pill(if (ui.battery >= 0) "${ui.battery}%" else "电量")
            }
            Spacer(Modifier.height(8.dp))
            Text(
                listOfNotNull(
                    ui.camera.ifBlank { null },
                    if (ui.imuHz > 0) "IMU ${ui.imuHz} Hz" else null,
                    if (ui.marks > 0) "标记 ${ui.marks}" else null,
                    if (ui.impactHits > 0) "颠簸 ${ui.impactHits}" else null,
                ).joinToString("  ·  "),
                color = RideColors.muted,
                fontSize = 12.sp,
            )
            Spacer(Modifier.weight(1f))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.Bottom) {
                Column {
                    Text("点「标记」记下坑/带。回家把 ZIP 拷到电脑，用控制台回放。", color = RideColors.muted, fontSize = 12.sp)
                    if (ui.sessions.isNotEmpty()) {
                        Text("最近：${ui.sessions.first()}", color = RideColors.muted, fontSize = 11.sp)
                    }
                    ui.lastError?.let { Text(it, color = RideColors.rec, fontSize = 12.sp) }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = onShare) { Text("导出", color = RideColors.text) }
                    Spacer(Modifier.width(8.dp))
                    Button(
                        onClick = onMark,
                        colors = ButtonDefaults.buttonColors(containerColor = RideColors.panel),
                    ) { Text("标记", color = RideColors.text, fontSize = 18.sp) }
                    Spacer(Modifier.width(12.dp))
                    Button(
                        onClick = { preview?.let(onToggle) },
                        modifier = Modifier.size(88.dp),
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(containerColor = if (ui.recording) RideColors.rec else RideColors.accent),
                    ) {
                        Text(if (ui.recording) "停" else "录", color = Color.White, fontSize = 22.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun Pill(text: String, rec: Boolean = false) {
    Box(
        Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(RideColors.panel)
            .border(1.dp, if (rec) RideColors.rec else RideColors.accent, RoundedCornerShape(999.dp))
            .padding(horizontal = 12.dp, vertical = 6.dp),
    ) {
        Text(text, color = RideColors.text, fontSize = 15.sp)
    }
}

private fun recClock(s: Long): String {
    val h = s / 3600
    val m = (s % 3600) / 60
    val sec = s % 60
    return if (h > 0) "%d:%02d:%02d".format(h, m, sec) else "%02d:%02d".format(m, sec)
}
