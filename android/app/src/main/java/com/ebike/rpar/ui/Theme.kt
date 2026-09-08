package com.ebike.rpar.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val HudBg = Color(0xFF080A0E)
val HudAccent = Color(0xFF28D2C8)
val HudText = Color(0xFFECF4F8)
val HudMuted = Color(0xFF9AA8B0)
val HudPanel = Color(0xCC0A1016)

private val scheme = darkColorScheme(
    primary = HudAccent,
    onPrimary = HudBg,
    background = HudBg,
    surface = HudPanel,
    onBackground = HudText,
    onSurface = HudText,
)

@Composable
fun RparTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = scheme, content = content)
}
