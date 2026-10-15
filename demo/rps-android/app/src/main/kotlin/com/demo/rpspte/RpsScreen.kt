// Copyright (c) 2026 EdgeImpulse Inc.
// SPDX-License-Identifier: BSD-3-Clause-Clear

package com.demo.rpspte

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.Canvas
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.demo.rpspte.core.GameConfig
import com.demo.rpspte.core.GameState
import com.demo.rpspte.core.Move
import com.demo.rpspte.core.Outcome
import com.demo.rpspte.core.Phase

private val Green = Color(0xFF4ADE80)
private val Red = Color(0xFFF87171)
private val Amber = Color(0xFFFBBF24)
private val Muted = Color(0xFF9CA3AF)

private fun Move.emoji() = when (this) {
    Move.ROCK -> "✊"
    Move.PAPER -> "✋"
    Move.SCISSORS -> "✌️"
}

@Composable
fun RpsApp(vm: RpsViewModel) {
    MaterialTheme(colorScheme = darkColorScheme()) {
        RpsScreen(vm)
    }
}

@Composable
private fun RpsScreen(vm: RpsViewModel) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    val context = LocalContext.current

    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) launcher.launch(Manifest.permission.CAMERA) }

    var front by remember { mutableStateOf(true) }

    Column(
        Modifier
            .fillMaxSize()
            .background(Color.Black)
            .systemBarsPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Header()

        when (val status = ui.status) {
            is ModelStatus.Loading -> Box(Modifier.fillMaxWidth().padding(48.dp), Alignment.Center) {
                CircularProgressIndicator()
            }
            is ModelStatus.Error -> Text(
                status.message,
                color = Red,
                fontSize = 15.sp,
                modifier = Modifier.padding(vertical = 24.dp),
            )
            is ModelStatus.Ready -> {
                CameraBox(vm, ui.game, granted, front)
                LiveRow(ui)
                StatsRow(ui)
                ScoreRow(ui.game)
                Controls(
                    game = ui.game,
                    onPlay = vm::play,
                    onFlip = { front = !front },
                    onSelfTest = vm::runSelfTest,
                )
                ui.selfTest?.let {
                    Text(it, color = Muted, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                }
                Footer(ui)
            }
        }
    }
}

@Composable
private fun Header() {
    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
        Text("Rock · Paper · Scissors", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
        Text("on-device", color = Muted, fontSize = 13.sp)
    }
}

@Composable
private fun CameraBox(vm: RpsViewModel, game: GameState, granted: Boolean, front: Boolean) {
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(3f / 4f)
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF111111)),
        contentAlignment = Alignment.Center,
    ) {
        if (granted) {
            CameraPreview(front = front, onFrame = vm::analyze, modifier = Modifier.fillMaxSize())
        } else {
            Text("Camera permission is needed to play.", color = Muted, textAlign = TextAlign.Center)
        }

        val roiColor = when (game.phase) {
            Phase.IDLE -> Color.White.copy(alpha = 0.7f)
            Phase.COUNTDOWN -> Amber
            Phase.SHOOT, Phase.COLLECTING -> Green
            Phase.RESULT -> when (game.lastResult?.outcome) {
                Outcome.WIN -> Green
                Outcome.LOSE -> Red
                else -> Muted
            }
        }
        // The classified square: centred, ROI_PERCENT of the frame's shorter side (its width).
        Canvas(Modifier.fillMaxSize()) {
            val side = size.width * RpsViewModel.ROI_PERCENT / 100f
            drawRoundRect(
                color = roiColor,
                topLeft = Offset((size.width - side) / 2f, (size.height - side) / 2f),
                size = Size(side, side),
                cornerRadius = CornerRadius(20f, 20f),
                style = Stroke(width = 6f),
            )
        }

        when (game.phase) {
            Phase.COUNTDOWN -> BigText(game.countdown.toString(), Amber)
            Phase.SHOOT, Phase.COLLECTING -> BigText("SHOOT!", Green)
            Phase.RESULT -> game.lastResult?.let { r ->
                val (headline, color) = when (r.outcome) {
                    Outcome.WIN -> "YOU WIN" to Green
                    Outcome.LOSE -> "PHONE WINS" to Red
                    Outcome.TIE -> "TIE" to Amber
                    Outcome.NO_GESTURE -> "NO GESTURE" to Muted
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    BigText(headline, color, size = 44)
                    val you = r.player
                    val phone = r.phone
                    if (you != null && phone != null) {
                        Text(
                            "You ${you.emoji()}   Phone ${phone.emoji()}",
                            color = Color.White,
                            fontSize = 28.sp,
                            modifier = Modifier.padding(top = 4.dp),
                        )
                    } else {
                        Text("Hold your hand inside the box", color = Color.White, fontSize = 16.sp)
                    }
                }
            }
            Phase.IDLE -> Unit
        }
    }
}

@Composable
private fun BigText(text: String, color: Color, size: Int = 96) {
    Text(text, color = color, fontSize = size.sp, fontWeight = FontWeight.ExtraBold, textAlign = TextAlign.Center)
}

@Composable
private fun LiveRow(ui: UiState) {
    val live = ui.live
    val confident = live != null && live.confidence >= GameConfig().confidenceMin && live.move != null
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
            Text(
                if (confident) live!!.label.replaceFirstChar { it.uppercase() } else "…",
                color = Color.White,
                fontSize = 22.sp,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                if (live != null) "${(live.confidence * 100).toInt()}%" else "",
                color = Muted,
                fontSize = 18.sp,
            )
        }
        LinearProgressIndicator(
            progress = { live?.confidence ?: 0f },
            modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
            color = if (confident) Green else Muted,
        )
    }
}

@Composable
private fun StatsRow(ui: UiState) {
    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
        Stat("inference", if (ui.avgLatencyMs > 0) "%.2f ms".format(ui.avgLatencyMs) else "–")
        Stat("camera", if (ui.fps > 0) "%.0f fps".format(ui.fps) else "–")
        Stat("input", ui.info?.let { "${it.inputSize}×${it.inputSize}" } ?: "–")
    }
}

@Composable
private fun Stat(label: String, value: String) {
    Column {
        Text(label, color = Muted, fontSize = 11.sp)
        Text(value, color = Color.White, fontSize = 16.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun ScoreRow(game: GameState) {
    Row(Modifier.fillMaxWidth(), Arrangement.SpaceEvenly) {
        Text("Wins ${game.wins}", color = Green, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        Text("Ties ${game.ties}", color = Amber, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        Text("Losses ${game.losses}", color = Red, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun Controls(game: GameState, onPlay: () -> Unit, onFlip: () -> Unit, onSelfTest: () -> Unit) {
    val canPlay = game.phase == Phase.IDLE || game.phase == Phase.RESULT
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Button(onClick = onPlay, enabled = canPlay, modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
            Text(if (canPlay) "PLAY" else "…", fontSize = 20.sp, fontWeight = FontWeight.Bold)
        }
        Row(Modifier.fillMaxWidth(), Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onFlip, modifier = Modifier.weight(1f)) { Text("Flip camera") }
            TextButton(onClick = onSelfTest, modifier = Modifier.weight(1f)) { Text("Self-test") }
        }
    }
}

@Composable
private fun Footer(ui: UiState) {
    val info = ui.info ?: return
    Text(
        "model.pte · sha256 ${info.fingerprint} · ${"%.1f".format(info.sizeMb)} MB\n" +
            "ExecuTorch ${BuildConfig.EXECUTORCH_VERSION} · ${BuildConfig.BACKEND} · ${info.source}",
        color = Muted,
        fontSize = 11.sp,
        fontFamily = FontFamily.Monospace,
    )
}
