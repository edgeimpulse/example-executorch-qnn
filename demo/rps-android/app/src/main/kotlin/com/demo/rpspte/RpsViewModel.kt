package com.demo.rpspte

import android.app.Application
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.demo.rpspte.core.GameEngine
import com.demo.rpspte.core.GameState
import com.demo.rpspte.core.Prediction
import com.demo.rpspte.core.RgbaFrame
import com.demo.rpspte.core.Scores
import java.io.File
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class ModelInfo(
    val fingerprint: String,
    val sizeMb: Double,
    val inputSize: Int,
    val labels: List<String>,
    val source: String,
)

sealed interface ModelStatus {
    data object Loading : ModelStatus
    data object Ready : ModelStatus
    data class Error(val message: String) : ModelStatus
}

data class UiState(
    val status: ModelStatus = ModelStatus.Loading,
    val info: ModelInfo? = null,
    val live: Prediction? = null,
    /** rolling mean of the last 30 forward() calls, in ms */
    val avgLatencyMs: Double = 0.0,
    val fps: Double = 0.0,
    val game: GameState = GameState(),
    val selfTest: String? = null,
)

class RpsViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui

    private val engine = GameEngine()

    // Module is not thread-safe: the camera thread and the self-test share this lock.
    private val classifierLock = Any()
    @Volatile private var classifier: PteClassifier? = null

    // Touched only from the camera analysis thread.
    private val latencies = ArrayDeque<Double>()
    private var lastFrameNs = 0L
    private var fpsEma = 0.0

    init {
        viewModelScope.launch(Dispatchers.Default) { loadModel() }
        // Drives the countdown and the result hold even when no frame arrives.
        viewModelScope.launch {
            while (isActive) {
                delay(33)
                engine.tick(now())
                _ui.update { it.copy(game = engine.state()) }
            }
        }
    }

    private fun now() = SystemClock.elapsedRealtime()

    private fun loadModel() {
        val ctx = getApplication<Application>()
        val resolved = ModelFiles.resolve(ctx)
        if (resolved == null) {
            fail(
                "No model found.\nCopy model.pte and labels.txt from deploy.zip into " +
                    "app/src/main/assets/ and rebuild, or adb push them to " +
                    "/sdcard/Android/data/${ctx.packageName}/files/ (see the README)."
            )
            return
        }
        try {
            val labels = Scores.parseLabels(resolved.labels.readText())
            val c = PteClassifier(resolved.model.absolutePath, labels)
            if (c.numOutputs != labels.size) {
                fail(
                    "The model returns ${c.numOutputs} values per frame but labels.txt has " +
                        "${labels.size} lines. labels.txt must come from the same deploy.zip as the model."
                )
                return
            }
            val info = ModelInfo(
                fingerprint = ModelFiles.fingerprint(resolved.model),
                sizeMb = resolved.model.length() / 1_048_576.0,
                inputSize = c.inputSize,
                labels = labels,
                source = resolved.source,
            )
            classifier = c
            _ui.update { it.copy(status = ModelStatus.Ready, info = info) }
        } catch (t: Throwable) {
            fail(
                "Could not load the model: ${t.message ?: t.javaClass.simpleName}\n" +
                    "A load failure that mentions the program or its version usually means the .pte " +
                    "was exported by a different ExecuTorch release than the Android runtime " +
                    "(README, \"Versions\")."
            )
        }
    }

    private fun fail(message: String) {
        _ui.update { it.copy(status = ModelStatus.Error(message)) }
    }

    /** Called on the camera analysis thread for every frame. */
    fun analyze(frame: RgbaFrame) {
        val c = classifier ?: return
        val prediction = try {
            synchronized(classifierLock) { c.classify(frame, ROI_PERCENT) }
        } catch (t: Throwable) {
            return
        }

        engine.onFrame(prediction, now())

        latencies.addLast(prediction.latencyMs)
        if (latencies.size > 30) latencies.removeFirst()

        val t = System.nanoTime()
        if (lastFrameNs != 0L) {
            val inst = 1e9 / (t - lastFrameNs)
            fpsEma = if (fpsEma == 0.0) inst else fpsEma * 0.9 + inst * 0.1
        }
        lastFrameNs = t

        val avg = latencies.average()
        val fps = fpsEma
        _ui.update { it.copy(live = prediction, avgLatencyMs = avg, fps = fps, game = engine.state()) }
    }

    fun play() {
        if (_ui.value.status != ModelStatus.Ready) return
        engine.start(now())
        _ui.update { it.copy(game = engine.state()) }
    }

    /**
     * Runs one known photo through the model with the same preprocessing as
     * `rps.py --image` (centre-crop, no ROI), so the scores can be compared line by line
     * with the IQ9. Looks for selftest.jpg in the adb folder first, then in assets.
     */
    fun runSelfTest() {
        val c = classifier ?: return
        viewModelScope.launch(Dispatchers.Default) {
            val ctx = getApplication<Application>()
            val external = ctx.getExternalFilesDir(null)?.let { File(it, "selftest.jpg") }
            val bitmap = if (external != null && external.isFile) {
                BitmapFactory.decodeFile(external.path)
            } else {
                try {
                    ctx.assets.open("selftest.jpg").use { BitmapFactory.decodeStream(it) }
                } catch (e: IOException) {
                    null
                }
            }
            if (bitmap == null) {
                _ui.update {
                    it.copy(selfTest = "No selftest.jpg found. Put one in assets/ or adb push it (see README).")
                }
                return@launch
            }

            val w = bitmap.width
            val h = bitmap.height
            val prediction = try {
                val px = IntArray(w * h)
                bitmap.getPixels(px, 0, w, 0, 0, w, h)
                val rgba = ByteArray(w * h * 4)
                for (i in px.indices) {
                    val p = px[i]
                    rgba[i * 4] = (p shr 16).toByte()
                    rgba[i * 4 + 1] = (p shr 8).toByte()
                    rgba[i * 4 + 2] = p.toByte()
                    rgba[i * 4 + 3] = 0xFF.toByte()
                }
                synchronized(classifierLock) { c.classify(RgbaFrame(rgba, w, h), 100) }
            } catch (t: Throwable) {
                // Includes OutOfMemoryError: a 12 MP photo needs ~150 MB here. Use a photo of
                // about 1600 px or less on the long side (and the same file in reference_scores.py).
                _ui.update { it.copy(selfTest = "Self-test failed (${w}x$h): ${t.message ?: t.javaClass.simpleName}") }
                return@launch
            }

            val labels = c.labels
            val scores = labels.indices.joinToString("   ") { i ->
                "${labels[i].lowercase()} ${"%.3f".format(prediction.probs[i])}"
            }
            _ui.update {
                it.copy(
                    selfTest = "${w}x$h -> ${c.inputSize}x${c.inputSize}\n$scores\n" +
                        "${"%.2f".format(prediction.latencyMs)} ms"
                )
            }
        }
    }

    companion object {
        /** The classified square is this % of the frame's shorter side (pte_perception ROI_FRACTION 0.6). */
        const val ROI_PERCENT = 60
    }
}
