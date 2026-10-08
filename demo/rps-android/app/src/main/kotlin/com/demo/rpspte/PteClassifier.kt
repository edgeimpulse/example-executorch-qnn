package com.demo.rpspte

import com.demo.rpspte.core.Prediction
import com.demo.rpspte.core.Preprocess
import com.demo.rpspte.core.RgbaFrame
import com.demo.rpspte.core.Scores
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * Loads a .pte with the ExecuTorch Android runtime and classifies camera frames.
 * The Android twin of `Classifier` in rps.py: the same model file, the same probing of
 * the input size, the same preprocessing (see core/Preprocess.kt).
 *
 * Not thread-safe: call [classify] from one thread (the camera analysis thread).
 */
class PteClassifier(modelPath: String, val labels: List<String>) {

    private val module: Module = Module.load(modelPath)

    /** Side of the square input the model wants, found by trying candidate sizes. */
    val inputSize: Int

    /** Number of values the model returns per frame; should equal labels.size. */
    val numOutputs: Int

    /** The last error seen while probing; module loading can be lazy, so a bad .pte surfaces here. */
    private var lastError: Throwable? = null

    init {
        var found = 0
        var outputs = 0
        for (size in CANDIDATE_SIZES) {
            val r = tryForward(FloatArray(3 * size * size) { 0.5f }, size)
            if (r != null && r.first.isNotEmpty()) {
                found = size
                outputs = r.first.size
                break
            }
        }
        if (found == 0) {
            throw IllegalStateException(
                "Could not find an input size the model accepts (tried ${CANDIDATE_SIZES.joinToString()}). " +
                    "Last runtime error: ${lastError?.message ?: "none"}",
                lastError,
            )
        }
        inputSize = found
        numOutputs = outputs
        // The first inferences are slow while XNNPACK packs weights; do them now.
        repeat(2) { tryForward(FloatArray(3 * found * found) { 0.5f }, found) }
    }

    /** (output values, milliseconds spent in forward) or null if the runtime rejected the input. */
    private fun tryForward(input: FloatArray, size: Int): Pair<FloatArray, Double>? = try {
        val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, size.toLong(), size.toLong()))
        val t0 = System.nanoTime()
        val out = module.forward(EValue.from(tensor))
        val ms = (System.nanoTime() - t0) / 1e6
        if (out.isEmpty()) null else Pair(out[0].toTensor().dataAsFloatArray, ms)
    } catch (t: Throwable) {
        lastError = t
        null
    }

    /**
     * @param roiPercent side of the classified square as a percentage of the frame's
     *                   shorter side; 100 reproduces `rps.py --image`
     */
    fun classify(frame: RgbaFrame, roiPercent: Int): Prediction {
        val input = Preprocess.toNchw(frame, inputSize, roiPercent)
        val (raw, ms) = checkNotNull(tryForward(input, inputSize)) { "forward() failed" }
        val probs = Scores.probabilities(raw)
        val best = Scores.argmax(probs)
        val label = labels.getOrElse(best) { "?" }.lowercase()
        return Prediction(label, probs[best], probs, ms)
    }

    companion object {
        /** Same list, same order, as CANDIDATE_SIZES in rps.py. */
        private val CANDIDATE_SIZES = intArrayOf(96, 160, 128, 224, 192, 320, 64, 48)
    }
}
