package com.demo.rpspte.core

import kotlin.math.exp

/** What the classifier says about one frame. */
data class Prediction(
    /** Lower-cased class name, e.g. "rock". "unknown" and anything else is not a gesture. */
    val label: String,
    val confidence: Float,
    /** Probability per class, in the model's output order. */
    val probs: FloatArray,
    /** Time spent inside Module.forward, in milliseconds. */
    val latencyMs: Double,
) {
    val move: Move? get() = Move.fromLabel(label)
}

object Scores {

    /**
     * The CPU `model.pte` from the learning block has softmax baked in, the QNN one
     * returns logits. Accept either: if the numbers already look like a probability
     * distribution they are returned untouched, otherwise a softmax is applied.
     */
    fun probabilities(raw: FloatArray): FloatArray {
        if (raw.isEmpty()) return raw
        val sum = raw.sum()
        val looksLikeProbs = raw.all { it in -1e-4f..1.0001f } && sum in 0.98f..1.02f
        if (looksLikeProbs) return raw
        val max = raw.max()
        val e = FloatArray(raw.size) { exp((raw[it] - max).toDouble()).toFloat() }
        val total = e.sum()
        return FloatArray(raw.size) { e[it] / total }
    }

    fun argmax(values: FloatArray): Int {
        var best = 0
        for (i in values.indices) if (values[i] > values[best]) best = i
        return best
    }

    /** One label per line, in the model's output order (deploy.zip's labels.txt). */
    fun parseLabels(text: String): List<String> =
        text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.toList()
}
