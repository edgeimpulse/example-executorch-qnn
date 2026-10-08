package com.demo.rpspte.core

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ScoresTest {

    @Test
    fun probabilitiesPassThroughUntouched() {
        val p = floatArrayOf(0.05f, 0.88f, 0.04f, 0.03f)
        assertTrue(Scores.probabilities(p) === p)
    }

    @Test
    fun logitsGetSoftmaxed() {
        val out = Scores.probabilities(floatArrayOf(2.0f, 0.0f, -1.0f, 5.0f))
        assertTrue(abs(out.sum() - 1f) < 1e-5f)
        assertEquals(3, Scores.argmax(out))
    }

    @Test
    fun anAllZeroOutputDoesNotPassAsProbabilities() {
        val out = Scores.probabilities(floatArrayOf(0f, 0f, 0f, 0f))
        assertTrue(out.all { abs(it - 0.25f) < 1e-6f })
    }

    @Test
    fun labelsKeepModelOrderAndDropBlankLines() {
        val labels = Scores.parseLabels("Paper\r\nRock\n\n  Scissors \nunknown\n")
        assertEquals(listOf("Paper", "Rock", "Scissors", "unknown"), labels)
    }

    @Test
    fun labelsMapToMoves() {
        assertEquals(Move.ROCK, Move.fromLabel("Rock"))
        assertEquals(Move.SCISSORS, Move.fromLabel(" SCISSORS "))
        assertEquals(null, Move.fromLabel("unknown"))
    }
}
