// Copyright (c) 2026 EdgeImpulse Inc.
// SPDX-License-Identifier: BSD-3-Clause-Clear

package com.demo.rpspte.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class GameEngineTest {

    private fun pred(label: String, conf: Float = 0.9f, ms: Double = 1.0) =
        Prediction(label, conf, floatArrayOf(), ms)

    private fun engine(phone: Move = Move.SCISSORS, config: GameConfig = GameConfig()) =
        GameEngine(config) { phone }

    /** Runs the countdown and the settle period; returns the time at which voting starts. */
    private fun toCollecting(e: GameEngine, start: Long = 0): Long {
        e.start(start)
        e.tick(start + 3000)          // countdown over -> SHOOT
        assertEquals(Phase.SHOOT, e.state().phase)
        e.tick(start + 3000 + 200)    // settled -> COLLECTING
        assertEquals(Phase.COLLECTING, e.state().phase)
        return start + 3200
    }

    @Test
    fun countdownCountsThreeTwoOne() {
        val e = engine()
        e.start(1000)
        assertEquals(Phase.COUNTDOWN, e.state().phase)
        e.tick(1000); assertEquals(3, e.state().countdown)
        e.tick(2000); assertEquals(2, e.state().countdown)
        e.tick(3000); assertEquals(1, e.state().countdown)
        e.tick(3999); assertEquals(1, e.state().countdown)
        e.tick(4000); assertEquals(Phase.SHOOT, e.state().phase)
    }

    @Test
    fun framesBeforeCollectingAreIgnored() {
        val e = engine()
        e.start(0)
        repeat(20) { e.onFrame(pred("rock"), 500L + it) }
        assertEquals(Phase.COUNTDOWN, e.state().phase)
        assertNull(e.state().lastResult)
    }

    @Test
    fun playerWinsWithRockAgainstScissors() {
        val e = engine(phone = Move.SCISSORS)
        var t = toCollecting(e)
        repeat(8) { e.onFrame(pred("Rock"), t++) }
        val s = e.state()
        assertEquals(Phase.RESULT, s.phase)
        val r = assertNotNull(s.lastResult)
        assertEquals(Outcome.WIN, r.outcome)
        assertEquals(Move.ROCK, r.player)
        assertEquals(8, r.framesSampled)
        assertEquals(1, s.wins)
    }

    @Test
    fun outcomeTable() {
        val cases = listOf(
            Triple(Move.ROCK, Move.SCISSORS, Outcome.WIN),
            Triple(Move.ROCK, Move.PAPER, Outcome.LOSE),
            Triple(Move.ROCK, Move.ROCK, Outcome.TIE),
            Triple(Move.PAPER, Move.ROCK, Outcome.WIN),
            Triple(Move.PAPER, Move.SCISSORS, Outcome.LOSE),
            Triple(Move.SCISSORS, Move.PAPER, Outcome.WIN),
            Triple(Move.SCISSORS, Move.ROCK, Outcome.LOSE),
        )
        for ((player, phone, expected) in cases) {
            val e = engine(phone = phone)
            var t = toCollecting(e)
            repeat(8) { e.onFrame(pred(player.name.lowercase()), t++) }
            assertEquals(expected, e.state().lastResult?.outcome, "$player vs $phone")
        }
    }

    @Test
    fun majorityVoteBeatsAFlickerFrame() {
        val e = engine()
        var t = toCollecting(e)
        listOf("paper", "paper", "rock", "paper", "paper", "scissors", "paper", "paper")
            .forEach { e.onFrame(pred(it), t++) }
        assertEquals(Move.PAPER, e.state().lastResult?.player)
    }

    @Test
    fun lowConfidenceAndUnknownFramesGetNoVote() {
        val e = engine()
        var t = toCollecting(e)
        repeat(4) { e.onFrame(pred("rock", conf = 0.39f), t++) }
        repeat(4) { e.onFrame(pred("unknown", conf = 0.99f), t++) }
        val r = assertNotNull(e.state().lastResult)
        assertEquals(Outcome.NO_GESTURE, r.outcome)
        assertNull(r.player)
        assertEquals(0, e.state().wins + e.state().losses + e.state().ties)
    }

    @Test
    fun tooFewVotesIsNoGesture() {
        val e = engine()
        var t = toCollecting(e)
        repeat(2) { e.onFrame(pred("rock"), t++) }
        repeat(6) { e.onFrame(pred("unknown"), t++) }
        assertEquals(Outcome.NO_GESTURE, e.state().lastResult?.outcome)
    }

    @Test
    fun aTieAtTheTopIsNoGesture() {
        val e = engine()
        var t = toCollecting(e, start = 0)
        repeat(4) { e.onFrame(pred("rock"), t++) }
        repeat(4) { e.onFrame(pred("paper"), t++) }
        assertEquals(Outcome.NO_GESTURE, e.state().lastResult?.outcome)
    }

    @Test
    fun timeoutFinishesARoundThatNeverGetsEnoughFrames() {
        val e = engine()
        val t = toCollecting(e)
        repeat(3) { e.onFrame(pred("rock"), t + it) }
        assertEquals(Phase.COLLECTING, e.state().phase)
        e.tick(t + 2000)
        val r = assertNotNull(e.state().lastResult)
        assertEquals(3, r.framesSampled)
        assertEquals(Move.ROCK, r.player)
    }

    @Test
    fun resultHoldsThenReturnsToIdleAndKeepsTheScore() {
        val e = engine(phone = Move.SCISSORS)
        var t = toCollecting(e)
        repeat(8) { e.onFrame(pred("rock"), t++) }
        val finishedAt = t - 1
        e.tick(finishedAt + 2999); assertEquals(Phase.RESULT, e.state().phase)
        e.tick(finishedAt + 3000); assertEquals(Phase.IDLE, e.state().phase)
        assertEquals(1, e.state().wins)
        assertNotNull(e.state().lastResult)
    }

    @Test
    fun startIsIgnoredMidRoundButAllowedFromResult() {
        val e = engine()
        e.start(0)
        e.start(500)                           // ignored: already counting down
        e.tick(2999)
        assertEquals(1, e.state().countdown)   // still on the first start's clock
        // fresh engine: finish a round, then start again straight from RESULT
        val e2 = engine()
        var t = toCollecting(e2)
        repeat(8) { e2.onFrame(pred("rock"), t++) }
        assertEquals(Phase.RESULT, e2.state().phase)
        e2.start(t)
        assertEquals(Phase.COUNTDOWN, e2.state().phase)
    }

    @Test
    fun averageLatencyIsReported() {
        val e = engine()
        var t = toCollecting(e)
        repeat(8) { e.onFrame(pred("rock", ms = 2.0), t++) }
        assertEquals(2.0, e.state().lastResult!!.avgLatencyMs, 1e-9)
    }
}
