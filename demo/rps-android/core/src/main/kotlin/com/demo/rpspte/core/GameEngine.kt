// Copyright (c) 2026 EdgeImpulse Inc.
// SPDX-License-Identifier: BSD-3-Clause-Clear

package com.demo.rpspte.core

import kotlin.random.Random

enum class Move {
    ROCK, PAPER, SCISSORS;

    fun beats(other: Move): Boolean = when (this) {
        ROCK -> other == SCISSORS
        PAPER -> other == ROCK
        SCISSORS -> other == PAPER
    }

    companion object {
        /** "Rock", "rock", " ROCK " -> ROCK; "unknown" or anything else -> null. */
        fun fromLabel(label: String): Move? = when (label.trim().lowercase()) {
            "rock" -> ROCK
            "paper" -> PAPER
            "scissors" -> SCISSORS
            else -> null
        }
    }
}

enum class Outcome { WIN, LOSE, TIE, NO_GESTURE }

enum class Phase { IDLE, COUNTDOWN, SHOOT, COLLECTING, RESULT }

data class RoundResult(
    val player: Move?,
    val phone: Move?,
    val outcome: Outcome,
    val votes: Map<Move, Int>,
    val framesSampled: Int,
    val avgLatencyMs: Double,
)

data class GameState(
    val phase: Phase = Phase.IDLE,
    /** 3, 2, 1 while phase == COUNTDOWN. */
    val countdown: Int = 0,
    val lastResult: RoundResult? = null,
    val wins: Int = 0,
    val losses: Int = 0,
    val ties: Int = 0,
)

/**
 * Timings and thresholds. The defaults are the ones rps.py ran with on the IQ9.
 */
data class GameConfig(
    val countdownSeconds: Int = 3,
    /** pause after "SHOOT!" so the hand stops moving */
    val settleMs: Long = 200,
    /** frames sampled for the vote */
    val voteFrames: Int = 8,
    /** a frame below this confidence does not get a vote */
    val confidenceMin: Float = 0.40f,
    /** the winning gesture needs at least this many votes, otherwise "no gesture" */
    val minVotes: Int = 3,
    /** give up collecting after this long even if fewer than voteFrames arrived */
    val collectTimeoutMs: Long = 2000,
    val resultHoldMs: Long = 3000,
)

/**
 * The rock-paper-scissors round, ported from rps.py's play_round():
 * countdown -> SHOOT -> settle -> sample [voteFrames] frames -> majority vote ->
 * the phone picks at random -> hold the result -> idle.
 *
 * The engine owns no threads and no clock: callers pass `nowMs` (any monotonic
 * millisecond clock). Camera frames arrive via [onFrame]; [tick] moves the clock
 * forward when no frame arrives (countdown, result hold). Every method is
 * synchronized so the camera thread and the UI ticker can both call in.
 */
class GameEngine(
    private val config: GameConfig = GameConfig(),
    private val pickPhoneMove: () -> Move = { Move.entries[Random.nextInt(Move.entries.size)] },
) {
    private var phase = Phase.IDLE
    private var phaseStartMs = 0L
    private var countdown = 0
    private var lastResult: RoundResult? = null
    private var wins = 0
    private var losses = 0
    private var ties = 0

    private val votes = HashMap<Move, Int>()
    private var framesSampled = 0
    private var latencySumMs = 0.0

    @Synchronized
    fun state(): GameState = GameState(phase, countdown, lastResult, wins, losses, ties)

    /** Start a round. Ignored while one is already running. */
    @Synchronized
    fun start(nowMs: Long) {
        if (phase != Phase.IDLE && phase != Phase.RESULT) return
        votes.clear()
        framesSampled = 0
        latencySumMs = 0.0
        phase = Phase.COUNTDOWN
        phaseStartMs = nowMs
        countdown = config.countdownSeconds
    }

    @Synchronized
    fun tick(nowMs: Long) {
        advance(nowMs)
    }

    /** Feed one classified camera frame. Only frames in the COLLECTING phase count. */
    @Synchronized
    fun onFrame(prediction: Prediction, nowMs: Long) {
        advance(nowMs)
        if (phase != Phase.COLLECTING) return

        framesSampled++
        latencySumMs += prediction.latencyMs
        val move = prediction.move
        if (move != null && prediction.confidence >= config.confidenceMin) {
            votes[move] = (votes[move] ?: 0) + 1
        }
        if (framesSampled >= config.voteFrames) finish(nowMs)
    }

    private fun advance(nowMs: Long) {
        val elapsed = nowMs - phaseStartMs
        when (phase) {
            Phase.IDLE -> Unit
            Phase.COUNTDOWN -> {
                val total = config.countdownSeconds * 1000L
                if (elapsed >= total) {
                    phase = Phase.SHOOT
                    phaseStartMs = nowMs
                    countdown = 0
                    advance(nowMs)
                } else {
                    countdown = config.countdownSeconds - (elapsed / 1000L).toInt()
                }
            }
            Phase.SHOOT -> if (elapsed >= config.settleMs) {
                phase = Phase.COLLECTING
                phaseStartMs = nowMs
            }
            Phase.COLLECTING -> if (elapsed >= config.collectTimeoutMs) finish(nowMs)
            Phase.RESULT -> if (elapsed >= config.resultHoldMs) {
                phase = Phase.IDLE
            }
        }
    }

    private fun finish(nowMs: Long) {
        val top = votes.maxByOrNull { it.value }
        val topVotes = top?.value ?: 0
        val tiedAtTop = votes.count { it.value == topVotes } > 1
        val player = if (top != null && topVotes >= config.minVotes && !tiedAtTop) top.key else null

        val avgMs = if (framesSampled > 0) latencySumMs / framesSampled else 0.0
        val result = if (player == null) {
            RoundResult(null, null, Outcome.NO_GESTURE, votes.toMap(), framesSampled, avgMs)
        } else {
            val phoneMove = pickPhoneMove()
            val outcome = when {
                player == phoneMove -> Outcome.TIE
                player.beats(phoneMove) -> Outcome.WIN
                else -> Outcome.LOSE
            }
            when (outcome) {
                Outcome.WIN -> wins++
                Outcome.LOSE -> losses++
                else -> ties++
            }
            RoundResult(player, phoneMove, outcome, votes.toMap(), framesSampled, avgMs)
        }
        lastResult = result
        phase = Phase.RESULT
        phaseStartMs = nowMs
    }
}
