# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Game rules + the match state machine."""

import asyncio
import random
import time
from contextlib import aclosing, suppress
from dataclasses import replace
from typing import AsyncIterator, Awaitable, Callable

import config
from actuation import Actuator
from commentary import Commentator
from data_model import BEATS, Detection, Gesture, Match, MatchConfig, Outcome, Phase, Round, Strategy
from perception import GestureDetector
from speech import Speaker

# Inverse of BEATS: each gesture -> the gesture that beats it.
_BEATEN_BY = {loser: winner for winner, loser in BEATS.items()}


def counter_gesture(gesture: Gesture) -> Gesture:
    return _BEATEN_BY[gesture]


def choose_machine_gesture(strategy: Strategy, human: Gesture | None) -> Gesture:
    if strategy is Strategy.RANDOM or human is None:
        return random.choice(list(Gesture))
    if strategy is Strategy.ALWAYS_WIN:
        return counter_gesture(human)
    if strategy is Strategy.ALWAYS_LOSE:
        return BEATS[human]
    raise ValueError(strategy)


def resolve_outcome(human: Gesture, machine: Gesture) -> Outcome:
    if human == machine:
        return Outcome.DRAW
    return Outcome.HUMAN if BEATS[human] == machine else Outcome.MACHINE


BroadcastFn = Callable[[Match], Awaitable[None]]
CommentaryFn = Callable[[dict], Awaitable[None]]
SnapshotFn = Callable[[Detection | None], bytes | None]

_COMMENTARY_LINGER_SECONDS = config.COMMENTARY_LINGER_SECONDS
_TTS_CHUNK_MIN_CHARS = config.TTS_CHUNK_MIN_CHARS
_TTS_CHUNK_MAX_CHARS = config.TTS_CHUNK_MAX_CHARS
_TTS_SAMPLE_RATE = config.TTS_SAMPLE_RATE
_READING_CHARS_PER_SEC = config.COMMENTARY_READING_CHARS_PER_SEC
_SEGMENT_BREATH_SECONDS = config.TTS_SEGMENT_BREATH_SECONDS
_COUNTDOWN_READY_SECONDS = config.COUNTDOWN_READY_SECONDS
_COUNTDOWN_READY_PAUSE_SECONDS = config.COUNTDOWN_READY_PAUSE_SECONDS
_COUNTDOWN_WORD_PAUSE_SECONDS = config.COUNTDOWN_WORD_PAUSE_SECONDS
_AUDIO_ONSET_SECONDS = config.AUDIO_ONSET_SECONDS

# Sentinel that tells the playback worker no more audio is coming.
_PLAYBACK_DONE = object()
# Sentinel that clears the browser panel before the next spoken segment (so a
# filler lead-in is overwritten when the real remark begins).
_SEGMENT_START = object()


def _speech_chunk_cut(buffer: str) -> int:
    """Length of the leading speakable chunk to synthesize now, or 0 to wait for
    more text. Prefers a sentence end and falls back to a clause break only once
    the buffer grows long, so playback can start before the whole line is generated."""
    for i, ch in enumerate(buffer):
        if ch in ".!?" and i + 1 >= _TTS_CHUNK_MIN_CHARS:
            return i + 1
    if len(buffer) >= _TTS_CHUNK_MAX_CHARS:
        window = buffer[:_TTS_CHUNK_MAX_CHARS]
        for sep in (",", ";", ":", " "):
            idx = window.rfind(sep)
            if idx + 1 >= _TTS_CHUNK_MIN_CHARS:
                return idx + 1
        return _TTS_CHUNK_MAX_CHARS
    return 0


class GameEngine:
    """Owns the Match and drives it through the phase loop.

    Does NOT start/stop the detector. The detector is a shared resource
    that runs for the whole app's lifetime (so idle/match_over box overlay
    keeps working across matches) -- main.py owns that lifecycle. This
    engine only calls detector.detections() to read from it.
    """

    def __init__(
        self,
        detector: GestureDetector,
        actuator: Actuator,
        commentator: Commentator,
        broadcast: BroadcastFn,
        config: MatchConfig | None = None,
        snapshot: SnapshotFn | None = None,
        commentary: CommentaryFn | None = None,
        speaker: Speaker | None = None,
    ) -> None:
        self._detector = detector
        self._actuator = actuator
        self._commentator = commentator
        self._broadcast = broadcast
        self._snapshot = snapshot
        self._commentary = commentary
        self._speaker = speaker
        self.match = Match(config=config or MatchConfig())
        self.last_capture_jpeg: bytes | None = None
        # Tracks a backgrounded actuator move (e.g. play()) so a later command
        # (home()/wiggle()/release()) never races it on the same serial link.
        self._actuator_task: asyncio.Task | None = None
        # Holds the countdown's trailing word-audio tasks so they aren't
        # garbage-collected while finishing after the phase ends.
        self._countdown_plays: list[asyncio.Task] = []

    async def _await_actuator_task(self) -> None:
        if self._actuator_task is not None:
            with suppress(asyncio.CancelledError):
                await self._actuator_task
            self._actuator_task = None

    async def reset(self) -> None:
        self.match = Match(config=self.match.config)
        self.last_capture_jpeg = None
        await self._set_phase(Phase.IDLE)
        await self._await_actuator_task()
        await self._actuator.home()  # back to the default rock pose

    async def run_match(self) -> None:
        # Broadcast immediately so the browser reflects a running match before
        # the (potentially slow) opening narration and actuator prep complete.
        await self._set_phase(Phase.OPENING)
        # Run the hand's open/close in the background so it overlaps the opening
        # narration instead of delaying the speech.
        prepare_task = asyncio.create_task(self._actuator.prepare())
        try:
            await self._narrate_opening()
            # Let the hand settle home before round 1's countdown moves it.
            await prepare_task
            for round_index in range(1, self.match.config.rounds + 1):
                self.match.current_round = round_index
                await self._run_round(round_index)
            await self._set_phase(Phase.MATCH_OVER)
        except asyncio.CancelledError:
            await self._set_phase(Phase.MATCH_OVER)
            raise
        finally:
            prepare_task.cancel()
            with suppress(asyncio.CancelledError):
                await prepare_task
            await self._await_actuator_task()
            await self._actuator.release()

    def _filler_getter(self, name: str):
        """The speaker's canned lead-in getter, or None when it would dangle.

        A lead-in ("let me consider that...") is a cover for LLM generation
        latency. With no commentator there is nothing to cover, so playing one
        leaves the hand promising a verdict that never arrives -- which is what
        happens with speech enabled and the GGUF missing. Both conditions have
        to hold, not just the speaker.
        """
        if self._speaker is None or not getattr(self._commentator, "enabled", True):
            return None
        return getattr(self._speaker, name)

    async def _narrate_opening(self) -> None:
        opening_filler = self._filler_getter("opening_filler")
        _, spoke = await self._run_commentary(
            0, self._commentator.opening(), filler=opening_filler
        )
        if spoke:
            await asyncio.sleep(_COMMENTARY_LINGER_SECONDS)

    async def _run_round(self, index: int) -> None:
        # Let the previous round's throw finish before this round's countdown
        # sends the hand home/wiggling on the same link.
        await self._await_actuator_task()
        self.last_capture_jpeg = None

        strategy = self.match.config.strategy
        # RANDOM's move doesn't depend on the human, so pick it now and start the
        # throw during "Shoot!" -- the hand's travel time overlaps the capture
        # window instead of adding a delay after the reveal.
        prechosen = choose_machine_gesture(strategy, None) if strategy is Strategy.RANDOM else None

        detection = await self._countdown_and_capture(prechosen)

        human_gesture = detection.gesture if detection else None
        machine_gesture = (
            prechosen if prechosen is not None
            else choose_machine_gesture(strategy, human_gesture)
        )
        outcome = (
            resolve_outcome(human_gesture, machine_gesture)
            if human_gesture is not None
            else Outcome.MACHINE
        )
        rnd = Round(
            index=index,
            human_gesture=human_gesture,
            machine_gesture=machine_gesture,
            outcome=outcome,
            confidence=detection.confidence if detection else 0.0,
            human_box=detection.box if detection else None,
        )

        if outcome is Outcome.HUMAN:
            self.match.human_score += 1
        elif outcome is Outcome.MACHINE:
            self.match.machine_score += 1

        # The round is recorded before the reveal so RESOLVE already carries the
        # machine's gesture (emoji). Throw now only if it wasn't already started
        # during "Shoot!" (RANDOM), and in the background so the on-screen reveal
        # doesn't wait on the physical hand.
        self.match.rounds.append(rnd)
        if prechosen is None:
            self._actuator_task = asyncio.create_task(self._actuator.play(machine_gesture))
        await self._set_phase(Phase.RESOLVE)
        await self._set_phase(Phase.ROUND_RESULT)

        # Stream the comment into the browser as the LLM generates it, then let
        # the last words linger before the next round. The freeze-frame result
        # stays on screen throughout.
        await self._narrate(index, rnd)

    async def _narrate(self, index: int, rnd: Round) -> None:
        round_filler = self._filler_getter("filler")
        text, spoke = await self._run_commentary(
            index, self._commentator.comment(self.match, rnd), filler=round_filler
        )
        if spoke:
            self.match.rounds[-1] = replace(rnd, commentary=text)
            await asyncio.sleep(_COMMENTARY_LINGER_SECONDS)
        elif index < self.match.config.rounds:
            await asyncio.sleep(self.match.config.result_display_seconds)

    async def _run_commentary(
        self,
        round_index: int,
        stream: AsyncIterator[str],
        *,
        filler: Callable[[], tuple[str, bytes] | None] | None = None,
    ) -> tuple[str, bool]:
        """Stream one LLM remark as pipelined speech with a synced on-screen
        reveal: each sentence-sized chunk is synthesized while later chunks are
        still being generated, and the browser types each chunk in time with its
        playback. Returns the full text and whether any commentary was produced.

        When a filler getter is given, its pre-synthesized lead-in is queued
        first so it speaks the instant this segment starts while the real remark
        is still being generated; the real remark's segment start then overwrites
        it in the browser.

        Detection stays paused for all NPU work (generation + synthesis) and
        resumes while the queued audio, which does not touch the NPU, plays out.
        """
        pieces: list[str] = []
        pending = ""
        speaker = self._speaker
        play_queue: asyncio.Queue = asyncio.Queue()

        async def play_worker() -> None:
            prev_audio_end: float | None = None
            segment_boundary = False
            # Clear the panel only when the next segment's text is ready to show,
            # so a filler stays up until the real remark arrives (no empty gap).
            pending_clear = False
            while True:
                item = await play_queue.get()
                try:
                    if item is _PLAYBACK_DONE:
                        return
                    if item is _SEGMENT_START:
                        segment_boundary = True
                        pending_clear = True
                        continue
                    text, pcm = item
                    # A new segment (the real remark after a filler) gets a short
                    # breath, but only if its audio was ready right away; if we
                    # had to wait for it, that wait already served as the gap.
                    if segment_boundary and prev_audio_end is not None:
                        idle = time.monotonic() - prev_audio_end
                        if idle < _SEGMENT_BREATH_SECONDS:
                            await asyncio.sleep(_SEGMENT_BREATH_SECONDS - idle)
                    segment_boundary = False
                    # PCM is 16-bit mono, so its playout length sets the reveal
                    # pace; fall back to a reading speed when there is no audio.
                    seconds = (
                        len(pcm) / (_TTS_SAMPLE_RATE * 2)
                        if pcm
                        else len(text) / _READING_CHARS_PER_SEC
                    )
                    # Start audio first, then reveal the text on its audible
                    # onset so the typewriter tracks the voice (the same constant
                    # output-latency delay the countdown uses).
                    play_task = asyncio.create_task(speaker.play(pcm)) if pcm else None
                    try:
                        if play_task is not None and _AUDIO_ONSET_SECONDS > 0:
                            await asyncio.sleep(_AUDIO_ONSET_SECONDS)
                        if self._commentary is not None:
                            # Clear the previous segment right as this one's first
                            # chunk appears, so nothing is left blank in between.
                            if pending_clear:
                                await self._commentary(
                                    {"round_index": round_index, "operation": "start"}
                                )
                            await self._commentary(
                                {
                                    "round_index": round_index,
                                    "operation": "say",
                                    "text": text,
                                    "duration_ms": int(seconds * 1000),
                                }
                            )
                        pending_clear = False
                        if play_task is not None:
                            await play_task
                        else:
                            await asyncio.sleep(seconds)
                    except asyncio.CancelledError:
                        if play_task is not None and not play_task.done():
                            play_task.cancel()
                        raise
                    prev_audio_end = time.monotonic()
                finally:
                    play_queue.task_done()

        player = asyncio.create_task(play_worker())

        async def emit(text: str) -> None:
            pcm = await speaker.synthesize(text) if speaker is not None else None
            await play_queue.put((text, pcm))

        self._detector.pause()
        detector_live = False
        real_started = False
        try:
            # A canned lead-in (already synthesized) covers the generation delay;
            # the sequential worker plays it before any real chunk, and the real
            # remark's own segment start overwrites it when it begins.
            if filler is not None:
                clip = filler()
                if clip is not None:
                    filler_text, filler_pcm = clip
                    await play_queue.put(_SEGMENT_START)
                    await play_queue.put((filler_text, filler_pcm))
            async with aclosing(stream) as gen:
                async for token in gen:
                    pieces.append(token)
                    pending += token
                    cut = _speech_chunk_cut(pending)
                    if cut > 0:
                        chunk, pending = pending[:cut], pending[cut:]
                        if not real_started:
                            await play_queue.put(_SEGMENT_START)
                            real_started = True
                        await emit(chunk)
            if pending.strip():
                if not real_started:
                    await play_queue.put(_SEGMENT_START)
                    real_started = True
                await emit(pending)
            # Generation and synthesis (the NPU work) are done; let detection
            # resume while the remaining audio plays and types out.
            self._detector.resume()
            detector_live = True
            await play_queue.put(_PLAYBACK_DONE)
            await player
        finally:
            if not detector_live:
                self._detector.resume()
            if not player.done():
                player.cancel()
                with suppress(asyncio.CancelledError):
                    await player

        return "".join(pieces), bool(pieces)

    async def _run_countdown_cues(self, seconds: float) -> None:
        """Run the countdown paced by speech: audio is the slowest cue, so each
        word's spoken clip is the clock. Its clip starts playing, then the word's
        on-screen text and wiggle fire once the audio is audible (after
        AUDIO_ONSET_SECONDS), and the next word waits for this clip to
        finish -- so the display, wiggle, and voice stay together instead of the
        audio queuing up behind a fixed beat.

        With audio unavailable, each word falls back to a fixed slot so the
        countdown still runs (`seconds` total), just silent.
        """
        speaker = self._speaker
        words = config.TTS_COUNTDOWN_WORDS
        display = config.COUNTDOWN_DISPLAY_WORDS
        ready_fallback = _COUNTDOWN_READY_SECONDS
        word_fallback = (seconds - _COUNTDOWN_READY_SECONDS) / len(words) if words else 0.0

        ready_pcm = speaker.countdown_clip(config.TTS_COUNTDOWN_READY) if speaker else None
        # "Ready?" resets the hand to its default pose; the words each wiggle.
        await self._countdown_step(
            config.TTS_COUNTDOWN_READY, ready_pcm, ready_fallback, self._actuator.home
        )
        # A short beat on "Ready?" before the words, so it doesn't launch straight
        # into Rock the instant "Ready?" finishes.
        if _COUNTDOWN_READY_PAUSE_SECONDS > 0:
            await asyncio.sleep(_COUNTDOWN_READY_PAUSE_SECONDS)
        for disp, word in zip(display, words):
            pcm = speaker.countdown_clip(word) if speaker else None
            await self._countdown_step(disp, pcm, word_fallback, self._actuator.wiggle)
            # With audio on, a beat is only as long as its clip, and "Rock." is
            # short -- without this the three words run together. Applied after
            # the last word too, so there is a gap before "Shoot!".
            if _COUNTDOWN_WORD_PAUSE_SECONDS > 0:
                await asyncio.sleep(_COUNTDOWN_WORD_PAUSE_SECONDS)

    async def _countdown_step(self, display_word, pcm, fallback, motion) -> None:
        """One countdown beat: start the clip, reveal the word + hand motion on
        its audible onset, then hold until the clip finishes (or `fallback`
        seconds when there is no audio)."""
        onset = _AUDIO_ONSET_SECONDS

        async def reveal() -> None:
            self.match.countdown_word = display_word
            await self._broadcast(self.match)
            await motion()

        if pcm and self._speaker is not None:
            play_task = asyncio.create_task(self._speaker.play(pcm))
            try:
                if onset > 0:
                    await asyncio.sleep(onset)
                await reveal()
                await play_task
            except asyncio.CancelledError:
                play_task.cancel()
                raise
        else:
            # No audio: show the word and run its hand motion concurrently, then
            # hold a fixed `fallback` so every word shows for the same time.
            self.match.countdown_word = display_word
            await self._broadcast(self.match)
            motion_task = asyncio.create_task(motion())
            try:
                await asyncio.sleep(fallback)
                await motion_task
            except asyncio.CancelledError:
                motion_task.cancel()
                raise

    async def _countdown_and_capture(self, machine_gesture: Gesture | None = None):
        cfg = self.match.config
        stable_gesture = None
        stable_count = 0
        captured = None
        got_gesture = asyncio.Event()

        def record_detection(det):
            nonlocal stable_gesture, stable_count, captured
            if det.confidence < cfg.min_confidence:
                return
            if det.gesture == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = det.gesture
                stable_count = 1
            if stable_count >= cfg.stable_frames:
                captured = det
                if self._snapshot is not None:
                    self.last_capture_jpeg = self._snapshot(det)
                got_gesture.set()

        # Countdown is animation only -- a gesture must not lock in here (the
        # hand is often mid-throw at "Shoot!"), so detection sampling starts
        # only once the capture grace window opens.
        self.match.countdown_word = config.TTS_COUNTDOWN_READY
        await self._set_phase(Phase.COUNTDOWN, duration=cfg.countdown_seconds)
        # Speech is the slowest cue, so it paces the countdown: the on-screen word
        # and the hand's wiggle follow each spoken word's audible onset.
        await self._run_countdown_cues(cfg.countdown_seconds)

        # Consume the detector in the background so phase timing is driven by
        # the event/timeout rather than bounding __anext__ with wait_for
        # (which would cancel and close the generator).
        async def consume():
            async with aclosing(self._detector.detections()) as gen:
                async for det in gen:
                    record_detection(det)
                    if got_gesture.is_set():
                        return

        shoot_play = None
        consumer = None
        try:
            # "Shoot!" is its own beat, like Rock/Paper/Scissors: show it and let
            # the whole word play out before reading the gesture, so the capture
            # isn't taken while the hand is still moving. The text appears on the
            # audible onset, and the capture window opens once the spoken word has
            # played (its short silent output tail may overlap the capture).
            shoot_pcm = (
                self._speaker.countdown_clip(config.TTS_SHOOT_WORD)
                if self._speaker is not None
                else None
            )
            if shoot_pcm:
                shoot_play = asyncio.create_task(self._speaker.play(shoot_pcm))
                self._countdown_plays.append(shoot_play)
                if _AUDIO_ONSET_SECONDS > 0:
                    await asyncio.sleep(_AUDIO_ONSET_SECONDS)
                await self._set_phase(Phase.CAPTURE, duration=cfg.capture_window_seconds)
                # Pre-chosen (RANDOM) move throws on "Shoot!" so the hand's travel
                # overlaps the capture instead of delaying the reveal.
                if machine_gesture is not None:
                    self._actuator_task = asyncio.create_task(
                        self._actuator.play(machine_gesture)
                    )
                await shoot_play
            else:
                await self._set_phase(Phase.CAPTURE, duration=cfg.capture_window_seconds)
                if machine_gesture is not None:
                    self._actuator_task = asyncio.create_task(
                        self._actuator.play(machine_gesture)
                    )
                # No audio: still hold "Shoot!" for a fixed beat (matching the
                # countdown words) before sampling, so the gesture isn't read
                # mid-throw.
                words = config.TTS_COUNTDOWN_WORDS
                shoot_beat = (
                    (cfg.countdown_seconds - _COUNTDOWN_READY_SECONDS) / len(words)
                    if words
                    else _COUNTDOWN_READY_SECONDS
                )
                await asyncio.sleep(shoot_beat)

            # Only now, with the hand settled after "Shoot!", start sampling.
            consumer = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(got_gesture.wait(), timeout=cfg.capture_window_seconds)
            except asyncio.TimeoutError:
                pass
            if captured is not None:
                return captured

            if self._snapshot is not None:
                self.last_capture_jpeg = self._snapshot(None)
            return None
        finally:
            if consumer is not None:
                consumer.cancel()
                with suppress(asyncio.CancelledError):
                    await consumer
            if shoot_play is not None and not shoot_play.done():
                shoot_play.cancel()

    async def _set_phase(self, phase: Phase, duration: float | None = None) -> None:
        self.match.phase = phase
        self.match.phase_duration = duration
        await self._broadcast(self.match)
