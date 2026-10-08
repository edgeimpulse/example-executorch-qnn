"""Shared data model. Imports only config; every other module imports this."""

from dataclasses import dataclass, field
from enum import Enum

import config


class Gesture(str, Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


# Each gesture beats the gesture it maps to (e.g. ROCK beats SCISSORS). Shared
# here (rather than defined per-module) so game rules and commentary agree.
BEATS: dict[Gesture, Gesture] = {
    Gesture.ROCK: Gesture.SCISSORS,
    Gesture.PAPER: Gesture.ROCK,
    Gesture.SCISSORS: Gesture.PAPER,
}


class Outcome(str, Enum):
    HUMAN = "human"
    MACHINE = "machine"
    DRAW = "draw"


class Phase(str, Enum):
    IDLE = "idle"
    OPENING = "opening"
    COUNTDOWN = "countdown"
    CAPTURE = "capture"
    RESOLVE = "resolve"
    ROUND_RESULT = "round_result"
    MATCH_OVER = "match_over"


class Strategy(str, Enum):
    RANDOM = "random"
    ALWAYS_WIN = "always_win"
    ALWAYS_LOSE = "always_lose"


@dataclass(frozen=True)
class Detection:
    gesture: Gesture
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class InferenceResult:
    detections: list[Detection]
    timestamp: float


@dataclass(frozen=True)
class Round:
    index: int
    human_gesture: Gesture | None
    machine_gesture: Gesture
    outcome: Outcome
    confidence: float
    human_box: tuple[int, int, int, int] | None = None  # for drawing over /captured.jpg
    commentary: str | None = None


@dataclass
class MatchConfig:
    rounds: int = config.MATCH_ROUNDS
    countdown_seconds: float = config.COUNTDOWN_SECONDS
    capture_window_seconds: float = config.CAPTURE_WINDOW_SECONDS
    result_display_seconds: float = config.RESULT_DISPLAY_SECONDS
    min_confidence: float = config.MIN_CONFIDENCE
    stable_frames: int = config.STABLE_FRAMES
    strategy: Strategy = Strategy.RANDOM


@dataclass
class Match:
    phase: Phase = Phase.IDLE
    config: MatchConfig = field(default_factory=MatchConfig)
    rounds: list[Round] = field(default_factory=list)
    current_round: int = 0
    human_score: int = 0
    machine_score: int = 0
    phase_duration: float | None = None
    # The countdown word to show now ("Ready?"/"Rock…"/...), driven by the
    # server so the on-screen word, the wiggle, and the spoken word share one
    # clock instead of a separate browser timer.
    countdown_word: str | None = None
