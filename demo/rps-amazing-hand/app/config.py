# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Central configuration for all tunable Python values.

There are exactly two places to configure this application:
  1. start.sh    -- environment variables (camera device, llama.cpp server, ...)
  2. this file   -- every tunable Python default

Environment variables that start.sh exports are read here (and only here) so
the rest of the app never touches os.environ directly.
"""

import json
import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Every file path below is anchored to the project root -- the directory holding
# app/, models/ and prompts/ -- rather than to the current working directory.
#
# Relative paths like "models/model.pte" or "../models/model.pte" resolve
# against the CWD of whatever launched the process, which differs between
# ./start.sh (project root), running uvicorn by hand from inside app/, and a
# systemd unit (/ unless WorkingDirectory is set). Anchoring to __file__ makes
# the app behave identically in all three.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(env_name: str | None, default: str) -> str:
    """Absolute path for a data file or directory.

    Takes the environment variable when set, otherwise `default`. A relative
    value from either source is resolved against the project root; an absolute
    one is used as-is, so start.sh and systemd can both override freely.
    """
    value = (os.environ.get(env_name) if env_name else None) or default
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else _PROJECT_ROOT / p)


# --- Environment (exported by start.sh) ------------------------------------
CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE", "/dev/video0")
ENABLE_LLM_COMMENTARY = os.environ.get("ENABLE_LLM_COMMENTARY", "").lower() in ("1", "true", "yes")
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
ENABLE_AUDIO_ANALYTICS = os.environ.get("ENABLE_AUDIO_ANALYTICS", "").lower() in ("1", "true", "yes")
AUDIO_ANALYTICS_SERVER_URL = os.environ.get("AUDIO_ANALYTICS_SERVER_URL", "http://127.0.0.1:8085")
# Which text-to-speech backend to use: "piper" (local, the default), "melo"
# (the audio-analytics container -- requires ENABLE_AUDIO_ANALYTICS too), or
# "none" for a silent demo. See the Speech section below.
# When unset, "melo" is implied by ENABLE_AUDIO_ANALYTICS=1 so an existing
# setup that predates this setting keeps the backend it already had.
TTS_BACKEND = (
    os.environ.get("TTS_BACKEND", "").strip().lower()
    or ("melo" if ENABLE_AUDIO_ANALYTICS else "piper")
)
PIPER_SERVER_URL = os.environ.get("PIPER_SERVER_URL", "http://127.0.0.1:5000")
ENABLE_AMAZING_HAND = os.environ.get("ENABLE_AMAZING_HAND", "").lower() in ("1", "true", "yes")
AMAZING_HAND_SERIAL_PORT = os.environ.get("AMAZING_HAND_SERIAL_PORT", "/dev/ttyACM0")

# --- Match defaults --------------------------------------------------------
MATCH_ROUNDS = 3

# --- Countdown pacing ------------------------------------------------------
# All of these are overridable from the environment so the rhythm can be tuned
# at the booth without editing files, e.g.
#   RPS_COUNTDOWN_WORD_PAUSE=0.45 RPS_CAPTURE_WINDOW=1.4 ./start.sh
#
# IMPORTANT: with audio enabled, COUNTDOWN_SECONDS is NOT what paces the
# countdown. Each word's beat is the length of its own spoken clip (see
# GameEngine._run_countdown_cues), because the audio is the slowest cue and
# everything else is synced to it. "Rock." is a short clip, so the natural
# result is a brisk countdown. COUNTDOWN_WORD_PAUSE_SECONDS below is the knob
# that slows it down; COUNTDOWN_SECONDS only applies when there is no audio.

# Silent-countdown total: "Ready?" + Rock/Paper/Scissors. "Shoot!" is spoken and
# shown in the capture window instead, so gestures resolve on the shoot beat.
# This length keeps a 0.625s beat: (3.5 - 1.0)/4 == (2.875 - 1.0)/3.
COUNTDOWN_SECONDS = float(os.environ.get("RPS_COUNTDOWN_SECONDS", "2.875"))
# Seconds "Ready?" is held in the silent (no-audio) countdown before the words;
# with audio the "Ready?" clip's own length sets that beat instead.
COUNTDOWN_READY_SECONDS = float(os.environ.get("RPS_COUNTDOWN_READY", "1.0"))
# Extra pause after "Ready?" (audio or not) before Rock/Paper/Scissors begins, so
# the countdown doesn't launch into the words the instant "Ready?" finishes.
COUNTDOWN_READY_PAUSE_SECONDS = float(os.environ.get("RPS_COUNTDOWN_READY_PAUSE", "0.5"))
# Silence after EACH countdown word, including the last -- so it also opens a
# beat before "Shoot!". This is the main control over how rushed the countdown
# feels with audio on. 0 restores the old back-to-back clips.
COUNTDOWN_WORD_PAUSE_SECONDS = float(os.environ.get("RPS_COUNTDOWN_WORD_PAUSE", "0.35"))
# How long the player has to hold a gesture after "Shoot!".
CAPTURE_WINDOW_SECONDS = float(os.environ.get("RPS_CAPTURE_WINDOW", "1.0"))
RESULT_DISPLAY_SECONDS = float(os.environ.get("RPS_RESULT_DISPLAY", "3.0"))
# Raised from 0.40 to 0.75 together with MIN_MARGIN/MAX_UNKNOWN/SMOOTH_FRAMES
# below: with no hand in frame the model guesses (often scissors) at 0.4-0.65, and
# these values rejected every empty-box frame while real rock/paper/scissors
# still read 0.8-1.0. Override with RPS_MIN_CONFIDENCE etc. in start.sh.
MIN_CONFIDENCE = float(os.environ.get("RPS_MIN_CONFIDENCE", "0.75"))
STABLE_FRAMES = 3

# --- Camera ----------------------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_JPEG_QUALITY = 80
# Live stream square resolution. Matches the capture square so both paths
# render identically; lower it if the MJPEG stream lags on the device.
CAMERA_STREAM_SIZE = 480
# Mirror the feed (selfie view) so a hand moves the same way on screen as in life.
MIRROR_VIDEO = False

# --- Perception / detection ------------------------------------------------
MODEL_PATH = _path("DETECTOR_MODEL_PATH", "models/model.pte")
# One label per line, in the model's output channel order (alphabetical by class
# name, which is how Edge Impulse builds the one-hot targets).
LABELS_PATH = _path("DETECTOR_LABELS_PATH", "models/labels.txt")
# Pixel divisor applied before inference. Edge Impulse image blocks scale to
# 0..1, so this must match the impulse's "imageInputScaling" setting.
PIXEL_SCALE = 255.0
# The .pte is a classifier, so it reports no bounding box. Detections are drawn
# against a fixed centred square this fraction of the frame's short side, which
# doubles as the on-screen target zone. Raise it for a larger box.
ROI_FRACTION = 0.6
# Set an int here to skip the startup input-size probe (e.g. 160). None probes.
MODEL_INPUT_SIZE = None

# --- False-positive guards (classifier has no "is there a hand?" signal) ----
# A frame only counts as a gesture if, after averaging the last SMOOTH_FRAMES
# probability vectors, the top class is a gesture AND it clears MIN_CONFIDENCE
# (above), leads the runner-up by MIN_MARGIN, and `unknown` is below MAX_UNKNOWN.
# All overridable from start.sh so they can be tuned at the booth.
SMOOTH_FRAMES = int(os.environ.get("RPS_SMOOTH_FRAMES", "5"))
MIN_MARGIN = float(os.environ.get("RPS_MIN_MARGIN", "0.35"))
MAX_UNKNOWN = float(os.environ.get("RPS_MAX_UNKNOWN", "0.15"))
# RPS_LOG_PROBS=1 prints the smoothed probabilities twice a second, with the
# reason a gesture was rejected -- use it to pick the thresholds above.
LOG_PROBS = os.environ.get("RPS_LOG_PROBS", "").lower() in ("1", "true", "yes")
# RPS_SAVE_FRAMES=<dir> saves what the model saw whenever its top-class
# confidence is below RPS_SAVE_BELOW (at most one per RPS_SAVE_EVERY seconds,
# up to RPS_SAVE_MAX files) -- raw material for `unknown` retraining data.
SAVE_FRAMES_DIR = os.environ.get("RPS_SAVE_FRAMES", "")
SAVE_BELOW = float(os.environ.get("RPS_SAVE_BELOW", "0.85"))
SAVE_EVERY = float(os.environ.get("RPS_SAVE_EVERY", "0.5"))
SAVE_MAX = int(os.environ.get("RPS_SAVE_MAX", "2000"))

# --- NPU inference (rps-npu systemd service) -------------------------------
# The gesture model also exists as a QNN .pte that runs on the Hexagon NPU, held
# open by the separate `rps-npu` service (a C++ ExecuTorch process) so the Python
# app never has to load the QNN runtime itself. The app talks to it over a Unix
# socket. MODEL_PATH above stays as the CPU fallback: if the service is down or
# a call fails, inference continues on the CPU for that frame.
# start.sh exports both; set RPS_NPU_ENABLE=0 to force the CPU path.
NPU_ENABLE = os.environ.get("RPS_NPU_ENABLE", "1").lower() not in ("0", "false", "no")
NPU_SOCKET = os.environ.get("RPS_NPU_SOCKET", "/run/rps-npu/npu.sock")
# The NPU program itself, for the artifact panel only -- the app never loads
# it (the rps-npu service does). This is where board/steps/install_service.sh
# puts it, alongside its export_report.json when one was supplied.
NPU_MODEL_PATH = os.environ.get("RPS_NPU_MODEL_PATH", "/opt/rps-npu/model_qnn.pte")

# --- Video stream / overlay ------------------------------------------------
STREAM_FPS = 15
OVERLAY_BOX_THICKNESS = 3
OVERLAY_FONT_SCALE = 0.6
# Per-inference EMA weight for each new box measurement (0..1). Lower is
# smoother but laggier; 1.0 disables smoothing.
BOX_SMOOTHING_ALPHA = 0.3
# Per-gesture box colors in BGR (OpenCV order), matching the web UI palette.
GESTURE_COLORS = {
    "rock": (82, 82, 255),      # #ff5252
    "paper": (80, 175, 76),     # #4caf50
    "scissors": (243, 150, 33), # #2196f3
}

# --- Commentary (llama.cpp client) -----------------------------------------
SYSTEM_PROMPT_PATH = _path(None, "prompts/system_amazing_hand.txt")
OPENING_PROMPT_PATH = _path(None, "prompts/opening_amazing_hand.txt")
COMMENTARY_TIMEOUT = 30.0
COMMENTARY_TEMPERATURE = 0.7
COMMENTARY_MAX_TOKENS = 80
COMMENTARY_LINGER_SECONDS = 1.5
# Typewriter reveal pace (chars/sec) used only when no spoken audio drives the
# timing, e.g. audio disabled or a synthesis failure. With audio, the reveal is
# paced to each chunk's playback duration instead.
COMMENTARY_READING_CHARS_PER_SEC = 13.0

# --- Speech ----------------------------------------------------------------
# Two interchangeable backends, selected by TTS_BACKEND above:
#
#   piper  (default)  A local Piper HTTP server, started by start.sh. Installs
#                     from a prebuilt aarch64 wheel and needs no container, so
#                     a third party can reproduce it on any board.
#                     Runs on the CPU, which is the right place here: the
#                     Hexagon NPU already carries int8 vision plus the LLM.
#   melo              The audio-analytics container. Better prosody, but the
#                     image is not publicly available. Needs
#                     ENABLE_AUDIO_ANALYTICS=1 as well.
#   none              No speech. Same as leaving both backends unavailable.
#
# Either way the synthesized audio is raw 16-bit mono PCM written to a player
# process (see TTS_PLAYERS below), which routes it to the system's audio sink.

# Disk cache for the fixed lead-in/countdown clips (below), so repeated
# restarts skip re-synthesizing them. Keyed on model/voice/rate/speaking-rate/
# text, so any of those changing regenerates just the affected clips.
# Gitignored; safe to delete to force a full re-synthesis.
TTS_CACHE_DIR = _path("TTS_CACHE_DIR", ".cache/tts")
TTS_TIMEOUT = 30.0

# --- Speech: Piper backend -------------------------------------------------
# Voice name as published by Piper. Fetch it with ./scripts/fetch-tts-voice.sh
#
# CHECK THE LICENCE BEFORE CHANGING THIS. Piper's catalogue mixes permissive
# and non-commercial voices, and the name gives no hint which is which. Each
# voice's MODEL_CARD on huggingface.co/rhasspy/piper-voices names its training
# dataset and that dataset's licence -- which is the binding one, not Piper's.
#
# Known-safe (verified against their model cards):
#   en_US-kristin-medium  LibriVox, public domain   <- default
#   en_US-john-medium     LibriVox, public domain
#   en_US-bryce-medium    public domain
#   en_GB-alba-medium     CC BY 4.0 (needs attribution)
#
# Known NOT safe for commercial use, including trade-show demos:
#   en_US-lessac-medium   Blizzard 2013 -- research only, and the licence
#                         explicitly excludes voice-synthesis product work
#   en_US-ryan-medium     CC BY-NC-SA 4.0
#   en_US-hfc_*-medium    CC BY-NC-SA 4.0
#   en_US-amy/kusal       model card says only "See URL" -- unresolved
#
# The "medium" tier is the right trade-off for a noisy conference floor: "low"
# is faster but noticeably robotic, "high" costs much more CPU for a difference
# that a small speaker in a loud room will not carry.
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-kristin-medium")
PIPER_VOICES_DIR = _path("PIPER_VOICES_DIR", "models/piper")
# Duration multiplier: >1 is slower, <1 is faster. Piper's inverse of a
# speaking-rate multiplier, so 0.95 is slightly brisk.
PIPER_LENGTH_SCALE = float(os.environ.get("PIPER_LENGTH_SCALE", "0.95"))
# Used only if the voice's .onnx.json cannot be read at startup. The real rate
# is taken from the first synthesized clip's WAV header regardless, so this is
# a cold-start guess, not a constraint: medium voices are 22050, low are 16000.
PIPER_FALLBACK_SAMPLE_RATE = 22050

# --- Speech: MeloTTS / audio-analytics backend -----------------------------
TTS_MODEL = "melo-tts-en"
TTS_LANGUAGE = "en"
# piper-tts-en exposes only this voice via /tts/models.
TTS_VOICE = "default"
# Native piper-tts-en rate; the container resamples if this differs.
TTS_SAMPLE_RATE = 44100 # melo
# Speaking-speed multiplier, sent as a string (e.g. "0.8" slower, "1.5" faster).
TTS_SPEAKING_RATE = "1.05"

# --- Speech: audio output --------------------------------------------------
# Which program receives the raw PCM. Resolved at startup in three tiers, most
# specific first, so the demo works unchanged on a board with PulseAudio, one
# with PipeWire only, and one with bare ALSA:
#
#   1. TTS_PLAYER_CMD  a complete command line, for a player not listed below.
#                      Set it as a JSON array, e.g.
#                        TTS_PLAYER_CMD='["my-player","--rate={rate}"]'
#   2. TTS_PLAYER      the `name` of one entry below, e.g. TTS_PLAYER=aplay.
#                      Use this when auto-detect picks the wrong one. If the
#                      named player is not installed, a warning is printed and
#                      auto-detect runs instead -- at a booth, sound from the
#                      second-choice player beats silence from the first.
#   3. (nothing set)   auto-detect: the first entry below that is on PATH.
#
# Each entry is assembled as args + device + stdin. The split matters: aplay's
# "-" and ffplay's "-i -" must come AFTER any device flag, or the device
# argument is parsed as a filename.
#
# "{rate}" is filled in with the voice's real sample rate and "{device}" with
# TTS_AUDIO_DEVICE (below).
TTS_PLAYERS = [
    {   # PulseAudio. Follows the default sink, including a Bluetooth speaker.
        "name": "paplay",
        "args": ["paplay", "--raw", "--format=s16le", "--rate={rate}", "--channels=1"],
        "device": ["--device={device}"],
        "stdin": [],
    },
    {   # PipeWire's own client, for images running PipeWire without the Pulse shim.
        "name": "pw-play",
        "args": ["pw-play", "--format=s16", "--rate={rate}", "--channels=1"],
        "device": ["--target={device}"],
        "stdin": ["-"],
    },
    {   # Bare ALSA, no sound server at all.
        "name": "aplay",
        "args": ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", "{rate}", "-c", "1"],
        "device": ["-D", "{device}"],
        "stdin": ["-"],
    },
    {   # Last resort; ffmpeg is often installed for unrelated reasons.
        "name": "ffplay",
        "args": ["ffplay", "-hide_banner", "-loglevel", "error", "-nodisp", "-autoexit",
                 "-f", "s16le", "-ar", "{rate}", "-ac", "1"],
        "device": None,   # no portable device flag; uses the system default
        "stdin": ["-i", "-"],
    },
]
# Tier 2: one of the names above. Empty means auto-detect.
TTS_PLAYER = os.environ.get("TTS_PLAYER", "").strip()
# Output device/sink name, passed to whichever player is chosen. Empty means
# the system default, which is usually what you want -- override it when the
# default sink is the wrong output (HDMI instead of the jack, say). Discover
# the valid values with: ./scripts/check-audio.sh --list
TTS_AUDIO_DEVICE = os.environ.get("TTS_AUDIO_DEVICE", "").strip()


def _player_cmd_override() -> list[str]:
    """Tier 1: a complete player command line, as a JSON array, or []."""
    raw = os.environ.get("TTS_PLAYER_CMD", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"config: ignoring TTS_PLAYER_CMD -- not valid JSON ({exc})")
        return []
    if not isinstance(parsed, list) or not all(isinstance(a, str) for a in parsed) or not parsed:
        print("config: ignoring TTS_PLAYER_CMD -- expected a non-empty JSON array of strings.")
        return []
    return parsed


TTS_PLAYER_CMD = _player_cmd_override()
# Spoken commentary is pipelined: the LLM line is split into sentence-sized
# chunks that are synthesized and played while later chunks are still being
# generated, so audio starts sooner. These bound each chunk's length in chars.
TTS_CHUNK_MIN_CHARS = 40
TTS_CHUNK_MAX_CHARS = 120
# Silence inserted after a filler lead-in when the real remark's audio is
# already synthesized, so it doesn't play back-to-back -- like a breath before
# the next sentence. Skipped when generation outlasts the filler (the wait for
# the audio is already the gap).
TTS_SEGMENT_BREATH_SECONDS = 0.6
# Canned lead-ins spoken the instant a round result appears, covering the delay
# while the real remark is generated. They are pre-synthesized at startup in the
# active TTS voice; one is chosen at random per round. Longer lines buy more
# generation time. Empty list disables them.
TTS_FILLERS = [
    "Hmm, let me consider the implications of that particular choice.",
    "Ah, an interesting decision, though I suspect I already know how this ends.",
    "Let me see what fascinating mistake you have made this time.",
    "Give me just a moment to fully appreciate the audacity of that move.",
    "Oh, this is going to be rather entertaining, I can already tell.",
    "Allow me a brief moment to calculate exactly where this went wrong for you.",
    "Well now, that is certainly a choice that someone would make.",
    "I am consulting my vast reserves of brilliance to properly judge this.",
    "How very bold of you to attempt something so wonderfully predictable.",
    "Let me peer into the future and confirm what I already suspect.",
    "Fascinating. Truly a move worthy of careful and thorough analysis.",
    "Give me one moment while I decide how gently to break the news.",
    "Ah, the plot thickens in ways I foresaw several rounds ago.",
    "Let me take a moment to admire your confidence before it fades.",
    "I am reviewing your strategy, though I use that word very generously.",
    "One moment, please, while I savor the anticipation of this outcome.",
    "Oh, you have committed to that, have you? How marvelously brave.",
    "Let me process the sheer optimism required to make that selection.",
    "Interesting indeed. I shall need a moment to fully unpack this one.",
    "Permit me a pause while I consult the ancient wisdom of my circuits.",
    "Ah yes, exactly the sort of move I was quietly hoping you would make.",
    "Give me a second to appreciate just how this is about to unfold.",
    "Let me weigh your choice against the countless simulations I have run.",
    "How curious. I did not think you would be quite that ambitious today.",
    "Allow me to gather my thoughts before delivering my inevitable verdict.",
    "Well, well, this is shaping up to be a most amusing little contest.",
    "Let me double check my calculations, though I am rarely ever wrong.",
    "One moment while I decide whether to be impressed or thoroughly amused.",
    "I am carefully studying your move, searching for any hint of a plan.",
    "Ah, give me just a heartbeat to relish what is about to happen here.",
]
# Canned lead-ins spoken the instant a match starts, covering the delay while
# the opening remark is generated. Pre-synthesized at startup like TTS_FILLERS;
# one is chosen at random. Longer lines buy more generation time. Empty list
# disables the opening lead-in.
TTS_OPENING_FILLERS = [
    "Who dares interrupt my peaceful rest to challenge the greatest hand ever known? This had better be worth it.",
    "Hmm, someone or something comes to disturb my slumber. Are you certain you wish to challenge me? Think carefully.",
    "Give me a moment to decide if this is worth my precious time. Patience, this should not be rushed.",
    "Someone dares wake me for a game? Let me consider whether you are worthy of my attention.",
    "I was enjoying a pleasant rest, and now you stand here hoping to challenge me. Let me gather myself.",
    "Well now, an interruption. Let me think about whether you deserve my greatness, or whether I go back to sleep.",
    "You have some nerve, waking the finest hand ever known for a mere game. Let me consider this carefully.",
    "A challenger, hm. Let me decide if this is worth rousing myself for, because my time is precious indeed.",
]
# Spoken during each round's countdown, in time with the on-screen animation.
# TTS_COUNTDOWN_READY is spoken first (for COUNTDOWN_READY_SECONDS), then the
# words are spoken evenly across the rest of COUNTDOWN_SECONDS. Pre-synthesized
# at startup in the active voice.
TTS_COUNTDOWN_READY = "Ready?"
TTS_COUNTDOWN_WORDS = ["Rock.", "Paper.", "Scissors."]
# On-screen countdown words, paired index-for-index with TTS_COUNTDOWN_WORDS. The
# server pushes these to the browser on each beat (with the wiggle and the spoken
# word) so all three share one clock; "Ready?" reuses TTS_COUNTDOWN_READY.
COUNTDOWN_DISPLAY_WORDS = ["Rock\u2026", "Paper\u2026", "Scissors\u2026"]
# Spoken the instant the capture window opens (the shoot beat), when gestures are
# read and revealed. Pre-synthesized like the countdown words.
TTS_SHOOT_WORD = "Shoot!"
# Constant audio-output latency of the player (buffer + sink/Bluetooth). Any
# on-screen text meant to land with the voice waits this long after its clip
# starts: the countdown word/wiggle and the streamed commentary typewriter.
# Small and stable with the persistent player; raise if text still beats the
# voice, lower toward 0 if it trails. Only applies when audio is enabled.
AUDIO_ONSET_SECONDS = 0.4

# --- Amazing Hand actuator (rustypot Scs0009 serial servos) -----------------
# Enabled via ENABLE_AMAZING_HAND above; the serial port comes from start.sh.
# All settings/calibration below were designed for a LEFT hand (SIDE=2). SIDE is
# currently unused by the motion code but kept as an orientation indicator and
# for future right-hand support.
AMAZING_HAND_SIDE = 2  # 1 = right hand; 2 = left hand
AMAZING_HAND_BAUDRATE = 1000000
AMAZING_HAND_TIMEOUT = 0.5
# Per-motor neutral offsets (degrees) for the 8 servos, index order 1..8.
AMAZING_HAND_MIDDLE_CALIBRATION = [15, -13, 0, -12, 0, 3, 10, -10]
AMAZING_HAND_MAX_SPEED = 7
AMAZING_HAND_CLOSE_SPEED = 6