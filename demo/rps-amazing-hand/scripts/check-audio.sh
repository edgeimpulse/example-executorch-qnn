#!/usr/bin/env bash
# Verify that this board can speak, and say which stage fails if it cannot.
#
#   ./scripts/check-audio.sh                          run every check
#   ./scripts/check-audio.sh --list                   what is installed, what
#                                                     would be chosen, and the
#                                                     valid device names
#   ./scripts/check-audio.sh --player aplay \
#                           --device hw:0,0           try one combination
#                                                     before committing to it
#   ./scripts/check-audio.sh --no-synth               tone only (no Piper)
#
# There are three separate things that can be wrong, and "no sound" looks
# identical for all of them, which is why they are tested apart:
#
#   1. the player   -- no paplay/pw-play/aplay/ffplay installed at all
#   2. the output   -- a player exists but cannot open the sink, or opens the
#                      wrong one (HDMI instead of the jack is the classic)
#   3. the voice    -- piper or its voice files are missing
#
# Player resolution here is not a reimplementation: it imports app/config.py
# and app/speech.py, so what this script tests is exactly what start.sh will
# do, including TTS_PLAYER / TTS_PLAYER_CMD / TTS_AUDIO_DEVICE.
set -uo pipefail

cd "$(dirname "$0")/.."

LIST_ONLY=0
RUN_SYNTH=1
TEST_TEXT="Rock, paper, scissors. Audio is working."

while (( $# > 0 )); do
  case "$1" in
    --list)      LIST_ONLY=1; shift ;;
    --no-synth)  RUN_SYNTH=0; shift ;;
    --player)    [[ $# -ge 2 ]] || { echo "--player needs a name" >&2; exit 2; }
                 export TTS_PLAYER="$2"; shift 2 ;;
    --device)    [[ $# -ge 2 ]] || { echo "--device needs a name" >&2; exit 2; }
                 export TTS_AUDIO_DEVICE="$2"; shift 2 ;;
    --voice)     [[ $# -ge 2 ]] || { echo "--voice needs a name" >&2; exit 2; }
                 export PIPER_VOICE="$2"; shift 2 ;;
    --text)      [[ $# -ge 2 ]] || { echo "--text needs a string" >&2; exit 2; }
                 TEST_TEXT="$2"; shift 2 ;;
    -h|--help)   sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# --- Interpreter -----------------------------------------------------------
# The venv has httpx, which app/speech.py imports. Fall back to the system
# python3 so --list still works before ./install.sh has been run.
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
  echo "Note: .venv not found, using system python3." >&2
else
  echo "FAIL: no python3 available." >&2
  exit 1
fi

TMPDIR_AUDIO="$(mktemp -d -t rps-audio.XXXXXX)"
trap 'rm -rf "$TMPDIR_AUDIO"' EXIT

# --- Stage 1: which player, at which rate ----------------------------------
# One Python call resolves everything and writes it out as shell assignments,
# so the rest of this script works with the app's own answer.
resolved="$TMPDIR_AUDIO/resolved.sh"
if ! "$PY" - "$resolved" <<'PYEOF' ; then
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "app"))

try:
    import config
    import speech
except ImportError as exc:
    print(f"Could not import the app's own configuration: {exc}", file=sys.stderr)
    print("Run ./install.sh first so the dependencies are present.", file=sys.stderr)
    raise SystemExit(1)

voice = config.PIPER_VOICE
rate = speech.voice_sample_rate(config.PIPER_VOICES_DIR, voice)
rate_source = f"{voice}.onnx.json"
if rate is None:
    rate = config.PIPER_FALLBACK_SAMPLE_RATE
    rate_source = "fallback guess (voice metadata not readable)"

resolved = speech.resolve_player(rate)

with open(sys.argv[1], "w", encoding="utf-8") as out:
    out.write(f"RATE={rate}\n")
    out.write(f"RATE_SOURCE={shlex.quote(rate_source)}\n")
    out.write(f"VOICE={shlex.quote(voice)}\n")
    out.write(f"VOICES_DIR={shlex.quote(config.PIPER_VOICES_DIR)}\n")
    out.write(f"ALL_PLAYERS={shlex.quote(' '.join(speech.player_names()))}\n")
    out.write(f"INSTALLED_PLAYERS={shlex.quote(' '.join(speech.installed_players()))}\n")
    out.write(f"DEVICE={shlex.quote(config.TTS_AUDIO_DEVICE)}\n")
    if resolved is None:
        out.write("PLAYER_CMD=\nPLAYER_NAME=\n")
    else:
        cmd, name = resolved
        out.write(f"PLAYER_CMD={shlex.quote(shlex.join(cmd))}\n")
        out.write(f"PLAYER_NAME={shlex.quote(name)}\n")
PYEOF
  echo "" >&2
  echo "FAIL: could not read the app configuration (see above)." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$resolved"

echo "Voice        : ${VOICE}"
echo "Sample rate  : ${RATE} Hz  (${RATE_SOURCE})"
echo "Players known: ${ALL_PLAYERS}"
echo "  installed  : ${INSTALLED_PLAYERS:-none}"
if [[ -n "${PLAYER_CMD:-}" ]]; then
  echo "Chosen       : ${PLAYER_NAME}"
  echo "  command    : ${PLAYER_CMD}"
else
  echo "Chosen       : NONE"
fi
echo "Output device: ${DEVICE:-(system default)}"
echo ""

# --- Available output devices ----------------------------------------------
# Printed for --list, and also on a failure, since "wrong sink" is the most
# common cause of a silent-but-healthy player.
show_devices() {
  echo "Output devices this board reports:"
  local found=0
  if command -v pactl >/dev/null 2>&1; then
    echo "  pactl sinks (use the NAME column as --device):"
    pactl list short sinks 2>/dev/null | sed 's/^/    /' || echo "    (pactl failed; no sound server running?)"
    found=1
  fi
  if command -v wpctl >/dev/null 2>&1; then
    echo "  wpctl status (PipeWire):"
    wpctl status 2>/dev/null | sed -n '/Sinks:/,/^$/p' | sed 's/^/    /' || true
    found=1
  fi
  if command -v aplay >/dev/null 2>&1; then
    echo "  aplay -L (use one of these as --device for aplay):"
    aplay -L 2>/dev/null | grep -E '^(default|sysdefault|plughw|hw):' | head -20 | sed 's/^/    /' \
      || echo "    (none reported)"
    found=1
  fi
  if (( found == 0 )); then
    echo "  (no pactl, wpctl or aplay available -- cannot enumerate devices)"
  fi
}

if (( LIST_ONLY == 1 )); then
  show_devices
  echo ""
  echo "Set a choice for the app with, e.g.:"
  echo "  TTS_PLAYER=aplay TTS_AUDIO_DEVICE=plughw:0,0 ./start.sh"
  exit 0
fi

if [[ -z "${PLAYER_CMD:-}" ]]; then
  echo "FAIL (stage 1/3): no audio player is installed." >&2
  echo "" >&2
  echo "Install one of: ${ALL_PLAYERS}" >&2
  echo "  sudo apt install alsa-utils         # aplay, the smallest option" >&2
  echo "  sudo apt install pulseaudio-utils   # paplay" >&2
  echo "Or set TTS_PLAYER_CMD to a player of your own." >&2
  exit 1
fi

# A TTS_PLAYER_CMD override is taken at its word by resolve_player, so confirm
# the binary exists before claiming this stage passed.
if ! command -v "$PLAYER_NAME" >/dev/null 2>&1; then
  echo "FAIL (stage 1/3): '${PLAYER_NAME}' is not on PATH." >&2
  echo "" >&2
  echo "It came from TTS_PLAYER_CMD, which is used verbatim. Either install it," >&2
  echo "correct the setting, or unset TTS_PLAYER_CMD to auto-detect from:" >&2
  echo "  ${INSTALLED_PLAYERS:-(nothing installed)}" >&2
  exit 1
fi
echo "PASS (stage 1/3): audio player '${PLAYER_NAME}' is installed."

# --- Stage 2: can it actually make a sound? --------------------------------
# A 440 Hz tone at a quarter amplitude, generated from the standard library so
# this stage needs neither piper nor a voice file. If stage 2 passes and you
# hear nothing, the player opened the wrong output -- see show_devices below.
tone="$TMPDIR_AUDIO/tone.pcm"
"$PY" - "$tone" "$RATE" <<'PYEOF'
import math
import struct
import sys

path, rate = sys.argv[1], int(sys.argv[2])
frames = []
for n in range(rate):  # one second
    # Fade the last 10% out so the tone does not end on a click.
    envelope = min(1.0, (rate - n) / (rate * 0.1))
    frames.append(int(8000 * envelope * math.sin(2 * math.pi * 440 * n / rate)))
with open(path, "wb") as out:
    out.write(struct.pack(f"<{len(frames)}h", *frames))
PYEOF

echo ""
echo "Playing a 1-second 440 Hz tone ..."
# shellcheck disable=SC2086  # PLAYER_CMD is a resolved command line.
if timeout 15 bash -c "${PLAYER_CMD} < '${tone}'" 2>"$TMPDIR_AUDIO/tone.err"; then
  echo "PASS (stage 2/3): the player accepted the audio and exited cleanly."
else
  rc=$?
  echo "FAIL (stage 2/3): the player exited with status ${rc}." >&2
  echo "" >&2
  if [[ -s "$TMPDIR_AUDIO/tone.err" ]]; then
    echo "Its error output:" >&2
    sed 's/^/  /' "$TMPDIR_AUDIO/tone.err" >&2
    echo "" >&2
  fi
  echo "Usually this means no sound server is running, or the device name is" >&2
  echo "wrong. Pick one of these and retry with --device:" >&2
  echo "" >&2
  show_devices >&2
  exit 1
fi

if (( RUN_SYNTH == 0 )); then
  echo ""
  echo "Skipping the voice check (--no-synth)."
  echo "If you heard the tone, stages 1 and 2 are good."
  exit 0
fi

# --- Stage 3: the voice ----------------------------------------------------
# Uses the piper CLI rather than the HTTP server, so this works whether or not
# start.sh is running and tests the model/voice independently of the server.
echo ""
if ! "$PY" -c 'import piper' >/dev/null 2>&1; then
  echo "FAIL (stage 3/3): piper is not installed in this environment." >&2
  echo "" >&2
  echo "  ./install.sh                  (installs it from a prebuilt wheel)" >&2
  echo "  ${PY} -m pip install 'piper-tts[http]'" >&2
  echo "" >&2
  echo "Stages 1 and 2 passed, so the board can play audio -- only the voice" >&2
  echo "is missing. The demo will still run, silently." >&2
  exit 1
fi

if [[ ! -f "${VOICES_DIR}/${VOICE}.onnx" ]]; then
  echo "FAIL (stage 3/3): voice model not found at ${VOICES_DIR}/${VOICE}.onnx" >&2
  echo "" >&2
  echo "  ./scripts/fetch-tts-voice.sh ${VOICE}" >&2
  exit 1
fi

wav="$TMPDIR_AUDIO/line.wav"
echo "Synthesizing with ${VOICE} ..."
# -m / -f are piper's documented short flags; the "--" separates the text from
# the options so a line starting with a dash is not parsed as one.
if ! "$PY" -m piper \
      -m "${VOICES_DIR}/${VOICE}.onnx" \
      -f "$wav" \
      -- "$TEST_TEXT" 2>"$TMPDIR_AUDIO/synth.err"; then
  echo "FAIL (stage 3/3): piper could not synthesize." >&2
  echo "" >&2
  sed 's/^/  /' "$TMPDIR_AUDIO/synth.err" >&2
  exit 1
fi

# Strip the WAV container to the raw PCM the app feeds its player, report the
# real header values, and re-resolve the player if the voice's rate differs
# from the metadata guess -- the same correction the app makes at runtime.
line="$TMPDIR_AUDIO/line.pcm"
play_cmd="$("$PY" - "$wav" "$line" "$RATE" <<'PYEOF'
import shlex
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "app"))
import speech

wav_path, pcm_path, configured = sys.argv[1], sys.argv[2], int(sys.argv[3])

with wave.open(wav_path, "rb") as w:
    rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
    frames = w.getnframes()
    pcm = w.readframes(frames)
Path(pcm_path).write_bytes(pcm)

info = f"  {rate} Hz, {channels} channel(s), {width * 8}-bit, {frames / rate:.2f}s"
print(info, file=sys.stderr)

cmd = ""
if rate != configured:
    print(
        f"  note: the player was configured for {configured} Hz. The app adopts "
        f"{rate} Hz from this same header at startup; doing the same here.",
        file=sys.stderr,
    )
    resolved = speech.resolve_player(rate, quiet=True)
    if resolved is not None:
        cmd = shlex.join(resolved[0])
print(cmd)
PYEOF
)"
[[ -n "$play_cmd" ]] || play_cmd="$PLAYER_CMD"

echo "Playing the synthesized line ..."
# shellcheck disable=SC2086
if timeout 30 bash -c "${play_cmd} < '${line}'"; then
  echo "PASS (stage 3/3): synthesized and played."
else
  echo "FAIL (stage 3/3): the player rejected the synthesized audio." >&2
  exit 1
fi

cat <<EOF

All three stages passed. If you heard both the tone and the line, audio is
ready; start the demo with:

  ./start.sh

If a stage passed but you heard nothing, the player is writing to the wrong
output. List the alternatives and try one:

  ./scripts/check-audio.sh --list
  ./scripts/check-audio.sh --device <name>
EOF
