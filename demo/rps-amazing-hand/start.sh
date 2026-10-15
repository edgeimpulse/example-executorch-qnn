#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Starts the rock-paper-scissors server.
# Run from the project root: ./start.sh
# Select a camera device without prompting: ./start.sh /dev/video2
set -euo pipefail

cd "$(dirname "$0")"

# --- Configuration ---------------------------------------------------------
# All user-tunable settings live here. Each may be overridden from the
# environment, e.g. LLAMA_SERVER_PORT=9000 ./start.sh

# --- Detector --------------------------------------------------------------
DETECTOR_MODEL_PATH="${DETECTOR_MODEL_PATH:-models/model.pte}"
DETECTOR_LABELS_PATH="${DETECTOR_LABELS_PATH:-models/labels.txt}"
export DETECTOR_MODEL_PATH DETECTOR_LABELS_PATH

# --- NPU inference ---------------------------------------------------------
# DETECTOR_MODEL_PATH above is the CPU model and stays as the per-frame fallback.
# The NPU model is served by the rps-npu systemd service (see ../../board/README.md);
# the app only needs its socket. 0 forces the CPU path.
RPS_NPU_ENABLE="${RPS_NPU_ENABLE:-1}"
RPS_NPU_SOCKET="${RPS_NPU_SOCKET:-/run/rps-npu/npu.sock}"
export RPS_NPU_ENABLE RPS_NPU_SOCKET

# False-positive guards (see app/config.py). Tune live, e.g.:
#   RPS_LOG_PROBS=1 RPS_MIN_CONFIDENCE=0.7 ./start.sh
#   RPS_SAVE_FRAMES=$HOME/unknown-frames ./start.sh     # collect retraining frames
export RPS_MIN_CONFIDENCE RPS_SMOOTH_FRAMES RPS_MIN_MARGIN RPS_MAX_UNKNOWN RPS_LOG_PROBS
export RPS_SAVE_FRAMES RPS_SAVE_BELOW RPS_SAVE_EVERY RPS_SAVE_MAX

# --- LLM commentary --------------------------------------------------------
ENABLE_LLM_COMMENTARY="${ENABLE_LLM_COMMENTARY:-1}"   # on by default; 0 to disable. If llama-server or the model
                                                        # is missing, commentary is skipped with a message and the app still starts.
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
# Where a locally built llama.cpp lives. The Snapdragon build installs as
# <prefix>/bin + <prefix>/lib; if llama-server is not on PATH we look here, so
# nobody has to edit PATH just to run the demo.
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/pkg-linux}"
# Relative paths resolve against the project root (this script cd's there).
# Q4_0 is deliberate: every upstream Hexagon example uses it, and the HTP
# kernels are built around that block format.
# Fetch it with: ./scripts/fetch-llm-model.sh
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-models/Llama-3.2-1B-Instruct-Q4_0.gguf}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8080}"
LLAMA_DEVICE_ARGS="${LLAMA_DEVICE_ARGS:---device HTP0 -ngl 99}"
LLAMA_STARTUP_TIMEOUT="${LLAMA_STARTUP_TIMEOUT:-60}"
LLAMA_LIBRARY_PATH="${LLAMA_LIBRARY_PATH:-}"
LLAMA_CPP_SNAPDRAGON_ENV_FILE="${LLAMA_CPP_SNAPDRAGON_ENV_FILE:-/etc/llama.cpp-snapdragon.env}"

export LLAMA_SERVER_URL="http://127.0.0.1:${LLAMA_SERVER_PORT}"

# --- Speech ----------------------------------------------------------------
# Which text-to-speech engine to use:
#   piper   local Piper HTTP server, started below. The default: it installs
#           from a prebuilt wheel and needs no container, so anyone can
#           reproduce it. Runs on the CPU, leaving the NPU to vision + the LLM.
#   melo    the audio-analytics container (needs ENABLE_AUDIO_ANALYTICS=1).
#   none    silent demo.
# A missing engine, voice or audio player always degrades to silence with a
# message rather than stopping the demo.
TTS_BACKEND="${TTS_BACKEND:-piper}"

# Audio output. Leave both unset to auto-detect the player and use the system
# default sink; see ./scripts/check-audio.sh --list for the valid values.
#   TTS_PLAYER=aplay              pick a known player by name
#   TTS_AUDIO_DEVICE=plughw:0,0   pick the output device
#   TTS_PLAYER_CMD='["..."]'      a complete command line, as a JSON array
export TTS_PLAYER TTS_AUDIO_DEVICE TTS_PLAYER_CMD

# --- Speech: Piper ---------------------------------------------------------
# Fetch the voice with: ./scripts/fetch-tts-voice.sh
# Public domain (LibriVox). Piper's catalogue mixes permissive and
# non-commercial voices -- see app/config.py before changing this.
PIPER_VOICE="${PIPER_VOICE:-en_US-kristin-medium}"
PIPER_VOICES_DIR="${PIPER_VOICES_DIR:-models/piper}"
PIPER_PORT="${PIPER_PORT:-5000}"
PIPER_STARTUP_TIMEOUT="${PIPER_STARTUP_TIMEOUT:-60}"
PIPER_LENGTH_SCALE="${PIPER_LENGTH_SCALE:-0.95}"
export PIPER_VOICE PIPER_VOICES_DIR PIPER_LENGTH_SCALE
export PIPER_SERVER_URL="http://127.0.0.1:${PIPER_PORT}"

# --- Speech: audio analytics (MeloTTS container) ---------------------------
ENABLE_AUDIO_ANALYTICS="${ENABLE_AUDIO_ANALYTICS:-0}"  # phase 3          # set 0 to run without the TTS/audio-analytics container
AUDIO_COMPOSE_FILE="${AUDIO_COMPOSE_FILE:-$HOME/dev/docker-compose-audio.yml}"
AUDIO_ANALYTICS_PORT="${AUDIO_ANALYTICS_PORT:-8085}"
AUDIO_ANALYTICS_STARTUP_TIMEOUT="${AUDIO_ANALYTICS_STARTUP_TIMEOUT:-60}"

export AUDIO_ANALYTICS_SERVER_URL="http://127.0.0.1:${AUDIO_ANALYTICS_PORT}"
# ---------------------------------------------------------------------------

# --- Amazing Hand actuator -------------------------------------------------
ENABLE_AMAZING_HAND="${ENABLE_AMAZING_HAND:-0}"      # phase 2               # set 0 to run without the physical hand
AMAZING_HAND_SERIAL_PORT="${AMAZING_HAND_SERIAL_PORT:-/dev/ttyACM0}"
export ENABLE_AMAZING_HAND AMAZING_HAND_SERIAL_PORT
# ---------------------------------------------------------------------------

if (( $# > 1 )); then
  echo "Usage: $0 [camera-device]" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Cannot start: the Python environment is not installed." >&2
  echo "Run ./install.sh from the project root, then try again." >&2
  exit 1
fi

if ! .venv/bin/python -m pip check >/dev/null; then
  echo "Cannot start: the Python environment has unmet dependencies." >&2
  echo "Run ./install.sh from the project root, then try again." >&2
  exit 1
fi

source .venv/bin/activate

spinner_pid=""
start_spinner() {
  if [[ ! -t 1 ]]; then
    return
  fi
  (
    local frames='|/-\\'
    local index=0
    while true; do
      printf '\r%s' "${frames:index % ${#frames}:1}"
      sleep 0.15
      ((index += 1))
    done
  ) &
  spinner_pid=$!
}

stop_spinner() {
  local message="$1"
  if [[ -z "$spinner_pid" ]]; then
    if [[ -n "$message" ]]; then
      echo "$message"
    fi
    return
  fi
  kill "$spinner_pid" 2>/dev/null || true
  wait "$spinner_pid" 2>/dev/null || true
  spinner_pid=""
  printf '\r\033[K'
  if [[ -n "$message" ]]; then
    echo "$message"
  fi
}

CAMERA_DEVICE="${1:-}"

if [[ -z "$CAMERA_DEVICE" ]]; then
  shopt -s nullglob
  available_cameras=(/dev/video*)
  shopt -u nullglob

  if (( ${#available_cameras[@]} == 0 )); then
    echo "Cannot start: no camera devices were found under /dev/video*." >&2
    echo "Connect a USB camera and try again." >&2
    exit 1
  fi

  # Names live in sysfs; devices without one just show as "Unknown camera".
  camera_names=()
  for camera in "${available_cameras[@]}"; do
    name_file="/sys/class/video4linux/$(basename "$camera")/name"
    if [[ -r "$name_file" ]]; then
      camera_names+=("$(<"$name_file")")
    else
      camera_names+=("Unknown camera")
    fi
  done

  # Some platforms (e.g. Qualcomm camx) expose internal request-manager/sync
  # nodes as /dev/videoN with no capture support; drop anything that doesn't
  # actually report Video Capture, rather than keeping it when unsure.
  if command -v v4l2-ctl >/dev/null 2>&1; then
    capture_cameras=()
    capture_names=()
    for i in "${!available_cameras[@]}"; do
      # "Capabilities" is the whole physical device's union (always includes
      # Video Capture); only "Device Caps" reflects this specific node, e.g.
      # a webcam's metadata-only sibling node.
      caps="$(v4l2-ctl --device="${available_cameras[$i]}" --info 2>/dev/null \
        | sed -n '/Device Caps/,$p' || true)"
      if [[ "$caps" == *"Video Capture"* ]]; then
        capture_cameras+=("${available_cameras[$i]}")
        capture_names+=("${camera_names[$i]}")
      fi
    done
    if (( ${#capture_cameras[@]} > 0 )); then
      available_cameras=("${capture_cameras[@]}")
      camera_names=("${capture_names[@]}")
    fi
  elif command -v udevadm >/dev/null 2>&1; then
    capture_cameras=()
    capture_names=()
    for i in "${!available_cameras[@]}"; do
      caps="$(udevadm info --query=property --name="${available_cameras[$i]}" 2>/dev/null \
        | sed -n 's/^ID_V4L_CAPABILITIES=//p' || true)"
      if [[ "$caps" == *capture* ]]; then
        capture_cameras+=("${available_cameras[$i]}")
        capture_names+=("${camera_names[$i]}")
      fi
    done
    if (( ${#capture_cameras[@]} > 0 )); then
      available_cameras=("${capture_cameras[@]}")
      camera_names=("${capture_names[@]}")
    fi
  fi

  if (( ${#available_cameras[@]} == 1 )); then
    CAMERA_DEVICE="${available_cameras[0]}"
    echo "Using camera: ${camera_names[0]} (${CAMERA_DEVICE})"
  fi

  if [[ -z "$CAMERA_DEVICE" ]]; then
    if [[ ! -t 0 || ! -t 1 ]]; then
      echo "Cannot start: no terminal is attached to prompt for a camera." >&2
      echo "Pass one explicitly with ./start.sh /dev/videoN." >&2
      exit 1
    fi

    selected=0
    last=$(( ${#available_cameras[@]} - 1 ))

    draw_camera_menu() {
      echo "Select a camera (up/down arrows to move, Enter to confirm):"
      for i in "${!available_cameras[@]}"; do
        if (( i == selected )); then
          printf '  > %s (%s)\n' "${camera_names[$i]}" "${available_cameras[$i]}"
        else
          printf '    %s (%s)\n' "${camera_names[$i]}" "${available_cameras[$i]}"
        fi
      done
    }

    tput civis 2>/dev/null || true
    trap 'tput cnorm 2>/dev/null || true' EXIT
    draw_camera_menu
    while true; do
      IFS= read -rsn1 key
      if [[ "$key" == $'\x1b' ]]; then
        read -rsn2 -t 0.01 rest || true
        key+="$rest"
      fi
      case "$key" in
        $'\x1b[A')
          if (( selected == 0 )); then
            selected="$last"
          else
            selected=$(( selected - 1 ))
          fi
          ;;
        $'\x1b[B')
          if (( selected == last )); then
            selected=0
          else
            selected=$(( selected + 1 ))
          fi
          ;;
        '') break ;;
      esac
      tput cuu $(( ${#available_cameras[@]} + 1 ))
      draw_camera_menu
    done
    tput cnorm 2>/dev/null || true
    trap - EXIT

    CAMERA_DEVICE="${available_cameras[$selected]}"
  fi
fi
export CAMERA_DEVICE

if [[ ! -c "$CAMERA_DEVICE" ]]; then
  echo "Cannot start: no USB camera was found at $CAMERA_DEVICE." >&2
  echo "Connect the camera or choose a different device." >&2
  exit 1
fi

if [[ ! -r "$CAMERA_DEVICE" ]]; then
  echo "Cannot start: $CAMERA_DEVICE exists but is not readable by the current user." >&2
  echo "Check the camera permissions or add the user to the video's device-access group." >&2
  exit 1
fi

# --- Detector --------------------------------------------------------------
if [[ ! -f "$DETECTOR_MODEL_PATH" ]]; then
  echo "Cannot start: detector model was not found at $DETECTOR_MODEL_PATH." >&2
  echo "Place the detector model there and try again." >&2
  exit 1
fi

if [[ ! -f "$DETECTOR_LABELS_PATH" ]]; then
  echo "Cannot start: detector labels were not found at $DETECTOR_LABELS_PATH." >&2
  echo "Write one label per line, in the model's output channel order." >&2
  exit 1
fi

# --- NPU service (optional; the app falls back to CPU without it) -----------
if [[ "$RPS_NPU_ENABLE" != "0" ]]; then
  if [[ -S "$RPS_NPU_SOCKET" ]]; then
    echo "NPU: rps-npu service socket found at $RPS_NPU_SOCKET."
    # The LLM (llama-server) and this service both open sessions on the same
    # cDSP. They are separate processes with their own library paths, so they
    # do not share environment -- but they do share the NPU.
  else
    echo "NPU: no socket at $RPS_NPU_SOCKET -- detection will run on the CPU." >&2
    echo "     Start it with: sudo systemctl start rps-npu" >&2
  fi
fi

# NOTE: an Edge Impulse .eim model is a compiled executable and needed chmod +x.
# A .pte is plain data read by the ExecuTorch runtime, so only readability
# matters -- marking it executable would be meaningless.
if [[ ! -r "$DETECTOR_MODEL_PATH" ]]; then
  echo "Cannot start: detector model at $DETECTOR_MODEL_PATH is not readable." >&2
  echo "Check the file permissions and try again." >&2
  exit 1
fi

# --- Amazing Hand actuator -------------------------------------------------
# Optional like the LLM/audio features: if the serial port is missing or not
# accessible, warn with the fix and run without the hand rather than crashing.
if [[ "$ENABLE_AMAZING_HAND" != "0" ]]; then
  if [[ ! -e "$AMAZING_HAND_SERIAL_PORT" ]]; then
    echo "Amazing Hand disabled: no serial device at $AMAZING_HAND_SERIAL_PORT." >&2
    echo "Connect the hand, or set AMAZING_HAND_SERIAL_PORT to its port, then try again." >&2
    ENABLE_AMAZING_HAND=0
  elif [[ ! -r "$AMAZING_HAND_SERIAL_PORT" || ! -w "$AMAZING_HAND_SERIAL_PORT" ]]; then
    echo "Amazing Hand disabled: $AMAZING_HAND_SERIAL_PORT is not accessible by the current user." >&2
    echo "Add your user to the 'dialout' group, then log in again (or run 'newgrp dialout'):" >&2
    echo "  sudo usermod -aG dialout \$USER" >&2
    ENABLE_AMAZING_HAND=0
  fi
fi
export ENABLE_AMAZING_HAND

# --- Shared cleanup for the background processes started below ------------
llama_pid=""
piper_pid=""
audio_analytics_up=0
cleanup() {
  stop_spinner ""
  if [[ -n "$llama_pid" ]]; then
    echo "Stopping llama-server..."
    kill "$llama_pid" 2>/dev/null || true
    wait "$llama_pid" 2>/dev/null || true
  fi
  if [[ -n "$piper_pid" ]]; then
    echo "Stopping the Piper TTS server..."
    kill "$piper_pid" 2>/dev/null || true
    wait "$piper_pid" 2>/dev/null || true
  fi
  if [[ "$audio_analytics_up" == "1" ]]; then
    echo "Stopping audio-analytics container..."
    "${DOCKER_COMPOSE_CMD[@]}" -f "$AUDIO_COMPOSE_FILE" down >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- Start the llama.cpp model server (settings are in Configuration, above) ---
if [[ "$ENABLE_LLM_COMMENTARY" != "0" ]]; then
  if [[ ! -r "$LLAMA_MODEL_PATH" ]]; then
    echo "Commentary disabled: model not found at $LLAMA_MODEL_PATH." >&2
    ENABLE_LLM_COMMENTARY=0
  elif ! command -v "$LLAMA_SERVER_BIN" >/dev/null 2>&1 \
       && [[ ! -x "$LLAMA_CPP_DIR/bin/$LLAMA_SERVER_BIN" ]]; then
    echo "Commentary disabled: '$LLAMA_SERVER_BIN' is not on PATH and was not" >&2
    echo "found at $LLAMA_CPP_DIR/bin/. Build llama.cpp with the Hexagon" >&2
    echo "backend (see README) or set LLAMA_CPP_DIR to where it is installed." >&2
    ENABLE_LLM_COMMENTARY=0
  else
    # Prefer PATH, fall back to LLAMA_CPP_DIR. Either way the block below
    # derives LD_LIBRARY_PATH/ADSP_LIBRARY_PATH from the binary's own location,
    # so a self-contained <prefix>/bin + <prefix>/lib install just works.
    if ! command -v "$LLAMA_SERVER_BIN" >/dev/null 2>&1; then
      LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/bin/$LLAMA_SERVER_BIN"
    fi
    if [[ -r "$LLAMA_CPP_SNAPDRAGON_ENV_FILE" ]]; then
      source "$LLAMA_CPP_SNAPDRAGON_ENV_FILE"
    fi
    llama_server_path="$(command -v "$LLAMA_SERVER_BIN")"
    llama_server_dir="$(cd "$(dirname "$llama_server_path")" && pwd)"
    llama_library_path="${LLAMA_LIBRARY_PATH:-${LD_LIBRARY_PATH:-$llama_server_dir/../lib}}"
    adsp_library_path="${ADSP_LIBRARY_PATH:-$llama_server_dir/../lib}"
    echo "Starting llama-server on 127.0.0.1:${LLAMA_SERVER_PORT} ..."
    # shellcheck disable=SC2086  # LLAMA_DEVICE_ARGS is intentionally word-split.
    LD_LIBRARY_PATH="$llama_library_path" \
    ADSP_LIBRARY_PATH="$adsp_library_path" \
    "$llama_server_path" \
      -m "$LLAMA_MODEL_PATH" \
      $LLAMA_DEVICE_ARGS \
      --cache-prompt \
      --host 127.0.0.1 \
      --port "$LLAMA_SERVER_PORT" \
      >/tmp/llama-server.log 2>&1 &
    llama_pid=$!

    ready=0
    health_status="000"
    start_spinner
    for _ in $(seq 1 "$LLAMA_STARTUP_TIMEOUT"); do
      if ! kill -0 "$llama_pid" 2>/dev/null; then
        break  # process exited early
      fi
      health_status="$(
        curl -sS --connect-timeout 1 --max-time 2 \
          -o /dev/null -w "%{http_code}" \
          "http://127.0.0.1:${LLAMA_SERVER_PORT}/health" 2>/dev/null || true
      )"
      if [[ "$health_status" =~ ^2[0-9][0-9]$ ]]; then
        ready=1
        break
      fi
      sleep 1
    done

    if [[ "$ready" == "1" ]]; then
      stop_spinner "llama-server is ready."
    else
      stop_spinner ""
      if kill -0 "$llama_pid" 2>/dev/null; then
        echo "Commentary disabled: /health returned HTTP ${health_status} after ${LLAMA_STARTUP_TIMEOUT}s." >&2
      else
        echo "Commentary disabled: llama-server exited before becoming healthy." >&2
      fi
      tail -n 20 /tmp/llama-server.log >&2 || true
      cleanup
      llama_pid=""
      ENABLE_LLM_COMMENTARY=0
    fi
  fi
fi
export ENABLE_LLM_COMMENTARY

# --- Start the Piper TTS server (settings are in Configuration, above) -----
# Piper runs as its own process rather than being imported by the app: the
# model stays loaded between clips, and the GPL-licensed engine stays at arm's
# length from this code -- the same arrangement the MeloTTS container had.
if [[ "$TTS_BACKEND" == "piper" ]]; then
  piper_voice_file="${PIPER_VOICES_DIR}/${PIPER_VOICE}.onnx"

  if ! .venv/bin/python -c 'import piper' >/dev/null 2>&1; then
    echo "Speech disabled: the 'piper-tts' package is not installed." >&2
    echo "  ./install.sh                 (installs it from a prebuilt wheel)" >&2
    TTS_BACKEND=none
  elif [[ ! -r "$piper_voice_file" ]]; then
    echo "Speech disabled: no voice at $piper_voice_file." >&2
    echo "  ./scripts/fetch-tts-voice.sh ${PIPER_VOICE}" >&2
    TTS_BACKEND=none
  elif [[ ! -r "${piper_voice_file}.json" ]]; then
    echo "Speech disabled: ${piper_voice_file}.json is missing." >&2
    echo "A voice is two files; the app reads the sample rate from the JSON." >&2
    echo "  ./scripts/fetch-tts-voice.sh ${PIPER_VOICE}" >&2
    TTS_BACKEND=none
  else
    echo "Starting the Piper TTS server on 127.0.0.1:${PIPER_PORT} ..."
    .venv/bin/python -m piper.http_server \
      -m "$piper_voice_file" \
      --host 127.0.0.1 \
      --port "$PIPER_PORT" \
      >/tmp/piper-server.log 2>&1 &
    piper_pid=$!

    ready=0
    start_spinner
    for _ in $(seq 1 "$PIPER_STARTUP_TIMEOUT"); do
      if ! kill -0 "$piper_pid" 2>/dev/null; then
        break  # process exited early
      fi
      # Any 2xx/4xx means the HTTP server is answering; only a refused
      # connection (000) means it is still coming up.
      status="$(
        curl -sS --connect-timeout 1 --max-time 2 \
          -o /dev/null -w "%{http_code}" \
          "http://127.0.0.1:${PIPER_PORT}/voices" 2>/dev/null || true
      )"
      if [[ "$status" != "000" && -n "$status" ]]; then
        ready=1
        break
      fi
      sleep 1
    done

    if [[ "$ready" == "1" ]]; then
      stop_spinner "Piper TTS is ready (voice: ${PIPER_VOICE})."
    else
      stop_spinner ""
      if kill -0 "$piper_pid" 2>/dev/null; then
        # Leave it running: the app's own warm-up will try it and fall back to
        # silence if it really is broken. A slow first load is not a failure.
        echo "Warning: Piper did not answer within ${PIPER_STARTUP_TIMEOUT}s; continuing anyway." >&2
      else
        echo "Speech disabled: the Piper server exited before answering." >&2
        tail -n 20 /tmp/piper-server.log >&2 || true
        piper_pid=""
        TTS_BACKEND=none
      fi
    fi
  fi
fi

# --- Start the audio-analytics container (settings are in Configuration, above) ---
if [[ "$ENABLE_AUDIO_ANALYTICS" != "0" ]]; then
  if [[ ! -r "$AUDIO_COMPOSE_FILE" ]]; then
    echo "Audio analytics disabled: compose file not found at $AUDIO_COMPOSE_FILE." >&2
    ENABLE_AUDIO_ANALYTICS=0
  elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD=(docker-compose)
  elif docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD=(docker compose)
  else
    echo "Audio analytics disabled: neither 'docker-compose' nor 'docker compose' is available." >&2
    ENABLE_AUDIO_ANALYTICS=0
  fi
fi

if [[ "$ENABLE_AUDIO_ANALYTICS" != "0" ]]; then
  echo "Starting audio-analytics container (from $AUDIO_COMPOSE_FILE) ..."
  if "${DOCKER_COMPOSE_CMD[@]}" -f "$AUDIO_COMPOSE_FILE" up -d; then
    audio_analytics_up=1

    ready=0
    health_status="000"
    start_spinner
    for _ in $(seq 1 "$AUDIO_ANALYTICS_STARTUP_TIMEOUT"); do
      health_status="$(
        curl -sS --connect-timeout 1 --max-time 2 \
          -o /dev/null -w "%{http_code}" \
          "http://127.0.0.1:${AUDIO_ANALYTICS_PORT}/audio-analytics/v1/api/health" 2>/dev/null || true
      )"
      if [[ "$health_status" =~ ^2[0-9][0-9]$ ]]; then
        ready=1
        break
      fi
      sleep 1
    done

    if [[ "$ready" == "1" ]]; then
      stop_spinner "audio-analytics is ready on 127.0.0.1:${AUDIO_ANALYTICS_PORT}."
    else
      stop_spinner ""
      echo "Warning: audio-analytics did not report healthy after ${AUDIO_ANALYTICS_STARTUP_TIMEOUT}s (last status ${health_status}); continuing anyway." >&2
    fi
  else
    echo "Audio analytics disabled: 'docker compose up -d' failed." >&2
    ENABLE_AUDIO_ANALYTICS=0
  fi
fi
export ENABLE_AUDIO_ANALYTICS

if [[ "$TTS_BACKEND" == "melo" && "$ENABLE_AUDIO_ANALYTICS" == "0" ]]; then
  echo "Speech disabled: TTS_BACKEND=melo needs the audio-analytics container." >&2
  echo "Use the local engine instead with: TTS_BACKEND=piper ./start.sh" >&2
  TTS_BACKEND=none
fi
export TTS_BACKEND

if [[ "$TTS_BACKEND" == "none" ]]; then
  echo "Speech: disabled -- the demo will run silently."
fi

echo "Starting application server on 0.0.0.0:8000 ..."
uvicorn main:app --app-dir app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 2