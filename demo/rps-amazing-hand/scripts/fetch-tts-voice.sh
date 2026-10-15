#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Download a Piper voice into models/piper/ for spoken commentary.
#
#   ./scripts/fetch-tts-voice.sh                      # the default voice
#   ./scripts/fetch-tts-voice.sh en_GB-alba-medium    # a different one
#
# A voice is two files that must stay together: <name>.onnx (the model) and
# <name>.onnx.json (its metadata, including the sample rate the app reads so
# the audio player starts at the right rate).
#
# Browse the catalogue at https://rhasspy.github.io/piper-samples/ -- listen
# before you commit to one, because voice character is the single most visible
# choice in the demo.
#
# CHECK THE LICENCE TOO. Piper's catalogue mixes permissive and
# non-commercial voices, and the name gives no hint which is which. Each
# voice's MODEL_CARD names its training dataset and that dataset's licence,
# which is the binding one. app/config.py lists the ones verified safe;
# the popular en_US-lessac-medium is research-only and is NOT among them.
#
# The "medium" tier is the right default: "low" is faster but noticeably
# robotic, "high" costs much more CPU for a difference a small speaker in a
# loud room will not carry.
set -euo pipefail

cd "$(dirname "$0")/.."

VOICE="${1:-${PIPER_VOICE:-en_US-kristin-medium}}"
DEST_DIR="${PIPER_VOICES_DIR:-models/piper}"
# A voice ranges from ~10 MB (x_low) to ~110 MB (high). This floor only has to
# be above an HTML error page, which is a few kilobytes.
MIN_BYTES="${MIN_BYTES:-5000000}"
VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"

die() { echo "Error: $*" >&2; exit 1; }

onnx="${DEST_DIR}/${VOICE}.onnx"
meta="${onnx}.json"

mkdir -p "$DEST_DIR"

# --- Already have it? ------------------------------------------------------
if [[ -f "$onnx" && -f "$meta" ]]; then
  size=$(wc -c < "$onnx")
  if (( size >= MIN_BYTES )); then
    echo "Already present: $onnx ($(du -h "$onnx" | cut -f1))"
    echo "                 $meta"
    exit 0
  fi
  echo "Removing truncated voice ($size bytes): $onnx" >&2
  rm -f "$onnx" "$meta"
fi

echo "Voice : $VOICE"
echo "Into  : $DEST_DIR"
echo ""

# --- Preferred route: piper's own downloader -------------------------------
# It knows the repository layout, so it keeps working if that layout changes.
piper_python=""
if [[ -x "$VENV_PYTHON" ]] && "$VENV_PYTHON" -c 'import piper' >/dev/null 2>&1; then
  piper_python="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import piper' >/dev/null 2>&1; then
  piper_python="python3"
fi

if [[ -n "$piper_python" ]]; then
  echo "Downloading with piper's own downloader ($piper_python) ..."
  if "$piper_python" -m piper.download_voices "$VOICE" --data-dir "$DEST_DIR"; then
    downloaded=1
  else
    echo "" >&2
    echo "piper.download_voices failed. Falling back to a direct download." >&2
    echo "If the voice name is wrong, list the valid ones with:" >&2
    echo "  $piper_python -m piper.download_voices" >&2
    echo "" >&2
    downloaded=0
  fi
else
  echo "piper is not installed yet (that is fine -- ./install.sh adds it)."
  echo "Falling back to a direct download."
  echo ""
  downloaded=0
fi

# --- Fallback: construct the Hugging Face path -----------------------------
# Voice names are <lang>_<REGION>-<name>-<quality>, and the repository is laid
# out as <lang>/<lang>_<REGION>/<name>/<quality>/<full name>.onnx
if (( downloaded == 0 )); then
  locale="${VOICE%%-*}"                 # en_US
  rest="${VOICE#*-}"                    # kristin-medium
  speaker="${rest%-*}"                  # kristin
  quality="${rest##*-}"                 # medium
  lang="${locale%%_*}"                  # en

  if [[ -z "$locale" || -z "$speaker" || -z "$quality" || "$rest" != *-* ]]; then
    die "cannot parse '$VOICE'. Expected <lang>_<REGION>-<name>-<quality>,
  for example en_US-kristin-medium."
  fi

  base="https://huggingface.co/rhasspy/piper-voices/resolve/main/${lang}/${locale}/${speaker}/${quality}"

  command -v curl >/dev/null 2>&1 || die "curl is not installed."

  for suffix in onnx onnx.json; do
    url="${base}/${VOICE}.${suffix}"
    dest="${DEST_DIR}/${VOICE}.${suffix}"
    echo "  ${VOICE}.${suffix}"
    # -L follows the CDN redirect; -C - resumes a partial file.
    curl -L -C - --fail --progress-bar -o "$dest" "$url" || {
      echo "" >&2
      echo "Download failed: $url" >&2
      echo "Check the voice name against https://rhasspy.github.io/piper-samples/" >&2
      rm -f "$dest"
      exit 1
    }
  done
fi

# --- Verify what landed ----------------------------------------------------
[[ -f "$onnx" ]] || die "expected $onnx after the download, but it is missing.
  If piper's downloader placed it elsewhere, move both files into $DEST_DIR."
[[ -f "$meta" ]] || die "the model downloaded but $meta did not.
  Both files are required -- the app reads the sample rate from the JSON."

size=$(wc -c < "$onnx")
if (( size < MIN_BYTES )); then
  echo "" >&2
  echo "Downloaded only $size bytes -- that is an error page, not a model." >&2
  head -c 300 "$onnx" >&2
  rm -f "$onnx" "$meta"
  exit 1
fi

# The JSON is what the app parses for the sample rate, so a corrupt one is
# better caught here than at startup.
rate="$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("audio", {}).get("sample_rate", ""))
' "$meta" 2>/dev/null || true)"

echo ""
echo "Done: $onnx ($(du -h "$onnx" | cut -f1))"
if [[ -n "$rate" ]]; then
  echo "      ${rate} Hz, from $(basename "$meta")"
else
  echo "Warning: could not read audio.sample_rate from $meta." >&2
  echo "The app will fall back to a guess and correct itself from the first clip." >&2
fi

cat <<EOF

Next:
  ./scripts/check-audio.sh        confirm the board can actually play sound
  ./start.sh                      TTS_BACKEND defaults to piper

To use this voice by default, set it in app/config.py (PIPER_VOICE) or export
PIPER_VOICE=${VOICE} before ./start.sh.
EOF
