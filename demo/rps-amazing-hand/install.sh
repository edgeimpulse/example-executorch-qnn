#!/usr/bin/env bash
# Set up the Python environment needed to run this project.
# Run from the project root: ./install.sh
#
#   FORCE_REBUILD=1 ./install.sh     rebuild an existing .venv from scratch
#   TORCH_INDEX_URL=...              use a different torch index
#   ALLOW_CUDA=1 ./install.sh        don't fail if CUDA packages appear
#   SKIP_SPEECH=1 ./install.sh       leave out the text-to-speech engine
#                                    (see requirements-speech.txt)
#
# To reuse an environment that already works (e.g. one where you have already
# run a .pte), point .venv at it and add the three web packages:
#   <that venv>/bin/pip install fastapi "uvicorn[standard]" httpx
#   ln -s /path/to/that/venv .venv
# This script then only verifies it.
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
CONSTRAINTS="$(mktemp -t torch-constraint.XXXXXX)"
trap 'rm -f "$CONSTRAINTS"' EXIT

die() { echo "Error: $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 is not available."

# ---------------------------------------------------------------------------
# 0. Decide whether to build, rebuild, or just verify.
#
# This matters more than it looks. pip reports "Requirement already satisfied"
# for anything already present, so running this over an existing venv CANNOT
# undo a CUDA torch that is already installed -- the careful index handling
# below silently becomes a no-op. Refusing is better than pretending.
# ---------------------------------------------------------------------------
VERIFY_ONLY=0
if [[ -L "$VENV" ]]; then
  echo "$VENV is a symlink to $(readlink -f "$VENV") -- verifying it, not rebuilding."
  VERIFY_ONLY=1
elif [[ -d "$VENV" ]]; then
  if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
    echo "Removing the existing $VENV (FORCE_REBUILD=1) ..."
    rm -rf "$VENV"
  else
    die "$VENV already exists.
  Installing over it leaves existing packages untouched, so this script cannot
  fix a bad torch install in place. Choose one:
    FORCE_REBUILD=1 ./install.sh     rebuild from scratch (recommended)
    rm -rf $VENV && ./install.sh     same thing, by hand"
  fi
fi

if [[ "$VERIFY_ONLY" == "0" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip

  # -------------------------------------------------------------------------
  # 1. CPU-only torch, first and on its own.
  #
  # PyTorch's default PyPI wheels declare NVIDIA CUDA dependencies
  # (cuda-toolkit, nvidia-cudnn-cu13, nvidia-cusparselt-cu13, triton, ...).
  # On a Qualcomm board those are gigabytes that can never execute, and some
  # have no wheel for this platform at all, which fails the whole install with
  # "nvidia-cusparselt-cu13 ... is not supported on this platform".
  # -------------------------------------------------------------------------
  # Install the exact torch requirements.txt asks for, not the newest. The
  # `([=<>!~]|$)` guard stops this matching torchao, torchvision, etc.
  torch_spec="$(grep -iE '^torch([=<>!~]|$)' requirements.txt | head -1 || true)"
  if [[ -z "$torch_spec" ]]; then
    torch_spec="torch"
    echo "Note: requirements.txt does not pin torch; installing the latest." >&2
  fi

  echo "[1/3] Installing ${torch_spec} (CPU build) from ${TORCH_INDEX_URL} ..."
  if ! "$VENV/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" "$torch_spec"; then
    echo "" >&2
    echo "Warning: could not install ${torch_spec} from ${TORCH_INDEX_URL}." >&2
    echo "Common causes: a proxy blocking download.pytorch.org, or no wheel" >&2
    echo "for this Python version / architecture. Falling back to the default" >&2
    echo "index, which will likely pull the unused CUDA packages." >&2
    echo "" >&2
    "$VENV/bin/python" -m pip install "$torch_spec" \
      || die "${torch_spec} could not be installed from either index."
  fi

  # -------------------------------------------------------------------------
  # 2. Everything else.
  #
  # requirements.txt pins torch itself, so it acts as its own constraint: the
  # CPU build installed above already satisfies the pin and pip will not
  # re-resolve it from the default index. Belt and braces, the installed
  # version is also written to a constraint file (local tags like "+cpu"
  # stripped) so a transitive dependency cannot force an upgrade either.
  # -------------------------------------------------------------------------
  torch_version="$("$VENV/bin/python" -c \
    'import torch; print(torch.__version__.split("+")[0])')"
  echo "torch==${torch_version}" > "$CONSTRAINTS"

  echo "[2/3] Installing the remaining requirements (torch held at ${torch_version}) ..."
  if ! "$VENV/bin/python" -m pip install -c "$CONSTRAINTS" -r requirements.txt; then
    echo "" >&2
    die "Dependency resolution failed with torch held at ${torch_version}.
  This usually means the pinned executorch wants a different torch than
  requirements.txt asks for. Find what it wants:
    $VENV/bin/python -m pip install --dry-run --no-deps executorch==<version>
    $VENV/bin/python -m pip index versions torch
  then align the two pins in requirements.txt and rebuild:
    FORCE_REBUILD=1 ./install.sh"
  fi

  # -------------------------------------------------------------------------
  # 3. Text-to-speech, separately and non-fatally.
  #
  # It is optional (the demo runs silently without it) and separately licensed
  # -- see requirements-speech.txt. A failure here must not sink an otherwise
  # working install, so this warns rather than dies.
  # -------------------------------------------------------------------------
  if [[ "${SKIP_SPEECH:-0}" == "1" ]]; then
    echo "[3/3] Skipping the speech engine (SKIP_SPEECH=1)."
  else
    echo "[3/3] Installing the speech engine from requirements-speech.txt ..."
    if ! "$VENV/bin/python" -m pip install -c "$CONSTRAINTS" -r requirements-speech.txt; then
      echo "" >&2
      echo "Warning: the speech engine could not be installed." >&2
      echo "Everything else is fine -- the demo will run without a voice." >&2
      echo "Retry it on its own with:" >&2
      echo "  $VENV/bin/python -m pip install -r requirements-speech.txt" >&2
      echo "" >&2
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 4. Verify the result rather than assuming it.
# ---------------------------------------------------------------------------
echo ""
echo "Verifying the environment ..."

# Capture into a variable instead of piping into `grep -q`: grep exits on the
# first match, closing the pipe, which makes pip die with
# "ERROR: Pipe to stdout was broken".
cuda_pkgs="$("$VENV/bin/python" -m pip list --format=freeze \
             | grep -iE '^(nvidia-|cuda-|triton)' || true)"

# executorch does not expose __version__, so read distribution metadata, which
# works whether or not a module defines it.
"$VENV/bin/python" - <<'PYEOF'
import importlib.metadata as md
import sys

missing = []
try:
    import torch
    print(f"  torch          {torch.__version__}  (cuda_build={torch.version.cuda})")
except ImportError:
    print("  torch          NOT INSTALLED")
    missing.append("torch")

for dist in ("executorch", "opencv-python", "numpy", "fastapi", "uvicorn"):
    try:
        print(f"  {dist:<14} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  {dist:<14} NOT INSTALLED")
        missing.append(dist)

try:
    from executorch.runtime import Runtime  # noqa: F401
    print("  executorch.runtime imports OK")
except Exception as exc:
    print(f"  executorch.runtime FAILED to import: {exc}")
    missing.append("executorch.runtime")

if missing:
    sys.exit(f"\nMissing or broken: {', '.join(missing)}")
PYEOF

if [[ -n "$cuda_pkgs" ]]; then
  echo ""
  echo "CUDA/GPU packages are installed and cannot run on this board:" >&2
  echo "$cuda_pkgs" | sed 's/^/  /' >&2
  echo "Disk used by the environment: $(du -sh "$VENV" 2>/dev/null | cut -f1)" >&2
  if [[ "${ALLOW_CUDA:-0}" != "1" ]]; then
    die "Refusing to finish with CUDA packages present.
  torch came from an index that bundles them. Re-run with a reachable CPU
  index, or accept them with: ALLOW_CUDA=1 ./install.sh"
  fi
  echo "Continuing anyway (ALLOW_CUDA=1)." >&2
fi

# The NPU path needs only the standard library plus numpy (already required); the
# NPU model itself is served by the separate rps-npu systemd service, installed
# once from the board scripts. This script does not touch it. Report its state.
echo ""
if systemctl is-active --quiet rps-npu 2>/dev/null; then
  echo "  rps-npu service: active (detection will use the NPU)"
else
  echo "  rps-npu service: not running (detection will use the CPU .pte)"
fi

# --- Speech: report what is and is not ready, without failing --------------
# Three independent things are needed for a voice, and each has its own fix.
echo ""
if "$VENV/bin/python" -c 'import piper' >/dev/null 2>&1; then
  echo "  speech engine:   installed ($("$VENV/bin/python" -m pip show piper-tts 2>/dev/null \
    | sed -n 's/^Version: //p' || echo 'version unknown'))"
else
  echo "  speech engine:   not installed (the demo will run silently)"
  echo "                   add it: $VENV/bin/pip install -r requirements-speech.txt"
fi

# The voice name and directory come from app/config.py, so this cannot drift
# from what start.sh will look for.
speech_voice_report="$(
  "$VENV/bin/python" - <<'PYEOF' 2>/dev/null || true
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "app"))
import config

onnx = Path(config.PIPER_VOICES_DIR) / f"{config.PIPER_VOICE}.onnx"
if onnx.is_file() and onnx.with_suffix(".onnx.json").is_file():
    size = onnx.stat().st_size / (1024 * 1024)
    print(f"ok|{config.PIPER_VOICE} ({size:.0f} MB)")
else:
    print(f"missing|{config.PIPER_VOICE}")
PYEOF
)"
case "$speech_voice_report" in
  ok\|*)      echo "  speech voice:    ${speech_voice_report#ok|}" ;;
  missing\|*) echo "  speech voice:    not downloaded"
              echo "                   get it: ./scripts/fetch-tts-voice.sh ${speech_voice_report#missing|}" ;;
  *)          echo "  speech voice:    could not be checked" ;;
esac

speech_players="$(for p in paplay pw-play aplay ffplay; do
  command -v "$p" >/dev/null 2>&1 && printf '%s ' "$p"
done)"
if [[ -n "$speech_players" ]]; then
  echo "  audio players:   ${speech_players% }"
else
  echo "  audio players:   none found (no sound will play)"
  echo "                   install one: sudo apt install alsa-utils"
fi

echo ""
echo "Installation complete. Start the application with ./start.sh."
echo ""
echo "Worth running once on a new board, before the demo:"
echo "  ./scripts/check-audio.sh                 confirm it can speak"
echo "  .venv/bin/python scripts/test-hand.py    confirm the hand moves"