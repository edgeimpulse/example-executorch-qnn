#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Verify that llama.cpp is really running on the Hexagon NPU on this board,
# and measure what it buys you over CPU.
#
# RUN THIS ON THE EVK, after unzipping pkg-linux and putting its bin/ on PATH.
#
#   ./check-llama-hexagon.sh ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf
#
# Uses llama-bench rather than llama-cli. Recent llama-cli builds (b11271+) ship
# an interactive TUI that suppresses the backend initialisation logs, so the
# ggml-hex: lines that prove an HTP session was created never appear. llama-bench
# is non-interactive and prints the backend it actually used as a column, which
# is the same evidence upstream's own docs use.
set -uo pipefail

MODEL="${1:-${LLAMA_MODEL_PATH:-}}"
DEVICE="${DEVICE:-HTP0}"
TIMEOUT="${TIMEOUT:-300}"
LOG_HTP="${LOG_HTP:-/tmp/htp-bench.log}"
LOG_CPU="${LOG_CPU:-/tmp/cpu-bench.log}"
SKIP_CPU="${SKIP_CPU:-0}"

die() { echo "FAIL: $*" >&2; exit 1; }

[[ -n "$MODEL" ]] || die "usage: $0 /path/to/model-Q4_0.gguf"
[[ -r "$MODEL" ]] || die "model not readable: $MODEL"

command -v llama-bench >/dev/null 2>&1 \
  || die "llama-bench is not on PATH. Add pkg-linux/bin to PATH first."

bin_dir="$(cd "$(dirname "$(command -v llama-bench)")" && pwd)"
lib_dir="$(cd "${bin_dir}/../lib" 2>/dev/null && pwd || true)"

echo "llama-bench : $(command -v llama-bench)"
echo "libraries   : ${lib_dir:-MISSING}"
echo "model       : $MODEL  ($(du -h "$MODEL" | cut -f1))"
echo ""

[[ -n "$lib_dir" ]] || die "no lib/ directory beside bin/. Keep them siblings --
  that is the layout pkg-linux ships and what the demo's start.sh assumes."

shopt -s nullglob
htp_libs=("$lib_dir"/libggml-htp-v*.so)
shopt -u nullglob
(( ${#htp_libs[@]} > 0 )) || die "no libggml-htp-v*.so in $lib_dir -- this build
  has no Hexagon backend and can only run on CPU."
echo "Hexagon kernels available:"
for lib in "${htp_libs[@]}"; do echo "  $(basename "$lib")"; done
echo ""

# ADSP_LIBRARY_PATH is how the DSP finds its skel libraries; omitting it is the
# usual reason the backend registers but no session can be created.
export LD_LIBRARY_PATH="$lib_dir"
export ADSP_LIBRARY_PATH="$lib_dir"

echo "Devices llama.cpp can see:"
llama-bench --list-devices 2>&1 | sed 's/^/  /' \
  || llama-cli --list-devices 2>&1 | sed 's/^/  /' || true
echo ""

case "$MODEL" in
  *Q4_0*|*q4_0*) : ;;
  *) echo "Note: $(basename "$MODEL") is not Q4_0. Every upstream Hexagon example" >&2
     echo "uses Q4_0; if the backend column says CPU, try a Q4_0 build." >&2
     echo "" >&2 ;;
esac

run_bench() {  # $1 = log file, rest = extra args
  local log="$1"; shift
  timeout --foreground "$TIMEOUT" \
    llama-bench -m "$MODEL" "$@" >"$log" 2>&1 </dev/null
}

echo "Benchmarking on ${DEVICE} (timeout ${TIMEOUT}s) ..."
run_bench "$LOG_HTP" --device "$DEVICE" -ngl 99
rc=$?
[[ $rc -eq 124 ]] && echo "  (timed out; partial results below)"

# llama-bench prints a markdown-ish table whose penultimate columns name the
# backend actually used. Pull any row mentioning the device.
echo ""
echo "Result rows:"
grep -E '^\|' "$LOG_HTP" | sed 's/^/  /' || sed 's/^/  /' "$LOG_HTP" | tail -20
echo ""

if grep -qiE '\| *HTP' "$LOG_HTP" || grep -qi 'hexagon' "$LOG_HTP"; then
  echo "PASS: llama-bench reports the Hexagon/HTP backend -- inference ran on the NPU."
else
  echo "FAIL: no HTP row in the results -- this was CPU inference." >&2
  cat >&2 <<EOF

Where to look, in order:
  1. ADSP_LIBRARY_PATH -- set here to $lib_dir. If the skel libraries live
     elsewhere, point it there.
  2. The board's Hexagon arch may be outside the versions this build shipped
     (listed above).
  3. fastrpc on this kernel may not reach the DSP. There is an open upstream
     issue about the Hexagon backend failing at the first compute round-trip
     on mainline fastrpc.

Full output: $LOG_HTP
EOF
  exit 1
fi

# --- CPU baseline, so the HTP number means something ----------------------
if [[ "$SKIP_CPU" == "1" ]]; then
  echo ""
  echo "(skipping the CPU baseline; SKIP_CPU=1)"
  exit 0
fi

echo ""
echo "Benchmarking on CPU for comparison (-ngl 0) ..."
run_bench "$LOG_CPU" -ngl 0
echo ""
echo "CPU rows:"
grep -E '^\|' "$LOG_CPU" | sed 's/^/  /' || tail -20 "$LOG_CPU" | sed 's/^/  /'

cat <<EOF

Compare the t/s columns between the two tables. That delta is what the NPU
buys you, and it is the number worth quoting -- generation speed on a 1B Q4_0
is plausible on either target, so neither figure means much alone.

Logs: $LOG_HTP and $LOG_CPU
EOF
