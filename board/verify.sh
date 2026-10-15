#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Prove the model runs on the NPU and matches the accuracy predicted at export.
#
#   ./verify.sh ~/npu-model
#
# The folder holds the files from Studio's deploy.zip: model_qnn.pte, export_report.json,
# and (from _inputs/) X_split_test.npy, Y_split_test.npy. Runs three checks:
#   1. one inference with qnn_executor_runner            (does it run on the NPU at all?)
#   2. the whole test split through the runner           (accuracy == report's int8 value?)
#   3. the same test split through the running service   (what apps will get)
# Checks 2-3 need a small Python env with numpy; it is created at ~/npu-tools if missing.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/steps/common.sh"
D=$(realpath "${1:?usage: ./verify.sh <folder with model_qnn.pte, export_report.json, X/Y_split_test.npy>}")
PTE="$D/model_qnn.pte"; REP="$D/export_report.json"; X="$D/X_split_test.npy"; Y="$D/Y_split_test.npy"
[ -f "$PTE" ] || die "no model_qnn.pte in $D"
results=()

# Input shape, in order of preference: an explicit SHAPE, then the report, then
# a guess. The guess is only ever right by luck -- feeding the wrong size makes
# qnn_executor_runner abort with a raw byte-count assert, which reads like a
# broken model rather than a missing file -- so say loudly when we are guessing.
#   SHAPE=1,3,160,160 ./verify.sh ~/npu-model
SHAPE_SOURCE="guess"
if [ -n "${SHAPE:-}" ]; then
  SHAPE_SOURCE="the SHAPE environment variable"
elif [ -f "$REP" ]; then
  s=$(python3 -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1])).get('input_shape') or [])))" "$REP" 2>/dev/null)
  if [ -n "$s" ]; then SHAPE=$s; SHAPE_SOURCE="export_report.json"; fi
fi

if [ -z "${SHAPE:-}" ]; then
  SHAPE="1,3,96,96"
  echo "Warning: no $REP, so the input shape is a guess ($SHAPE)." >&2
  echo "         If the model is not 96x96 the next check fails on a size mismatch," >&2
  echo "         which means the wrong input, not a broken model. Either copy the" >&2
  echo "         report in, or pass the real shape:" >&2
  echo "           SHAPE=1,3,160,160 ./verify.sh $D" >&2
  echo "" >&2
fi

say "1/3 One inference on the NPU (input $SHAPE, from $SHAPE_SOURCE)"
if out=$("$HERE/steps/run_runner.sh" "$PTE" "$SHAPE" 2>&1) && grep -q "inference took" <<<"$out"; then
  grep -E "backend type|RESTORE MODE|inference took|argmax" <<<"$out" | tail -4
  results+=("PASS  runs on the NPU")
else
  echo "$out" | tail -15
  # A size mismatch means we fed the wrong input, not that the model is broken.
  # The runner only reports raw byte counts, so translate them back into the
  # shape the model actually wants and print the command that would work.
  if want=$(sed -n 's/.*tensor bytes: \([0-9]*\).*/\1/p' <<<"$out" | tail -1) && [ -n "$want" ]; then
    echo ""
    echo "This is an input-size mismatch, not a failed model: the program ran, it" >&2
    echo "was handed the wrong number of bytes. The model wants $want bytes." >&2
    side=$(python3 -c "
import math, sys
n = int(sys.argv[1]) // (3 * 4)          # 3 channels, float32
s = int(math.isqrt(n))
print(s if s * s == n else 0)" "$want" 2>/dev/null || echo 0)
    if [ "${side:-0}" != "0" ]; then
      echo "That is a 3-channel float32 image of ${side}x${side}:" >&2
      echo "  SHAPE=1,3,${side},${side} ./verify.sh $D" >&2
      echo "Better still, copy export_report.json into $D so the shape is read, not guessed." >&2
    fi
  fi
  results+=("FAIL  single inference (output above)")
fi

if [ -f "$X" ] && [ -f "$Y" ]; then
  PY="$HOME/npu-tools/bin/python"
  if [ ! -x "$PY" ]; then
    say "Creating ~/npu-tools (numpy) for the accuracy checks"
    python3 -m venv "$HOME/npu-tools" && "$HOME/npu-tools/bin/pip" -q install numpy
  fi
  say "2/3 Test split on the NPU"
  rep_arg=(); [ -f "$REP" ] && rep_arg=("$REP")
  if "$PY" "$HERE/steps/accuracy_on_npu.py" "$PTE" "$X" "$Y" "${rep_arg[@]}" 2>&1 | tail -4; then
    results+=("DONE  accuracy on the NPU (compare with the report's int8 value above)")
  else results+=("FAIL  accuracy run"); fi

  SOCK="/run/$SVC/npu.sock"
  say "3/3 Through the $SVC service"
  if [ -S "$SOCK" ]; then
    if RPS_NPU_SOCKET="$SOCK" "$PY" "$HERE/steps/test_npu_service.py" "$X" "$Y" 2>&1 | tail -3; then
      results+=("DONE  service answers (accuracy should equal step 2)")
    else results+=("FAIL  service test"); fi
  else
    echo "No socket at $SOCK -- is the service installed? (./setup.sh or ./install-prebuilt.sh)"
    results+=("SKIP  service not running")
  fi
else
  results+=("SKIP  accuracy checks: put X_split_test.npy and Y_split_test.npy (deploy.zip _inputs/) in $D")
fi

say "Summary"; printf '  %s\n' "${results[@]}"
