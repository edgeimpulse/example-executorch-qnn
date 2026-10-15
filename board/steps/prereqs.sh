#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Install what the board needs. Idempotent: anything already present is skipped.
#   steps/prereqs.sh runtime   QNN runtime only (libqnn1)          -- for install-prebuilt.sh
#   steps/prereqs.sh build     runtime + headers + build tools     -- for setup.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"
MODE=${1:-build}

[ "$(uname -m)" = aarch64 ] || die "run this on the board (aarch64), not on $(uname -m)"
. /etc/os-release
[ "${ID:-}" = ubuntu ] || echo "Warning: tested on Ubuntu 24.04; this is ${PRETTY_NAME:-unknown}" >&2

pkgs=(libqnn1)
[ "$MODE" = build ] && pkgs+=(libqnn-dev git python3-venv build-essential libabsl-dev libgflags-dev)
missing=()
for p in "${pkgs[@]}"; do [ -n "$(pkg_version "$p")" ] || missing+=("$p"); done

if [ ${#missing[@]} -gt 0 ]; then
  say "Installing: ${missing[*]}"
  if [[ " ${missing[*]} " == *" libqnn1 "* || " ${missing[*]} " == *" libqnn-dev "* ]]; then
    # Qualcomm's QNN runtime for Ubuntu (the same PPA Edge Impulse's IQ-9075 guide uses)
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y "$QNN_PPA"
  fi
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi

echo "QNN runtime (libqnn1): $(pkg_version libqnn1)"
[ -d "$ADSP_DIR" ] || echo "Warning: $ADSP_DIR not found. On boards other than the IQ-9075, set ADSP_LIBRARY_PATH to the folder holding libQnnHtpV<NN>Skel.so (diagnostics/find_qnn_runtime.sh)." >&2
if [ "$MODE" = build ]; then
  [ -f /usr/include/QNN/QnnInterface.h ] || die "QNN headers missing (/usr/include/QNN). Is libqnn-dev installed?"
fi
