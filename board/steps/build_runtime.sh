#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Build ExecuTorch (ET_TAG, default v1.5.1) with the Qualcomm QNN backend natively on
# the board, plus qnn_executor_runner. Verified on an IQ-9075 EVK (Ubuntu 24.04,
# apt QNN 2.46). Based on backends/qualcomm/scripts/build.sh --enable_linux_embedded,
# minus the cross-compile flags. 30-60 min the first time; skipped when already built
# (FORCE_REBUILD=1 to redo).
#
# Compiling the QNN backend needs only the QNN *headers*: the backend dlopen()s the
# QNN libraries at run time. A tiny QNN_SDK_ROOT shim points at the apt headers
# (libqnn-dev), so nothing is downloaded from Qualcomm on the board.
set -euo pipefail
. "$(dirname "$0")/common.sh"

BUILD_ROOT="$WORK/executorch/build-native"
EX="$WORK/executorch/build-examples"
backend_lib() { find "$BUILD_ROOT" -name 'libqnn_executorch_backend.so' 2>/dev/null | head -1; }
runner_bin()  { find "$EX" -name qnn_executor_runner -type f 2>/dev/null | head -1; }

if [ "${FORCE_REBUILD:-0}" != 1 ] && [ -n "$(backend_lib)" ] && [ -n "$(runner_bin)" ]; then
  say "ExecuTorch + QNN already built in $WORK (FORCE_REBUILD=1 to rebuild)"
  exit 0
fi
mkdir -p "$WORK" && cd "$WORK"

# 0. QNN_SDK_ROOT shim: only include/QNN is needed to compile
mkdir -p qnn-shim && ln -sfn /usr/include qnn-shim/include
export QNN_SDK_ROOT="$WORK/qnn-shim"
test -f "$QNN_SDK_ROOT/include/QNN/QnnInterface.h" || die "missing QNN headers (apt: libqnn-dev)"

# 1. isolated build venv (does not touch any existing Python environment)
say "Build environment (cmake, ninja, torch for code generation)"
[ -d venv-build ] || python3 -m venv venv-build
. venv-build/bin/activate
pip install -q -U pip setuptools wheel cmake ninja pyyaml packaging tomli
pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # needed by ET codegen
cmake --version | head -1   # must be >= 3.29

# 2. source at the matching tag
[ -d executorch ] || git clone --branch "$ET_TAG" --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/pytorch/executorch.git
cd executorch
pip install -q -r backends/qualcomm/requirements.txt || true
PY=$(which python)

# 3. main ExecuTorch build + install (native: no sysroot / cross compiler)
say "Building ExecuTorch $ET_TAG with the QNN backend (logs: $WORK/build-main.log)"
echo "    30-60 minutes the first time. Follow along with: tail -f $WORK/build-main.log"
cmake . \
  -DCMAKE_INSTALL_PREFIX="$BUILD_ROOT" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DPYTHON_EXECUTABLE="$PY" -B"$BUILD_ROOT" > ../configure.log 2>&1 || die "configure failed: $WORK/configure.log"
cmake --build "$BUILD_ROOT" -j"$JOBS" --target install > ../build-main.log 2>&1 \
  || die "build failed: tail $WORK/build-main.log (out of memory? retry with JOBS=2)"

# 4. the runner (examples/qualcomm). The example CMake also builds LLM/whisper/t5/
# stable-diffusion runners that need tokenizers/re2; disable those (idempotent).
say "Building qnn_executor_runner (logs: $WORK/build-ex.log)"
sed -i -E 's|^(add_subdirectory\(\$\{CMAKE_CURRENT_SOURCE_DIR\}/(oss_scripts\|qaihub_scripts)/)|# \1|' examples/qualcomm/CMakeLists.txt
[ "${FORCE_REBUILD:-0}" = 1 ] && rm -rf "$EX"
cmake examples/qualcomm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$BUILD_ROOT;$BUILD_ROOT/third-party/gflags;" \
  -DSUPPORT_REGEX_LOOKAHEAD=ON -DBUILD_TESTING=OFF -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT" -DPYTHON_EXECUTABLE="$PY" -B"$EX" > ../configure-ex.log 2>&1 \
  || die "runner configure failed: $WORK/configure-ex.log"
cmake --build "$EX" -j"$JOBS" --target qnn_executor_runner > ../build-ex.log 2>&1 \
  || die "runner build failed: tail $WORK/build-ex.log"

echo "Runner:          $(runner_bin)"
echo "QNN backend lib: $(backend_lib)"
