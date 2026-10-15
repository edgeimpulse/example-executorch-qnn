#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Build board/npu_server/npu_server.cpp on a laptop against the ExecuTorch pip
# wheel's C++ SDK, with the XNNPACK (CPU) backend instead of QNN. Same source,
# same socket protocol -- it just runs a CPU .pte. Use it to read, debug or
# demo the sidecar without an IQ-9075. (On the board, use board/setup.sh.)
#
#   ./tools/build_npu_server_cpu.sh                # needs: python with executorch==1.5.1, cmake >= 3.28, g++
#   build-npu-cpu/npu_server --model demo/rps-amazing-hand/models/model.pte --selftest
#   build-npu-cpu/npu_server --model demo/rps-amazing-hand/models/model.pte --socket /tmp/rps-npu.sock
#
# Linux only (the wheel's shared libraries are ELF). The INFO reply still says
# "qnn-htp" because that string is hard-coded in the server.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
ET=$("$PY" -c 'import executorch; print(list(executorch.__path__)[0])')
OUT=${OUT:-$ROOT/build-npu-cpu}
mkdir -p "$OUT/src"
cat > "$OUT/src/CMakeLists.txt" <<CM
cmake_minimum_required(VERSION 3.28)
project(npu_server_cpu CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(executorch CONFIG REQUIRED)
add_executable(npu_server $ROOT/board/npu_server/npu_server.cpp)
# Backends register themselves from static initialisers, so force-link them.
target_link_libraries(npu_server executorch::runtime
  "-Wl,--push-state,--no-as-needed,$ET/lib/libexecutorch_backend_xnnpack.so,$ET/lib/libexecutorch_kernels_optimized.so,--pop-state")
set_target_properties(npu_server PROPERTIES BUILD_RPATH $ET/lib RUNTIME_OUTPUT_DIRECTORY $OUT)
CM
cmake -S "$OUT/src" -B "$OUT/cmake" -DCMAKE_PREFIX_PATH="$ET" -DCMAKE_BUILD_TYPE=Release
cmake --build "$OUT/cmake" -j
echo "Built: $OUT/npu_server"
