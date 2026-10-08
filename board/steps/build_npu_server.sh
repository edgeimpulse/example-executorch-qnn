#!/usr/bin/env bash
# Build the npu_server sidecar inside the ExecuTorch examples build (after
# build_runtime.sh), then stage the runtime into $DIST:
#   npu_server  libqnn_executorch_backend.so  qnn_executor_runner  VERSIONS
# $DIST is what install_service.sh installs and what package.sh ships.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/common.sh"
SRC="$HERE/../npu_server"
cd "$WORK/executorch"
. "$WORK/venv-build/bin/activate"
export QNN_SDK_ROOT="$WORK/qnn-shim"

# 1. add our source as one more example target
rm -rf examples/qualcomm/npu_server && cp -r "$SRC" examples/qualcomm/npu_server
grep -q "npu_server" examples/qualcomm/CMakeLists.txt || \
  printf '\n# npu_server sidecar\nadd_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/npu_server)\n' >> examples/qualcomm/CMakeLists.txt

# 2. reconfigure + build just that target
say "Building npu_server (logs: $WORK/build-npu-server.log)"
EX="$PWD/build-examples"
cmake -B"$EX" examples/qualcomm > /dev/null
cmake --build "$EX" -j"$JOBS" --target npu_server > ../build-npu-server.log 2>&1 \
  || die "npu_server build failed: tail $WORK/build-npu-server.log"

# 3. stage the runtime
BIN=$(find "$EX" -name npu_server -type f | head -1)
RUN=$(find "$EX" -name qnn_executor_runner -type f | head -1)
LIB=$(find "$WORK/executorch/build-native" -name libqnn_executorch_backend.so | head -1)
mkdir -p "$DIST"
install -m 0755 "$BIN" "$DIST/npu_server"
install -m 0644 "$LIB" "$DIST/libqnn_executorch_backend.so"
[ -n "$RUN" ] && install -m 0755 "$RUN" "$DIST/qnn_executor_runner"
. /etc/os-release
{
  echo "executorch=$ET_TAG ($(git -C "$WORK/executorch" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  echo "qnn_runtime_built_against=$(pkg_version libqnn1)"
  echo "os=${PRETTY_NAME:-?}"
  echo "glibc=$(ldd --version | head -1 | awk '{print $NF}')"
  echo "arch=$(uname -m)"
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$DIST/VERSIONS"
echo "Staged runtime in $DIST:"; ls -l "$DIST"
