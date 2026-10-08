#!/usr/bin/env bash
# Cross-compile llama.cpp with the Snapdragon Hexagon (HTP) backend.
#
# RUN THIS ON AN x86_64 MACHINE WITH DOCKER, not on the EVK. Qualcomm's toolchain
# image is linux/amd64 and this is a cross-compile. On Apple Silicon it runs under
# emulation: enable Docker Desktop's Rosetta option or expect a very long build.
#
#   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
#   /path/to/this/repo/demo/rps-amazing-hand/scripts/build-llama-hexagon.sh
#
# Produces pkg-linux.zip, whose layout (bin/ beside lib/) is what the demo's
# start.sh expects: it finds llama-server on PATH or in $LLAMA_CPP_DIR/bin and
# derives LD_LIBRARY_PATH and ADSP_LIBRARY_PATH from <bin dir>/../lib.
#
# Reconstructed from the version that produced the pkg-linux used on the
# IQ-9075 booth unit (llama.cpp build 6a2743f02 / b11271).
set -euo pipefail

TOOLCHAIN_IMAGE="${TOOLCHAIN_IMAGE:-ghcr.io/snapdragon-toolchain/arm64-linux:v0.7}"
PRESET="${PRESET:-arm64-linux-snapdragon-release}"
BUILD_DIR="${BUILD_DIR:-build-snapdragon}"
PKG_DIR="${PKG_DIR:-pkg-linux}"
# Parallel compile jobs (default: nproc inside the container). Lower it, e.g.
# JOBS=4, if the build gets OOM-killed -- common in Docker Desktop's VM.
JOBS="${JOBS:-}"

die() { echo "Error: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH."
docker info >/dev/null 2>&1 || die "the docker daemon is not reachable."

[[ -f docs/backend/snapdragon/CMakeUserPresets.json ]] \
  || die "run this from the root of a llama.cpp clone
  (docs/backend/snapdragon/CMakeUserPresets.json was not found).
    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"

host_arch="$(uname -m)"
if [[ "$host_arch" != "x86_64" ]]; then
  echo "Warning: host is ${host_arch}, but ${TOOLCHAIN_IMAGE} is linux/amd64." >&2
  echo "The build runs under emulation and can take a long time. Continue? [y/N]" >&2
  read -r reply
  [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

echo "Toolchain : ${TOOLCHAIN_IMAGE}"
echo "Preset    : ${PRESET}"
echo "Source    : $(pwd)"
echo "Commit    : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo ""

# -u keeps the output files owned by the invoking user rather than root.
docker run --rm \
  -u "$(id -u):$(id -g)" \
  --volume "$(pwd):/workspace" \
  --platform linux/amd64 \
  "$TOOLCHAIN_IMAGE" \
  bash -lc "
    set -euo pipefail
    cd /workspace
    cp docs/backend/snapdragon/CMakeUserPresets.json .
    jobs='${JOBS}'
    [[ -n \"\$jobs\" ]] || jobs=\$(nproc)
    echo \"Building with \$jobs parallel jobs ...\"
    cmake --preset ${PRESET} -B ${BUILD_DIR}
    cmake --build ${BUILD_DIR} -j \$jobs
    cmake --install ${BUILD_DIR} --prefix ${PKG_DIR}
  " || die "the build failed inside the toolchain container (see output above)."

# --- Sanity-check the output before packaging ------------------------------
[[ -x "${PKG_DIR}/bin/llama-server" ]] \
  || die "${PKG_DIR}/bin/llama-server is missing. The demo needs the HTTP server."

# One Hexagon kernel library per arch version (v73 is the IQ-9075). If none were
# built, the package would silently run on the CPU on the board.
# (No GNU find -printf here: this part runs on the host, which may be macOS.)
htp_libs="$(for f in "${PKG_DIR}"/lib/libggml-htp-v*.so; do [[ -e "$f" ]] && basename "$f"; done | sort || true)"
[[ -n "$htp_libs" ]] \
  || die "no libggml-htp-v*.so found in ${PKG_DIR}/lib.
  The Hexagon backend was not built, so this package would fall back to CPU."

echo ""
echo "Hexagon kernel libraries built:"
echo "$htp_libs" | sed 's/^/  /'

rm -f "${PKG_DIR}.zip"
zip -qr "${PKG_DIR}.zip" "${PKG_DIR}"

echo ""
echo "Built $(du -h "${PKG_DIR}.zip" | cut -f1) -> ${PKG_DIR}.zip"
cat <<EOF

Next, on the EVK:

  scp ${PKG_DIR}.zip ubuntu@<evk-ip>:~/
  ssh ubuntu@<evk-ip> 'unzip -q ${PKG_DIR}.zip'      # -> ~/${PKG_DIR}

Then confirm the NPU is really used (llama.cpp falls back to CPU silently):

  PATH=\$HOME/${PKG_DIR}/bin:\$PATH ./scripts/check-llama-hexagon.sh models/Llama-3.2-1B-Instruct-Q4_0.gguf
EOF
