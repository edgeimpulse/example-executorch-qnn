#!/usr/bin/env bash
# Install a prebuilt NPU runtime (made by package.sh on another board) -- no build.
#
#   ./install-prebuilt.sh npu-runtime-v1.5.1-qnn2.46.0-ubuntu24.04-aarch64.tar.gz \
#        ~/npu-model/model_qnn.pte [~/npu-model/export_report.json]
#
# Needs the same board family and OS as the package was built on (see VERSIONS in it).
# Installs the QNN runtime from Ubuntu's IoT PPA if missing, unpacks to ~/et-qnn/dist,
# self-tests the model on the NPU and installs the `rps-npu` service.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/steps/common.sh"
PKG=${1:?usage: ./install-prebuilt.sh <package.tar.gz|folder> model_qnn.pte [export_report.json]}
MODEL=${2:?usage: ./install-prebuilt.sh <package.tar.gz|folder> model_qnn.pte [export_report.json]}
REPORT=${3:-}
[ -f "$MODEL" ] || die "model not found: $MODEL"

say "1/4 QNN runtime"; "$HERE/steps/prereqs.sh" runtime

say "2/4 Unpacking the runtime into $DIST"
mkdir -p "$DIST"
if [ -d "$PKG" ]; then cp -a "$PKG"/. "$DIST"/
else tar -xzf "$PKG" -C "$DIST" --strip-components=1; fi
[ -x "$DIST/npu_server" ] && [ -f "$DIST/libqnn_executorch_backend.so" ] || die "$PKG is not an npu-runtime package"
cat "$DIST/VERSIONS" 2>/dev/null || true
grep -q "arch=$(uname -m)" "$DIST/VERSIONS" 2>/dev/null || echo "Warning: package built for a different architecture" >&2
missing=$(LD_LIBRARY_PATH="$DIST" ldd "$DIST/npu_server" | grep "not found" || true)
[ -z "$missing" ] || die "npu_server needs libraries this board lacks:
$missing"
if [ -x "$DIST/qnn_executor_runner" ] && LD_LIBRARY_PATH="$DIST" ldd "$DIST/qnn_executor_runner" | grep -q "not found"; then
  echo "Note: qnn_executor_runner (used only by verify.sh) is missing libraries:" >&2
  LD_LIBRARY_PATH="$DIST" ldd "$DIST/qnn_executor_runner" | grep "not found" >&2
fi
built=$(sed -n 's/^qnn_runtime_built_against=//p' "$DIST/VERSIONS" 2>/dev/null | cut -d- -f1)
have=$(pkg_version libqnn1 | cut -d- -f1)
if [ -n "$built" ] && [ "$built" != "$have" ]; then
  echo "Note: package built against QNN $built, this board has $have. A model compiled with an" >&2
  echo "      older QNN SDK runs on a newer runtime; the self-test below is the real check." >&2
fi

say "3/4 Self-test on the NPU"
npu_env
"$DIST/npu_server" --model "$MODEL" --selftest || die "the model did not run on the NPU (see the messages above)"

say "4/4 Service"; "$HERE/steps/install_service.sh" "$MODEL" ${REPORT:+"$REPORT"}
say "Done. Check it end to end with:  ./verify.sh $(dirname "$(realpath "$MODEL")")"
