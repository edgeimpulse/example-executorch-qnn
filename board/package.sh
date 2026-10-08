#!/usr/bin/env bash
# Package this board's built NPU runtime so other boards can skip the build.
#   ./package.sh [output-folder]       (after setup.sh)
# Produces npu-runtime-<executorch>-qnn<version>-<os>-aarch64.tar.gz with npu_server,
# libqnn_executorch_backend.so, qnn_executor_runner and VERSIONS. Share it (e.g. as a
# GitHub release asset); others install it with ./install-prebuilt.sh.
# It contains no Qualcomm SDK files: the QNN libraries come from each board's apt packages.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/steps/common.sh"
OUT=$(realpath "${1:-.}")
[ -x "$DIST/npu_server" ] && [ -f "$DIST/libqnn_executorch_backend.so" ] || die "nothing to package in $DIST -- run ./setup.sh first"
. /etc/os-release 2>/dev/null || true
qnn=$(sed -n 's/^qnn_runtime_built_against=//p' "$DIST/VERSIONS" | cut -d- -f1)
NAME="npu-runtime-${ET_TAG}-qnn${qnn:-unknown}-${ID:-linux}${VERSION_ID:-}-$(uname -m)"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir "$TMP/$NAME" && cp -a "$DIST"/. "$TMP/$NAME"/
(cd "$TMP/$NAME" && sha256sum npu_server libqnn_executorch_backend.so qnn_executor_runner 2>/dev/null > SHA256SUMS || true)
tar -czf "$OUT/$NAME.tar.gz" -C "$TMP" "$NAME"
echo "Package: $OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
echo "sha256:  $(sha256sum "$OUT/$NAME.tar.gz" | cut -d' ' -f1)"
echo "Others install it with: ./install-prebuilt.sh $NAME.tar.gz model_qnn.pte [export_report.json]"
