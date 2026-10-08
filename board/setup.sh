#!/usr/bin/env bash
# One command from a fresh IQ-9075 (Ubuntu 24.04) to a model running on the NPU as a service.
#
#   ./setup.sh ~/npu-model/model_qnn.pte [~/npu-model/export_report.json]
#
# 1. installs the prerequisites (Qualcomm QNN runtime + headers from Ubuntu's IoT PPA, build tools)
# 2. builds ExecuTorch v1.5.1 with the QNN backend, natively   (30-60 min, once; skipped if already built)
# 3. builds npu_server and stages the runtime in ~/et-qnn/dist
# 4. self-tests the model on the NPU
# 5. installs and starts the `rps-npu` systemd service (socket /run/rps-npu/npu.sock)
#
# Then: ./verify.sh ~/npu-model     to prove it    |   ./package.sh   to let others skip the build
# Settings (environment): SVC=<service name>  JOBS=<build jobs>  WORK=<build folder>  FORCE_REBUILD=1
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/steps/common.sh"
MODEL=${1:?usage: ./setup.sh model_qnn.pte [export_report.json]}
REPORT=${2:-}
[ -f "$MODEL" ] || die "model not found: $MODEL"

say "1/5 Prerequisites";            "$HERE/steps/prereqs.sh" build
say "2/5 ExecuTorch + QNN backend"; "$HERE/steps/build_runtime.sh"
say "3/5 npu_server";               "$HERE/steps/build_npu_server.sh"
say "4/5 Self-test on the NPU"
npu_env
"$DIST/npu_server" --model "$MODEL" --selftest || die "the model did not run on the NPU (see the messages above)"
say "5/5 Service";                  "$HERE/steps/install_service.sh" "$MODEL" ${REPORT:+"$REPORT"}

say "Done. Check it end to end with:  ./verify.sh $(dirname "$(realpath "$MODEL")")"
