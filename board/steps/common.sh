# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Shared settings for the board scripts. Sourced, not run.
# Every value can be overridden from the environment.
WORK=${WORK:-$HOME/et-qnn}          # ExecuTorch source + build tree
ET_TAG=${ET_TAG:-v1.5.1}            # must equal the ExecuTorch version that exported model_qnn.pte
JOBS=${JOBS:-4}                     # parallel build jobs; lower it if the build runs out of memory
DIST=${DIST:-$WORK/dist}            # the runtime: npu_server + libqnn_executorch_backend.so (+ qnn_executor_runner)
SVC=${SVC:-rps-npu}                 # systemd service name; socket at /run/$SVC/npu.sock
# Folder with this board's Hexagon skel library (libQnnHtpV73Skel.so on the IQ-9075 / SA8775P family)
ADSP_DIR=${ADSP_LIBRARY_PATH:-/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp}
QNN_PPA=ppa:ubuntu-qcom-iot/qcom-ppa

say() { printf '\n==> %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Library paths for anything that runs a QNN .pte from $DIST.
npu_env() {
  export LD_LIBRARY_PATH="$DIST:/usr/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export ADSP_LIBRARY_PATH="$ADSP_DIR"
}

pkg_version() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true; }
