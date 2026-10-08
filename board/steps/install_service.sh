#!/usr/bin/env bash
# Install the runtime in $DIST as a systemd service that keeps a model on the NPU.
#   steps/install_service.sh /path/to/model_qnn.pte [/path/to/export_report.json]
#   SVC=kws-npu steps/install_service.sh kws_qnn.pte     # another model: own service + socket
# Files go to /opt/$SVC, the socket is /run/$SVC/npu.sock (default SVC=rps-npu, what the demo uses).
# Re-running replaces the model and restarts the service.
# Remove: sudo systemctl disable --now $SVC && sudo rm -rf /opt/$SVC /etc/systemd/system/$SVC.service
set -euo pipefail
. "$(dirname "$0")/common.sh"
MODEL=$(realpath "${1:?usage: $0 model_qnn.pte [export_report.json]}")
REPORT=${2:-}
SVC_USER=${SVC_USER:-${USER:-$(id -un)}}
[ -x "$DIST/npu_server" ] && [ -f "$DIST/libqnn_executorch_backend.so" ] \
  || die "no runtime in $DIST -- run setup.sh (build) or install-prebuilt.sh first"

# Check the report BEFORE touching /opt. A path that was given but does not
# exist used to abort mid-install under `set -e`, after the model had already
# been copied but before the unit was written -- leaving a new model on disk
# and the old one still being served. The report is optional, so warn and go on.
if [ -n "$REPORT" ] && [ ! -f "$REPORT" ]; then
  echo "Warning: no export report at $REPORT -- installing without it." >&2
  echo "         verify.sh will then guess the input shape instead of reading it," >&2
  echo "         and the demo's artifact panel omits its delegation and target rows." >&2
  echo "         It is written next to model_qnn.pte by the learning block." >&2
  REPORT=""
fi

say "Installing service $SVC (/opt/$SVC, socket /run/$SVC/npu.sock)"
sudo mkdir -p "/opt/$SVC"
sudo install -m 0755 "$DIST/npu_server" "/opt/$SVC/npu_server"
sudo install -m 0644 "$DIST/libqnn_executorch_backend.so" "/opt/$SVC/libqnn_executorch_backend.so"
sudo install -m 0644 "$MODEL" "/opt/$SVC/model_qnn.pte"
[ -n "$REPORT" ] && sudo install -m 0644 "$REPORT" "/opt/$SVC/export_report.json"

sudo tee "/etc/systemd/system/$SVC.service" >/dev/null <<U
[Unit]
Description=$SVC NPU inference server (ExecuTorch QNN / Hexagon HTP)
After=local-fs.target

[Service]
User=$SVC_USER
SupplementaryGroups=fastrpc
RuntimeDirectory=$SVC
RuntimeDirectoryMode=0775
Environment=LD_LIBRARY_PATH=/opt/$SVC:/usr/lib:/usr/lib/aarch64-linux-gnu
Environment=ADSP_LIBRARY_PATH=$ADSP_DIR
# Fails the unit (so apps use their CPU fallback) unless the model really runs on the NPU
ExecStartPre=/opt/$SVC/npu_server --model /opt/$SVC/model_qnn.pte --selftest
ExecStart=/opt/$SVC/npu_server --model /opt/$SVC/model_qnn.pte --socket /run/$SVC/npu.sock
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
U
sudo systemctl daemon-reload
sudo systemctl enable "$SVC" >/dev/null
sudo systemctl restart "$SVC"
sleep 2
if systemctl is-active --quiet "$SVC"; then
  echo "Service $SVC is running: socket /run/$SVC/npu.sock   (logs: journalctl -u $SVC -f)"
else
  systemctl --no-pager status "$SVC" | head -15
  die "service $SVC did not start -- see: journalctl -u $SVC -e"
fi
