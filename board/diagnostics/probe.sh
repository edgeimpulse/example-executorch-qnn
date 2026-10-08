#!/usr/bin/env bash
# Read-only probe of the board: architecture, OS, QNN packages and libraries,
# DSP device nodes, and which ExecuTorch backends a Python env registers.
#   diagnostics/probe.sh            (PY=/path/to/python to check another env)
set -u
echo "== arch / os"; uname -m; . /etc/os-release; echo "$PRETTY_NAME"
echo "== libqnn packages"; dpkg -l | grep -i qnn || echo "(none)"
echo "== QNN libs on disk"
find / \( -name 'libQnnHtp*.so' -o -name 'libQnnSystem.so' -o -name 'libQnnHtpV73Skel.so' \) 2>/dev/null | head -20
echo "== QNN headers (needed for a native ET build)"
find / -name 'QnnInterface.h' 2>/dev/null | head -3 || true
echo "== DSP devices"; ls -l /dev/fastrpc-* /dev/adsprpc* 2>&1 | head
echo "== env"; echo "QNN_SDK_ROOT=${QNN_SDK_ROOT:-} ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH:-}"
echo "== python env"
PY=${PY:-python3}   # e.g. PY=~/npu-tools/bin/python
"$PY" - <<'P'
import sys, importlib.util
print(sys.version)
for m in ("torch","executorch","torchao","numpy"):
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m,"MISSING",e)
try:
    from executorch.extension.pybindings import portable_lib as pl
    print("backends registered:", pl._get_registered_backend_names())
except Exception as e:
    print("pybindings error:", e)
P
