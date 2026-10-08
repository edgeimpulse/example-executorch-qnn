#!/usr/bin/env bash
# Run model_qnn.pte on the Hexagon NPU with the C++ qnn_executor_runner.
# One random input, quickest "does it run at all" check. verify.sh calls this.
#   steps/run_runner.sh /path/to/model_qnn.pte [H W]        image models (3 channels, default 96 96)
#   steps/run_runner.sh /path/to/model_qnn.pte 1,1,49,13    any input shape, comma-separated (e.g. KWS)
#   steps/run_runner.sh /path/to/model_qnn.pte 1,39         (e.g. a timeseries MLP)
# The shape is in export_report.json -> input_shape.
set -euo pipefail
. "$(dirname "$0")/common.sh"
PTE=$(realpath "${1:?model_qnn.pte}")
if [[ "${2:-}" == *,* ]]; then SHAPE=$2; EXTRA_FROM=3; else SHAPE="1,3,${2:-96},${3:-96}"; EXTRA_FROM=4; fi
RUNNER="$DIST/qnn_executor_runner"
[ -x "$RUNNER" ] || die "no qnn_executor_runner in $DIST (run setup.sh, or use a package that includes it)"
# QNN runtime libs (libQnnHtp.so, libQnnSystem.so, stubs) come from apt; the skel from ADSP_DIR
npu_env

D=$(mktemp -d); cd "$D"
# stdlib only (system python has no numpy): random float32 NCHW input, values 0..1
python3 - "$SHAPE" <<'P'
import sys, random, array, math
n = math.prod(int(v) for v in sys.argv[1].split(","))
a = array.array("f", (random.random() for _ in range(n)))
open("input_0_0.raw", "wb").write(a.tobytes())
open("input_list.txt", "w").write("input_0_0.raw\n")
P
mkdir -p out   # the runner does not create its output folder
echo "ADSP_LIBRARY_PATH=$ADSP_LIBRARY_PATH"
"$RUNNER" --model_path "$PTE" --input_list_path input_list.txt --output_folder_path out --log_level 1 "${@:$EXTRA_FROM}" \
  2>&1 | tee run.log
echo "work dir: $D"; find out -type f | head
python3 - <<'P'
import glob, array
for f in sorted(glob.glob("out/**/*.raw", recursive=True)):
    a = array.array("f"); a.frombytes(open(f, "rb").read())
    v = list(a); print(f, [round(x, 4) for x in v[:8]], "argmax", v.index(max(v)))
P
