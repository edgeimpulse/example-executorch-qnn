#!/usr/bin/env python3
"""Run the real test split through model_qnn.pte on the NPU and report accuracy.

Works for any block's model: the input layout comes from export_report.json
(input_shape). Classifiers get top-1 accuracy; for other tasks (e.g. FOMO, whose
Y split is JSON) the raw outputs are saved to npu_outputs.npy for scoring.

    ~/npu-tools/bin/python steps/accuracy_on_npu.py MODEL_QNN.pte X_split_test.npy Y_split_test.npy \
        [export_report.json] [--runner PATH]      (verify.sh calls this)

X_split_test.npy / Y_split_test.npy are in the Studio deployment zip under _inputs/.
Uses numpy (so run with a venv that has it, e.g. ~/npu-tools/bin/python) and calls the C++ qnn_executor_runner once
for all samples.
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile
import numpy as np


def to_model_layout(x, shape):
    """Studio features (flat, channels-last) -> the model's input layout.

    One rule covers every block: [1,F] vectors as-is; [1,C,T] windows from
    (T,C) interleaved samples; [1,C,H,W] images/spectrograms from (H,W,C).
    """
    n = len(x)
    if len(shape) == 2:
        return x.reshape(n, shape[1]).copy()
    if len(shape) == 3:
        _, c, t = shape
        return x.reshape(n, t, c).transpose(0, 2, 1).copy()
    _, c, h, w = shape
    return x.reshape(n, h, w, c).transpose(0, 3, 1, 2).copy()


def load_labels(path):
    """Integer labels from Studio's one-hot Y split, or None for tasks whose Y is
    not a label matrix (FOMO's Y_split is JSON text despite the .npy name)."""
    try:
        y = np.load(path)
    except Exception:  # noqa: BLE001
        return None
    return y.argmax(1) if y.ndim > 1 else y.astype(int)


ap = argparse.ArgumentParser()
ap.add_argument("pte"); ap.add_argument("x"); ap.add_argument("y")
ap.add_argument("report", nargs="?")
ap.add_argument("--runner")
ap.add_argument("--size", type=int, default=0, help="image H=W if there is no export_report.json (assumes 3 channels)")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

work = os.path.expanduser(os.environ.get("WORK", "~/et-qnn"))
dist = os.path.expanduser(os.environ.get("DIST", os.path.join(work, "dist")))
runner = args.runner or os.path.join(dist, "qnn_executor_runner")
backend = os.path.join(dist, "libqnn_executorch_backend.so")
if not (os.path.exists(runner) and os.path.exists(backend)):
    sys.exit(f"runtime not found in {dist} (run setup.sh or install-prebuilt.sh)")

report = json.load(open(args.report)) if args.report else {}
labels = report.get("labels")
if args.report and not report.get("input_shape"):
    sys.exit(f"{args.report} has no input_shape (export status: {report.get('status')}). "
             "Use the report from a successful QNN export.")
shape = report.get("input_shape") or [1, 3, args.size or 96, args.size or 96]
X = np.load(args.x).astype(np.float32)
Y = load_labels(args.y)
if args.limit:
    X = X[:args.limit]
    Y = Y[:args.limit] if Y is not None else None
n = len(X)
X = to_model_layout(X, shape)   # same layout rule the blocks use for training

d = tempfile.mkdtemp(prefix="npu_acc_")
os.makedirs(os.path.join(d, "out"), exist_ok=True)   # the runner does not create it
with open(os.path.join(d, "input_list.txt"), "w") as f:
    for i in range(n):
        X[i:i+1].tofile(os.path.join(d, f"input_{i}_0.raw"))
        f.write(f"input_{i}_0.raw\n")

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f"{os.path.dirname(backend)}:/usr/lib:/usr/lib/aarch64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
env.setdefault("ADSP_LIBRARY_PATH", "/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp")
r = subprocess.run([runner, "--model_path", os.path.abspath(args.pte), "--input_list_path", "input_list.txt",
                    "--output_folder_path", "out", "--log_level", "1"],
                   cwd=d, env=env, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-500:])
if r.returncode:
    sys.exit(f"runner failed ({r.returncode}); work dir {d}")

outs = {}
for f in glob.glob(os.path.join(d, "out", "**", "*.raw"), recursive=True):
    m = re.search(r"output_(\d+)_(\d+)\.raw$", f)
    if m and m.group(2) == "0":
        outs[int(m.group(1))] = np.fromfile(f, dtype=np.float32)
if len(outs) != n:
    sys.exit(f"expected {n} outputs, found {len(outs)} under {d}/out; files: "
             f"{[os.path.relpath(p, d) for p in glob.glob(d + '/**/*', recursive=True)][:15]}")
P = np.stack([outs[i] for i in range(n)])
np.save(os.path.join(d, "npu_outputs.npy"), P)
if Y is None or P.shape[1] != len(labels or P[0]):
    print(f"\nRan {n} samples on the NPU; output shape per sample {P.shape[1:]}. "
          f"Not a plain classifier, so no top-1 accuracy here -- score "
          f"{d}/npu_outputs.npy with the task's own metric.")
    sys.exit(0)
pred = P.argmax(1)
acc = float((pred == Y).mean())
ref = report.get("accuracy", {})
print(f"\nNPU accuracy on {n} test samples: {acc:.4f}"
      + (f"  (report: float32 {ref.get('float32')}, int8 simulated {ref.get('int8_fake_quant')})" if ref else ""))
if labels:
    from collections import Counter
    print("confusion (true -> predicted):", dict(Counter((labels[t], labels[p]) for t, p in zip(Y, pred) if t != p)) or "none wrong")
print("work dir:", d)
