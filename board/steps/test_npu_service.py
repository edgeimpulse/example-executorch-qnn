#!/usr/bin/env python3
"""End-to-end test of the running NPU service through the Python client.

    ~/npu-tools/bin/python steps/test_npu_service.py X_split_test.npy Y_split_test.npy   (verify.sh calls this)
Reports accuracy (should match accuracy_on_npu.py exactly), per-call round-trip latency and the NPU-side time.
"""
import os, sys, time
import numpy as np
# npu_client.py lives with the demo app (single copy): ../demo/rps-amazing-hand/app
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "demo", "rps-amazing-hand", "app"))
from npu_client import NpuClient, softmax


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


X = np.load(sys.argv[1]).astype(np.float32); Y = load_labels(sys.argv[2])
c = NpuClient(os.environ.get("RPS_NPU_SOCKET", "/run/rps-npu/npu.sock"))
assert c.connect(), f"cannot connect to {c.sock_path}: is the service running? (systemctl status <name>, e.g. rps-npu)"
print("server info:", c.info)
X = to_model_layout(X, c.info["input_shape"])     # Studio's flat channels-last -> model layout
pred, rtt, npu = [], [], []
for x in X:
    t = time.perf_counter(); r = c.infer(x); rtt.append((time.perf_counter() - t) * 1e3)
    assert r is not None, "inference failed"
    pred.append(int(softmax(r[0]).argmax())); npu.append(r[1] / 1e3)
if Y is not None:
    acc = float((np.array(pred) == Y).mean())
    print(f"accuracy via service: {acc:.4f} (n={len(Y)})  (compare with accuracy_on_npu.py)")
else:
    print(f"ran {len(X)} samples via service (labels not a class matrix: no top-1 accuracy)")
print(f"round trip ms  median {np.median(rtt):.2f}  p95 {np.percentile(rtt, 95):.2f}")
print(f"NPU-side ms    median {np.median(npu):.3f}  p95 {np.percentile(npu, 95):.3f}")
