#!/usr/bin/env python3
"""Top-1 accuracy of a CPU ExecuTorch classifier .pte on Edge Impulse's test split -- on a laptop.

    python tools/evaluate_pte.py model.pte X_split_test.npy Y_split_test.npy [--labels labels.txt]

The .npy files are in the Studio deployment zip under _inputs/ (see
blocks/deploy-executorch). Studio stores features flat and channels-last; this
script reshapes them to the model's input (read from the .pte) with the same
rule the board scripts use -- [1,F] vectors, [1,C,T] windows, [1,C,H,W] images or
spectrograms -- so the number is comparable with board/verify.sh (steps/accuracy_on_npu.py).
Works for the image, KWS and timeseries blocks; FOMO needs its own metric.

Needs the `executorch` Python runtime with a matching torch (ExecuTorch 1.5.1 ->
torch 2.14). A QNN .pte cannot run here -- use board/verify.sh on
the device for that.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pte")
    ap.add_argument("x")
    ap.add_argument("y")
    ap.add_argument("--labels", help="labels.txt (one class per line, output order)")
    a = ap.parse_args()

    import torch
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(a.pte).load_method("forward")
    shape = list(method.metadata.input_tensor_meta(0).sizes())

    x = np.load(a.x).astype(np.float32)
    y = np.load(a.y)
    y = y.argmax(1) if y.ndim > 1 else y.astype(int)
    want = int(np.prod(shape[1:]))
    if x[0].size != want:
        raise SystemExit(f"features have {x[0].size} values per sample, the model wants "
                         f"{shape[1:]}={want}. Was it trained on different features or resolution?")
    n = len(x)
    if len(shape) == 2:
        x = x.reshape(n, shape[1]).copy()
    elif len(shape) == 3:                       # (T, C) interleaved samples -> (C, T)
        x = x.reshape(n, shape[2], shape[1]).transpose(0, 2, 1).copy()
    else:                                       # (H, W, C) -> (C, H, W)
        x = x.reshape(n, shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2).copy()

    labels = None
    if a.labels:
        labels = [l.strip() for l in open(a.labels) if l.strip()]

    pred = []
    for row in x:
        out = method.execute([torch.from_numpy(np.ascontiguousarray(row[None]))])[0]
        pred.append(int(np.asarray(out).argmax()))
    pred = np.array(pred)
    acc = float((pred == y).mean())
    print(f"{a.pte}: input {shape[1:]}, accuracy {acc:.4f} on {len(y)} samples")
    wrong = Counter((int(t), int(p)) for t, p in zip(y, pred) if t != p)
    if wrong:
        name = (lambda i: labels[i]) if labels else str
        print("confusion (true -> predicted):",
              ", ".join(f"{name(t)}->{name(p)} x{k}" for (t, p), k in wrong.most_common()))


if __name__ == "__main__":
    main()
