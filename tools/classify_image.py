#!/usr/bin/env python3
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Classify one photo with an image-classifier model.pte, on any computer.

    pip install executorch==1.5.1 torch==2.14.0 opencv-python numpy
    # Linux: add --extra-index-url https://download.pytorch.org/whl/cpu to skip CUDA
    python tools/classify_image.py model.pte labels.txt hand.jpg

Preprocessing is the same as the demos (IQ-9075 and Android): centre-crop the
largest square, resize to the model's input size (read from the .pte), RGB,
scale to 0..1, NCHW float32. Prints the class scores, best first, and the time
for one inference.

For the CPU model.pte (XNNPACK). Tested on x86_64 Linux; it should work wherever
the executorch pip wheel installs. A QNN model_qnn.pte only runs on the device it
was compiled for (see board/).
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime


def preprocess(path: str, size_hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"cannot read image: {path}")
    h, w = bgr.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    square = bgr[top:top + side, left:left + side]
    resized = cv2.resize(square, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])          # (1, 3, H, W)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="model.pte (CPU / XNNPACK)")
    ap.add_argument("labels", help="labels.txt, one class per line in output order")
    ap.add_argument("image", help="a photo (jpg/png)")
    args = ap.parse_args()

    labels = [l.strip() for l in open(args.labels) if l.strip()]
    try:
        method = Runtime.get().load_program(args.model).load_method("forward")
    except RuntimeError as e:
        sys.exit(f"could not load {args.model}: {e}\n(A model_qnn.pte only runs on the Qualcomm "
                 "device it was compiled for; use the CPU model.pte here.)")
    shape = tuple(method.metadata.input_tensor_meta(0).sizes())       # e.g. (1, 3, 96, 96)
    if len(shape) != 4 or shape[1] != 3:
        sys.exit(f"expected an image model with input [1, 3, H, W], got {list(shape)}")

    x = torch.from_numpy(preprocess(args.image, (shape[2], shape[3])))
    method.execute([x])                                                # warm-up
    t0 = time.perf_counter()
    out = method.execute([x])[0].detach().numpy().reshape(-1).astype(np.float64)
    ms = (time.perf_counter() - t0) * 1000

    is_probs = np.all(out >= -1e-4) and abs(out.sum() - 1.0) < 1e-2
    p = out if is_probs else np.exp(out - out.max()) / np.exp(out - out.max()).sum()
    if len(labels) != len(p):
        sys.exit(f"labels.txt has {len(labels)} classes but the model outputs {len(p)}")

    print(f"input {list(shape)}   one inference: {ms:.2f} ms")
    for i in np.argsort(-p):
        print(f"  {labels[i]:<12} {p[i]:.3f}")


if __name__ == "__main__":
    main()
