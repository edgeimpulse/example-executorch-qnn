#!/usr/bin/env python3
"""
reference_scores.py - the "Python side" of the parity check.

Runs one photo through the .pte with the ExecuTorch Python runtime and exactly the
preprocessing the IQ9 uses, and prints the scores in the same format as the app's
"Self-test" button. Run the same photo through both and compare:

    pip install executorch opencv-python numpy
    python3 tools/reference_scores.py --model model.pte --labels labels.txt --image rock.jpg

If the numbers agree to ~2 decimals, the Android preprocessing is faithful. If they do
not, the difference is the preprocessing (or labels.txt order), not the model.

The photo is read WITHOUT applying its EXIF rotation, like Android's BitmapFactory does,
so both sides see identical pixels. Use a photo that is already upright.
"""

import argparse
import sys
import time

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime

from reference_preprocess import cv2_reference

CANDIDATE_SIZES = (96, 160, 128, 224, 192, 320, 64, 48)   # same list as rps.py / the app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pte")
    ap.add_argument("--labels", default="labels.txt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--roi", type=int, default=100,
                    help="percent of the shorter side; 100 = what `rps.py --image` and the app's Self-test use")
    args = ap.parse_args()

    labels = [l.strip() for l in open(args.labels) if l.strip()]
    method = Runtime.get().load_program(args.model).load_method("forward")

    size = None
    for s in CANDIDATE_SIZES:
        try:
            method.execute([torch.rand(1, 3, s, s)])
            size = s
            break
        except Exception:
            continue
    if size is None:
        sys.exit("model rejected every candidate input size")

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        sys.exit(f"cannot read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    h, w = rgb.shape[:2]

    x = cv2_reference(rgba, 0, args.roi, size)[None]          # (1, 3, size, size)
    t0 = time.perf_counter()
    out = method.execute([torch.from_numpy(np.ascontiguousarray(x))])
    ms = (time.perf_counter() - t0) * 1000
    p = out[0].detach().numpy().reshape(-1).astype(np.float64)
    if not (np.all(p >= -1e-4) and np.all(p <= 1.0001) and 0.98 <= p.sum() <= 1.02):
        e = np.exp(p - p.max())
        p = e / e.sum()                                          # logits -> softmax, like the app

    print(f"{w}x{h} -> {size}x{size}")
    print("   ".join(f"{labels[i].lower()} {p[i]:.3f}" for i in range(len(labels))))
    print(f"{ms:.2f} ms")


if __name__ == "__main__":
    main()
