#!/usr/bin/env python3
"""Write a fake Edge Impulse dataset so a learning block can be run locally
without a Studio account.

    # image classification (default): X (n, H, W, 3) NHWC, Y one-hot
    python tools/make_fake_data.py /tmp/ei-data --size 96 --classes 4
    # FOMO object detection: X (n, H, W, 3), Y = JSON text with bounding boxes
    python tools/make_fake_data.py /tmp/ei-fomo --size 96 --classes 2 --fomo
    # flat features (timeseries, KWS/audio): X (n, F), Y one-hot
    python tools/make_fake_data.py /tmp/ei-ts  --flat 375 --classes 4     # e.g. 3 axes x 125 samples
    python tools/make_fake_data.py /tmp/ei-kws --flat 637 --classes 3     # e.g. 49 frames x 13 MFCC

Produces what Studio hands a learning block:
    X_split_train.npy, X_split_test.npy   float32, values 0..1
    Y_split_train.npy, Y_split_test.npy   one-hot float32 (n, classes) -- or, with
                                          --fomo, a JSON *text* file (Studio's
                                          object-detection format) despite the name

Random data carries no signal, so accuracy is meaningless -- this exercises the
pipeline (training, ONNX, XNNPACK .pte, QNN .pte, export report), nothing more.
"""
import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=96, help="image height = width")
    ap.add_argument("--flat", type=int, default=0, help="flat feature length instead of images")
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--fomo", action="store_true", help="bounding-box labels (images only)")
    ap.add_argument("--train", type=int, default=64)
    ap.add_argument("--test", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.fomo and a.flat:
        ap.error("--fomo needs image data, not --flat")

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    for split, n in (("train", a.train), ("test", a.test)):
        shape = (n, a.flat) if a.flat else (n, a.size, a.size, 3)
        x = rng.random(shape, dtype=np.float32)
        np.save(os.path.join(a.out_dir, f"X_split_{split}.npy"), x)
        y_path = os.path.join(a.out_dir, f"Y_split_{split}.npy")
        if a.fomo:
            labels = []
            for _ in range(n):
                boxes = []
                for _ in range(int(rng.integers(1, 3))):
                    w = h = int(a.size // 6)
                    boxes.append({"label": int(rng.integers(1, a.classes + 1)),
                                  "x": int(rng.integers(0, a.size - w)),
                                  "y": int(rng.integers(0, a.size - h)), "w": w, "h": h})
                labels.append({"boundingBoxes": boxes})
            with open(y_path, "w") as f:
                json.dump(labels, f)
        else:
            y = np.eye(a.classes, dtype=np.float32)[rng.integers(0, a.classes, n)]
            np.save(y_path, y)
    kind = f"flat {a.flat}" if a.flat else f"{a.size}x{a.size}x3"
    print(f"Wrote {a.train}+{a.test} samples ({kind}, {a.classes} classes"
          f"{', FOMO boxes' if a.fomo else ''}) to {a.out_dir}")


if __name__ == "__main__":
    main()
