#!/usr/bin/env python3
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
reference_preprocess.py - the ground truth for the Android preprocessing.

Two implementations of the same thing:

  cv2_reference()   what rps.py / pte_perception.py do on the IQ9:
                    centre-crop -> cv2.resize (INTER_LINEAR) -> RGB -> /255 -> NCHW
  kotlin_mirror()   a line-for-line numpy mirror of core/.../Preprocess.kt
                    (no cv2, no Bitmap), used to prove the Kotlin algorithm
                    reproduces cv2 before it is ever compiled.

`make_golden.py` writes cv2_reference() outputs into core/src/test/resources so the
Kotlin unit test can assert the same thing on the JVM.

Frame convention (same as CameraX ImageProxy): RGBA bytes, plus a clockwise
`rotation` in degrees that has to be applied for the image to be upright.
"""

import numpy as np

try:
    import cv2
except ImportError:  # the mirror and golden check still work without cv2
    cv2 = None


def upright(rgb, rotation):
    """Apply CameraX's clockwise rotation so the image is upright."""
    k = {0: 0, 90: -1, 180: 2, 270: 1}[rotation]  # np.rot90 is counter-clockwise
    return np.rot90(rgb, k)


def crop_box(uw, uh, roi_percent):
    side = max(1, min(uw, uh) * roi_percent // 100)
    return (uw - side) // 2, (uh - side) // 2, side   # left, top, side


def cv2_reference(rgba, rotation, roi_percent, size):
    """What the IQ9 does. rgba: (H, W, 4) uint8. Returns (3, size, size) float32."""
    rgb = upright(rgba[..., :3], rotation)
    uh, uw = rgb.shape[:2]
    left, top, side = crop_box(uw, uh, roi_percent)
    square = np.ascontiguousarray(rgb[top:top + side, left:left + side])
    resized = cv2.resize(square, (size, size))            # default = INTER_LINEAR
    arr = resized.astype(np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def kotlin_mirror(rgba, rotation, roi_percent, size):
    """Same maths as Preprocess.kt, written the way the Kotlin is written."""
    rgb = upright(rgba[..., :3], rotation)
    uh, uw = rgb.shape[:2]
    left, top, side = crop_box(uw, uh, roi_percent)

    def axis(n):
        f = (np.arange(n) + 0.5) * side / n - 0.5
        i0 = np.floor(f).astype(int)
        w = f - i0
        return np.clip(i0, 0, side - 1), np.clip(i0 + 1, 0, side - 1), w

    x0, x1, wx = axis(size)
    y0, y1, wy = axis(size)
    crop = rgb[top:top + side, left:left + side].astype(np.float64)

    out = np.empty((3, size, size), np.float32)
    for c in range(3):
        p = crop[..., c]
        top_row = p[y0][:, x0] * (1 - wx) + p[y0][:, x1] * wx
        bot_row = p[y1][:, x0] * (1 - wx) + p[y1][:, x1] * wx
        v = top_row * (1 - wy)[:, None] + bot_row * wy[:, None]
        out[c] = np.clip(np.floor(v + 0.5), 0, 255).astype(np.int32) / np.float32(255.0)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst = 0.0
    for rot in (0, 90, 180, 270):
        for roi in (100, 60):
            for size in (96, 160):
                img = rng.integers(0, 256, (240, 320, 4), dtype=np.uint8)
                a = cv2_reference(img, rot, roi, size)
                b = kotlin_mirror(img, rot, roi, size)
                d = float(np.abs(a - b).max()) * 255
                worst = max(worst, d)
                print(f"rot={rot:3d} roi={roi:3d} size={size:3d}  max diff = {d:.2f} levels")
    print(f"worst: {worst:.2f} levels of 255 (cv2 uses fixed-point weights, so <=1 is expected)")
