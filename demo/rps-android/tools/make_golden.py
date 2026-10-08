#!/usr/bin/env python3
"""
make_golden.py - write the golden vectors the Kotlin unit tests check against.

    python3 tools/make_golden.py

For each case it writes, into core/src/test/resources/golden/:
    <name>.rgba    raw RGBA bytes of the camera frame (as CameraX hands it over)
    <name>.nchw    expected float32 little-endian output of cv2_reference()
    index.txt      one line per case:  name width height rotation roiPercent size

It also checks the rotation index formulas used by Preprocess.kt (the `when (rotation)`
block) against numpy's rot90, so a wrong rotation shows up here and not on the phone.
"""

import pathlib
import numpy as np
from reference_preprocess import cv2_reference, kotlin_mirror, upright

OUT = pathlib.Path(__file__).resolve().parent.parent / "core/src/test/resources/golden"

# (name, width, height, rotation, roi_percent, size)
CASES = [
    ("landscape_rot0_roi100_s96",   320, 240,   0, 100,  96),   # what rps.py does
    ("landscape_rot0_roi60_s160",   320, 240,   0,  60, 160),
    ("portrait_rot90_roi60_s96",    320, 240,  90,  60,  96),   # back camera, portrait
    ("portrait_rot270_roi60_s96",   320, 240, 270,  60,  96),   # front camera, portrait
    ("upside_rot180_roi100_s96",    320, 240, 180, 100,  96),
    ("full_640x480_rot270_roi60_s160", 640, 480, 270, 60, 160), # real analysis size
]


def synthetic_frame(w, h, seed):
    """Smooth gradients plus a bright blob plus noise: not a random-noise worst case."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.empty((h, w, 4), np.uint8)
    img[..., 0] = (255 * x / w).astype(np.uint8)
    img[..., 1] = (255 * y / h).astype(np.uint8)
    img[..., 2] = (127 + 120 * np.sin(x / 17.0) * np.cos(y / 23.0)).astype(np.uint8)
    blob = ((x - w * 0.6) ** 2 + (y - h * 0.4) ** 2) < (min(w, h) * 0.2) ** 2
    img[blob, :3] = (240, 200, 40)
    noise = rng.integers(-12, 13, (h, w, 3))
    img[..., :3] = np.clip(img[..., :3].astype(int) + noise, 0, 255).astype(np.uint8)
    img[..., 3] = 255
    return img


def src_index(rotation, ux, uy, sw, sh):
    """Mirror of Preprocess.kt: upright (ux, uy) -> source (sx, sy)."""
    if rotation == 0:
        return ux, uy
    if rotation == 90:
        return uy, sh - 1 - ux
    if rotation == 180:
        return sw - 1 - ux, sh - 1 - uy
    if rotation == 270:
        return sw - 1 - uy, ux
    raise ValueError(rotation)


def check_rotation_formulas():
    rng = np.random.default_rng(1)
    for rot in (0, 90, 180, 270):
        src = rng.integers(0, 256, (6, 10, 3), dtype=np.uint8)   # h=6, w=10
        sh, sw = src.shape[:2]
        up = upright(src, rot)
        uh, uw = up.shape[:2]
        for uy in range(uh):
            for ux in range(uw):
                sx, sy = src_index(rot, ux, uy, sw, sh)
                assert (src[sy, sx] == up[uy, ux]).all(), (rot, ux, uy)
    print("rotation index formulas match np.rot90 for 0/90/180/270")


def main():
    check_rotation_formulas()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (name, w, h, rot, roi, size) in enumerate(CASES):
        frame = synthetic_frame(w, h, seed=100 + i)
        expected = cv2_reference(frame, rot, roi, size)
        mirror = kotlin_mirror(frame, rot, roi, size)
        diff = float(np.abs(expected - mirror).max()) * 255
        assert diff <= 1.01, (name, diff)   # 1 level of 255, plus float32 rounding
        (OUT / f"{name}.rgba").write_bytes(frame.tobytes())
        (OUT / f"{name}.nchw").write_bytes(expected.astype("<f4").tobytes())
        lines.append(f"{name} {w} {h} {rot} {roi} {size}")
        print(f"{name:34s} ok  (mirror vs cv2: {diff:.2f} levels)")
    (OUT / "index.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
