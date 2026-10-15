# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Server-side detection overlay: draws boxes onto a frame and JPEG-encodes it.

Boxes and labels mirror the former client canvas (per-gesture color, "gesture NN%"
label) so the composited stream and freeze-frame look identical to the old UI.
"""

import cv2

import config
from data_model import Detection

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame, detections: list[Detection], *, scale: float = 1.0) -> None:
    """Draw each detection's box + label onto `frame` in place.

    `scale` maps detection box coordinates (in square_size units) onto the
    resolution of `frame`, e.g. stream_size / square_size for the live stream.
    Line width and font scale with it too so the overlay looks the same on
    screen whether drawn on the smaller stream frame or the full capture square.
    """
    thickness = max(1, round(config.OVERLAY_BOX_THICKNESS * scale))
    font_scale = config.OVERLAY_FONT_SCALE * scale
    for det in detections:
        color = config.GESTURE_COLORS.get(det.gesture.value, (255, 255, 255))
        x, y, w, h = (round(v * scale) for v in det.box)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        label = f"{det.gesture.value} {det.confidence * 100:.0f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, 1)
        top = max(0, y - text_h - baseline - 4)
        cv2.rectangle(frame, (x, top), (x + text_w + 8, top + text_h + baseline + 4), color, -1)
        cv2.putText(
            frame, label, (x + 4, top + text_h + 2), _FONT, font_scale, (0, 0, 0), 1, cv2.LINE_AA
        )


def render_jpeg(
    frame, detections: list[Detection], *, scale: float = 1.0, quality: int | None = None
) -> bytes | None:
    """Return a JPEG of `frame` with detection boxes drawn on a copy of it."""
    annotated = frame.copy()
    draw_detections(annotated, detections, scale=scale)
    q = config.CAMERA_JPEG_QUALITY if quality is None else quality
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None
