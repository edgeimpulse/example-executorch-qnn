# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Camera capture + gesture detection."""

import asyncio
import threading
import time
from dataclasses import replace
from typing import AsyncIterator, Protocol

import cv2

import config
from data_model import Detection, Gesture, InferenceResult


class CameraSource:
    def __init__(
        self,
        device: str | None = None,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
        stream_size: int = config.CAMERA_STREAM_SIZE,
        mirror: bool = config.MIRROR_VIDEO,
    ) -> None:
        self._device = device or config.CAMERA_DEVICE
        self._width = width
        self._height = height
        self._fps = fps
        self._stream_size = stream_size
        self._mirror = mirror

        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        self.latest_frame = None
        # Resized square BGR frame (no boxes); the compositor overlays and encodes it.
        self.latest_display = None
        self.square_size: int = min(width, height)

    def start(self) -> None:
        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {self._device}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"camera: {self.describe()}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _loop(self) -> None:
        assert self._capture is not None
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                continue

            # Flip before anything reads the frame so the live feed, inference,
            # boxes, and capture all share one mirrored coordinate space.
            if self._mirror:
                frame = cv2.flip(frame, 1)

            self.latest_frame = frame

            square = _center_crop_square(frame)
            self.square_size = square.shape[0]
            display = square
            if self._stream_size != square.shape[0]:
                display = cv2.resize(
                    square, (self._stream_size, self._stream_size), interpolation=cv2.INTER_AREA
                )
            self.latest_display = display

    def describe(self) -> str:
        if self._capture is None:
            return "not started"
        cap = self._capture
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        return (
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"@{cap.get(cv2.CAP_PROP_FPS):.0f}fps {fourcc_str}"
        )


def _center_crop_square(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


class GestureDetector(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def detections(self) -> AsyncIterator[Detection]: ...


class FakeGestureDetector:
    """Replays a scripted gesture list. Wire this into main.py in place of the
    real detector to exercise the web UI, game engine and scoring with no model
    and no camera.

    main.py reaches past the GestureDetector protocol for `latest_result` and
    `result_square()`, so both are implemented here too -- without them the
    box broadcaster and /captured.jpg raise AttributeError on the first frame.
    """

    def __init__(
        self,
        script: list[Gesture | None],
        interval: float = 0.05,
        camera: CameraSource | None = None,
    ) -> None:
        self._script = script
        self._interval = interval
        self._camera = camera
        self._running = False
        self._paused = False
        self.latest_result: InferenceResult | None = None

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def new_result(self, since: float | None) -> InferenceResult | None:
        result = self.latest_result
        if result is not None and result.timestamp != since:
            return result
        return None

    def result_square(self):
        """Latest camera square, or None when running without a camera -- which
        makes main.py's _snapshot() fall back to the live display frame."""
        if self._camera is None or self._camera.latest_frame is None:
            return None
        return _center_crop_square(self._camera.latest_frame)

    async def detections(self) -> AsyncIterator[Detection]:
        while self._running:
            for gesture in self._script:
                if not self._running:
                    return
                if gesture is not None:
                    det = Detection(gesture=gesture, confidence=0.95, box=(0, 0, 100, 100))
                    self.latest_result = InferenceResult(
                        detections=[det], timestamp=time.time()
                    )
                    yield det
                await asyncio.sleep(self._interval)


class EIGestureDetector:
    def __init__(self, camera: CameraSource, model_path: str = config.MODEL_PATH) -> None:
        self._camera = camera
        self._model_path = model_path
        self._runner = None
        self._thread: threading.Thread | None = None
        self._running = False
        # Skips inference (e.g. so it doesn't contend with the LLM on the NPU).
        self._paused = False

        self.latest_result: InferenceResult | None = None
        # The exact frame latest_result was computed from, so a captured
        # detection's box and freeze-frame image come from the same frame.
        self.latest_result_frame = None
        # Per-gesture EMA box state (in display coords) that smooths jitter.
        self._smoothed_boxes: dict[str, tuple[float, float, float, float]] = {}

    async def start(self) -> None:
        from edge_impulse_linux.image import ImageImpulseRunner

        self._runner = ImageImpulseRunner(self._model_path)
        info = self._runner.init()
        params = info["model_parameters"]
        print(
            f"detector: {info['project']['name']} "
            f"input={params['image_input_width']}x{params['image_input_height']} "
            f"labels={params['labels']}"
        )

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._runner is not None:
            self._runner.stop()
            self._runner = None

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        # Drop stale EMA state so boxes don't glide from a pre-pause position.
        self._smoothed_boxes.clear()

    def new_result(self, since: float | None) -> InferenceResult | None:
        """Latest inference result if newer than `since`, else None. Single
        source of truth for both the engine's detections() stream and
        main.py's idle box broadcaster."""
        result = self.latest_result
        if result is not None and result.timestamp != since:
            return result
        return None

    async def detections(self) -> AsyncIterator[Detection]:
        # Start from the current result's timestamp so the first thing yielded
        # is a genuinely new inference, not the one already sitting from before
        # (e.g. the mid-throw "Shoot!" frame when capture begins).
        last_timestamp = self.latest_result.timestamp if self.latest_result is not None else None
        while self._running:
            result = self.new_result(last_timestamp)
            if result is not None:
                last_timestamp = result.timestamp
                best = max(result.detections, key=lambda d: d.confidence, default=None)
                if best is not None:
                    yield best
            await asyncio.sleep(0.01)

    def _loop(self) -> None:
        assert self._runner is not None
        while self._running:
            if self._paused:
                time.sleep(0.02)
                continue

            frame = self._camera.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                features, cropped = self._runner.get_features_from_image(rgb)
                result = self._runner.classify(features)
            except (OSError, IndexError):
                # The runner's subprocess dies mid-call when it gets SIGINT
                # directly (e.g. Ctrl-C), before stop() clears self._running.
                break

            cropped_size = cropped.shape[0]
            self.latest_result_frame = frame
            parsed = _parse_result(
                result,
                cropped_size=cropped_size,
                display_size=self._camera.square_size,
                timestamp=time.time(),
            )
            self.latest_result = replace(
                parsed, detections=self._smooth_boxes(parsed.detections)
            )

    def _smooth_boxes(self, detections: list[Detection]) -> list[Detection]:
        alpha = config.BOX_SMOOTHING_ALPHA
        seen: set[str] = set()
        smoothed: list[Detection] = []
        for det in detections:
            key = det.gesture.value
            seen.add(key)
            prev = self._smoothed_boxes.get(key)
            if prev is None:
                cur = tuple(float(v) for v in det.box)
            else:
                cur = tuple(alpha * m + (1.0 - alpha) * p for m, p in zip(det.box, prev))
            self._smoothed_boxes[key] = cur
            smoothed.append(replace(det, box=tuple(round(v) for v in cur)))
        # Forget gestures that vanished so a re-entry starts fresh, not lerped.
        for key in tuple(self._smoothed_boxes):
            if key not in seen:
                del self._smoothed_boxes[key]
        return smoothed

    def result_square(self):
        """Center-cropped square of the frame the latest inference ran on -- the
        frame a captured detection's box was computed against."""
        frame = self.latest_result_frame
        if frame is None:
            return None
        return _center_crop_square(frame)


def _parse_result(result, cropped_size: int, display_size: int, timestamp: float) -> InferenceResult:
    scale = display_size / cropped_size
    detections = []
    for bb in result.get("result", {}).get("bounding_boxes", []):
        try:
            gesture = Gesture(bb["label"].lower())
        except ValueError:
            continue
        box = (
            round(bb["x"] * scale),
            round(bb["y"] * scale),
            round(bb["width"] * scale),
            round(bb["height"] * scale),
        )
        detections.append(Detection(gesture=gesture, confidence=bb["value"], box=box))
    return InferenceResult(detections=detections, timestamp=timestamp)
