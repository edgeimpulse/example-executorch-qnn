# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""ExecuTorch gesture detection.

Drop-in replacement for EIGestureDetector that runs a .pte exported from an
Edge Impulse PyTorch learning block instead of an Edge Impulse .eim model.

Two differences from the .eim path that shape this module:

1. The .pte is an image *classifier*, not an object detector, so there are no
   bounding boxes. Every detection is reported against a fixed centred ROI
   (config.ROI_FRACTION) which doubles as an on-screen target zone -- put your
   hand in the box. Because the box never moves, the EMA smoothing the .eim
   detector needs is unnecessary here.

2. The model's `unknown` class has no Gesture member, so a frame classified as
   unknown produces NO detection. That is deliberate: it is what lets the
   engine's stable-frame counter stall and resolve the round as "no gesture
   captured" instead of guessing.

Preprocessing must match what Edge Impulse did at training time. The input
resolution and class count are probed from the model at startup, so retraining
at a different resolution needs no code change here, but NORMALIZE and the
centre-crop below have to agree with the impulse's image block.
"""

import asyncio
import collections
import logging
import os
import threading
import time
from typing import AsyncIterator

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime

import config
from npu_inference import HybridClassifier
from data_model import Detection, Gesture, InferenceResult
from perception import _center_crop_square

# npu_inference logs "NPU ACTIVE" / "CPU FALLBACK" on every backend change; make
# sure those reach the console even though the app doesn't configure logging.
_hybrid_log = logging.getLogger("hybrid")
if not _hybrid_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("detector: %(message)s"))
    _hybrid_log.addHandler(_h)
    _hybrid_log.setLevel(logging.INFO)
    _hybrid_log.propagate = False

# Input sizes tried when probing the model, most likely first.
_CANDIDATE_SIZES = (96, 160, 128, 224, 192, 320, 64, 48)


class PteGestureDetector:
    def __init__(self, camera, model_path: str = config.MODEL_PATH) -> None:
        self._camera = camera
        self._model_path = model_path
        self._labels = _load_labels(config.LABELS_PATH)
        self._program = None
        self._method = None
        self._thread: threading.Thread | None = None
        self._running = False
        # Skips inference (e.g. so it doesn't contend with the LLM on the NPU).
        self._paused = False
        self._classifier: HybridClassifier | None = None
        self._current_frame = None
        self._recent: collections.deque = collections.deque(
            maxlen=max(1, config.SMOOTH_FRAMES)
        )
        self._last_log = 0.0
        self._last_save = 0.0
        self._saved = 0

        self.input_size: int | None = None
        self.latest_result: InferenceResult | None = None
        # The exact frame latest_result was computed from, so a captured
        # detection's box and freeze-frame image come from the same frame.
        self.latest_result_frame = None

    async def start(self) -> None:
        runtime = Runtime.get()
        self._program = runtime.load_program(self._model_path)
        self._method = self._program.load_method("forward")

        self.input_size, num_classes = self._probe()
        if num_classes != len(self._labels):
            raise RuntimeError(
                f"{self._model_path} has {num_classes} output classes but "
                f"{config.LABELS_PATH} lists {len(self._labels)} "
                f"({self._labels}). Fix the labels file before running."
            )
        print(
            f"detector: ExecuTorch {self._model_path} "
            f"input={self.input_size}x{self.input_size} labels={self._labels}"
        )

        # NPU first, CPU .pte as the per-frame fallback. Needs to agree with the
        # model loaded above on input shape and class count, or we'd feed the
        # NPU a frame it rejects (or label the wrong class) on every frame.
        self._classifier = HybridClassifier(
            cpu_predict=self._cpu_probs,
            sock_path=config.NPU_SOCKET,
        )
        self._classifier.enabled = self._classifier.enabled and config.NPU_ENABLE
        self._check_npu_matches_cpu_model(num_classes)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._method = None
        self._program = None
        if self._classifier is not None:
            self._classifier.client.close()
            self._classifier = None

    def status(self) -> dict:
        """Backend in use and frame counts, for logs or a UI badge."""
        return self._classifier.status() if self._classifier else {"backend": "none"}

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._recent.clear()  # don't average in frames from before the pause
        self._paused = False

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

    def result_square(self):
        """Center-cropped square of the frame the latest inference ran on -- the
        frame a captured detection's box was computed against."""
        frame = self.latest_result_frame
        if frame is None:
            return None
        return _center_crop_square(frame)

    def roi_box(self, square_size: int) -> tuple[int, int, int, int]:
        """The fixed target zone, in square_size coordinates (what overlay.py
        and Detection.box expect)."""
        side = int(round(square_size * config.ROI_FRACTION))
        offset = (square_size - side) // 2
        return (offset, offset, side, side)

    # --- Inference ---------------------------------------------------------

    def _probe(self) -> tuple[int, int]:
        """Find the input resolution and class count the model actually wants."""
        forced = getattr(config, "MODEL_INPUT_SIZE", None)
        sizes = (forced,) if forced else _CANDIDATE_SIZES
        for size in sizes:
            try:
                out = self._method.execute([torch.rand(1, 3, size, size)])
            except Exception:
                continue
            return size, int(np.asarray(out[0].detach()).reshape(-1).size)
        raise RuntimeError(
            f"Could not determine the input size of {self._model_path}. Tried "
            f"{', '.join(str(s) for s in sizes)}. Set MODEL_INPUT_SIZE in config.py."
        )

    def _cpu_probs(self, _npu_chw: np.ndarray) -> np.ndarray:
        """CPU fallback. Ignores the NPU-sized tensor and re-preprocesses the
        current frame at the CPU model's own size. ClassifierExport bakes
        softmax into the CPU .pte, so its output is already probabilities (the
        NPU .pte emits logits and is softmaxed in HybridClassifier)."""
        out = self._method.execute([self._preprocess(self._current_frame)])
        return np.asarray(out[0].detach()).reshape(-1)

    def _check_npu_matches_cpu_model(self, num_classes: int) -> None:
        """The two models may be different exports (different input sizes are
        fine: each backend gets a frame preprocessed to its own size). What
        must agree is the class count, since both are labelled by labels.txt."""
        clf = self._classifier
        if not (clf.enabled and clf.client.connected):
            return
        shape = clf.client.info.get("input_shape")
        n_out = clf.client.info.get("num_outputs")
        print(f"detector: NPU model input={shape}, CPU model input="
              f"{self.input_size}x{self.input_size}", flush=True)
        if n_out is not None and int(n_out) != num_classes:
            print(f"detector: NPU model has {n_out} outputs but labels.txt lists "
                  f"{num_classes}; using CPU only.", flush=True)
            clf.enabled = False

    def _npu_input_size(self) -> int | None:
        clf = self._classifier
        if not clf.enabled:
            return None
        if not clf.client.connected:
            clf.client.maybe_reconnect()
        shape = clf.client.info.get("input_shape") if clf.client.connected else None
        return int(shape[2]) if shape and len(shape) == 4 else None

    def _preprocess(self, frame_bgr, size: int | None = None) -> torch.Tensor:
        size = size or self.input_size
        # Centre-crop to a square so the resize doesn't distort the hand. If the
        # impulse used "squash" instead of a crop, resize frame_bgr directly.
        square = _center_crop_square(frame_bgr)
        resized = cv2.resize(square, (size, size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        arr = rgb.astype(np.float32) / config.PIXEL_SCALE   # HWC, 0..1
        arr = np.transpose(arr, (2, 0, 1))                  # CHW
        arr = np.expand_dims(arr, 0)                        # NCHW
        # XNNPACK-lowered programs reject channels-last placeholders.
        return torch.from_numpy(np.ascontiguousarray(arr))

    def _probs(self, frame_bgr) -> np.ndarray:
        """Class probabilities for one frame, from whichever backend answered
        (NPU service, or the CPU .pte as the per-frame fallback)."""
        self._current_frame = frame_bgr
        chw = self._preprocess(frame_bgr, self._npu_input_size()).numpy()
        probs, _backend = self._classifier.predict(chw)
        return np.asarray(probs, dtype=np.float32).reshape(-1)

    def _decide(self, probs: np.ndarray) -> tuple[str, float, str | None]:
        """Smooth, then gate. Returns (label, confidence, reject_reason); a
        non-None reason means the frame must NOT produce a detection."""
        self._recent.append(probs)
        avg = np.mean(self._recent, axis=0)
        order = np.argsort(avg)[::-1]
        top, second = int(order[0]), int(order[1])
        label, conf = self._labels[top], float(avg[top])
        try:
            Gesture(label.lower())
        except ValueError:
            return label, conf, "unknown"
        if len(self._recent) < self._recent.maxlen:
            return label, conf, "warming up"
        if conf < config.MIN_CONFIDENCE:
            return label, conf, f"confidence {conf:.2f} < {config.MIN_CONFIDENCE}"
        margin = conf - float(avg[second])
        if margin < config.MIN_MARGIN:
            return label, conf, f"margin {margin:.2f} < {config.MIN_MARGIN}"
        for i, name in enumerate(self._labels):
            if name.lower() == "unknown" and avg[i] > config.MAX_UNKNOWN:
                return label, conf, f"unknown {avg[i]:.2f} > {config.MAX_UNKNOWN}"
        return label, conf, None

    def _log_and_save(self, frame, avg_note: str, label: str, conf: float,
                      reason: str | None) -> None:
        now = time.time()
        if config.LOG_PROBS and now - self._last_log >= 0.5:
            self._last_log = now
            avg = np.mean(self._recent, axis=0)
            table = " ".join(f"{n}={p:.2f}" for n, p in zip(self._labels, avg))
            backend = self._classifier.backend if self._classifier else "?"
            print(f"probs[{backend}] {table} -> "
                  f"{'ACCEPT ' + label if reason is None else 'reject (' + reason + ')'}",
                  flush=True)
        if (config.SAVE_FRAMES_DIR and conf < config.SAVE_BELOW
                and self._saved < config.SAVE_MAX
                and now - self._last_save >= config.SAVE_EVERY):
            self._last_save = now
            try:
                os.makedirs(config.SAVE_FRAMES_DIR, exist_ok=True)
                name = f"{int(now * 1000)}_{label}_{conf:.2f}.jpg"
                cv2.imwrite(os.path.join(config.SAVE_FRAMES_DIR, name),
                            _center_crop_square(frame))
                self._saved += 1
            except Exception as exc:  # noqa: BLE001 - never break detection
                print(f"detector: frame save failed ({exc})", flush=True)

    def _loop(self) -> None:
        # Only infer on frames we haven't seen. The camera thread assigns a new
        # array per capture, so identity is a reliable "is this new?" test.
        # Without this the loop re-infers the same frame: a .pte at ~1ms against
        # a 30fps camera burns a core doing ~98% redundant work, which the much
        # slower .eim path self-limited into never noticing.
        last_frame = None
        while self._running:
            if self._paused:
                time.sleep(0.02)
                continue

            frame = self._camera.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            if frame is last_frame:
                time.sleep(0.002)
                continue
            last_frame = frame

            try:
                probs = self._probs(frame)
                label, confidence, reject = self._decide(probs)
                self._log_and_save(frame, "", label, confidence, reject)
            except Exception as exc:  # noqa: BLE001 - keep the app alive
                print(f"detector: inference failed ({exc}); stopping loop.", flush=True)
                break

            detections: list[Detection] = []
            if reject is None:
                # Rejected frames (unknown, low confidence, narrow margin, high
                # `unknown`) yield nothing, so the engine's stable-frame counter
                # stalls and the round resolves as "no gesture captured".
                detections.append(
                    Detection(
                        gesture=Gesture(label.lower()),
                        confidence=confidence,
                        box=self.roi_box(self._camera.square_size),
                    )
                )

            self.latest_result_frame = frame
            self.latest_result = InferenceResult(
                detections=detections, timestamp=time.time()
            )


def _load_labels(path: str) -> list[str]:
    """One label per line, in the model's output channel order.

    This must match the class order Edge Impulse used to build the one-hot
    targets (alphabetical by class name), NOT the order you created them in.
    Getting it wrong gives a model that works perfectly and is wrong every time.
    """
    with open(path) as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise RuntimeError(f"{path} is empty.")
    return labels
