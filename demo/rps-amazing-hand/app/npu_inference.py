"""NPU-first inference with automatic CPU fallback, for the RPS app.

    from npu_inference import HybridClassifier
    clf = HybridClassifier(cpu_predict=my_cpu_probs_fn)      # cpu_predict(chw)->probs
    probs, backend = clf.predict(chw_float32)                 # backend: "npu" | "cpu"
    clf.status()                                              # dict for logs / a UI badge

Environment:
    RPS_NPU_ENABLE=0          disable the NPU path entirely (default 1)
    RPS_NPU_SOCKET=/run/rps-npu/npu.sock

Behaviour: tries the NPU service each frame; on any failure it answers from the CPU
model for that frame (no gap), and retries the NPU in the background every few seconds.
Logs one line on every backend change: "NPU ACTIVE" / "CPU FALLBACK".
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Tuple

import numpy as np

from npu_client import NpuClient, softmax

log = logging.getLogger("hybrid")


class HybridClassifier:
    def __init__(self, cpu_predict: Callable[[np.ndarray], np.ndarray],
                 sock_path: str | None = None):
        self.cpu_predict = cpu_predict
        self.enabled = os.environ.get("RPS_NPU_ENABLE", "1") not in ("0", "false", "no")
        self.client = NpuClient(sock_path or os.environ.get("RPS_NPU_SOCKET", "/run/rps-npu/npu.sock"))
        self.backend = "cpu"
        self.npu_frames = self.cpu_frames = 0
        self.last_npu_us = 0
        if self.enabled:
            self.client.connect()
        self._announce()

    def _announce(self) -> None:
        now = "npu" if (self.enabled and self.client.connected) else "cpu"
        if now != self.backend or self.npu_frames + self.cpu_frames == 0:
            self.backend = now
            log.info("NPU ACTIVE (%s)" % self.client.info.get("input_shape")
                     if now == "npu" else "CPU FALLBACK")

    def predict(self, chw: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.enabled:
            if not self.client.connected:
                self.client.maybe_reconnect()
            r = self.client.infer(chw)
            if r is not None:
                logits, us = r
                self.last_npu_us = us
                self.npu_frames += 1
                self._announce()
                return softmax(logits), "npu"
        self.cpu_frames += 1
        probs = self.cpu_predict(chw)
        self._announce()
        return probs, "cpu"

    def status(self) -> dict:
        return {"backend": self.backend, "npu_frames": self.npu_frames,
                "cpu_frames": self.cpu_frames, "last_npu_us": self.last_npu_us,
                "npu_enabled": self.enabled, "npu_connected": self.client.connected}
