"""Client for the npu_server sidecar (QNN / Hexagon NPU inference).

    from npu_client import NpuClient
    npu = NpuClient("/run/rps-npu/npu.sock")
    if npu.connect():
        logits, infer_us = npu.infer(chw_float32)      # (C,H,W) or (1,C,H,W), values 0..1
        probs = softmax(logits)

Design: never raises into the caller's frame loop. connect()/infer() return
False/None on any failure; the caller falls back to the CPU model and may call
connect() again later (see retry_after_s).
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import time
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("npu_client")
_INFO = 0xFFFFFFFF


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


class NpuClient:
    def __init__(self, sock_path: str = "/run/rps-npu/npu.sock",
                 timeout_s: float = 1.0, retry_after_s: float = 5.0):
        self.sock_path = sock_path
        self.timeout_s = timeout_s
        self.retry_after_s = retry_after_s
        self._sock: Optional[socket.socket] = None
        self._last_fail = 0.0
        self.info: dict = {}

    # -- connection ---------------------------------------------------------
    def connect(self) -> bool:
        """(Re)connect and fetch model info. False if the server is unavailable."""
        self.close()
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout_s)
            s.connect(self.sock_path)
            self._sock = s
            self.info = self._get_info()
            self._in_numel = int(np.prod(self.info["input_shape"]))
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail(f"connect: {exc}")
            return False

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _fail(self, why: str) -> None:
        if time.monotonic() - self._last_fail > 30:  # don't spam logs
            log.warning("NPU server unavailable (%s)", why)
        self._last_fail = time.monotonic()
        self.close()

    def maybe_reconnect(self) -> bool:
        """Rate-limited reconnect; call from the frame loop when not connected."""
        if self.connected:
            return True
        if time.monotonic() - self._last_fail < self.retry_after_s:
            return False
        return self.connect()

    # -- io -----------------------------------------------------------------
    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("server closed the connection")
            buf += chunk
        return bytes(buf)

    def _get_info(self) -> dict:
        self._sock.sendall(struct.pack("<I", _INFO))
        (n,) = struct.unpack("<I", self._recv_exact(4))
        return json.loads(self._recv_exact(n).decode())

    def infer(self, x: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
        """Return (logits float32, infer_us) or None on failure (caller falls back)."""
        if not self.connected:
            return None
        try:
            a = np.ascontiguousarray(x, dtype=np.float32)
            if a.size != self._in_numel:
                log.error("NPU input has %d values, model wants %d (%s)",
                          a.size, self._in_numel, self.info.get("input_shape"))
                return None
            data = a.tobytes()
            self._sock.sendall(struct.pack("<I", len(data)) + data)
            status, n, us = struct.unpack("<III", self._recv_exact(12))
            if status != 0:
                log.error("NPU server returned status %d", status)
                return None
            out = np.frombuffer(self._recv_exact(4 * n), dtype=np.float32).copy()
            return out, us
        except Exception as exc:  # noqa: BLE001
            self._fail(f"infer: {exc}")
            return None
