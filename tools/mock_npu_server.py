#!/usr/bin/env python3
"""A stand-in for board/npu_server that runs a CPU .pte instead of the NPU.

It speaks the exact same UNIX-socket protocol, so the demo app's NPU path
(npu_client.py -> HybridClassifier -> per-frame CPU fallback) can be exercised
on a laptop with no Qualcomm hardware:

    python tools/mock_npu_server.py --model demo/rps-amazing-hand/models/model.pte \
        --socket /tmp/rps-npu.sock
    # in another shell
    cd demo/rps-amazing-hand && RPS_NPU_SOCKET=/tmp/rps-npu.sock ./start.sh

The app will log "NPU ACTIVE"; stop this script and it logs "CPU FALLBACK" and
keeps playing. NOTHING here touches an NPU -- it is a protocol double.

Protocol (little-endian; see board/npu_server/npu_server.cpp):
    request : u32 n_bytes + float32 NCHW   (n_bytes == 0xFFFFFFFF -> INFO)
    response: u32 status, u32 n_floats, u32 infer_us, then float32 logits
    INFO    : u32 json_len + JSON {"backend", "input_shape", "num_outputs", ...}
The real NPU model returns logits; the CPU .pte has softmax baked in, so this
mock returns log(probs), which the client's softmax maps back to probs.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import time

import numpy as np

INFO = 0xFFFFFFFF


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return bytes(buf)


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU mock of the rps-npu service")
    ap.add_argument("--model", required=True, help="a CPU (XNNPACK) .pte")
    ap.add_argument("--socket", default="/tmp/rps-npu.sock")
    a = ap.parse_args()

    import torch
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(a.model).load_method("forward")
    shape = list(method.metadata.input_tensor_meta(0).sizes())
    numel = int(np.prod(shape))
    n_out = int(np.prod(method.metadata.output_tensor_meta(0).sizes()))
    info = json.dumps({"backend": "mock-cpu", "model": a.model, "input_dtype": "float32",
                       "input_layout": "NCHW", "input_shape": shape, "num_outputs": n_out}).encode()

    # Clean up the socket on SIGTERM too (like the real npu_server), so a
    # restarted server is not mistaken for a live one by a stale socket file.
    def _on_sigterm(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(4)
    print(f"mock npu_server (CPU!) on {a.socket}: input {shape}, {n_out} outputs", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                try:
                    while True:
                        (n,) = struct.unpack("<I", recv_exact(conn, 4))
                        if n == INFO:
                            conn.sendall(struct.pack("<I", len(info)) + info)
                            continue
                        data = recv_exact(conn, n)
                        if n != numel * 4:
                            conn.sendall(struct.pack("<III", 1, 0, 0))
                            continue
                        x = torch.from_numpy(np.frombuffer(data, np.float32).reshape(shape).copy())
                        t0 = time.perf_counter()
                        probs = np.asarray(method.execute([x])[0], dtype=np.float32).reshape(-1)
                        us = int((time.perf_counter() - t0) * 1e6)
                        logits = np.log(np.clip(probs, 1e-9, 1.0)).astype(np.float32)
                        conn.sendall(struct.pack("<III", 0, logits.size, us) + logits.tobytes())
                except ConnectionError:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        if os.path.exists(a.socket):
            os.unlink(a.socket)


if __name__ == "__main__":
    main()
