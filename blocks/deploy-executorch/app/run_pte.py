# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Standalone ExecuTorch runtime harness.

Loads a `.pte` program and runs a single forward pass. Use this to smoke-test the
deployment package on a Linux/ARM target that has the ExecuTorch Python runtime
installed.

Usage:
    python app/run_pte.py model.pte --labels labels.txt
"""

from __future__ import annotations

import argparse

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ExecuTorch .pte program")
    parser.add_argument("model", help="Path to the .pte program")
    parser.add_argument("--labels", default=None, help="Optional labels.txt file")
    parser.add_argument("--shape", default=None,
                        help="Comma-separated input shape (NCHW); default: read from the program")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    method = program.load_method("forward")

    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))
    else:
        shape = tuple(method.metadata.input_tensor_meta(0).sizes())
    example = torch.rand(*shape)   # random image-like input in 0..1
    outputs = method.execute([example])

    out = outputs[0].squeeze(0)
    # model.pte from this block has softmax built in (outputs sum to 1);
    # model_qnn.pte outputs logits. Only apply softmax when it is needed.
    is_probs = bool((out >= 0).all()) and abs(float(out.sum()) - 1.0) < 1e-3
    probs = out if is_probs else torch.softmax(out, dim=-1)
    top = int(torch.argmax(probs))

    labels = None
    if args.labels:
        with open(args.labels) as f:
            labels = [line.strip() for line in f if line.strip()]

    label = labels[top] if labels and top < len(labels) else str(top)
    print(f"Predicted class: {label} (index {top}, p={probs[top]:.4f})")


if __name__ == "__main__":
    main()
