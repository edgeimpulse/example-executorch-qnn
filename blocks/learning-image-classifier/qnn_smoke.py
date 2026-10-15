# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Build-time smoke test: lower a tiny conv net to QNN HTP.

Run from the Dockerfile so a broken QNN setup (missing libs, wrong
LD_LIBRARY_PATH, SDK/libc++ mismatch) fails the image build, not a training run.
"""
import sys

import numpy as np
import torch
from torch import nn

from qnn_export import export_qnn

net = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(8, 4),
).eval()
x = torch.rand(1, 3, 32, 32)
calib = np.random.rand(8, 3, 32, 32).astype(np.float32)
report = export_qnn(net, x, calib, "/tmp/smoke_qnn.pte")
print(report)
if report["status"] != "ok":
    sys.exit(f"QNN smoke test failed: {report['reason']}")
print("QNN smoke test OK")
