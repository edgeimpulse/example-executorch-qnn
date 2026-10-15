# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Offline tests for the learning block. No network, no Edge Impulse account,
no Qualcomm hardware.

    pip install -r requirements.txt pytest
    python -m pytest -q

Tests that need ExecuTorch are skipped when it is not installed. The QNN
lowering itself is exercised by qnn_smoke.py during `docker build` (it needs
the Qualcomm SDK, which the Dockerfile bakes in).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from model import ClassifierExport, build_model  # noqa: E402
from train import to_labels, to_nchw  # noqa: E402

try:
    import executorch  # noqa: F401
    HAVE_ET = True
except ImportError:
    HAVE_ET = False


# --- model ------------------------------------------------------------------

def test_smallcnn_forward_shape():
    model = build_model(in_channels=3, num_classes=10, pretrained=False)
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_smallcnn_grayscale():
    model = build_model(in_channels=1, num_classes=2, pretrained=False)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 2)


def test_classifier_export_softmax_flag():
    """CPU/ONNX export bakes softmax in; the QNN export keeps raw logits."""
    torch.manual_seed(0)
    base = build_model(in_channels=3, num_classes=4, pretrained=False).eval()
    x = torch.rand(1, 3, 32, 32)
    probs = ClassifierExport(base, softmax=True)(x)
    logits = ClassifierExport(base, softmax=False)(x)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(torch.softmax(logits, dim=1), probs, atol=1e-6)


# --- data handling ----------------------------------------------------------

def test_to_nchw_is_contiguous():
    nhwc = np.zeros((5, 28, 28, 3), dtype=np.float32)
    nchw = to_nchw(nhwc)
    assert nchw.shape == (5, 3, 28, 28)
    # XNNPACK lowering rejects channels-last views, so this must be a real copy.
    assert nchw.flags["C_CONTIGUOUS"]


def test_to_labels_from_one_hot():
    y = np.eye(4)[[2, 0, 3]]
    assert to_labels(y).tolist() == [2, 0, 3]


# --- QNN export contract: optional artifact, never raises ---------------------

def test_qnn_export_never_raises_on_bad_scheme(tmp_path):
    from qnn_export import export_qnn

    net = build_model(in_channels=3, num_classes=4, pretrained=False).eval()
    report_path = tmp_path / "export_report.json"
    report = export_qnn(
        net, torch.rand(1, 3, 32, 32), np.random.rand(2, 3, 32, 32).astype(np.float32),
        str(tmp_path / "model_qnn.pte"), quant_scheme="nope", report_path=str(report_path),
    )
    assert report["status"] == "failed"
    assert report["reason"]
    assert json.loads(report_path.read_text())["status"] == "failed"
    assert not (tmp_path / "model_qnn.pte").exists()


def test_qnn_export_refuses_non_x86(tmp_path, monkeypatch):
    import qnn_export

    monkeypatch.setattr(qnn_export.platform, "machine", lambda: "aarch64")
    net = build_model(in_channels=3, num_classes=4, pretrained=False).eval()
    report = qnn_export.export_qnn(
        net, torch.rand(1, 3, 32, 32), np.random.rand(2, 3, 32, 32).astype(np.float32),
        str(tmp_path / "model_qnn.pte"),
    )
    assert report["status"] == "failed"
    assert "x86_64" in report["reason"]


# --- end to end on synthetic data (what Studio does, minus Studio) -----------

def _fake_split(d, n, size=32, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, size, size, 3), dtype=np.float32)          # NHWC, 0..1
    y = np.eye(classes, dtype=np.float32)[rng.integers(0, classes, n)]
    return x, y


@pytest.mark.skipif(not HAVE_ET, reason="executorch not installed")
def test_train_end_to_end_exports_runnable_pte(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    for split, n in (("train", 24), ("test", 8)):
        x, y = _fake_split(data, n, seed=n)
        np.save(data / f"X_split_{split}.npy", x)
        np.save(data / f"Y_split_{split}.npy", y)

    subprocess.run(
        [sys.executable, os.path.join(HERE, "train.py"),
         "--data-directory", str(data), "--out-directory", str(out),
         "--epochs", "1", "--no-pretrained", "--learning-rate", "0.001", "--export-pte"],
        check=True, cwd=HERE,
    )
    assert (out / "model.onnx").exists()
    assert (out / "model.pte").exists()

    # Static check: the whole graph went to XNNPACK.
    from executorch.exir._serialize._program import deserialize_pte_binary

    plan = deserialize_pte_binary((out / "model.pte").read_bytes()).program.execution_plan[0]
    assert [d.id for d in plan.delegates] == ["XnnpackBackend"]

    # Runtime check. ExecuTorch 1.5.1's Python runtime is built against torch
    # 2.14; with the block's pinned torch 2.12.1, exporting works but executing
    # fails ("tensor does not have a device"). See docs/versions.md.
    if not torch.__version__.startswith("2.14"):
        pytest.skip(f"runtime check needs torch 2.14 for executorch 1.5.1 (have {torch.__version__})")

    from executorch.runtime import Runtime

    method = Runtime.get().load_program(str(out / "model.pte")).load_method("forward")
    probs = method.execute([torch.rand(1, 3, 32, 32)])[0]
    assert probs.shape == (1, 3)
    assert abs(float(probs.sum()) - 1.0) < 1e-4      # softmax baked in


def test_qnn_export_reports_a_broken_sdk_path_instead_of_crashing(tmp_path, monkeypatch):
    """ExecuTorch's C++ layer aborts the process on a bad QNN_SDK_ROOT; the
    pre-check must turn that into a failed report."""
    from qnn_export import export_qnn

    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path / "no-sdk-here"))
    net = build_model(in_channels=3, num_classes=4, pretrained=False).eval()
    report = export_qnn(net, torch.rand(1, 3, 32, 32),
                        np.random.rand(2, 3, 32, 32).astype(np.float32),
                        str(tmp_path / "model_qnn.pte"))
    assert report["status"] == "failed"
    assert "libQnnHtp.so" in report["reason"]


@pytest.mark.skipif(not HAVE_ET, reason="executorch not installed")
def test_qnn_export_custom_metric_survives_missing_sdk(tmp_path, monkeypatch):
    """Non-classifier blocks (e.g. FOMO) pass eval_fn. Quantization runs before
    the SDK is needed, so the float-vs-int8 numbers land in the report even when
    lowering cannot happen."""
    import platform

    if platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("QNN quantizer path is x86_64 only")
    import executorch.backends.qualcomm.utils.qnn_sdk_setup as setup_mod

    def no_sdk():
        raise RuntimeError("no SDK in this test")

    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setattr(setup_mod, "setup_qnn_sdk", no_sdk)
    net = build_model(in_channels=3, num_classes=4, pretrained=False).eval()
    x = torch.rand(1, 3, 32, 32)
    calls = []

    def score(m):
        calls.append(1)
        return float(m(x).shape[1]) / 4.0          # any deterministic number

    from qnn_export import export_qnn

    report = export_qnn(net, x, np.random.rand(4, 3, 32, 32).astype(np.float32),
                        str(tmp_path / "model_qnn.pte"),
                        eval_fn=score, metric_name="my_metric")
    assert len(calls) == 2                          # float module, then int8 graph
    assert report["accuracy"]["metric"] == "my_metric"
    assert report["accuracy"]["float32"] == report["accuracy"]["int8_fake_quant"] == 1.0
    assert report["status"] == "failed" and "no SDK" in report["reason"]
