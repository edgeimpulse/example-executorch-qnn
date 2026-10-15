# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Lower a trained PyTorch classifier to an ExecuTorch .pte for the Qualcomm
QNN backend (Hexagon HTP / NPU).

Flow (PT2E post-training quantization, then QNN lowering):

    torch.export -> prepare_pt2e(QnnQuantizer) -> calibrate -> convert_pt2e
        -> to_edge_transform_and_lower_to_qnn -> .pte with a QNN context binary

Requirements
------------
* x86_64 Linux. The AOT lowering libraries ship for x86_64 only.
* The Qualcomm AI Runtime SDK. ExecuTorch downloads its pinned version on
  first use unless ``QNN_SDK_ROOT`` points at an existing install. Custom
  blocks run without network access, so bake the SDK into the Docker image
  (see the Dockerfile) and set ``QNN_SDK_ROOT``.

Design rules
------------
* ``export_qnn`` NEVER raises. QNN is an optional artifact, exactly like the
  XNNPACK .pte: on any failure it returns a report with ``status: "failed"``
  and a reason, and training still succeeds.
* A model that quantizes to nonsense must not ship quietly. The report carries
  float32 vs int8 quality and a ``delegation`` section; ``status`` is "ok"
  only if the QNN delegate is present in the final graph.
* Task-agnostic. Inputs can be any single float32 tensor (NCHW images,
  (N, features) vectors, (N, axes, time) windows). Classifiers get top-1
  accuracy for free from ``eval_x``/``eval_y``; any other task passes
  ``eval_fn`` (e.g. FOMO per-cell accuracy). See docs/porting-to-other-blocks.md.
"""

from __future__ import annotations

import json
import os
import platform
import time
from typing import Callable, Optional

import numpy as np
import torch

# Compile target. The IQ-9075 EVK (QCS9075, Hexagon v73) is not named in
# ExecuTorch's QcomChipset enum. QCS9100 is the same v73 / 8 MB VTCM part, so
# it is the working assumption. VERIFY ON DEVICE: if the context binary fails
# to load, try the other v73 entries in qc_schema.py.
DEFAULT_SOC = "QCS9100"
DEFAULT_QUANT = "8a8w"

QUANT_SCHEMES = {
    "8a8w": "use_8a8w",
    "16a8w": "use_16a8w",
    "16a16w": "use_16a16w",
    "16a4w": "use_16a4w",
}


def _qnn_sdk_version() -> Optional[str]:
    try:
        from executorch.backends.qualcomm.scripts import download_qnn_sdk

        return str(download_qnn_sdk.QNN_VERSION)
    except Exception:  # noqa: BLE001
        return None


def _argmax_accuracy(module, x: np.ndarray, y: np.ndarray) -> float:
    """Top-1 accuracy of a batch-1 exported module over (N, C, H, W) data."""
    if len(x) == 0:
        return float("nan")
    correct = 0
    with torch.no_grad():
        for row, label in zip(x, y):
            out = module(torch.from_numpy(np.ascontiguousarray(row[None])))
            if int(out.argmax()) == int(label):
                correct += 1
    return correct / len(x)


def export_qnn(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    calibration: np.ndarray,
    out_path: str,
    *,
    soc_model: str = DEFAULT_SOC,
    quant_scheme: str = DEFAULT_QUANT,
    eval_x: Optional[np.ndarray] = None,
    eval_y: Optional[np.ndarray] = None,
    report_path: Optional[str] = None,
    extra_report: Optional[dict] = None,
    eval_fn: Optional[Callable[[Callable], float]] = None,
    metric_name: str = "accuracy",
) -> dict:
    """Quantize, lower to QNN HTP and write ``out_path``. Returns the report.

    Args:
        model: eval-mode module that outputs LOGITS (no softmax) so the int8
            output keeps its dynamic range. Apply softmax at inference.
        sample_input: batch-1 float32 input in the model's layout, e.g.
            (1, C, H, W) for images, (1, F) for feature vectors.
        calibration: (N, ...) float32 samples in the same layout, used to
            calibrate activation ranges. ~200 representative samples is plenty.
        out_path: where to write the .pte.
        soc_model: ``QcomChipset`` member name.
        quant_scheme: one of ``QUANT_SCHEMES``.
        eval_x, eval_y: held-out inputs and integer labels for a classifier's
            float32 vs int8 top-1 accuracy check.
        eval_fn: for non-classifiers: called with a callable model (the float
            module, then the quantized graph) and returns a score. It must only
            call the model, not ``.eval()``/``.train()`` on it (exported graphs
            refuse those). Takes precedence over eval_x/eval_y.
        metric_name: label for eval_fn's score in the report.
        report_path: if given, the report is written there as JSON.
        extra_report: merged into the report (labels, input size, ...).
    """
    report: dict = {
        "backend": "qnn-htp",
        "status": "failed",
        "soc_model": soc_model,
        "quant_scheme": quant_scheme,
        "executorch_qnn_sdk": None,
        "host": f"{platform.system()} {platform.machine()}",
        "output": None,
        "reason": None,
        "calibration_samples": int(len(calibration)),
    }
    if extra_report:
        report.update(extra_report)
    t0 = time.time()

    def finish(reason: Optional[str] = None) -> dict:
        if reason:
            report["reason"] = reason
            print(f"WARNING: QNN export skipped/failed: {reason}", flush=True)
        report["seconds"] = round(time.time() - t0, 1)
        if report_path:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
        return report

    if platform.machine().lower() not in ("x86_64", "amd64") or platform.system() != "Linux":
        return finish(
            "QNN lowering needs x86_64 Linux (this host is "
            f"{platform.system()} {platform.machine()})."
        )
    if quant_scheme not in QUANT_SCHEMES:
        return finish(f"unknown quant scheme {quant_scheme!r}; use {sorted(QUANT_SCHEMES)}")

    try:
        from executorch.backends.qualcomm.quantizer.quantizer import (
            QnnQuantizer,
            QuantDtype,
        )
        from executorch.backends.qualcomm.serialization.qc_schema import (
            QcomChipset,
            QnnExecuTorchBackendType,
        )
        from executorch.backends.qualcomm.utils.qnn_sdk_setup import setup_qnn_sdk
        from executorch.backends.qualcomm.utils.utils import (
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
            get_qnn_context_binary_alignment as _align,
            to_edge_transform_and_lower_to_qnn,
        )
        from executorch.devtools.backend_debug import get_delegation_info
        from executorch.exir.backend.utils import format_delegated_graph
        from executorch.exir.capture._config import ExecutorchBackendConfig
        from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )
    except Exception as exc:  # noqa: BLE001
        return finish(f"ExecuTorch Qualcomm backend not importable: {exc}")

    # ExecuTorch's C++ QNN layer ABORTS the process (not a Python exception) when
    # QNN_SDK_ROOT is set but its libraries are missing, so check before touching it.
    sdk_root = os.environ.get("QNN_SDK_ROOT")
    if sdk_root and not os.path.exists(
        os.path.join(sdk_root, "lib", "x86_64-linux-clang", "libQnnHtp.so")
    ):
        return finish(
            f"QNN_SDK_ROOT={sdk_root} has no lib/x86_64-linux-clang/libQnnHtp.so "
            "(a broken SDK path would abort the process)."
        )

    try:
        if not hasattr(QcomChipset, soc_model):
            return finish(f"unknown soc_model {soc_model!r} (see qc_schema.QcomChipset)")
        soc = getattr(QcomChipset, soc_model)

        model = model.eval()
        sample_input = sample_input.contiguous()
        example = (sample_input,)

        # ---- float reference (accuracy baseline) --------------------------
        def score(m) -> Optional[float]:
            if eval_fn is not None:
                with torch.no_grad():
                    return float(eval_fn(m))
            if eval_x is not None and eval_y is not None and len(eval_x):
                return _argmax_accuracy(m, eval_x, eval_y)
            return None

        if eval_fn is None:
            metric_name = "accuracy"
        float_acc = score(model)

        # ---- PT2E quantization --------------------------------------------
        captured = torch.export.export(model, example, strict=False).module()
        quantizer = QnnQuantizer(
            backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=soc
        )
        quantizer.set_default_quant_config(
            getattr(QuantDtype, QUANT_SCHEMES[quant_scheme]),
            is_qat=False,
            is_conv_per_channel=True,
            is_linear_per_channel=True,
        )
        prepared = prepare_pt2e(captured, quantizer)
        with torch.no_grad():
            for row in calibration:
                prepared(torch.from_numpy(np.ascontiguousarray(row[None])))
        quantized = convert_pt2e(prepared)

        if float_acc is not None:
            q_acc = score(quantized)
            report["accuracy"] = {
                "metric": metric_name,
                "float32": round(float_acc, 4),
                "int8_fake_quant": round(q_acc, 4),
                "delta_pp": round((q_acc - float_acc) * 100, 2),
                "n": int(len(eval_x)) if eval_x is not None else None,
                "note": "Quantized graph run on CPU (simulated). Final on-NPU "
                        "accuracy must be re-checked on the device.",
            }
            print(
                f"[qnn] {metric_name}: float32 {float_acc:.4f} | {quant_scheme} {q_acc:.4f} "
                f"| delta {(q_acc - float_acc) * 100:+.2f} pp",
                flush=True,
            )

        # ---- lower to QNN HTP ---------------------------------------------
        # The SDK is only needed from here on, so a missing SDK still leaves
        # the float-vs-int8 numbers above in the report.
        setup_qnn_sdk()  # uses QNN_SDK_ROOT if set, else downloads the pinned SDK
        report["executorch_qnn_sdk"] = _qnn_sdk_version()
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compile_spec = generate_qnn_executorch_compiler_spec(
            soc_model=soc, backend_options=backend_options
        )
        edge = to_edge_transform_and_lower_to_qnn(quantized, example, compile_spec)

        graph = edge.exported_program().graph_module
        info = get_delegation_info(graph)
        delegated = "QnnBackend" in format_delegated_graph(graph)
        report["delegation"] = {
            "qnn_delegate_present": bool(delegated),
            "delegated_subgraphs": info.num_delegated_subgraphs,
            "delegated_nodes": info.num_delegated_nodes,
            "non_delegated_nodes": info.num_non_delegated_nodes,
            "non_delegated_ops": sorted(
                op for op, b in info.delegation_by_operator.items() if b.non_delegated
            ),
        }
        print(f"[qnn] delegation: {report['delegation']}", flush=True)
        if not delegated:
            return finish("QNN delegate absent from the lowered graph; nothing runs on the NPU.")

        program = edge.to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=True, alloc_graph_output=True
                ),
                segment_alignment=_align(),
            )
        )
        with open(out_path, "wb") as f:
            program.write_to_file(f)

        report["status"] = "ok"
        report["output"] = os.path.basename(out_path)
        report["size_kb"] = round(os.path.getsize(out_path) / 1024.0, 1)
        report["input_shape"] = list(sample_input.shape)
        print(f"Saved QNN ExecuTorch program to {out_path} ({report['size_kb']} KB)", flush=True)
        return finish()
    except Exception as exc:  # noqa: BLE001 - optional artifact, never fail training
        return finish(f"{type(exc).__name__}: {exc}")
