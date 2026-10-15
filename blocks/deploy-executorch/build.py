# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Edge Impulse custom deployment block: export to ExecuTorch (.pte).

Edge Impulse invokes this with:

    build.py --metadata <path to deployment-metadata.json>

The metadata file describes the impulse and points to the input and output
folders. The input folder contains the trained model in several formats plus the
Edge Impulse SDK. We package an ExecuTorch program together with a small runtime
harness into `deploy.zip` in the output folder.

Conversion path
---------------
ExecuTorch consumes PyTorch programs, while Edge Impulse provides TensorFlow /
TFLite / ONNX artifacts. This block converts the exported **ONNX** model to a
PyTorch module (via onnx2torch), then lowers it to a `.pte` with the selected
backend. If ONNX is not available or conversion fails, the block still packages
the original model plus instructions so the deployment never hard-fails.

Quantization
------------
Studio signals the user's int8/float32 choice through
`tfliteModels[0].details.modelType`. The ONNX handed to this block is float32 in
both cases - Studio only quantizes on the TFLite side - so when int8 is
requested we quantize here with the PT2E flow, calibrating on the learn block's
feature data mounted at /data (requires "mountLearnBlock": true in
parameters.json).

Every quantization failure path degrades to float32 *and says so in the log*.
Silently shipping a float32 model to a user who asked for int8 is the bug this
code exists to fix; do not reintroduce it.

Layout note
-----------
Edge Impulse supplies image features flat in **NHWC** order. onnx2torch models
converted from Studio's exports expect **NCHW**. Reshaping the flat buffer
straight to NCHW silently interleaves the channels and produces chance-level
accuracy, so `to_model_input()` permutes explicitly. Do not remove it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile

import numpy as np

APP_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "app")
BUILD_DIR = "/tmp/build"
DATA_DIR = "/data"
CALIBRATION_PATH = os.path.join(DATA_DIR, "X_split_train.npy")

# Copies the mounted learn-block inputs into deploy.zip (under _inputs/) so the
# block can be run locally against real Studio inputs. Set to False once the
# local harness works.
DUMP_INPUTS = True
DUMP_SUBDIR = "_inputs"
DUMP_FILES = [
    "X_split_train.npy",
    "X_split_test.npy",
    "Y_split_test.npy",
    "dsp_metadata.json",
    "model.pte",
    "model.onnx",
    "model_qnn.pte",      # QNN/NPU program from the learn block (--export-qnn)
    "export_report.json", # QNN delegation + accuracy report
]


# --------------------------------------------------------------------------
# Debug helpers - remove once the block is stable
# --------------------------------------------------------------------------

def scout() -> None:
    """Print the container's filesystem layout. Diagnostic only."""
    print("=" * 60, flush=True)
    print("MOUNT PROBE", flush=True)
    print("=" * 60, flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"argv: {sys.argv}", flush=True)

    for root in [DATA_DIR, "/app"]:
        print(f"\n--- {root} ---", flush=True)
        if not os.path.isdir(root):
            print("  (does not exist)", flush=True)
            continue
        count = 0
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = -1
                print(f"  {size:>12,}  {p}", flush=True)
                count += 1
                if count > 60:
                    print("  ... truncated at 60 files", flush=True)
                    break
            if count > 60:
                break
        if count == 0:
            print("  (empty)", flush=True)

    print("=" * 60, flush=True)


def dump_inputs(metadata_path: str) -> None:
    """Copy the learn-block inputs into BUILD_DIR/_inputs for local testing.

    IMPORTANT: this must write to a SUBDIRECTORY. Copying /data/model.pte into
    BUILD_DIR directly overwrites the .pte this block just produced.
    """
    dest = os.path.join(BUILD_DIR, DUMP_SUBDIR)
    os.makedirs(dest, exist_ok=True)

    copied = []
    for name in DUMP_FILES:
        src = os.path.join(DATA_DIR, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest, name))
            copied.append(name)
    try:
        shutil.copy(metadata_path, os.path.join(dest, "deployment-metadata.json"))
        copied.append("deployment-metadata.json")
    except OSError as exc:
        print(f"[dump] Could not copy metadata: {exc}", flush=True)
    print(f"[dump] Wrote to {DUMP_SUBDIR}/: {', '.join(copied) or 'nothing'}", flush=True)


# --------------------------------------------------------------------------
# Input layout
# --------------------------------------------------------------------------

def is_nchw(input_shape: tuple) -> bool:
    """True when a 4D model input is channels-first.

    Studio image models come through as (N, C, H, W) with C in {1, 3, 4}.
    A channels-last model would instead be (N, H, W, C).
    """
    if len(input_shape) != 4:
        return False
    _, d1, d2, d3 = input_shape
    if d1 in (1, 3, 4) and d2 == d3:
        return True
    if d3 in (1, 3, 4) and d1 == d2:
        return False
    # Ambiguous (e.g. non-square). Assume channels-first, which is what
    # onnx2torch produces from Studio's exports.
    return True


def to_model_input(row: np.ndarray, input_shape: tuple):
    """Turn one flat Edge Impulse feature row into a model-ready tensor.

    Edge Impulse stores image features flat in NHWC order. When the model wants
    NCHW we must reshape to NHWC first and then permute - reshaping the flat
    buffer straight to NCHW scrambles the channels.
    """
    import torch

    t = torch.from_numpy(np.ascontiguousarray(row)).float()

    if len(input_shape) == 4 and is_nchw(input_shape):
        n, c, h, w = input_shape
        t = t.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()
    elif len(input_shape) == 3:
        # (batch, axes, time) Conv1d models: Studio's raw window is interleaved
        # samples (time, axes), so the same reshape-then-permute applies.
        n, c, length = input_shape
        t = t.reshape(n, length, c).permute(0, 2, 1).contiguous()
    else:
        t = t.reshape(*input_shape)

    return t


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------

def wants_int8(metadata: dict) -> bool:
    """True when the user selected the quantized (int8) optimization.

    Confirmed empirically: Studio reports the choice as
    tfliteModels[0].details.modelType, which is "int8" or "float32". The ONNX
    at onnxModelPath is float32 in both cases.
    """
    models = metadata.get("tfliteModels", [])
    if not models:
        return False
    return models[0].get("details", {}).get("modelType") == "int8"


def load_calibration(input_shape: tuple, max_samples: int = 500):
    """Return calibration samples, or None if int8 is not safely possible.

    Returning None must always be survivable - the caller falls back to float32.
    """
    if not os.path.exists(CALIBRATION_PATH):
        print(f"[quant] No calibration data at {CALIBRATION_PATH} - float32 only. "
              "(Is \"mountLearnBlock\": true set in parameters.json?)", flush=True)
        return None

    try:
        X = np.load(CALIBRATION_PATH).astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - degrade rather than fail the build
        print(f"[quant] Could not read calibration data ({exc}) - float32 only.", flush=True)
        return None

    if X.ndim < 2:
        print(f"[quant] Calibration array has shape {X.shape}, expected at least 2D "
              "- float32 only.", flush=True)
        return None

    per_sample = int(np.prod(input_shape[1:]))
    have = int(np.prod(X.shape[1:]))
    if have != per_sample:
        print(f"[quant] Calibration provides {have} values per sample but the model "
              f"expects {per_sample} (input shape {tuple(input_shape)}) - float32 only.",
              flush=True)
        return None

    n = min(len(X), max_samples)
    layout = "NCHW (permuting from NHWC)" if (len(input_shape) == 4 and is_nchw(input_shape)) \
        else "flat"
    print(f"[quant] Calibrating on {n} of {len(X)} samples "
          f"(input shape {tuple(input_shape)}, layout {layout}).", flush=True)
    return X[:n]


def quantize_static(model, example_inputs: tuple, calibration, input_shape: tuple):
    """PT2E static quantization for the XNNPACK backend."""
    import torch
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    m = torch.export.export(model.eval(), example_inputs).module()

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

    prepared = prepare_pt2e(m, quantizer)

    with torch.no_grad():
        for row in calibration:
            prepared(to_model_input(row, input_shape))

    return convert_pt2e(prepared)


def report_quantization_impact(float_model, quant_model, input_shape: tuple) -> None:
    """Print the accuracy delta on the held-out test split, if available."""
    import torch

    xp = os.path.join(DATA_DIR, "X_split_test.npy")
    yp = os.path.join(DATA_DIR, "Y_split_test.npy")
    if not (os.path.exists(xp) and os.path.exists(yp)):
        print("[quant] No test split available - skipping accuracy check.", flush=True)
        return

    try:
        X = np.load(xp).astype(np.float32)
        Y = np.load(yp)
    except Exception as exc:  # noqa: BLE001
        print(f"[quant] Could not read test split ({exc}) - skipping accuracy check.", flush=True)
        return

    if Y.ndim > 1:
        Y = Y.argmax(axis=1)
    n_classes = int(Y.max()) + 1 if len(Y) else 0

    def accuracy(m) -> float:
        correct = 0
        with torch.no_grad():
            for row, label in zip(X, Y):
                out = m(to_model_input(row, input_shape))
                if int(out.argmax()) == int(label):
                    correct += 1
        return correct / len(X)

    try:
        a_float = accuracy(float_model)
        a_quant = accuracy(quant_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[quant] Accuracy check failed ({exc}).", flush=True)
        return

    print(f"[quant] float32 accuracy {a_float:.4f} | int8 accuracy {a_quant:.4f} "
          f"| delta {(a_quant - a_float) * 100:+.2f} pp  (n={len(X)})", flush=True)

    # A float32 model scoring at chance means the inputs are being fed in the
    # wrong layout or scale - the quantization numbers above are then worthless.
    if n_classes > 1 and a_float <= (1.0 / n_classes) + 0.05:
        print(f"[quant] WARNING: float32 accuracy is at chance level for "
              f"{n_classes} classes. The calibration inputs are probably in the "
              "wrong layout or scale, so the int8 model is calibrated on "
              "meaningless data. Investigate before shipping this model.",
              flush=True)


# --------------------------------------------------------------------------
# Metadata / model discovery
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExecuTorch deployment block")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--backend", type=str, default="xnnpack",
                        choices=["xnnpack", "portable"])
    args, _ = parser.parse_known_args()
    return args


def load_metadata(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def find_onnx_model(metadata: dict, input_dir: str) -> str | None:
    """Locate an ONNX model, preferring the path in the metadata."""
    for model in metadata.get("tfliteModels", []):
        onnx_path = model.get("onnxModelPath")
        if onnx_path and os.path.exists(onnx_path):
            return onnx_path
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".onnx"):
                return os.path.join(root, name)
    return None


def convert_onnx_to_pte(onnx_path: str, backend: str, out_path: str,
                        want_int8: bool = False) -> bool:
    """Convert an ONNX model to an ExecuTorch program. Returns success."""
    try:
        import onnx
        import torch
        from onnx2torch import convert
    except ImportError as exc:
        print(f"NOTE: ONNX->PyTorch tooling unavailable ({exc}); skipping .pte.")
        return False

    try:
        onnx_model = onnx.load(onnx_path)
        torch_model = convert(onnx_model).eval()

        graph_input = onnx_model.graph.input[0]
        dims = graph_input.type.tensor_type.shape.dim
        shape = tuple((d.dim_value if d.dim_value > 0 else 1) for d in dims)
        example = (torch.randn(*shape),)

        from executorch.exir import to_edge_transform_and_lower

        model_to_export = torch_model
        precision = "float32"
        if want_int8:
            calibration = load_calibration(shape)
            if calibration is None:
                print("[quant] int8 requested but calibration unavailable - "
                      "producing float32.", flush=True)
            else:
                try:
                    model_to_export = quantize_static(torch_model, example,
                                                      calibration, shape)
                    precision = "int8"
                    report_quantization_impact(torch_model, model_to_export, shape)
                except Exception as exc:  # noqa: BLE001
                    print(f"[quant] int8 requested but quantization failed ({exc}) - "
                          "producing float32.", flush=True)
                    model_to_export = torch_model
                    precision = "float32"

        exported = torch.export.export(model_to_export, example)
        if backend == "xnnpack":
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
            program = to_edge_transform_and_lower(
                exported, partitioner=[XnnpackPartitioner()]
            ).to_executorch()
        else:
            program = to_edge_transform_and_lower(exported).to_executorch()

        with open(out_path, "wb") as f:
            f.write(program.buffer)
        size_kb = os.path.getsize(out_path) / 1024.0
        print(f"Wrote {precision} ExecuTorch program to {out_path} "
              f"({size_kb:.1f} KB, backend={backend}).")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue packaging
        print(f"WARNING: ExecuTorch conversion failed: {exc}")
        return False


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------

def make_build_dir() -> None:
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)


def zip_dir(src_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, src_dir)
                zf.write(abs_path, rel_path)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)

    input_dir = metadata["folders"]["input"]
    output_dir = metadata["folders"]["output"]

    make_build_dir()

    # 1. Copy the runtime harness.
    shutil.copytree(APP_DIR, os.path.join(BUILD_DIR, "app"))

    # 2. Write the label map for the runtime.
    classes = metadata.get("classes", [])
    with open(os.path.join(BUILD_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(classes))

    # 3. Attempt ExecuTorch conversion from ONNX.
    want_int8 = wants_int8(metadata)
    print(f"[quant] Requested precision: {'int8' if want_int8 else 'float32'}", flush=True)

    onnx_path = find_onnx_model(metadata, input_dir)
    pte_path = os.path.join(BUILD_DIR, "model.pte")
    converted = False
    if onnx_path:
        converted = convert_onnx_to_pte(onnx_path, args.backend, pte_path, want_int8)
        shutil.copy(onnx_path, os.path.join(BUILD_DIR, "model.onnx"))
    else:
        print("NOTE: no ONNX model found in the deployment inputs.")

    if not converted:
        with open(os.path.join(BUILD_DIR, "CONVERSION_NOTE.txt"), "w") as f:
            f.write(
                "ExecuTorch (.pte) was not produced automatically.\n"
                "The trained model is included as model.onnx. Convert it with:\n\n"
                "    python app/convert.py model.onnx model.pte --backend "
                f"{args.backend}\n"
            )

    # 3a. Pass the learn block's QNN program (if any) straight through, next to
    # model.pte, so it is easy to find in the downloaded deployment.
    for name in ("model_qnn.pte", "export_report.json"):
        src = os.path.join(DATA_DIR, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(BUILD_DIR, name))
            print(f"[qnn] Included {name} from the learn block.", flush=True)
    if not os.path.exists(os.path.join(DATA_DIR, "model_qnn.pte")):
        print("[qnn] No model_qnn.pte in the learn block output "
              "(was 'Export QNN .pte' enabled and did it succeed?).", flush=True)

    # 3b. Bundle the learn-block inputs (in a subdirectory - see dump_inputs).
    if DUMP_INPUTS:
        dump_inputs(args.metadata)

    # 4. Package everything into deploy.zip.
    os.makedirs(output_dir, exist_ok=True)
    deploy_zip = os.path.join(output_dir, "deploy.zip")
    zip_dir(BUILD_DIR, deploy_zip)
    final_kb = os.path.getsize(pte_path) / 1024.0 if os.path.exists(pte_path) else 0.0
    print(f"Deployment package written to {deploy_zip} "
          f"(model.pte in package: {final_kb:.1f} KB)")


if __name__ == "__main__":
    scout()
    main()
