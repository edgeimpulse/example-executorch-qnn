# Deployment block: ExecuTorch `.pte` (CPU int8 + QNN pass-through)

This is an Edge Impulse [custom deployment block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-deployment-blocks). It turns a trained impulse into a `deploy.zip` you can download from the project's **Deployment** page. It has three jobs:

1. **CPU model.**
   - Converts Studio's ONNX export to PyTorch with `onnx2torch` and lowers it to `model.pte` with XNNPACK.
   - If you pick **Quantized (int8)** on the Deployment page, it applies PT2E static quantization with `XNNPACKQuantizer` first. Calibration uses the learning block's training data, and the log shows float32 vs int8 accuracy.
2. **NPU model pass-through.** If the learning block produced `model_qnn.pte` and `export_report.json` (with **Export QNN .pte** on), they are copied into the package **unchanged**. This block does no QNN work.
3. **Delivery.** Studio only exposes learning-block artifacts through a deployment. The package also includes the learning block's inputs and outputs under `_inputs/`, for testing on a laptop or the board.

Deployment blocks need an **Enterprise** organization. Without one, run the learning block locally to get the same files (see [the learning block README](../learning-image-classifier/README.md#run-it-locally)).

## Prerequisites

**Background.** You should know Studio's **Deployment** page, and what a [custom deployment block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-deployment-blocks) is. Studio runs this folder's Docker image with the trained impulse and writes whatever the block produces as the download.

**What you need:**

- **An Edge Impulse Enterprise organization.** The block is pushed to the organization, and the project must belong to that organization to see the block.
- **A project trained with the [learning block](../learning-image-classifier/)** from this repository. With **Export QNN .pte** checked, the package also carries `model_qnn.pte`. The CPU conversion was written and tested for this repository's blocks; with other learning blocks it may work, but there is no NPU file.
- **The Edge Impulse CLI** (`npm install -g edge-impulse-cli`, Node.js 16 or later, 22 recommended), to push the block.
- **Docker**, only to test the block locally with `edge-impulse-blocks runner`.

**To use what is in `deploy.zip`:**

- **`model_qnn.pte`:** an IQ‑9075 EVK set up with [board/](../../board/README.md).
- **`model.pte`:** any Linux machine (x86_64 or Arm) with Python and `pip install executorch==1.3.1`. That is the release that built it, and a `.pte` loads on the release that built it and the next one ([versions](../../docs/versions.md)).

## What is in `deploy.zip`

```text
model.pte              CPU program (XNNPACK; int8 if "Quantized (int8)" was chosen, else float32)
model.onnx             Studio's ONNX export (the source of model.pte above)
labels.txt             class names, in output order (alphabetical, as Edge Impulse orders them)
model_qnn.pte          NPU program from the learning block (only if --export-qnn succeeded)
export_report.json     its report: accuracy, delegation, SoC, SDK, input shape, labels
app/run_pte.py         minimal Python runner:  python app/run_pte.py model.pte --labels labels.txt
app/convert.py         the same ONNX → .pte conversion, offline
_inputs/               learning-block inputs and outputs, copied for local testing:
  X_split_train.npy  X_split_test.npy  Y_split_test.npy   (Studio features, flat NHWC)
  model.pte          the learning block's own float32 XNNPACK program (no ONNX round trip)
  model.onnx  model_qnn.pte  export_report.json  dsp_metadata.json  deployment-metadata.json
```

The board scripts use `_inputs/X_split_test.npy` and `Y_split_test.npy` to measure accuracy on the NPU.

## Files

| File | Purpose |
|---|---|
| `parameters.json` | Block metadata: `"mountLearnBlock": true` (mounts the learning block's data and outputs at `/data`), `"showOptimizations": true` (shows the int8/float32 choice), a backend selector (`xnnpack` / `portable`) |
| `build.py` | Entry point: reads `deployment-metadata.json`, converts, quantizes, passes QNN files through, and writes `deploy.zip` |
| `app/` | Runtime harness copied into the package |
| `Dockerfile`, `requirements.txt` | Container definition (ExecuTorch 1.3.1, onnx2torch) |

## Design notes

- **How the int8 choice is detected.** Studio reports the user's choice as `tfliteModels[0].details.modelType`, either `int8` or `float32`. The ONNX file is float32 either way, so quantization happens here. Every failure falls back to float32 *and logs it*, so a float32 model is never shipped silently when int8 was asked for.
- **Layout.** Studio's features are flat **NHWC**, while onnx2torch models expect **NCHW**. `to_model_input()` reshapes to NHWC first and then permutes. Reshaping the flat buffer straight to NCHW scrambles the channels and gives chance-level accuracy, and the block warns if float32 accuracy is at chance.
- **Why this block doesn't lower to QNN itself.** The learning block already has the trained PyTorch module, the calibration images and the test split. It is the natural place for the NPU export. Here, the only PyTorch model available is one rebuilt from ONNX.

## Test locally

```bash
edge-impulse-blocks runner        # downloads the project's deployment inputs and runs this block in Docker
# (edge-impulse-blocks runner --download-data only downloads; the inputs land under ./ei-block-data/)
```

`deploy.zip` is written to the output folder named in `deployment-metadata.json`. The QNN pass-through only appears when the learning block's outputs are mounted at `/data`. Studio does that because `mountLearnBlock` is set. A local run only does it if you mount a folder containing `model_qnn.pte` at `/data` yourself.

## Push

```bash
edge-impulse-blocks init     # choose "Deployment block"; writes .ei-block-config (gitignored)
edge-impulse-blocks push
```

The block then appears on the project's **Deployment** page under custom blocks.

## Known limitations

- The CPU `model.pte` comes from an **ONNX round trip**, not from the PyTorch module that was trained. For standard CNNs it reaches the same accuracy, but unusual ops may not convert. The learning block's native float32 `model.pte` is in `_inputs/`; making it the main CPU artifact is on the [roadmap](../../docs/roadmap.md).
- `build.py` still prints a filesystem scan (`scout()`) and dumps `_inputs/` (`DUMP_INPUTS = True`). Both are deliberate during development and can be turned off.
