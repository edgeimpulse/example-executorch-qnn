# Learning block: PyTorch image classifier → ExecuTorch (CPU + Qualcomm NPU)

This is an Edge Impulse [custom learning block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks). It fine-tunes an ImageNet-pretrained MobileNetV2 in PyTorch on your Studio image data, then exports the trained network in up to four forms:

| Output | Used by | When |
|---|---|---|
| `model.onnx` | Studio, for live classification, model testing and profiling (Studio converts it to TFLite) | Always |
| `model.pte` | ExecuTorch on any CPU (XNNPACK, float32, softmax built in) | **Export ExecuTorch .pte** is on (the default) |
| `model_qnn.pte` | ExecuTorch with the QNN backend on the Hexagon NPU (int8, outputs logits) | **Export QNN .pte** is on |
| `export_report.json` | You and the device: what was built, how accurate it is, how much runs on the NPU | **Export QNN .pte** is on |

The QNN export is **optional and can never fail a training run**. If anything goes wrong, training still succeeds and `export_report.json` says why the export failed.

## Prerequisites

**Background.** You should know how to build and train an impulse in Edge Impulse Studio: create an image project, label data, add an Image processing block, and pick a learning block. It also helps to know what a [custom learning block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks) is: Studio runs this folder's Docker image with your data at `--data-directory`, and the block writes model files to `--out-directory`.

**What you need, depending on what you want to do:**

| To… | You need |
|---|---|
| Run the tests ([below](#tests-no-docker-no-network)) | <ul><li>Python 3.10–3.13. The image uses 3.11.</li><li>About 3 GB of disk for torch and ExecuTorch.</li><li>No Docker, network or hardware.</li></ul> |
| Build the image and run the block with the NPU export | <ul><li>Docker on an **x86_64** machine. Qualcomm's QNN compiler only ships for x86_64 Linux. On Apple Silicon it runs under emulation, much more slowly.</li><li>Internet access during `docker build`. It downloads the Qualcomm AI Runtime SDK and a libc++, about 1.9 GB.</li><li>About 10 GB free disk.</li><li>No Qualcomm account. ExecuTorch fetches the SDK itself, under Qualcomm's licence.</li></ul> |
| Train in Studio | <ul><li>An [Edge Impulse account](https://studio.edgeimpulse.com/signup). A free developer profile can push learning blocks.</li><li>The Edge Impulse CLI: `npm install -g edge-impulse-cli`, which needs Node.js 16 or later (22 recommended).</li><li>An image classification project with labelled images and an impulse that has an **Image** processing block in RGB. Generate its features before you train.</li></ul> |
| Run on your project's data locally | <ul><li>Everything for "Train in Studio".</li><li>Docker, for `edge-impulse-blocks runner`.</li></ul> |
| Get the files out of Studio as `deploy.zip` | The [deployment block](../deploy-executorch/), which needs an **Enterprise** organization. Without one, run the block locally on your project's data. |
| Run `model_qnn.pte` | <ul><li>A Qualcomm Dragonwing IQ‑9075 EVK. See [board/](../../board/README.md).</li><li>For another chip, see [retargeting](../../docs/retargeting.md).</li></ul> |

## Files

| File | Purpose |
|---|---|
| `parameters.json` | Block metadata and the settings shown in Studio |
| `train.py` | Entry point: loads Studio's data, trains, exports ONNX, `.pte` and QNN `.pte` |
| `model.py` | `PretrainedClassifier` (MobileNetV2), `SmallCNN` (from scratch), `ClassifierExport` (with or without softmax for export) |
| `qnn_export.py` | PT2E quantization with `QnnQuantizer`, then QNN lowering and the export report (never raises) |
| `qnn_smoke.py` | Lowers a tiny network to QNN **during `docker build`**, so a broken SDK setup fails the image build and not your training job |
| `Dockerfile` | x86_64 image with torch, ExecuTorch 1.5.1, the ImageNet weights and the QNN SDK built in |
| `tests/` | Offline unit and end-to-end tests (no network, no hardware) |

## Settings in Studio

| Setting | Flag | Default | Notes |
|---|---|---|---|
| Number of training epochs | `--epochs` | 30 | |
| Learning rate | `--learning-rate` | 0.0001 | Use about 0.001 with **Train from scratch** |
| Batch size | `--batch-size` | 32 | |
| Train from scratch | `--no-pretrained` | off | Uses `SmallCNN` (~25K params). Needs much more data. |
| Freeze backbone | `--freeze-backbone` | off | Trains only the head |
| Disable data augmentation | `--no-augment` | off | Augmentation: flip, affine, color jitter |
| Export ExecuTorch .pte | `--export-pte` | **on** | CPU `model.pte` (XNNPACK) |
| **Export QNN .pte (Qualcomm NPU)** | `--export-qnn` | off | Writes `model_qnn.pte` and `export_report.json` |
| QNN SoC model | `--soc-model` | `QCS9100` | Use `QCS9100` for the IQ‑9075 (Hexagon v73) |
| QNN quantization | `--quant-scheme` | `8a8w` | Also `16a8w`, `16a16w`, `16a4w` |
| QNN calibration samples | `--calibration-samples` | 200 | Random training images, fixed seed |

## How it works

```text
Studio data (NHWC float32, 0..1)
   │  to_nchw()  ── contiguous NCHW copy (XNNPACK rejects channels-last)
   ▼
MobileNetV2 (ImageNet weights, new head) ── train with Adam + cross-entropy, keep best epoch
   │
   ├─► ClassifierExport(softmax=True) ─► torch.onnx.export ───────────────► model.onnx
   ├─► ClassifierExport(softmax=True) ─► torch.export ─► XnnpackPartitioner ─► model.pte
   └─► ClassifierExport(softmax=False)            (only with --export-qnn)
         ─► torch.export ─► prepare_pt2e(QnnQuantizer 8a8w, per-channel)
         ─► calibrate on 200 training images ─► convert_pt2e
         ─► accuracy: float32 vs int8 (simulated on CPU) on the test split
         ─► to_edge_transform_and_lower_to_qnn(QCS9100 HTP spec)
         ─► delegation check (is QnnBackend in the graph?) ─► model_qnn.pte + export_report.json
```

Key points:

- **Logits, not softmax, for the NPU.** An int8 softmax output has only 256 levels, so its probabilities are very coarse. The NPU model outputs logits and the device applies softmax. The CPU and ONNX models keep softmax built in, because Studio's evaluator expects probabilities.
- **`status: "ok"` only if the NPU is actually used.** `export_qnn` checks that `QnnBackend` appears in the lowered graph. A model where every op fell back to the CPU is reported as `failed` and no file is written.
- **QNN lowering is x86_64 Linux only.** ExecuTorch's QNN ahead-of-time compiler ships only in the x86_64 wheel. The Dockerfile pins `--platform=linux/amd64`, which matters on Apple Silicon.
- **The SDK is built into the image.** Custom blocks have no network access while they run. ExecuTorch normally downloads the Qualcomm AI Runtime SDK (2.37.0.250724, about 1.3 GB) plus a libc++ (about 575 MB) on first use, so the Dockerfile triggers that download during `docker build`. It also sets `QNN_SDK_ROOT` and `LD_LIBRARY_PATH` with `ENV`: the loader reads `LD_LIBRARY_PATH` only when a process starts, so setting it from Python has no effect.

### `export_report.json`

This is the shape of the report. The numbers are from the verified rock‑paper‑scissors run, and `…` marks values not recorded.

```json
{
  "backend": "qnn-htp",
  "status": "ok",
  "soc_model": "QCS9100",
  "quant_scheme": "8a8w",
  "executorch_qnn_sdk": "2.37.0.250724",
  "calibration_samples": 200,
  "accuracy": { "float32": 0.9744, "int8_fake_quant": 0.9744, "delta_pp": 0.0, "n": 78,
                "note": "Quantized graph run on CPU (simulated). Final on-NPU accuracy must be re-checked on the device." },
  "delegation": { "qnn_delegate_present": true, "delegated_subgraphs": "…", "delegated_nodes": "…",
                  "non_delegated_nodes": "…", "non_delegated_ops": ["…"] },
  "output": "model_qnn.pte", "reason": null, "host": "Linux x86_64", "seconds": "…",
  "size_kb": "≈2600",
  "input_shape": [1, 3, 96, 96],
  "labels": ["Paper", "Rock", "Scissors", "unknown"],
  "input_layout": "NCHW", "input_dtype": "float32", "input_scaling": "0..1",
  "output_semantics": "logits (apply softmax on host)"
}
```

The device reads `labels` and `input_shape` from this file. The board's accuracy script compares its on-NPU result with the `accuracy` block.

## Run it locally

### Tests: no Docker, no network

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q
```

With the pinned torch 2.12.1, expect 7 passed and 1 skipped. The skipped step loads the `.pte` back into the ExecuTorch 1.5.1 runtime, which needs torch 2.14 (see [docs/versions.md](../../docs/versions.md)). Install `torch==2.14.0 "torchvision==0.29.*"` to run it as well.

The tests cover:
- model shapes and the contiguous NCHW conversion
- the softmax and logits export variants
- the contract that the QNN export reports failure and never raises
- an end-to-end run of `train.py` on synthetic data that exports a `.pte`, checks that the whole graph is in one XNNPACK delegate, and, under torch 2.14, runs it in the ExecuTorch runtime

### The full block in Docker, including the NPU export

**With fake data** (no Edge Impulse account needed):

```bash
docker build -t ei-executorch-qnn .            # x86_64; the build runs qnn_smoke.py
python ../../tools/make_fake_data.py /tmp/ei-data --size 96 --classes 4
mkdir -p /tmp/ei-out
docker run --rm -v /tmp/ei-data:/data -v /tmp/ei-out:/out ei-executorch-qnn \
    --data-directory /data --out-directory /out \
    --epochs 2 --export-pte --export-qnn
ls -l /tmp/ei-out        # model.onnx  model.pte  model_qnn.pte  export_report.json
```

Accuracy on random data is meaningless. This run only proves the pipeline works. Note that `--export-pte` is off by default on the command line, though it is on by default in Studio.

**Labels.** Studio passes `--info-file train_input.json`, which is how the report gets real class names. Without it, as in these local runs, `labels` in the report are indices. Edge Impulse orders classes alphabetically, so write `labels.txt` in that order.

**With your Studio project's data:**

```bash
edge-impulse-blocks runner --download-data          # downloads your project's processed features under ./ei-block-data/
DATA=$(dirname "$(find ei-block-data -name X_split_train.npy | head -1)")
mkdir -p out
docker run --rm -v "$PWD/$DATA":/data -v "$PWD/out":/out ei-executorch-qnn \
    --data-directory /data --out-directory /out --epochs 30 --export-pte --export-qnn
```

This is also how to get `model_qnn.pte` **without an Enterprise organization**: the deployment block needs Enterprise, but a learning block run locally writes everything to `out/`. Running `edge-impulse-blocks runner` without `--download-data` also builds and runs the block for you. Pass the block's flags with `--extra-args "--export-pte --export-qnn"`.

## Push it to Edge Impulse

```bash
edge-impulse-blocks init     # first time: pick your org/profile; writes .ei-block-config (gitignored)
edge-impulse-blocks push
```

Learning blocks can be pushed to a free developer profile.

In Studio:
1. **Create impulse**:
   - image input 96×96 with resize mode **Fit shortest axis** (a center crop, which matches the demo's preprocessing)
   - an **Image** processing block in RGB
   - **PyTorch image classifier (ExecuTorch + QNN)** as the learning block
2. Train with **Export QNN .pte** checked. The log shows lines like:
   ```text
   [qnn] accuracy: float32 0.9744 | 8a8w 0.9744 | delta +0.00 pp
   [qnn] delegation: {...}
   Saved QNN ExecuTorch program to /out/model_qnn.pte (... KB)
   QNN export status: ok
   ```
3. Get the files out with the [deployment block](../deploy-executorch/), or with the local runner above.

> **Keeping a working pipeline safe.** A pushed block's identity is the `id` in `.ei-block-config`. To try changes without touching a block a project already uses, copy the folder, delete `.ei-block-config`, change `info.name` in `parameters.json`, and push. That creates a new block.

## Limitations

- **Calibration and quantization.** Calibration uses random training images, and quantization is post-training only (no QAT). If accuracy drops, try more calibration samples, then `16a8w`.
- **Fixed shapes.** The input shape and batch size are fixed (batch 1), which the HTP needs.
- **Optimistic accuracy.** `train.py` picks the best epoch on the test split, so the reported accuracy is optimistic. Studio's **Model testing** uses ONNX/TFLite, not the `.pte`.
- **One chipset per file.** A QNN `.pte` is compiled for one SoC and one SDK range. The CPU `model.pte` is the portable one.
- **Large image.** The Docker image is large (several GB) because of the SDK.
