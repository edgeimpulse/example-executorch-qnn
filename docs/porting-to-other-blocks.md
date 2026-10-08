# Add the NPU export to the other ExecuTorch blocks

The same six changes that gave the image classifier a NPU output work for Edge Impulse's other public ExecuTorch learning blocks:
- [timeseries](https://github.com/edgeimpulse/executorch-pytorch-timeseries-block)
- [keyword spotting (KWS)](https://github.com/edgeimpulse/executorch-pytorch-kws-block)
- [FOMO object detection](https://github.com/edgeimpulse/executorch-pytorch-object-detection-fomo-block)

One command applies them. The deployment block and the board runtime need no changes. What differs per block is everything *around* the model: how the device produces the model's input, and how it reads the output.

This guide starts from those repositories **as published**, on ExecuTorch 0.4.0 and torch 2.5.0.

## At a glance

| Block | Model (input) | Port | NPU benefit | Extra work on the device |
|---|---|---|---|---|
| KWS | 2D CNN on an MFE/MFCC spectrogram `[1, 1, frames, coeffs]` | `apply_port.py` | Modest | Compute the **same MFE/MFCC features** Studio computed, from live audio |
| FOMO | Fully convolutional, per-cell classes `[1, 3, H, W] → [1, C, H/8, W/8]` | `apply_port.py` (FOMO recipe) | **Best of the three:** the largest model and inputs | Softmax per cell, then decode cells into object centroids |
| Timeseries, MLP | `[1, F]` features from any DSP block | `apply_port.py` | Little: the model has about 4K parameters, so the fixed cost of each NPU call outweighs the compute | Compute the **same DSP features** (e.g. spectral analysis) on the device |
| Timeseries, Conv1d | `[1, axes, samples]` raw window | `apply_port.py` | Little (about 8K parameters) | Window the raw sensor stream the way Studio did |

What was checked in this repository, without hardware, for all four models:
- They port cleanly, train, and export ONNX and a CPU `.pte`.
- They quantize with `QnnQuantizer` 8a8w for `QCS9100`, and the report shows float32 vs int8.
- Their own unit tests still pass.
- The real `npu_server` (CPU build) serves them, and the board's test script reproduces the training accuracy.

What was **not** run: the final QNN lowering. The build-time smoke test and the report's `delegation` section tell you on the first real run.

## Before you start

- An x86_64 machine with Docker. The QNN compiler ships only in the x86_64 ExecuTorch wheel.
- This repository cloned next to the block you are porting.
- An Edge Impulse project of the matching type: time series with a DSP block, audio with MFE/MFCC, or object detection with FOMO-style labels.
- For the board: the IQ‑9075 set up as in [board/README.md](../board/README.md). The ExecuTorch + QNN build there is shared by every block.

## Option A: one command

From the folder that holds both this repository and the block:

```bash
git clone https://github.com/edgeimpulse/executorch-pytorch-kws-block
python edge-impulse-executorch-qnn/porting/apply_port.py executorch-pytorch-kws-block
cd executorch-pytorch-kws-block && git status && git diff     # two files are new, four changed
```

[`porting/apply_port.py`](../porting/apply_port.py):
- recognizes FOMO by its `FomoExport` wrapper and uses the FOMO recipe for it
- refuses to run twice
- checks every edit point before writing anything: if the block has changed upstream, it names what it couldn't find and modifies nothing

The same changes are also available as plain patches, for review or for `git apply`: [`porting/patches/`](../porting/patches/). They were generated against the repositories as published in October 2026. If you are unsure, use `apply_port.py`: it always copies the current `qnn_export.py`.

## Option B: by hand (what changes, and why)

Every file mentioned is in [`porting/kit/`](../porting/kit/) or [`blocks/learning-image-classifier/`](../blocks/learning-image-classifier/).

**1. `requirements.txt`: move to ExecuTorch 1.5.1.**
Copy `porting/kit/requirements.txt` (torch 2.12.1, numpy 2.0.2, onnx 1.17.0, executorch 1.5.1). There are two reasons:
- 0.4.0 has none of the QNN quantizer and lowering APIs used here.
- The board runtime is built from the 1.5.1 tag, and a `.pte` must be run by the release that made it.

These are the same pins that are verified in Studio for the image block (see [versions](versions.md)).

**2. `train.py`: add `dynamo=False` to `torch.onnx.export`.**
Since torch 2.9, `torch.onnx.export` uses the new dynamo-based exporter by default, and that exporter needs the `onnxscript` package. Without this line, the upgraded block fails with `No module named 'onnxscript'`. This was confirmed on all three public blocks. One line, inside `export_onnx`:

```python
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,  # torch >= 2.9 defaults to the new exporter, which needs onnxscript
```

**3. Copy `qnn_export.py` and `qnn_smoke.py`** from `blocks/learning-image-classifier/`, unchanged. `qnn_export()` is task-agnostic:
- It takes any single float32 input layout.
- It reports top-1 accuracy for classifiers, or the score from an `eval_fn` you pass for any other task.
- It never raises, so training always succeeds and `export_report.json` explains any NPU failure.

**4. `Dockerfile`: replace it with `porting/kit/Dockerfile`.** Compared with the public one, it:
- pins `linux/amd64`
- adds `libatomic1`
- downloads the QNN SDK (about 1.9 GB with libc++) at **build** time, because blocks have no network at run time
- sets `QNN_SDK_ROOT` and `LD_LIBRARY_PATH` with `ENV`
- copies the two new files
- runs `qnn_smoke.py`, so a broken SDK setup fails the image build, not a training job

**5. `parameters.json`: append the four entries** in `porting/kit/qnn-parameters.json`. Studio then shows **Export QNN .pte**, **QNN SoC model**, **QNN quantization** and **QNN calibration samples**.

**6. `train.py`: three additions** (all in [`porting/kit/train-snippets.py`](../porting/kit/train-snippets.py)):
- **(a)** Four flags in `parse_args()`, next to `--export-pte`.
- **(b)** Two functions above `main()`: `load_class_names()` and `export_qnn_pte()`. The second picks calibration samples from `x_train` (already in the model's layout), calls `export_qnn()` and prints the status.
- **(c)** One call at the end of `main()`, after the existing `export_pte` call.

For the timeseries and KWS blocks (classifiers), the call in (c) is:

```python
    if args.export_qnn:
        # `model` outputs logits; ClassifierExport only adds the softmax Studio needs.
        has_test = len(x_test) > 0
        export_qnn_pte(
            model, x_train, args,
            eval_x=x_test if has_test else None,
            eval_y=label_test if has_test else None,
            labels=load_class_names(args.info_file, num_classes),
        )
```

For FOMO, it scores the int8 model with the block's own metric (accuracy on object cells) through `eval_fn`:

```python
    if args.export_qnn:
        def object_cell_accuracy(m) -> float:
            correct = total = 0
            for x, t in zip(x_test, t_test):
                pred = m(torch.from_numpy(np.ascontiguousarray(x[None]))).argmax(dim=1)[0]
                target = torch.from_numpy(t)
                fg = target > 0
                total += int(fg.sum())
                correct += int((pred[fg] == target[fg]).sum())
            return correct / total if total else float("nan")

        export_qnn_pte(
            model, x_train, args,
            eval_fn=object_cell_accuracy if len(x_test) else None,
            metric_name="object_cell_accuracy",
            labels=["background"] + load_class_names(args.info_file, num_classes - 1),
        )
```

**The one rule that matters: pass the trained `model`, not the export wrapper.** `ClassifierExport` and `FomoExport` add a softmax for Studio. An int8 softmax output has only 256 levels, so the NPU program outputs **logits** and the device applies softmax.

**For your own metric:** an `eval_fn` receives a callable model. It must only call it, not `.eval()` or `.train()` on it, because the quantized graph refuses those.

## Test it without Studio

[`tools/make_fake_data.py`](../tools/make_fake_data.py) writes data in Studio's format for each block type. Accuracy on random data is meaningless; this run proves the pipeline works.

| Block | Fake data | Block flags |
|---|---|---|
| Timeseries MLP | `--flat 39 --classes 4` | `--model-type mlp` |
| Timeseries Conv1d | `--flat 375 --classes 4` (3 axes × 125 samples) | `--model-type conv1d --axes 3` |
| KWS | `--flat 637 --classes 3` (49 frames × 13 MFCC) | `--columns 13` |
| FOMO | `--size 96 --classes 2 --fomo` | (none) |

```bash
cd executorch-pytorch-kws-block
docker build -t kws-qnn .                               # runs the QNN smoke test
python ../edge-impulse-executorch-qnn/tools/make_fake_data.py /tmp/kws-data --flat 637 --classes 3   # needs numpy
mkdir -p /tmp/kws-out
docker run --rm -v /tmp/kws-data:/data -v /tmp/kws-out:/out kws-qnn \
    --data-directory /data --out-directory /out --epochs 2 --columns 13 --export-pte --export-qnn
```

You should see:

```text
[qnn] accuracy: float32 0.25… | 8a8w 0.25… | delta +0.00 pp        (FOMO: "object_cell_accuracy")
QNN export status: ok
```

The output folder then holds `model.onnx`, `model.pte`, `model_qnn.pte` and `export_report.json`.
- **Check the report's `delegation.non_delegated_ops`.** It should be empty or small: whatever is listed there runs on the CPU, not the NPU.
- **Check the backend.** `python ../edge-impulse-executorch-qnn/tools/inspect_pte.py /tmp/kws-out/model_qnn.pte` should show `QnnBackend`. This needs a Python with `executorch` installed; see [tools/README.md](../tools/README.md).

**Running `train.py` directly, outside the Docker image.** Install the block's `requirements.txt` first. What you get depends on the host:

| Host | QNN result |
|---|---|
| x86_64 Linux, `QNN_SDK_ROOT` unset, internet | ExecuTorch downloads the SDK (about 1.9 GB) on first use, then exports `model_qnn.pte` |
| x86_64 Linux, `QNN_SDK_ROOT` unset, offline | The report holds the float32 vs int8 numbers and the reason lowering was skipped |
| `QNN_SDK_ROOT` set to a folder without the SDK | `failed` straight away, with no numbers. Unset it to get the numbers. |
| macOS or Arm | `failed` straight away. QNN lowering is x86_64 Linux only, so use the Docker image. |

In every case, `model.onnx` and `model.pte` are written and the job succeeds.

## Studio

Push and train exactly as for the image block ([reproduce § 1](reproduce-on-iq9.md#1-train-in-studio)):
1. `edge-impulse-blocks init` (a new block id), then `edge-impulse-blocks push`.
2. In Studio, select the block in the impulse and train with **Export QNN .pte** on.

Labels come from Studio's `--info-file`. Local runs without it write indices.

**Getting the files out works unchanged:**
- Through [the deployment block](../blocks/deploy-executorch/), which copies `model_qnn.pte` and `export_report.json` through for any block. It needs Enterprise.
- Or by running the block locally (any plan).

The deployment block's own CPU int8 conversion now handles Conv1d inputs, which have interleaved sensor axes. That was fixed alongside this guide.

## On the board

Nothing is rebuilt. The ExecuTorch + QNN build from [board/README.md](../board/README.md) runs any of these models, because `npu_server` reads the input shape from the `.pte`.

First, copy the four files into a folder on the board, for example `~/kws/`:
- From Studio's `deploy.zip`: `model_qnn.pte` and `export_report.json` at the top level, plus `X_split_test.npy` and `Y_split_test.npy` from `_inputs/` (`unzip -j`).
- From a local run: `/tmp/kws-out/model_qnn.pte`, `/tmp/kws-out/export_report.json`, and the data folder's `X_split_test.npy` and `Y_split_test.npy`.

Use the report from a run whose QNN export succeeded. Only that report carries `input_shape`.

```bash
cd board
SVC=kws-npu ./setup.sh ~/kws/model_qnn.pte ~/kws/export_report.json   # its own service + socket; the build is reused if it exists
SVC=kws-npu ./verify.sh ~/kws                                          # input shape comes from export_report.json
```

`SVC=kws-npu` gives the model its own `/opt/kws-npu`, systemd unit and `/run/kws-npu/npu.sock`, so it runs alongside `rps-npu`. Leave `SVC` out to replace the RPS model instead.

- `verify.sh`'s accuracy checks (`steps/accuracy_on_npu.py` and `steps/test_npu_service.py`) turn Studio's flat features into the model's layout with one rule that covers every block:
  - `[1, F]` vectors are used as they are.
  - `[1, C, T]` windows come from `(T, C)` interleaved samples.
  - `[1, C, H, W]` images and spectrograms come from `(H, W, C)`.

  This rule was checked against each block's training code, and it reproduces the training accuracy exactly.
- For FOMO, `verify.sh` saves the raw outputs, because its labels are boxes, not classes. The file is `npu_outputs.npy` in the temporary work folder the script prints at the end, with one row of logits per sample. To score it:
  1. Reshape each row to `[C, H/8, W/8]` and take the argmax over `C`.
  2. Build the targets with the block's own `build_label_map()` and `boxes_to_grid()` from its `train.py`.
  3. Compare them on object cells, the same `object_cell_accuracy` the report used.
- Each model gets its own service. `SVC=` sets the name, and the socket is `/run/$SVC/npu.sock`. The RPS demo keeps using `rps-npu`.

**Your application** talks to the service with [`npu_client.py`](../demo/rps-amazing-hand/app/npu_client.py), about 120 lines that depend only on numpy. Give it your service's socket, for example `NpuClient("/run/kws-npu/npu.sock")`:
1. Send one float32 sample in the model's layout.
2. Get the logits back.
3. Apply softmax: over the classes for KWS and timeseries, or over the channel axis of each cell for FOMO, after reshaping to `[C, H/8, W/8]`.

The `HybridClassifier` pattern (NPU first, CPU `.pte` fallback on each call; [`npu_inference.py`](../demo/rps-amazing-hand/app/npu_inference.py)) carries over unchanged. Pass `sock_path`, or set `RPS_NPU_SOCKET`, because its default is the RPS service.

**The real work for KWS and timeseries is the input.** The model expects the **same features** Studio's DSP block produced: MFE or MFCC frames, spectral-analysis values, or a raw window with the same length and axis order. Those have to be computed on the device from live audio or sensor data, with the same parameters. Those are shown on Studio's DSP block page, and in `dsp_metadata.json`, which the deployment block copies into `_inputs/` when Studio provides it. This is independent of the NPU. It is the same open item as the CPU path in the [blog post](https://www.edgeimpulse.com/blog/from-pytorch-to-the-edge-getting-started-with-executorch-and-edge-impulse/), "ship DSP parameters in the deployment package". Until it exists, check the models on the device with `05` and `08` and Studio's test features.

## Known caveats

- **A broken SDK path aborts the process.** If `QNN_SDK_ROOT` points at a folder without the QNN libraries, ExecuTorch's C++ layer aborts. That is not a Python exception, so the "never fails training" rule could not catch it. `qnn_export.py` now checks the path first and reports `failed` instead. The Docker image's smoke test also prevents it.
- **Tiny models.** Every NPU call has a fixed cost: a FastRPC round trip to the DSP, plus the socket. For the image model the service measured 0.70 ms per call from Python, of which 0.23 ms was NPU time. A model with a few thousand parameters computes in far less than that on the CPU, so expect the timeseries models to be faster there. Port them for one consistent pipeline and CPU offload, not speed.
- **Conv1d pooling.** In the CPU `.pte`, XNNPACK leaves the Conv1d model's max-pool on portable CPU kernels (`tools/inspect_pte.py` shows `max_pool2d_with_indices`, `squeeze`, `unsqueeze` outside the delegate). The QNN backend has a builder for that op, so it should stay on the NPU. Confirm it in `export_report.json` → `delegation.non_delegated_ops`.
- **FOMO accuracy.** Studio's FOMO (62% F1 on the RPS data) beat the public `FomoNet`, whose receptive field is about 15 px. A MobileNetV2-backbone rewrite matched Studio offline. Port the version you intend to use; the recipe covers both.
- **Optimistic accuracy.** The public blocks keep the last epoch, while the image block keeps the best epoch on the test split. Either way, add a validation split before trusting the absolute numbers.
- **Patches can go stale.** If the public repositories change, `apply_port.py` names the edit points it cannot find and changes nothing. Port that block by hand with Option B. Use a clean checkout, so `git checkout . && git clean -fd` gets you back.
