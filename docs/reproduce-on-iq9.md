# Reproduce it end to end

This guide goes from an empty Studio project to the game running on the IQ-9075 NPU.

## What you need

| | Required for |
|---|---|
| An Edge Impulse account (a free developer profile is enough for the learning block) | Training |
| An Edge Impulse **Enterprise** org | The deployment block. Optional: you can run the learning block locally instead. |
| An x86_64 machine with Docker (Linux, or a Mac with Rosetta enabled in Docker Desktop) | Building and testing the blocks |
| A Qualcomm Dragonwing **IQ-9075 EVK** with Ubuntu 24.04, network access, a monitor or SSH | The NPU |
| *Optional:* a USB webcam (V4L2, e.g. Logitech C920) | Running an application against the service |

## 1. Train in Studio

1. **Create an image project** and collect or upload images per class. This project used Paper, Rock, Scissors and **unknown**:
   - About 100 images per class.
   - `unknown` should cover empty scenes, faces, torsos and hands at rest, in varied light.
   - Use the camera and distance of the final setup where you can.
2. **Push the learning block** from your x86 machine:
   ```bash
   cd blocks/learning-image-classifier
   edge-impulse-blocks init      # pick your profile/org
   edge-impulse-blocks push      # the image build takes a while: it downloads ~1.9 GB of QNN SDK and libc++ and runs the QNN smoke test
   ```
3. **Create the impulse:**
   - Image data at 96×96, resize mode **Fit shortest axis**.
   - An **Image** processing block (RGB), then **Generate features**.
   - Learning block: **PyTorch image classifier (ExecuTorch + QNN)**.
4. **Train** with these settings:
   - **Export ExecuTorch .pte**: on
   - **Export QNN .pte**: on
   - **QNN SoC model**: `QCS9100`
   - **QNN quantization**: `8a8w`
   - **Calibration samples**: 200

   In the training log, check for:
   ```text
   [qnn] accuracy: float32 0.97… | 8a8w 0.97… | delta … pp
   QNN export status: ok
   ```

## 2. Get the files out

**Option A: deployment block (Enterprise).**

```bash
cd blocks/deploy-executorch && edge-impulse-blocks init && edge-impulse-blocks push
```

In Studio, go to **Deployment**, pick **ExecuTorch .pte (CPU int8 + QNN pass-through)**, choose **Quantized (int8)**, and click **Build**. Download `deploy.zip`. It contains:
- `model.pte` (CPU int8)
- `model_qnn.pte` and `export_report.json`
- `labels.txt`
- `_inputs/X_split_test.npy` and `_inputs/Y_split_test.npy`

**Option B: run the learning block locally (any plan).**

```bash
cd blocks/learning-image-classifier
edge-impulse-blocks runner --download-data               # features land under ./ei-block-data/
DATA=$(dirname "$(find ei-block-data -name X_split_train.npy | head -1)")
docker build -t ei-executorch-qnn .
mkdir -p out          # create it yourself, or Docker creates it owned by root
docker run --rm -v "$PWD/$DATA":/data -v "$PWD/out":/out ei-executorch-qnn \
    --data-directory /data --out-directory /out --epochs 30 --export-pte --export-qnn
```

You get `out/model.pte`, `out/model_qnn.pte` and `out/export_report.json`. The test split (`X_split_test.npy`, `Y_split_test.npy`) is in `$DATA`. Write `labels.txt` yourself: the class names in **alphabetical order**, one per line (for this project: `Paper`, `Rock`, `Scissors`, `unknown`). That is the order of Edge Impulse's one-hot labels. Without Studio's `--info-file`, `export_report.json` → `labels` contains only indices (`"0"`, `"1"`, …). Put `labels.txt` in `~/npu-model/` with the other files.

**Optional sanity checks on your laptop**, using the torch 2.14 + ExecuTorch 1.5.1 venv from [tools/README.md](../tools/README.md), still in `blocks/learning-image-classifier`:

```bash
python ../../tools/inspect_pte.py out/model.pte out/model_qnn.pte     # XnnpackBackend vs QnnBackend
python ../../tools/evaluate_pte.py out/model.pte $DATA/X_split_test.npy $DATA/Y_split_test.npy
```

## 3. Prepare the board

On the EVK:

```bash
git clone <this repository's URL> ~/edge-impulse-executorch-qnn      # the whole repo: board/ uses demo/'s client
mkdir -p ~/npu-model     # copy model_qnn.pte, export_report.json, X_split_test.npy, Y_split_test.npy here
```

The board scripts install the Qualcomm QNN packages (Ubuntu's IoT PPA) and the build tools themselves.

## 4. Build the NPU runtime and install the service

Details are in [board/README.md](../board/README.md).

```bash
cd ~/edge-impulse-executorch-qnn/board
./setup.sh  ~/npu-model/model_qnn.pte ~/npu-model/export_report.json   # 30–60 min the first time
./verify.sh ~/npu-model
```

`setup.sh` installs the prerequisites, builds ExecuTorch v1.5.1 with QNN natively, builds `npu_server`, self-tests the model on the NPU and installs the `rps-npu` service. `verify.sh` then runs one inference, the whole test split on the NPU, and the test split through the service. The NPU accuracy should equal `accuracy.int8_fake_quant` in the report. That was 0.9744 in our run.

**No build:** if you have a runtime package from `./package.sh` on another IQ-9075 with the same OS, run `./install-prebuilt.sh <package.tar.gz> ~/npu-model/model_qnn.pte ~/npu-model/export_report.json` instead of `setup.sh`.

## 5. Use the model from an application

At this point the NPU runs your model and `verify.sh` has proved it. Anything
that wants inferences talks to the `rps-npu` service over its UNIX socket.

```bash
# the service is already running after step 4
systemctl status rps-npu
journalctl -u rps-npu | tail             # FastRPC "domain 3" lines mean the cDSP

# call it from Python with the client in this repository
~/npu-tools/bin/python board/steps/test_npu_service.py \
    ~/npu-model/X_split_test.npy ~/npu-model/Y_split_test.npy
```

`board/steps/npu_client.py` is that client, about 120 lines depending only on
numpy. Copy it into your application, or use it as a reference. The README's
[Using it in your own application](../README.md#using-it-in-your-own-application)
section shows the calling pattern and the CPU-fallback behaviour.

A complete application built this way — a camera loop, per-frame fallback, and
an Android app running the portable `model.pte` — is in the separate
**demo-executorch-qnn** repository.

## 6. Checklist

- [ ] Training log: `QNN export status: ok`, delta <= 1-2 points
- [ ] `inspect_pte.py model_qnn.pte` -> `delegates : QnnBackend`
- [ ] `verify.sh`: all checks PASS/DONE, and the NPU accuracy matches the report's int8 accuracy
- [ ] `systemctl status rps-npu` shows active; `journalctl -u rps-npu` shows `domain 3` FastRPC lines
- [ ] `test_npu_service.py` reports the same accuracy as `verify.sh`, and a per-call latency you are happy with
- [ ] `sudo systemctl stop rps-npu` -> your client's calls return `None`, so it can fall back to `model.pte`. Restart it -> calls succeed again.
