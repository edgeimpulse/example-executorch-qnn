# Reproduce it end to end

This guide goes from an empty Studio project to the game running on the IQ-9075 NPU.

## What you need

| | Required for |
|---|---|
| An Edge Impulse account (a free developer profile is enough for the learning block) | Training |
| An Edge Impulse **Enterprise** org | The deployment block. Optional: you can run the learning block locally instead. |
| An x86_64 machine with Docker (Linux, or a Mac with Rosetta enabled in Docker Desktop) | Building and testing the blocks, and cross-compiling llama.cpp |
| A Qualcomm Dragonwing **IQ-9075 EVK** with Ubuntu 24.04, network access, a monitor or SSH | The NPU |
| A USB webcam (V4L2, e.g. Logitech C920) | The demo |
| *Optional:* a Pollen Robotics [Amazing Hand](https://github.com/pollen-robotics/AmazingHand) (8× Feetech SCS0009, serial bus adapter on `/dev/ttyACM0`) | The robot opponent |
| *Optional:* a speaker plus a TTS container | Spoken commentary  |

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

## 5. Run the demo

```bash
cd ~/edge-impulse-executorch-qnn/demo/rps-amazing-hand
cp ~/npu-model/labels.txt models/labels.txt     # must match the model's output order
# Optional: replace models/model.pte with the CPU model from the same training run
./install.sh
./start.sh                 # pick the camera; open http://<evk-ip>:8000
```

Look for these startup lines:

```text
NPU: rps-npu service socket found at /run/rps-npu/npu.sock.
detector: NPU ACTIVE ([1, 3, 96, 96])
```

**Optional extras.** Each is independent, and each is skipped with a message naming what is missing, so add them one at a time and confirm with the check script before moving on. Background for the last two is in [speech and actuation](speech-and-actuation.md).

- **LLM commentary.** On the x86 machine, clone llama.cpp, check out the tested build (`git checkout 6a2743f02`), build `pkg-linux.zip` with `demo/rps-amazing-hand/scripts/build-llama-hexagon.sh`, and unzip it to `~/pkg-linux` on the board. Then run `./scripts/fetch-llm-model.sh` and `./scripts/check-llama-hexagon.sh models/Llama-3.2-1B-Instruct-Q4_0.gguf`. `start.sh` picks it up automatically.

- **Spoken commentary.** Runs on the CPU, locally, with no container.

  ```bash
  ./scripts/fetch-tts-voice.sh        # a voice is two files: .onnx and .onnx.json
  ./scripts/check-audio.sh            # player, output device and voice, tested apart
  ```

  `install.sh` already added the engine (`SKIP_SPEECH=1` skips it). If the board has no PCM player, `check-audio.sh` says so: `sudo apt install alsa-utils` is the smallest fix. If every stage passes but you hear nothing, the player opened the wrong output — `./scripts/check-audio.sh --list`, then `--device <name>`, and make it permanent with `TTS_AUDIO_DEVICE`. `start.sh` then starts the Piper server itself.

- **Amazing Hand.**
  1. Add your user to the serial port's group: `sudo usermod -aG dialout $USER`, then log in again (or `newgrp dialout`).
  2. Verify it before involving the web app:

     ```bash
     .venv/bin/python scripts/test-hand.py
     ```

     This checks the port and permissions, then drives the real actuator through every pose. A clean exit only proves the servos accepted the commands, so watch the hand: paper should be flat with all four fingers extended, and scissors should show exactly two.
  3. If a finger sits visibly wrong, adjust that motor's offset in `AMAZING_HAND_MIDDLE_CALIBRATION` (`app/config.py`). Order is motors 1–8 = index, index, middle, middle, ring, ring, thumb, thumb. Try it with `--cal 15,-13,0,-12,0,3,10,-10` first. The committed values are for a left hand (`AMAZING_HAND_SIDE = 2`).
  4. Run `ENABLE_AMAZING_HAND=1 ./start.sh`.

## 6. Checklist

- [ ] Training log: `QNN export status: ok`, delta ≤ 1–2 points
- [ ] `inspect_pte.py model_qnn.pte` → `delegates : QnnBackend`
- [ ] `verify.sh`: all checks PASS/DONE, and the NPU accuracy matches the report's int8 accuracy
- [ ] `systemctl status rps-npu` shows active; `journalctl -u rps-npu` shows `domain 3` FastRPC lines
- [ ] The app prints `NPU ACTIVE`. `sudo systemctl stop rps-npu` → `CPU FALLBACK`, and the game continues. Restart it → `NPU ACTIVE` again.
- [ ] `RPS_LOG_PROBS=1 ./start.sh`: an empty box is rejected, and clear gestures are accepted

If you added the optional extras:

- [ ] `check-llama-hexagon.sh` shows an HTP row, and the app starts with commentary enabled
- [ ] `check-audio.sh` passes all three stages **and** you can hear both the tone and the spoken line
- [ ] `test-hand.py` completes, and paper, rock and scissors each look right on the physical hand
- [ ] One full match with everything on: the voice tracks the on-screen countdown, the hand throws on the shoot beat, and detection still reports `NPU ACTIVE` afterwards
