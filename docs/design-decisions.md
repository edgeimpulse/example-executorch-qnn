# Design decisions

Each entry gives what was chosen, why, the trade-offs, and the alternatives that were weighed. They are ordered from the core block design outwards to the demo.

---

### 1. ExecuTorch as the NPU path, not Edge Impulse's existing QNN `.eim`

**Chosen.** Lower the PyTorch model to an ExecuTorch `.pte` with the QNN backend.

**Why.** The aim of the project is a **PyTorch-native** path: a PyTorch developer trains in Studio and gets an ExecuTorch artifact, with no TFLite conversion in between. The NPU target was meant to be "the same block, one more flag."

| Pros | Cons |
|---|---|
| One source graph (`torch.export`), two outputs: portable CPU `.pte` and NPU `.pte` | Edge Impulse already has a productized NPU path for this board (`.eim` through the TFLite QNN delegate), and it is one click |
| No PyTorch → ONNX → TFLite conversion for the deployed artifact | The QNN SDK version is pinned by the ExecuTorch release (2.37), and the runtime must be at least that version |
| Partial delegation with CPU fallback inside one program; small C++ runtime | No aarch64 QNN runtime wheel, so the board needs a source build |
| Upstream Qualcomm maintains the ExecuTorch QNN backend; same API on Android Snapdragon | Not a first-class Studio deployment target yet: needs custom blocks plus Enterprise for the deployment block |

**Alternatives considered:**
- **`.eim` with QNN.** The right answer for TFLite or Keras users. It does not serve the "PyTorch in, ExecuTorch out" story.
- **ONNX Runtime with the QNN execution provider.** The block already emits ONNX, and Qualcomm now ships aarch64 `onnxruntime-qnn` wheels. Rejected because it moves the runtime off ExecuTorch, which was the goal.
- **Qualcomm AI Hub.** Cloud compile and profile, with targets TFLite, ONNX and QNN context binaries, but no ExecuTorch target. It also needs network access, which a block doesn't have at runtime.
- **QAIRT/QNN directly.** The most control, but no PyTorch-native flow, and you write the runtime glue yourself.

### 2. Lower to QNN in the **learning** block, not the deployment block

**Why.** The learning block has the trained `nn.Module`, the calibration images and the test split. Those are exactly what PT2E quantization and an honest accuracy check need. The deployment block only has ONNX, which it would have to convert back to PyTorch.

- **Pros:**
  - A native graph, with no round trip.
  - Quantization accuracy is measured where the model is trained.
- **Cons:**
  - Every training run pays for the lowering (seconds to minutes).
  - The learning-block image carries the SDK (about 1.9 GB more).
- **Alternative:** a QNN option in the deployment block. That would let the user choose the target at deploy time, which is closer to how Studio works. 

### 3. Fine-tune ImageNet MobileNetV2 instead of training a small CNN from scratch

**Why.** The original `SmallCNN` (about 25K parameters, random initialisation) reached **47.5%** on a few hundred hand images. Studio's own transfer learning reached 87% on the same data. Fine-tuning MobileNetV2 brought the block to 88–97%.

- **Pros:**
  - Works on small datasets.
  - It is ExecuTorch's own XNNPACK example model.
  - Its ops (convolution, ReLU6, pooling, linear) lower cleanly to both XNNPACK and HTP.
  - torchvision's BSD licence.
- **Cons:**
  - About 3.5M parameters: 8.9 MB float, about 2.5 MB int8.
  - Too large for an MCU.
- **Alternatives:** timm (more dependencies, same result). `SmallCNN` is still available with `--no-pretrained`.

### 4. 8a8w per-channel post-training quantization, 200 calibration images

**Why.** HTP needs static quantization. 8a8w is ExecuTorch's default and fastest scheme.

- **Result:** float32 0.9744, NPU 0.9744. **Nothing lost** on 78 test images.
- **Cons:**
  - Calibration images are random, not stratified.
  - Post-training quantization only, no QAT.
- **What to try if accuracy drops:**
  1. More calibration samples.
  2. `16a8w` (a block setting).
  3. Leave the last layer unquantized.
  4. QAT.
- **Alternative:** fp16 on HTP (`use_fp16=True`). It is only supported on newer SoCs and was not tried.

### 5. Compile for `QcomChipset.QCS9100`

**Why.** The IQ‑9075 is not listed by name in ExecuTorch's chipset enum. QCS9100 is the same silicon family (SA8775P), with **Hexagon v73** and 8 MB VTCM. The board's Qualcomm packages ship the v73 skel libraries under `sa8775p/`.

- **Status:** confirmed on the device. The model loads, runs and matches the expected accuracy.

### 6. The NPU model outputs logits; softmax runs on the host

**Why.** An int8-quantized softmax has 256 levels, which makes the probabilities too coarse for confidence gates. The CPU and ONNX models keep softmax inside, because Studio's evaluator expects probabilities.

- **Cost:** the two artifacts mean different things. `export_report.json` records `output_semantics`, and the app handles each backend correctly.

### 7. The QNN export is optional and never fails training; a build-time smoke test guards the setup

**Why.** NPU export is an extra. A user who only wants the CPU model should never lose a training run to an SDK problem.
- `export_qnn` turns every exception into a report with `status: failed` and a `reason`.
- Setup problems are caught earlier by `qnn_smoke.py` during `docker build`, so a broken image never reaches Studio.
- `status: ok` requires `QnnBackend` in the graph, so a model where every op silently ran on the CPU can't pass as an NPU model.

### 8. Build the Qualcomm SDK into the Docker image

**Why.** Blocks have no network when they run, and ExecuTorch downloads the SDK on first use.

- **Pros:**
  - Deterministic: the SDK version is fixed by the image.
  - Studio's builder can reach Qualcomm's download server at build time (verified).
- **Cons:**
  - The image is several GB.
  - Each push rebuilds a large layer.
  - Redistribution terms for the Qualcomm SDK inside an image should be checked before publishing the image itself.

### 9. Pass the NPU program through the deployment block

**Why.** Studio does not show learning-block output files in the UI. A deployment is the only supported way to download them, and `mountLearnBlock` exposes them at `/data`.

- **Pros:**
  - The NPU artifact is exactly what the learning block measured.
- **Cons:**
  - The deployment block builds its own CPU model from ONNX in parallel. It could ship the learning block's native float32 `model.pte` instead. 


### 10. New block IDs for the QNN versions

**Why.** It keeps the working CPU pipeline untouched while QNN was being developed. A block's identity is the `id` in `.ei-block-config`. Copying the folder, deleting that file and renaming `info.name` creates a new block on `push`.

### 11. Build ExecuTorch natively on the board against the apt QNN headers

**Why.** The board already had g++ 13.3 and `libqnn-dev` 2.46. Compiling the backend only needs headers, so a `QNN_SDK_ROOT` shim pointing at `/usr/include` was enough.

- **Pros:**
  - No cross toolchain.
  - No 1.3 GB SDK download on the board.
  - No version juggling between the SDK and the runtime on the board.
- **Cons:**
  - 30–60 minutes on the board. Mitigated by `board/package.sh`: build once, and other boards with the same OS install the result with `board/install-prebuilt.sh` in about a minute.
  - It needs CMake ≥ 3.29 (from pip), abseil and gflags.
  - The LLM example runners have to be disabled.
- **Alternatives:**
  - Cross-compile with `backends/qualcomm/scripts/build.sh --enable_linux_embedded`. This is the documented route for OE Linux and needs a target sysroot.
  - Build the Python bindings with QNN (the original "Route A"). The build files only support QNN's Python adaptor on x86, so it was not attempted.

### 12. A C++ sidecar over a UNIX socket for the NPU, separate from the app

**Why.** `qnn_executor_runner` only processes batches of files, the board's Python ExecuTorch cannot load QNN, and the app needs a model that stays loaded at camera rate.

- **Pros:**
  - The Python app never loads QNN or needs its library paths.
  - It is isolated from llama.cpp's environment.
  - The model is loaded and warmed up once.
  - systemd gates the service on a real NPU self-test and restarts it if it dies.
  - Any language can call it with 12-byte headers.
- **Cons:**
  - About 0.4–0.5 ms extra per frame to copy 110 KB over the socket (0.70 ms round trip vs 0.23 ms on the NPU). That is negligible against a 33 ms frame.
  - One more process to deploy.
- **Alternatives:**
  - In-process Python bindings with QNN (a source build of the wheel).
  - Shared memory or `rpcmem` zero-copy, if latency ever matters.
  - gRPC or HTTP, which add overhead and aren't needed on one host.

### 13. Per-frame CPU fallback, with the CPU model loaded eagerly

**Why.** The NPU service dying mid-round must not stop the game. `HybridClassifier` answers that frame from the CPU and retries the NPU every 5 s. The CPU model is loaded at startup, because loading it at the first failure would stall at the worst moment. It also anchors the label-count check.

### 14. Classification with confidence gates, not object detection, for the demo

**Why.**
- The classification block was solid: 92–97%.
- The ExecuTorch FOMO detection block, after a rewrite on a pretrained backbone, matched Studio's FOMO offline (about 63% F1) but got only 4 of 15 right in a live trial on the board.
- YOLO via Ultralytics was ruled out (AGPL licence).


- **Cost:**
  - A classifier has no "is there a hand?" signal, so empty frames still get a class. The gates (smoothing, confidence, margin, maximum `unknown`) handle this, and a clean, varied `unknown` class is the real fix.


### 15. The LLM on the NPU through llama.cpp's Hexagon backend (Q4_0)

**Why.** Qualcomm's toolchain image cross-compiles llama.cpp with `GGML_HEXAGON`, and Q4_0 is the format the HTP kernels are built for.

**Measured on the IQ‑9075 (Llama 3.2 1B Q4_0):**

| | NPU | CPU | |
|---|---|---|---|
| Prompt processing | **1619 t/s** | 282 t/s | 5.7× faster on the NPU |
| Token generation | 45 t/s | **57 t/s** | 25% faster on the CPU |

So the NPU is not used for speed here. It is used to keep the CPU free for the camera, MJPEG encoding and the fallback.

- **Alternatives:** Ollama on the CPU (the fallback), OpenCL on the Adreno GPU.


### 16. Two artifacts, two claims

`model.pte` (XNNPACK) is portable: one file runs on any Arm or x86 CPU with ExecuTorch. `model_qnn.pte` contains a context binary for **one chipset and SDK range**. The honest claim is "one training run, two artifacts", not "the same file runs on the NPU and on a phone."
