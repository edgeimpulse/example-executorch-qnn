# Results

All numbers below were measured on a Qualcomm Dragonwing **IQ-9075 EVK**:
- Ubuntu 24.04
- apt `libqnn1` 2.46.0
- ExecuTorch 1.5.1 built natively

The training runs were in Edge Impulse Studio, Oct 1–2 2026, unless stated otherwise.

## Model and data

| | |
|---|---|
| Task | 4-class image classification: Paper, Rock, Scissors, unknown |
| Dataset | About 400 images (Paper 117, Rock 101, Scissors 91, unknown 91). The test split has 78 images. |
| Model | MobileNetV2 (ImageNet weights), fine-tuned, 96×96 RGB input |
| NPU export | 8a8w PTQ, per-channel, 200 calibration images, `QCS9100`, QNN SDK 2.37.0.250724 |

## Accuracy

| Model | Where it ran | Accuracy (n=78) | Source |
|---|---|---|---|
| float32 PyTorch | Studio learning block (CPU) | **0.9744** | `export_report.json` → `accuracy.float32` |
| int8 (QNN quantization, simulated) | Studio learning block (CPU) | **0.9744** | `export_report.json` → `accuracy.int8_fake_quant` |
| CPU int8 `.pte` (XNNPACK, from the deploy block) | Studio deploy block | 0.9487 | deploy-block log (`[quant] … int8 accuracy`) |
| **`model_qnn.pte`** | **IQ-9075 NPU**, `qnn_executor_runner` | **0.9744** | `board/verify.sh` check 2 (`steps/accuracy_on_npu.py`) |
| **`model_qnn.pte`** | **IQ-9075 NPU**, through the `rps-npu` service | **0.9744** | `board/verify.sh` check 3 (`steps/test_npu_service.py`) |

The NPU errors were `{Rock → Paper: 2}`.

**Caveat:** `train.py` keeps the epoch that scores best on the test split, so these figures are optimistic as a generalization estimate. They are exact as a comparison between float, CPU int8, and NPU on the same 78 images.

## Latency

| Measurement | Result | Source |
|---|---|---|
| NPU, `qnn_executor_runner`, 78 images | average **0.256 ms** per inference (19.97 ms total) | runner log |
| NPU, single cold run | 0.711 ms | runner log |
| `npu_server --selftest` (20 runs after warm-up) | median **233 µs**, p95 257 µs | self-test output |
| Through the service from Python (`verify.sh` check 3) | round trip median **0.70 ms**, p95 1.14 ms; NPU side median 0.233 ms | test script |
| Service while the game runs | mean 361–425 µs per frame over 500–9,500 frames, at about 25 fps. It rose briefly when another process started on the same cDSP. | `journalctl -u rps-npu` |
| CPU `.pte` (XNNPACK) on the IQ-9075 Kryo cores | 0.4–1.4 ms (earlier 96×96 model) | earlier on-board runs |
| CPU int8 `model.pte` (the release file) on x86_64 Linux, ExecuTorch 1.5.1 Python | about 1.0 ms (mean of 50 runs after warm-up) | `tools/classify_image.py`-style timing loop |
| CPU int8 `model.pte` on a Samsung Galaxy S26, ExecuTorch Android | *to be measured* (shown on the app's screen) | `demo-executorch-qnn` |

The NPU is faster per inference, but this model was already fast on the CPU. The benefits that matter are accuracy at int8 and **taking vision work off the CPU**. Measuring app CPU usage with `RPS_NPU_ENABLE=0` against the NPU path is still to do.

## Artifact sizes

| File | Size |
|---|---|
| float32 `model.pte` (learning block) | 8.9 MB |
| CPU int8 `model.pte` (deploy block) | 2.4 MB |
| `model_qnn.pte` (int8 + v73 context binary) | 2.6 MB |
| `export_report.json` | about 1 KB |

The CPU int8 `model.pte`, `model_qnn.pte`, `model.onnx`, `export_report.json` and `labels.txt` from this run are published on the repository's GitHub Releases page with a `SHA256SUMS` file (`model.pte` starts `2e4f8977`, `model_qnn.pte` starts `726b9955`).

## Evidence that it runs on the NPU

1. The `.pte` contains a Hexagon v73 QNN context binary, and the runtime logs `Initialize Qnn backend parameters for Qnn executorch backend type 2` and `Caching is in RESTORE MODE`. Type 2 is HTP, and RESTORE MODE means the precompiled binary was loaded. The graph does not have a CPU fallback in this path.
2. The `rps-npu` journal shows `fastrpc_apps_user.c … manage_poll_qos: poll mode updated to 3 for domain 3`. FastRPC domain 3 is the cDSP.
3. Accuracy is exactly the quantized model's predicted value, at about 0.23 ms per inference.
