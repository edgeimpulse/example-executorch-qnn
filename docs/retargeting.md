# Retargeting: other chips besides the IQ‑9075

The NPU export can be pointed at any chip that ExecuTorch has a **backend** for. A backend is the vendor plug-in that compiles part of a model for that chip and runs it there. How much work that takes depends on how far the target is from what was built here:

| Target | Example | What is new |
|---|---|---|
| Another Qualcomm chip | Dragonwing QCS6490 boards, Snapdragon phones | A setting plus the device runtime |
| Another vendor's NPU | MediaTek, Samsung Exynos, Intel, NXP i.MX | One export module plus the vendor's SDK and runtime |
| A microcontroller NPU | Arm Ethos-U, NXP MCX / i.MX RT | The export is similar; the device side is bare metal |
| No ExecuTorch backend | Anything else | — |

This page is a **map, not a tutorial**. Only the IQ‑9075 path is verified end to end. The vendor-specific setup lives in [ExecuTorch's backend docs](https://docs.pytorch.org/executorch/stable/backends-overview.html), and those change with each release. For the background on why the design looks the way it does, see [design-decisions.md](design-decisions.md).

## What is Qualcomm-specific, and what is not

Most of the repository does not care which chip it targets. These are the only places that do:

| Piece | Qualcomm-specific? | What changes for another target |
|---|---|---|
| `blocks/learning-image-classifier/qnn_export.py` | **Yes:** `QnnQuantizer`, the HTP compiler spec, `to_edge_transform_and_lower_to_qnn`, the QNN memory alignment, the `QnnBackend` delegation check | For another Qualcomm chip, only `--soc-model`. For another vendor, a sibling module with the same structure ([below](#another-vendors-npu)). |
| `qnn_smoke.py` | Calls `export_qnn` | Point it at the new module |
| `Dockerfile` | Yes: builds in the QNN SDK and sets `QNN_SDK_ROOT` / `LD_LIBRARY_PATH` | Build in the vendor's SDK instead |
| `parameters.json` | The SoC setting lists Qualcomm chips | A target dropdown and the vendor's options |
| `train.py` → `export_qnn_pte()` | No: it only picks calibration and test data | Unchanged, or renamed |
| Export report, never-fail rule, build-time smoke test | No | Unchanged |
| `blocks/deploy-executorch/` (pass-through) | No | Unchanged (it could also copy a `model_<target>.pte`) |
| `board/steps/build_runtime.sh`, `npu_server/CMakeLists.txt` | Yes: `EXECUTORCH_BUILD_QNN=ON`, links `qnn_executorch_backend`, QNN headers | Build ExecuTorch with the vendor's backend option and link its backend library |
| `board/npu_server/npu_server.cpp` | **No.** It uses ExecuTorch's generic `Module` API; the backend is chosen by what is linked in. (Its info reply says `qnn-htp`, which is cosmetic.) | Unchanged |
| `board/steps/` (runner, service), `ADSP_LIBRARY_PATH` | The runner and DSP library paths are QNN-specific | Qualcomm: the board's skel folder. Others: the vendor's runtime setup. |
| Demo app (`npu_client.py`, `HybridClassifier`, gates) | No | Unchanged |

## Where the target is chosen

Today the target is fixed **at training**: the learning block lowers the model while it still has the trained module, the calibration images and the test split. That is solid, but switching chips means retraining. There are two ways to move the choice to **deployment**:

1. **Without changing Studio.** The learning block also saves the un-lowered program, `model.pt2` from `torch.export.save`, as one more output file. A deployment block with a target dropdown then loads it, calibrates on the training data it already receives at `/data`, and lowers for the chosen chip. This uses only custom blocks; the deployment block still needs Enterprise.
2. **First-class in Studio.** ExecuTorch targets would appear next to the built-in deployments, and Studio would store and test the `.pte` itself. That requires platform work by Edge Impulse.

Either way, the export code per target is the same; only where it runs changes.

## Another Qualcomm chip

**The export is the same code with a different `--soc-model`.** ExecuTorch 1.5.1 knows these chips (`QcomChipset` in `backends/qualcomm/serialization/qc_schema.py`):

| `--soc-model` | Hexagon (HTP) | Typical products |
|---|---|---|
| `QCS9100` | v73 | Dragonwing IQ‑9075 / QCS9075 class (**the verified target**) |
| `SA8255` | v73 | Automotive (same v73 family) |
| `QCM6490` | v68 | Dragonwing QCS6490 class (e.g. RB3 Gen 2) |
| `SM8550` | v73 | Snapdragon 8 Gen 2 phones |
| `SM8650` | v75 | Snapdragon 8 Gen 3 phones |
| `SM8750` | v79 | Snapdragon 8 Elite phones |
| `SM8850`, `SM8845` | v81 | Newer Snapdragon 8-series phones |
| `SC8380XP` | v73 | Snapdragon X Elite laptops |

The rest of the enum covers XR, wearable and automotive parts (`SXR*`, `SSG*`, `SW6100`, `SAR2230P`, `SA8295`, `SA8797`, `SM8350`, `SM8450`, `SM8475`). Check the file in your ExecuTorch version, because the list grows with each release.

**If your chip is not listed:** the IQ‑9075 isn't either. Pick the entry from the same silicon family, or at least the same Hexagon version and VTCM size, then confirm on the device. That is how `QCS9100` was chosen and verified here.

**Chips without an HTP get the CPU path.** The Arduino UNO Q (QRB2210), for example, has no HTP entry, so it runs the portable `model.pte`.

**What differs on the device:**

- **Linux boards (Dragonwing).** Repeat the native build in [board/](../board/README.md) with that board's QNN packages. Point `ADSP_LIBRARY_PATH` at the folder that holds that board's `libQnnHtpV<NN>Skel.so`; the IQ‑9075's `sa8775p` folder is only the default. All the board scripts honour the variable. Everything else is unchanged.
- **Android phones.** Use ExecuTorch's Android library (AAR) with the Qualcomm backend instead of the board build.
    - Prebuilt QNN AARs have been published for some releases. Otherwise build one with ExecuTorch's `scripts/build_android_library.sh` and `QNN_SDK_ROOT` set ([Android guide](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)).
    - The AAR must be the **same ExecuTorch version** that exported the `.pte`.
    - The app replaces the socket service: the Kotlin/Java `Module` loads `model_qnn.pte` directly.
- **Versions, as here.** The QNN runtime on the device must be the same as or newer than the SDK used to compile, and ExecuTorch versions must match.

## Other NPUs

Every backend in ExecuTorch 1.5.1 follows the same pattern as Qualcomm's: a **quantizer** that sets up int8 (or the formats the chip wants), a **partitioner** that picks the parts of the graph the chip can run, and a **compile spec** that names the chip. A sibling of `qnn_export.py` keeps the same steps and swaps the vendor pieces:

| Step in `export_qnn()` | Qualcomm (here) | Other |
|---|---|---|
| Quantize (PT2E) | `QnnQuantizer` | MediaTek `NeuropilotQuantizer`, Samsung `EnnQuantizer`, NXP `NeutronQuantizer`, Intel `OpenVINOQuantizer`, Arm `EthosUQuantizer`, Apple `CoreMLQuantizer` |
| Lower | `to_edge_transform_and_lower_to_qnn` with the HTP compile spec | `to_edge_transform_and_lower(..., partitioner=[<Vendor>Partitioner(compile_spec)])`, e.g. `NeuropilotPartitioner`, `EnnPartitioner`, `NeutronPartitioner`, `OpenvinoPartitioner`, `EthosUPartitioner` |
| Delegation check | `QnnBackend` in the graph | That vendor's backend id (check with `tools/inspect_pte.py`) |
| SDK at build time | QNN SDK built into the image | The vendor's compiler and SDK. Often a registration download or licence terms; sometimes x86-only. |
| Everything else | Platform guard, never raise, float vs int8 report, smoke test | Unchanged |

The class names above exist in ExecuTorch 1.5.1. Their exact arguments and options are in each backend's docs, and none of these paths were run in this project.

**The device side is where most of the work is,** as it was for Qualcomm:
- Build ExecuTorch for the device's OS with that backend enabled (each backend has its own CMake option and build notes).
- Install the vendor's runtime drivers.
- Link `npu_server` against that backend's library instead of `qnn_executorch_backend`.

The server source, the socket protocol and the demo's CPU fallback stay as they are.

**Things to find out early for any vendor:**
- Which of the model's ops it supports. Run the export and read `non_delegated_ops` before investing in the device side.
- Which quantization formats it requires.
- Whether its SDK can be built into a Docker image legally and practically.
- Whether the compiled program is tied to an exact driver version on the device.

## Microcontroller NPUs

**The export looks similar.** For Arm Ethos-U, for example, the pair is `EthosUQuantizer` plus `EthosUPartitioner`, with Arm's Vela compiler. Everything after the export changes:

- **No Linux.** No sidecar process, socket or Python app. The model runs in a bare-metal C++ runner compiled into the firmware.
- **Memory.** Kilobytes to a few megabytes. MobileNetV2 at this size likely won't fit; the block's `--no-pretrained` small CNN, a smaller input, or a slimmer backbone would.
- **Formats.** Typically int8-only, with stricter op support.

That is closer to the MCU direction in the [ExecuTorch blog post](https://www.edgeimpulse.com/blog/from-pytorch-to-the-edge-getting-started-with-executorch-and-edge-impulse/) than to the IQ‑9075 setup.

## Hardware without a backend

- **It still runs on the CPU.** The portable `model.pte` (XNNPACK) works on Arm and x86 CPUs under Linux, Android and iOS. Portable C++ kernels cover almost anything with a C++ compiler.
- **New accelerators need a backend.** Getting a new accelerator into ExecuTorch means writing a backend: an ahead-of-time partitioner and preprocessing step, plus a runtime that executes the compiled blob on the device. That is a vendor-scale project.

## For any new target: what to check

The same evidence used for the IQ‑9075 works anywhere:

1. **The export report:** `status: ok`, the float32 vs int8 delta, and `delegation.non_delegated_ops` (empty or small).
2. **`tools/inspect_pte.py`:** the expected backend appears in `delegates`.
3. **On the device:** the accuracy matches the report's `int8_fake_quant` value. Use `board/verify.sh` or its equivalent.
4. **Versions:** the ExecuTorch runtime matches the exporting release, and the vendor runtime is compatible with the compile-time SDK.
5. **Proof the accelerator is used:** a log line, driver statistic or profiler trace. A backend falling back to the CPU looks exactly like success except for speed.

## Links

- [ExecuTorch backends overview](https://docs.pytorch.org/executorch/stable/backends-overview.html): every backend, the hardware it targets, and its docs
- [Qualcomm backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html) · [MediaTek](https://docs.pytorch.org/executorch/stable/backends-mediatek.html) · [Samsung Exynos](https://docs.pytorch.org/executorch/stable/backends/samsung/samsung-overview.html) · [NXP](https://docs.pytorch.org/executorch/stable/backends/nxp/nxp-overview.html) · [OpenVINO](https://docs.pytorch.org/executorch/stable/build-run-openvino.html) · [Arm Ethos-U](https://docs.pytorch.org/executorch/stable/backends/arm-ethos-u/arm-ethos-u-overview.html) · [Core ML](https://docs.pytorch.org/executorch/stable/backends/coreml/coreml-overview.html) · [Vulkan](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html)
- [Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html), including the QNN AAR
