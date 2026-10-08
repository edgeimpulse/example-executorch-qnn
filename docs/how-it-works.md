# How it works

This walks one trained model from Studio to the NPU and names the file that does each step. It starts with the custom blocks, because they are the reusable part. The board runtime and the demo come after.

## 1. The Edge Impulse block contracts used

### Learning block (`blocks/learning-image-classifier`)

**How Studio runs it.** Studio runs the block's Docker image (ENTRYPOINT `python3 train.py`) with:
- `--data-directory <dir>` containing `X_split_train.npy`, `X_split_test.npy`, `Y_split_train.npy`, `Y_split_test.npy`
  - The image features are flat float32 in **NHWC** order, already scaled by the impulse's image block (`imageInputScaling: "0..1"` in `parameters.json`).
  - The labels are one-hot.
- `--out-directory <dir>`, where the block writes its outputs
- `--info-file train_input.json` (training configuration, including class names)
- One `--flag value` for every entry in `parameters.json` → `parameters`. This is how **Export QNN .pte**, **QNN SoC model**, **QNN quantization** and **QNN calibration samples** appear as fields in Studio.

**What Studio needs back.** At least one model it understands. This block writes `model.onnx`, which Studio converts to TFLite for live classification, model testing and profiling.

**Extra files.** Other files in the output directory, here `model.pte`, `model_qnn.pte` and `export_report.json`, are **kept with the learning block's output**, but Studio does not show them in the UI. A deployment block can read them (next section). That is the hook this project uses.

**Network at runtime.** None. Learning blocks run with no network access, so anything downloaded on first use (ImageNet weights, the Qualcomm SDK) has to be built into the image at `docker build` time.

### Deployment block (`blocks/deploy-executorch`)

**How Studio runs it.** Studio runs the image with `--metadata deployment-metadata.json`. The metadata includes:
- `folders.input`, which holds the trained model in several formats, including `onnxModelPath`
- `folders.output`, where `deploy.zip` must be written
- `classes`
- `tfliteModels[0].details.modelType`, which is `int8` or `float32` depending on the optimization the user chose on the Deployment page (`"showOptimizations": true`)

**`"mountLearnBlock": true`** mounts the learning block's data **and output files** at `/data`. Verified in this project: `/data` contains `X_split_train.npy`, `X_split_test.npy`, `Y_split_test.npy`, `dsp_metadata.json`, and the learning block's `model.pte`, `model.onnx`, `model_qnn.pte` and `export_report.json`.


## 2. Inside the learning block

`train.py`:
1. Loads the splits.
2. Converts NHWC to **contiguous** NCHW (`np.ascontiguousarray`). A transposed view keeps channels-last strides, and the XNNPACK partitioner rejects channels-last inputs.
3. Builds `PretrainedClassifier`: torchvision MobileNetV2 with ImageNet weights and a new linear head sized to the dataset. The weights are fetched during `docker build` into `TORCH_HOME=/opt/torch`.
4. Trains with Adam and cross-entropy, with flip/affine/colour augmentation, keeping the best epoch.

It then exports:

| Export | Wrapper | Path | Notes |
|---|---|---|---|
| `model.onnx` | `ClassifierExport(softmax=True)` | `torch.onnx.export` (opset 13) | Studio's evaluator expects probabilities |
| `model.pte` | `ClassifierExport(softmax=True)` | `torch.export` → `to_edge_transform_and_lower(XnnpackPartitioner)` → `to_executorch` | float32. The whole graph lands in one XNNPACK delegate (check with `tools/inspect_pte.py`). |
| `model_qnn.pte` | `ClassifierExport(softmax=False)` | `qnn_export.export_qnn()` (below) | int8, outputs logits |

### `qnn_export.export_qnn()`, step by step

```python
captured  = torch.export.export(model, (x,), strict=False).module()
quantizer = QnnQuantizer(backend=kHtpBackend, soc_model=QcomChipset.QCS9100)
quantizer.set_default_quant_config(QuantDtype.use_8a8w, is_qat=False,
                                   is_conv_per_channel=True, is_linear_per_channel=True)
prepared  = prepare_pt2e(captured, quantizer)      # torchao.quantization.pt2e
for img in calibration_200_training_images: prepared(img)
quantized = convert_pt2e(prepared)
# quality: float model vs quantized graph (simulated on CPU) -> report["accuracy"]
#   classifiers: top-1 on X_split_test; other tasks pass eval_fn (e.g. FOMO per-cell accuracy)
setup_qnn_sdk()                                    # uses QNN_SDK_ROOT (baked into the image)
spec  = generate_qnn_executorch_compiler_spec(soc_model=QCS9100,
                                              backend_options=generate_htp_compiler_spec(use_fp16=False))
edge  = to_edge_transform_and_lower_to_qnn(quantized, (x,), spec)
# delegation: get_delegation_info + "QnnBackend" in format_delegated_graph -> report["delegation"]
prog  = edge.to_executorch(ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True),
            segment_alignment=get_qnn_context_binary_alignment()))
```

The resulting `.pte` contains a **QNN context binary** compiled ahead of time for Hexagon v73. That is why the board doesn't compile anything at load time: the runtime logs `Caching is in RESTORE MODE`.

Four design rules hold this together:
- **It never raises.** Every failure, including a non-x86 host, a missing SDK, an unknown SoC, a lowering error, or a graph with no QNN delegate, becomes `status: "failed"` with a `reason`. Training still succeeds.
- **`ok` means the NPU really runs it.** The status is `ok` only if `QnnBackend` is in the lowered graph.
- **The image build tests the QNN path.** `qnn_smoke.py` runs during `docker build` and lowers a tiny convolutional network. If the SDK, libc++ or `LD_LIBRARY_PATH` setup is broken, the **image fails to build**. Nobody finds out from a failed training job.
- **The SDK is set up in the image.** The Dockerfile downloads it with `setup_qnn_sdk()` and `QNN_STAGING_DIR=/opt/qnn`, sets `QNN_SDK_ROOT` and `LD_LIBRARY_PATH` with `ENV`, adds the `libc++.so.1` soname symlinks, and installs `libatomic1`. Each of these fixed a real failure; see [troubleshooting](troubleshooting.md).

## 3. Inside the deployment block

`build.py`:
1. Copies the `app/` runtime harness and writes `labels.txt` from `metadata.classes`.
2. **Builds the CPU program.**
   - Converts the ONNX to PyTorch with `onnx2torch`.
   - If int8 was requested, applies PT2E static quantization with `XNNPACKQuantizer` (symmetric, per-channel), calibrated on `/data/X_split_train.npy` (500 samples at most). Each row is reshaped NHWC to NCHW explicitly.
   - Prints float32 vs int8 accuracy on `/data/X_split_test.npy`.
   - Lowers with `XnnpackPartitioner` to `model.pte`.
   - Any failure falls back to float32, and the log says so.
3. **Passes the NPU program through.** It copies `/data/model_qnn.pte` and `/data/export_report.json` next to `model.pte`. If they are missing, it logs `[qnn] No model_qnn.pte…`.
4. Copies the learning block's inputs and outputs into `_inputs/` for local testing.
5. Zips everything into `folders.output/deploy.zip`.

## 4. On the board

### Why a custom runtime build

| Fact | Consequence |
|---|---|
| The aarch64 ExecuTorch wheel has no QNN backend (QNN is linked only in the x86_64 wheel) | Build ExecuTorch from source on the board |
| A `.pte` must be run by the ExecuTorch release that made it | Build **v1.5.1**, the learning block's version. Keep it separate from any pip ExecuTorch on the board. |
| The QNN backend only needs QNN **headers** to compile (it `dlopen`s the libraries) | Compile against the apt `libqnn-dev` 2.46 headers through a `QNN_SDK_ROOT` shim. No SDK download on the board. |
| The board's QNN runtime (2.46) is newer than the compile SDK (2.37) | Works: an older context binary runs on a newer runtime. The reverse fails with "newer context binary on old SDK". |
| The Python runtime cannot load the QNN backend here | Run the NPU model in a separate C++ process: `npu_server` |

`board/setup.sh` does all of this in one command: prerequisites, the build, `npu_server`, an NPU self-test and the service. `board/verify.sh` then checks the model on the NPU against the export report. Because the build output is only three files, `board/package.sh` can hand it to other boards, which install it with `board/install-prebuilt.sh` instead of building.

### `npu_server` and the `rps-npu` service

`board/npu_server/npu_server.cpp` is about 300 lines of C++17 using ExecuTorch's `Module` API:
1. `load_forward()`, then read the input shape from `method_meta`.
2. Warm up.
3. Serve a UNIX socket.

The protocol:

```text
request : u32 n_bytes | n_bytes of float32 NCHW         (n_bytes == 0xFFFFFFFF → INFO request)
response: u32 status (0 ok · 1 bad input size · 2 inference failed) | u32 n_floats | u32 infer_us | logits
INFO    : u32 json_len | {"backend":"qnn-htp","input_shape":[1,3,96,96],"num_outputs":4,...}
```

The systemd unit:
- runs `npu_server --selftest` as `ExecStartPre`, so the service only comes up if 20 inferences succeed on the NPU
- runs as the user, with `SupplementaryGroups=fastrpc`
- sets `LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH`, the latter pointing at the v73 skel library directory the Qualcomm packages install for SA8775P/QCS9100
- restarts on failure

The same source builds on a laptop against the pip wheel's C++ SDK with XNNPACK instead (`tools/build_npu_server_cpu.sh`). That makes the sidecar testable without the board.

## 5. In the demo app

`app/pte_perception.py` → `HybridClassifier` (`app/npu_inference.py`) → `NpuClient` (`app/npu_client.py`):

- **Each frame:**
  1. Centre-crop, resize to the **NPU model's** input size (from INFO), RGB, /255, NCHW.
  2. Send it to the socket and apply softmax to the logits.
  3. If the call fails, reprocess the same frame at the **CPU model's** size and run `models/model.pte` with the ExecuTorch Python runtime. That model has softmax built in.
- **Reconnects** are attempted every 5 s, and a log line marks each change: `NPU ACTIVE` / `CPU FALLBACK`.
- **Gates.** The probabilities are averaged over 5 frames, then must pass confidence ≥ 0.75, a margin ≥ 0.35 over the runner-up, and `unknown` ≤ 0.15. Otherwise there is no detection, and an empty round resolves as "no gesture".
- **The LLM** is a separate process: `llama-server --device HTP0 -ngl 99`, with Llama‑3.2‑1B Q4_0. It opens its own FastRPC session on the same cDSP. The two processes share the NPU but nothing else.
- **Speech** is a third process, `python -m piper.http_server`, and the only one on the **CPU** — the NPU is already carrying the two above. `app/speech.py` POSTs a line, gets a WAV back, strips it to raw PCM and writes that into a long-lived audio-player pipe. The fixed clips (countdown words, lead-ins) are synthesized once at startup and cached on disk, so the cues that must land on a beat never wait on synthesis.
- **The hand** is `app/actuation.py` over a serial link to eight servos. Every method offloads its blocking I/O with `asyncio.to_thread`, because the same event loop times the countdown audio.

Both of those last two are optional and fail soft: each resolves to a null implementation that does nothing, so a missing voice or an unplugged hand costs that feature and nothing else. [Speech and actuation](speech-and-actuation.md) covers both in full.
- **`app/artifact.py`** closes the loop for a spectator: it hashes and parses both `.pte` files once at startup and serves the result at `GET /artifact`, next to the live backend from `detector.status()`. The sidebar panel turns the project's "one training run, two artifacts" claim into something you can read off the screen and test by stopping the service mid-match ([demo README](../demo/rps-amazing-hand/README.md#artifact-panel)).
