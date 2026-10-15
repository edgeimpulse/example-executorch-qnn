# Troubleshooting

Every problem below actually happened during this project. Each entry gives the symptom, the cause, and the fix that is now in the code.

## Learning block (Docker / Studio)

| Symptom | Cause | Fix (already applied) |
|---|---|---|
| `export_report.json` says `Failed to set up QNN SDK`, or the download fails | Blocks have **no network at run time**, and ExecuTorch downloads the QNN SDK on first use | The Dockerfile runs `setup_qnn_sdk()` at **build** time with `QNN_STAGING_DIR=/opt/qnn` and sets `QNN_SDK_ROOT`. Studio's builder can reach `softwarecenter.qualcomm.com`. Some corporate proxies cannot. |
| `libatomic.so.1: cannot open shared object file`, then `libQnnHtp.so` needs `__cxa_init_primary_exception` | ExecuTorch's downloaded LLVM 14 libc++ needs libatomic. Without it the loader fell back to the apt `libc++1`, which is too old. | Install `libatomic1`, and **do not** install `libc++1` / `libc++abi1` |
| `Failed to initialize QNN backend for kHtpBackend … LD_LIBRARY_PATH includes $QNN_SDK_ROOT/lib/x86_64-linux-clang/` | `LD_LIBRARY_PATH` is read when a process starts, so setting it from Python does nothing. ExecuTorch also stages only `libc++.so.1.0`, without the soname link. | `ENV LD_LIBRARY_PATH=…` in the Dockerfile, plus `ln -sf libc++.so.1.0 libc++.so.1` (and the same for libc++abi) |
| QNN export reports `QNN lowering needs x86_64 Linux` | The QNN ahead-of-time libraries ship only in the x86_64 wheel. On Apple Silicon, Docker builds arm64 by default. | `FROM --platform=linux/amd64` in the Dockerfile. Build with `docker build --platform linux/amd64` if needed. |
| `executorch.backends.qualcomm` imports fine but every op runs on the CPU | Nothing was delegated | `export_qnn` refuses to write the file (`QNN delegate absent…`). See `report.delegation.non_delegated_ops`. |
| `No module named 'onnxscript'` during ONNX export (after upgrading a public block's torch) | torch ≥ 2.9 defaults `torch.onnx.export` to the dynamo exporter | Pass `dynamo=False` (this repo's blocks do; `porting/apply_port.py` adds it) |
| Training job dies with `Fail to load Qnn library` / `assert failed` (an abort, no Python traceback) | `QNN_SDK_ROOT` set to a folder without the QNN libraries; ExecuTorch's C++ layer aborts the process | `qnn_export.py` now checks the path first and reports `failed`; fix `QNN_SDK_ROOT` (the Dockerfile sets it) |
| Chance-level accuracy after quantization | Calibration fed data in the wrong layout (flat NHWC reshaped straight to NCHW) | Reshape to NHWC, then permute (`to_nchw`, `to_model_input`). The deploy block warns when float32 accuracy is at chance. |
| `edge-impulse-blocks push` fails with `ECONNRESET` | The upload included `.venv`, `out/`, data and so on | `.ei-ignore` excludes them |
| `Can only build one Keras block` in Studio | Two learning blocks in one impulse | Use one learning block per impulse |
| Pushing updates the wrong block | The block's identity is the `id` in `.ei-block-config` | To create a new block: copy the folder, delete `.ei-block-config`, rename `info.name`, then `push` |

## Getting the files out of Studio

| Symptom | Cause | Fix |
|---|---|---|
| `model_qnn.pte` is nowhere in the Studio UI | Studio only exposes learning-block outputs through deployments | Use the deployment block (Enterprise), or run the learning block locally (`edge-impulse-blocks runner`, `docker run`) |
| Deploy log: `[qnn] No model_qnn.pte in the learn block output` | **Export QNN .pte** was off, or the export failed | Check the training log for `QNN export status:` and the `reason` in `export_report.json` |
| `model.pte` is float32 although int8 was requested | An earlier deploy-block bug: the int8 choice is in `tfliteModels[0].details.modelType` and was ignored | Fixed. The log shows `[quant] Requested precision` and the float-vs-int8 accuracy. |

## Board build (`board/setup.sh`)

| Symptom | Cause | Fix |
|---|---|---|
| `pip executorch` on the board lists `['VgfBackend', 'OpenvinoBackend', 'XnnpackBackend']`, with no QNN | The aarch64 wheel has no QNN | Build from source (`board/setup.sh`), or install a prebuilt runtime (`board/install-prebuilt.sh`) |
| A `.pte` from 1.5.1 doesn't load in the board's pip 1.4.1 | Version rule | Build **v1.5.1** in its own folder (`~/et-qnn`). Existing environments are untouched. |
| `cmake` is missing, or apt's 3.27 is too old | ExecuTorch needs CMake ≥ 3.29 | `pip install cmake ninja` in `venv-build` (done by the script) |
| `Could not find … absl` when configuring the runner | `examples/qualcomm` needs abseil and gflags | `sudo apt-get install -y libabsl-dev libgflags-dev` (installed by `steps/prereqs.sh`). After a failed build, rerun `./setup.sh …`: it continues from where the build stopped |
| `tokenizers::tokenizers` / `re2::re2` not found | The examples CMake also builds LLM, Whisper, T5 and Stable Diffusion runners | The script comments out the `oss_scripts/` and `qaihub_scripts/` `add_subdirectory` lines with `sed` (safe to run twice) |
| The build is OOM-killed | Too many parallel jobs | `JOBS=2 ./setup.sh …` (continues where it stopped) |

## Running on the NPU

| Symptom | Cause | Fix |
|---|---|---|
| `No module named numpy` | System python3 has no numpy | `verify.sh` runs the accuracy checks in `~/npu-tools`, which it creates with numpy. If that venv is broken, delete it and rerun. |
| `0 outputs found` | `qnn_executor_runner` doesn't create `--output_folder_path` | The scripts `mkdir -p out` first |
| `Output 0 is memory planned … set_output_data_ptr(): 0x2` | The runner tries to set an output buffer that is memory planned | Harmless: the runner ignores it |
| `QnnContextCustomProtocol expected magic number … but get …` (INFO) | Informational while loading the context | Harmless |
| `ADSP_LIBRARY_PATH=/nonexistent` still runs | The DSP also finds the skel library in default locations | Not a proof either way. Use the FastRPC `domain 3` log lines (see [results](results.md#evidence-that-it-runs-on-the-npu)). |
| `rps-npu` fails to start | `ExecStartPre` self-test failed: model missing, libraries not found, or the DSP unreachable | `journalctl -u rps-npu -e`. Run `/opt/rps-npu/npu_server --model /opt/rps-npu/model_qnn.pte --selftest` with the unit's environment. |
| A new model was installed but the old one still answers | An earlier install script used `systemctl enable --now`, which doesn't restart a running service | Fixed: `steps/install_service.sh` now restarts the service. On an old install, `sudo systemctl restart rps-npu`. |
| `install: cannot stat '…/export_report.json'` at step 5/5, then no "Service is running" line | A report path was passed to `setup.sh` but the file was not there. Under `set -e` the install aborted **after** copying the model into `/opt/rps-npu` but **before** writing the unit — a new model on disk, the old one still being served | Fixed: the report is checked before anything is copied, and a missing one warns and continues (it is optional). Re-run `setup.sh`, ideally with the report, and check `systemctl status rps-npu` after an aborted run |
| `assert failed (file_size == tensor_meta->nbytes()): Input(0) size mismatch. file bytes: 110592, tensor bytes: 307200` | **Not a broken model.** `verify.sh` fed the wrong input size. With no `export_report.json` it used to silently guess 96×96; 110592 bytes is 1×3×96×96 float32 and 307200 is 1×3×160×160, so this is a 160×160 model | Fixed: `verify.sh` now says where the shape came from, warns when guessing, decodes the byte counts back into a resolution, and accepts `SHAPE=1,3,160,160 ./verify.sh ~/npu-model`. The real fix is to copy the report in |
