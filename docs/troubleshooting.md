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

## Demo app

| Symptom | Cause | Fix |
|---|---|---|
| `NPU ACTIVE` immediately followed by `CPU FALLBACK` at start | An older check compared input shapes. The NPU model was 96×96 and the CPU model 160×160. | Each backend now gets its own preprocessing, and only the class count is checked |
| `Attempted to resize a static tensor … Expected (1,3,160,160)` at start | The startup probe tries several input sizes on the CPU model | Harmless. Set `MODEL_INPUT_SIZE = 160` in `app/config.py` to silence it. |
| `cpuinfo_utils … midr_el1` lines | XNNPACK reading CPU ID registers | Harmless |
| False gestures with an empty box (often "scissors") | The classifier has no "no hand" output, and `unknown` is under-trained | The gates (`RPS_MIN_CONFIDENCE`, `RPS_MIN_MARGIN`, `RPS_MAX_UNKNOWN`, `RPS_SMOOTH_FRAMES`). Retrain with a cleaner, more varied `unknown` class. Collect frames with `RPS_SAVE_FRAMES`. |
| Labels look shifted (a "rock" reads as "paper", every time) | `labels.txt` is not in output order | Use alphabetical order, matching Edge Impulse's one-hot order (`Paper, Rock, Scissors, unknown`). Case matters. |
| `Commentary disabled: 'llama-server' is not on PATH and was not found at …/pkg-linux/bin/` | llama.cpp isn't installed where `start.sh` looks | Unzip `pkg-linux.zip` to `~/pkg-linux` or set `LLAMA_CPP_DIR`. Put the GGUF in `models/`. |
| llama.cpp is slow and never touches the NPU | llama.cpp silently falls back to the CPU | Run `scripts/check-llama-hexagon.sh`. The model must be **Q4_0**, and `ADSP_LIBRARY_PATH` must point at `pkg-linux/lib`. |
| `Qt xcb: could not connect to display` over SSH (older OpenCV preview scripts) | No display in the SSH session | `DISPLAY=:0` with the right `XAUTHORITY`, or use the web UI |
| `pip install` on the board pulls CUDA packages (`nvidia-…`) | The PyPI torch for aarch64 brings CUDA dependencies | `install.sh` installs CPU torch first from the PyTorch CPU index, uses a constraint file, and refuses to finish if CUDA packages appear |
| Ctrl+C prints `CancelledError` / `timeout graceful shutdown exceeded` | Uvicorn waits on the endless MJPEG stream | Harmless |
| `ExecuTorch` Python on a laptop: `tensor does not have a device` | ExecuTorch 1.5.1's runtime with torch 2.12 | Use torch 2.14 with ExecuTorch 1.5.1 (see [versions](versions.md)) |

## Speech and the hand

Unlike the tables above, these are not all problems that have already happened
— speech and the hand have only been exercised off-board so far
([what is verified](speech-and-actuation.md#what-is-verified-and-what-is-not)).
Each row is either a failure that was hit and fixed, or a failure mode the code
now reports deliberately because it would otherwise be silent. Both subsystems
degrade quietly by design, so the first move for either is its check script.

| Symptom | Cause | What to do |
|---|---|---|
| The game runs but says nothing, and nothing looks wrong | Some piece of the speech chain is missing, and each one degrades quietly on purpose | `./scripts/check-audio.sh` tests the player, the output device and the voice separately and names the stage that failed |
| `speech: audio disabled -- none of these audio players are installed` | The board has no PCM player. `paplay` used to be hardcoded, which is why the original setup worked and a fresh board did not | `sudo apt install alsa-utils` (smallest) or `pulseaudio-utils`. The app now looks for `paplay`, `pw-play`, `aplay` and `ffplay` in that order; `TTS_PLAYER_CMD` takes anything else |
| `check-audio.sh` passes all three stages but the room stays quiet | The player opened an output nobody can hear — HDMI instead of the jack is the classic. No software check can detect this | `./scripts/check-audio.sh --list` for the valid device names, then `--device <name>` to try one. Make it permanent with `TTS_AUDIO_DEVICE` |
| `Speech disabled: no voice at models/piper/….onnx`, or `….onnx.json is missing` | The voice was never downloaded, or only one of its two files arrived | `./scripts/fetch-tts-voice.sh`. A voice is two files that must stay together — the app reads the sample rate from the JSON |
| The voice is the wrong pitch, and on-screen text drifts out of sync with it | A sample-rate mismatch. `play()` derives each clip's duration from the rate, so the wrong rate breaks pitch **and** timing | Fixed: the rate comes from the voice's `.onnx.json`, then is re-checked against the first clip's WAV header. If they disagree, the audio wins and the player is respawned |
| `TTS_PLAYER=aplay` is ignored and something else is used | That player is not installed. At a booth, sound from the second choice beats silence from the first, so it warns and auto-detects instead of refusing | Install it, or pick one that `--list` reports as installed |
| `TTS_PLAYER_CMD` is ignored entirely | It must be a JSON array of strings, e.g. `'["my-player","--rate={rate}"]'`. Anything else is rejected with a reason at startup | Check the quoting. `{rate}` and `{device}` are substituted; an unknown placeholder is passed through untouched rather than crashing |
| `TTS_BACKEND=melo` runs silently | The audio-analytics container is not running, or `ENABLE_AUDIO_ANALYTICS` is 0. That image is not part of this repository | Use `TTS_BACKEND=piper`, which is the default and needs no container |
| `Amazing Hand disabled: /dev/ttyACM0 is not accessible` | The user is not in the serial port's group | `sudo usermod -aG dialout $USER`, then `newgrp dialout` or log out and back in. `scripts/test-hand.py` reads the owning group off the device itself and prints the exact line |
| The hand opens the serial port but a movement raises | Usually power, not software: torque needs the external supply, not just USB. Or another process holds the port | `sudo lsof /dev/ttyACM0`. `scripts/test-hand.py` separates the port, the permissions and the movement so you can tell which one it is |
| The hand moves cleanly but a finger sits visibly wrong | Servo horns are installed by hand, so neutral offsets differ between assemblies. A clean exit only proves the servos accepted the commands | That motor's entry in `AMAZING_HAND_MIDDLE_CALIBRATION` — order is motors 1–8 = index, index, middle, middle, ring, ring, thumb, thumb. Try a value with `scripts/test-hand.py --cal …` before editing `app/config.py` |
| The hand says "let me consider that…" and then never delivers a verdict | The canned lead-ins exist only to cover LLM generation latency, and they were gated on the **speaker** rather than on the commentator. With speech on and no GGUF, the lead-in plays and nothing follows | Two parts. The cause: `Commentary disabled: model not found at models/…gguf` at startup — fix with `./scripts/fetch-llm-model.sh`. The symptom: fixed, the lead-in now needs both a speaker **and** `commentator.enabled` |
| `GET /artifact 200 OK` floods the terminal | The artifact panel polls every 2 s to keep its NPU/CPU badge live | Fixed: those access-log lines are filtered out. `RPS_LOG_ARTIFACT=1` brings them back when debugging the route |
| The countdown voice lags the on-screen animation | Blocking work on the event loop that also times the countdown | Already addressed: every actuator method wraps its `rustypot` calls in `asyncio.to_thread`, and the audio player is one long-lived pipe rather than a process per clip. If it reappears, tune `AUDIO_ONSET_SECONDS` |
