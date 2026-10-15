# Board: run `model_qnn.pte` on the IQ‑9075 Hexagon NPU

Two commands take a Qualcomm Dragonwing **IQ‑9075 EVK** running Ubuntu 24.04 from a fresh install to a model that runs on the NPU, is checked for accuracy, and is served to applications by a boot-time service:

```bash
./setup.sh  ~/npu-model/model_qnn.pte ~/npu-model/export_report.json   # build + install (30–60 min the first time)
./verify.sh ~/npu-model                                                # prove it
```

If someone has already built the runtime for your board (see [package.sh](#share-the-build-with-others)), you can skip the build:

```bash
./install-prebuilt.sh npu-runtime-v1.5.1-qnn2.46.0-ubuntu24.04-aarch64.tar.gz \
                      ~/npu-model/model_qnn.pte ~/npu-model/export_report.json   # about a minute
./verify.sh ~/npu-model
```

Run everything **on the board**, from a full clone of this repository (`verify.sh` uses the demo's Python client).

## Before you start

- An IQ‑9075 EVK on Ubuntu 24.04 with internet access, `sudo`, and about 10 GB free in `$HOME` for the build.
- The files from Studio's `deploy.zip` ([deployment block](../blocks/deploy-executorch/README.md)), copied to the board:

```bash
mkdir -p ~/npu-model && cd ~/npu-model
unzip -j ~/deploy.zip model_qnn.pte export_report.json labels.txt \
      _inputs/X_split_test.npy _inputs/Y_split_test.npy        # -j: flat, no _inputs/ folder
```

No deployment block? `edge-impulse-blocks runner` writes `model_qnn.pte` and `export_report.json` straight to your disk ([reproduce § 2](../docs/reproduce-on-iq9.md)). The test split is optional; without it `verify.sh` runs only the single-inference check.

The scripts install what they need: Qualcomm's QNN runtime and headers from Ubuntu's IoT PPA (`ppa:ubuntu-qcom-iot/qcom-ppa`, the same one Edge Impulse's IQ‑9075 guide uses) and the build tools.

## What each command does

| Command | Steps | What you should see |
|---|---|---|
| `./setup.sh MODEL [REPORT]` | 1. Prerequisites: `libqnn1`, `libqnn-dev`, build tools.<br>2. Clone ExecuTorch **v1.5.1** to `~/et-qnn` and build it with the QNN backend and `qnn_executor_runner`. **30–60 minutes, once;** skipped when a build is already there.<br>3. Build [`npu_server`](npu_server/npu_server.cpp) and stage the runtime in `~/et-qnn/dist`.<br>4. Self-test the model on the NPU.<br>5. Install and start the `rps-npu` service. | `selftest OK: 4 outputs, median ~233 us, p95 ~257 us`, then `Service rps-npu is running` |
| `./verify.sh DIR` | 1. One inference with `qnn_executor_runner`, input shape read from `export_report.json`.<br>2. The real test split on the NPU.<br>3. The same test split through the running service, the way apps use it. | `Caching is in RESTORE MODE`, `1 inference took ~0.7 ms`; `NPU accuracy on 78 test samples: 0.9744` (equal to the report's `accuracy.int8_fake_quant`); `accuracy via service: 0.9744`, round trip ~0.70 ms, NPU-side ~0.23 ms; then a PASS/DONE/SKIP/FAIL summary |
| `./install-prebuilt.sh PKG MODEL [REPORT]` | 1. QNN runtime from the PPA (no build tools).<br>2. Unpack the package to `~/et-qnn/dist` and check its libraries resolve on this board.<br>3. Self-test.<br>4. Install and start the service. | Same as `setup.sh` steps 4–5 |
| `./package.sh [OUT]` | Tar up `~/et-qnn/dist` with a `VERSIONS` file and checksums. | `Package: …/npu-runtime-v1.5.1-qnn2.46.0-ubuntu24.04-aarch64.tar.gz` and its sha256 |

`verify.sh` creates a small Python environment at `~/npu-tools` (numpy only) the first time it needs one.

### Settings

All optional, passed as environment variables, e.g. `JOBS=2 ./setup.sh …`:

| Variable | Default | Use |
|---|---|---|
| `JOBS` | `4` | Build parallelism. Use `2` if the build runs out of memory. |
| `WORK` | `~/et-qnn` | Build folder. The runtime is staged in `$WORK/dist`. |
| `SVC` | `rps-npu` | Service name. Gives `/opt/$SVC`, `$SVC.service` and `/run/$SVC/npu.sock`. Use a different name to run a second model alongside. |
| `FORCE_REBUILD` | unset | `1` rebuilds ExecuTorch even if a finished build exists. Not needed after a failed or interrupted build: rerunning `setup.sh` continues from where it stopped. |
| `ADSP_LIBRARY_PATH` | the IQ‑9075's `sa8775p/…/dsp/cdsp` | Folder with the board's `libQnnHtpV73Skel.so`. Change it for another Dragonwing board ([retargeting](../docs/retargeting.md)). |
| `ET_TAG` | `v1.5.1` | ExecuTorch release. Must equal the one that exported the `.pte`. |

### Common tasks

| Task | Command |
|---|---|
| Swap in a new model (runtime already built) | `./setup.sh new_model_qnn.pte new_report.json` (the build is skipped), or just `steps/install_service.sh new_model_qnn.pte` |
| A second model alongside | `SVC=kws-npu ./setup.sh ~/kws/model_qnn.pte ~/kws/export_report.json`, then `SVC=kws-npu ./verify.sh ~/kws` ([porting guide](../docs/porting-to-other-blocks.md)) |
| Logs | `journalctl -u rps-npu -f` (mean latency is printed every 500 frames) |
| Remove the service | `sudo systemctl disable --now rps-npu && sudo rm -rf /opt/rps-npu /etc/systemd/system/rps-npu.service` |
| Something is off before you start | `diagnostics/probe.sh` (architecture, OS, QNN packages, DSP device nodes) and `diagnostics/find_qnn_runtime.sh` (where the QNN libraries, skel files and headers are) |

## Why there is a build at all

The ExecuTorch pip wheel for **aarch64 has no QNN backend**; QNN is linked only into the x86_64 wheel. So the board builds ExecuTorch **v1.5.1** from source with `EXECUTORCH_BUILD_QNN=ON`. It must be exactly 1.5.1, the release that created the `.pte`.

Compiling the backend only needs the QNN **headers**, because the QNN libraries are loaded with `dlopen` at runtime. The build uses the headers and libraries from Ubuntu's Qualcomm packages (`libqnn-dev` / `libqnn1`, QNN 2.46) through a small `QNN_SDK_ROOT` shim folder. Nothing is downloaded from Qualcomm on the board.

A model compiled with QNN SDK 2.37 (in the learning block) runs on the 2.46 runtime: an older context binary on a newer runtime works. The reverse does not.

The build is the slow, one-time part, and its output is small: `npu_server`, `libqnn_executorch_backend.so` and `qnn_executor_runner`. That is why it can be packaged and shared.

## Share the build with others

After `setup.sh` has worked on one board:

```bash
./package.sh            # → npu-runtime-v1.5.1-qnn2.46.0-ubuntu24.04-aarch64.tar.gz (a few MB)
```

Attach the tarball to a GitHub release (or any file share) and publish its sha256. Others with the **same board and OS** run `./install-prebuilt.sh` with it and skip the 30–60 minute build.

- The package holds only ExecuTorch binaries built from open source. It contains **no Qualcomm SDK files**: each board gets the QNN libraries from its own apt packages.
- `install-prebuilt.sh` refuses a package whose `npu_server` needs libraries the board does not have, and notes a QNN version difference. The NPU self-test is the real check.
- A package is tied to its ExecuTorch release (v1.5.1 here), so it runs only `.pte` files exported by that release.

## How do I know it is really on the NPU?

There are three independent signs:

1. **The model can only run there.** The `.pte` holds a compiled Hexagon v73 context binary, and the runtime has no CPU path for it. The runner logs `Initialize Qnn backend parameters for Qnn executorch backend type 2` (HTP) and `Caching is in RESTORE MODE`, which means it is loading the precompiled binary.
2. **FastRPC logs.** `journalctl -u rps-npu` shows `fastrpc_apps_user.c … poll mode updated to 3 for domain 3`. FastRPC domain 3 is the **cDSP**, the compute DSP that hosts the NPU.
3. **The accuracy matches the quantized model.** `verify.sh` gives exactly the int8 result predicted at export time, at about 0.23 ms per inference.

## The `npu_server` sidecar

The service exists because `qnn_executor_runner` only processes batches of files. A camera loop needs a process that keeps the model loaded and answers each frame.

- **C++17**, using ExecuTorch's `Module` API. It reads the input shape from the model, so a retrained model at 128 or 160 pixels, or another block's model, needs no code change.
- **Protocol** (UNIX socket, little-endian):
  - Request: `u32 n_bytes` followed by float32 data in the model's input layout. Sending `n_bytes = 0xFFFFFFFF` asks for model info as JSON.
  - Response: `u32 status` (0 ok, 1 bad size, 2 failed), `u32 n_floats`, `u32 infer_us`, then the logits.
- **`--selftest`** runs 20 inferences and prints the median and p95 latency. The systemd unit runs it in `ExecStartPre`, so the service only starts if the NPU really works. Otherwise the app falls back to the CPU.
- The Python client is [`board/steps/npu_client.py`](steps/npu_client.py). It is about 120 lines and needs only numpy, so any app can use it.

### Service reference

| | |
|---|---|
| Files | `/opt/rps-npu/{npu_server, libqnn_executorch_backend.so, model_qnn.pte, export_report.json}` |
| Unit | `/etc/systemd/system/rps-npu.service` (`User=$USER`, `SupplementaryGroups=fastrpc`, `Restart=on-failure`) |
| Environment | `LD_LIBRARY_PATH=/opt/rps-npu:/usr/lib:/usr/lib/aarch64-linux-gnu`, `ADSP_LIBRARY_PATH=/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp` |
| Socket | `/run/rps-npu/npu.sock` (mode 0660) |

## What is in this folder

```text
setup.sh  verify.sh  install-prebuilt.sh  package.sh     the commands above
steps/                 what they call; each is short and can be run on its own
  common.sh            shared settings (WORK, JOBS, SVC, DIST, ADSP path)
  prereqs.sh           apt: QNN runtime (+ headers and build tools for `build`)
  build_runtime.sh     ExecuTorch v1.5.1 + QNN + qnn_executor_runner, native
  build_npu_server.sh  npu_server; stages $WORK/dist
  install_service.sh   /opt/$SVC + systemd unit, enable, start
  run_runner.sh        one inference with qnn_executor_runner
  accuracy_on_npu.py   the test split through qnn_executor_runner
  test_npu_service.py  the test split through the service socket
diagnostics/           read-only checks: probe.sh, find_qnn_runtime.sh
npu_server/            the sidecar: npu_server.cpp, CMakeLists.txt
```

What the build creates in `~/et-qnn`:

```text
venv-build/                cmake >= 3.29, ninja, torch 2.14.0 (CPU; needed by ExecuTorch's code generation)
qnn-shim/include -> /usr/include       QNN_SDK_ROOT points here (headers only)
executorch/                v1.5.1 source (examples/qualcomm/CMakeLists.txt is patched to skip LLM runners)
executorch/build-native/   installed ExecuTorch + lib/executorch/backends/qualcomm/libqnn_executorch_backend.so
executorch/build-examples/ qnn_executor_runner, npu_server
dist/                      the runtime: npu_server, libqnn_executorch_backend.so, qnn_executor_runner, VERSIONS
*.log                      one log per build stage
```

Build problems that came up, and their fixes, are listed in [docs/troubleshooting.md](../docs/troubleshooting.md).
