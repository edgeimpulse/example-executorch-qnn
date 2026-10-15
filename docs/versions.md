# Versions

Version mismatches caused most of the problems in this project. This page lists what is pinned where, the rules behind the pins, and the known gaps.

## The rules

1. **A `.pte` must be run by the ExecuTorch release that produced it.** In practice the next release often loads it too, but don't rely on that across the QNN backend.
2. **QNN context binaries are forward compatible only.** A model compiled with QNN SDK *X* runs on a runtime ≥ *X*. A newer binary on an older runtime fails ("newer context binary on old SDK").
3. **The ExecuTorch Python runtime needs the torch it was built against.** ExecuTorch 1.5.1's pybindings fail to execute any program under torch 2.12 (`tensor does not have a device`), and work under torch 2.14. *Exporting* under 2.12 works. This was verified in this repository's tests.
4. **ExecuTorch pins a QNN SDK.** 1.5.1 pins **2.37.0.250724** and downloads it unless `QNN_SDK_ROOT` is set.

## Matrix

| Where | ExecuTorch | torch | QNN | Why |
|---|---|---|---|---|
| Learning block image (`blocks/learning-image-classifier`) | **1.5.1** | 2.12.1 (Studio-verified) | SDK **2.37.0.250724** built in | Produces `model.pte` and `model_qnn.pte` |
| Board native build (`board/setup.sh` → `steps/build_runtime.sh`) | **v1.5.1** source | 2.14.0 CPU (build-time code generation only) | apt **`libqnn1` / `libqnn-dev` 2.46.0** (headers + runtime) | Runs `model_qnn.pte`. The 2.37 binary on the 2.46 runtime works (rule 2). |
| Deploy block image (`blocks/deploy-executorch`) | 1.3.1 | 2.12.1 | — | Builds the CPU int8 `model.pte` from ONNX. Passes `model_qnn.pte` through untouched. |
| Laptop tools (`tools/`) | 1.5.1 | **2.14.0** | — | Rule 3. The release's 1.3.1 CPU `model.pte` loads and runs here too (checked). |

Board OS: Ubuntu 24.04 (glibc 2.39, g++ 13.3, Python 3.12.3), Qualcomm IoT PPA `ppa:ubuntu-qcom-iot/qcom-ppa`.

## Known gaps

- **Learning block torch.** The block pins torch 2.12.1 because that combination is verified end to end in Studio, including the QNN export. ExecuTorch 1.5.1 is built against 2.14. Training and XNNPACK export under 2.14 were checked locally and work, and the `.pte` files then run in the 1.5.1 runtime. **The QNN export under 2.14 has not been run yet.** Moving the pin to `torch==2.14.0` / `torchvision==0.29.*` is the next step once one Studio QNN run confirms it.
- **Two ExecuTorch versions for CPU models.** The CPU fallback comes from the deploy block (1.3.1), while the learning block also writes a native float32 `model.pte` (1.5.1). Aligning on 1.5.1 everywhere, and using the learning block's native `.pte` as the fallback, would remove the 1.3.1 and 1.4.1 environments entirely. That needs the demo venv on 1.5.1 + torch 2.14.
- **QNN SDK for the board.** The board uses Ubuntu's 2.46 runtime. If the PPA moves to a version that drops support for 2.37 binaries, recompile with a matching SDK. ExecuTorch's `QNN_SDK_ROOT` can point at any installed SDK.

## Quick checks

```bash
# What ExecuTorch / torch does this Python have?
python -c "import importlib.metadata as m; print(m.version('executorch'), m.version('torch'))"
# What QNN runtime does the board have?
apt policy libqnn1 | head -3
# What backend is inside a .pte and what input does it expect?
python tools/inspect_pte.py model_qnn.pte
# What SDK compiled this NPU model?
python -c "import json; r=json.load(open('export_report.json')); print(r['executorch_qnn_sdk'], r['soc_model'], r['quant_scheme'])"
```
