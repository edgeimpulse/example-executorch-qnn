# Tools: understand and test the outputs without the board

These tools run on a laptop (Linux or macOS, x86 or Arm). They need the `executorch` pip package with the torch it was built against: **ExecuTorch 1.5.1 → torch 2.14** (see [docs/versions.md](../docs/versions.md)).

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    torch==2.14.0 executorch==1.5.1 numpy opencv-python
```

| Tool | What it answers |
|---|---|
| `classify_image.py model.pte labels.txt photo.jpg` | What does this model say about a photo, and how fast is one inference on this computer? Uses the same preprocessing as the demos (centre-crop, resize to the model's input, RGB, 0..1). CPU `model.pte` only. Needs `opencv-python`. |
| `inspect_pte.py model.pte [model_qnn.pte]` | What backend is inside this file? Did the whole graph get delegated, or did some ops fall back to portable CPU kernels? What are the input and output shapes? Parses the file only, so it also works on a QNN `.pte` on a machine without QNN. |
| `evaluate_pte.py model.pte X_split_test.npy Y_split_test.npy --labels labels.txt` | Accuracy of a CPU `.pte` on Studio's test split. The `.npy` files are in `deploy.zip` under `_inputs/`. Compare it with the NPU figure from `board/verify.sh`. |
| `make_fake_data.py /tmp/ei-data --size 96 --classes 4` | Writes a dataset in Studio's format, so the learning block can run locally (including the QNN export) with no Edge Impulse account |
| `build_npu_server_cpu.sh` | Builds the **real** [`board/npu_server`](../board/npu_server/npu_server.cpp) C++ source against the pip wheel's C++ SDK, with the XNNPACK (CPU) backend instead of QNN. Same code and same protocol, so `board/steps/test_npu_service.py` works against it (Linux, CMake ≥ 3.28). One caveat: a CPU `.pte` has softmax built in, but this server returns its output as if it were logits, and the client applies softmax again. Accuracy is unchanged, but probabilities are flattened, so the demo's 0.75 confidence gate rejects everything. For the demo, use `mock_npu_server.py`. |
| `mock_npu_server.py --model model.pte --socket /tmp/rps-npu.sock` | A CPU stand-in for `board/npu_server` that speaks the same socket protocol. Run the demo with `RPS_NPU_SOCKET=/tmp/rps-npu.sock` to see `NPU ACTIVE`. Stop the mock to see `CPU FALLBACK`, start it again to see it reconnect. **It does not use an NPU.** |

Example:

```text
$ python tools/inspect_pte.py demo/rps-amazing-hand/models/model.pte
demo/rps-amazing-hand/models/model.pte
  size      : 2418.3 KB   sha256 3c0a837d3e5ada18
  method    : forward
  inputs    : float32[1, 3, 160, 160]
  outputs   : float32[1, 4]
  delegates : XnnpackBackend x1
  portable ops: none -- the whole graph is inside the delegate(s)
```

For `model_qnn.pte`, the delegate line reads `QnnBackend`, and the compiled Hexagon context binary is stored as a segment inside the file.
