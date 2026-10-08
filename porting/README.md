# Porting kit: NPU export for the other ExecuTorch blocks

Adds the Qualcomm NPU (`model_qnn.pte`) export to Edge Impulse's public ExecuTorch learning blocks for **timeseries**, **keyword spotting** and **FOMO object detection**. They start from the repositories as published, on ExecuTorch 0.4.0. The full explanation is in [docs/porting-to-other-blocks.md](../docs/porting-to-other-blocks.md).

```bash
# from the folder that holds both this repository and the block
git clone https://github.com/edgeimpulse/executorch-pytorch-timeseries-block
python edge-impulse-executorch-qnn/porting/apply_port.py executorch-pytorch-timeseries-block
```

| Path | What it is |
|---|---|
| `apply_port.py` | Applies the guide's six changes in place. It detects FOMO, refuses to run twice, and checks every edit point before writing anything, so it never leaves a half-ported block. |
| `kit/requirements.txt` | ExecuTorch 1.5.1 / torch 2.12.1 pins, the same as the image block |
| `kit/Dockerfile` | x86_64 image with the QNN SDK built in, plus the build-time smoke test |
| `kit/qnn-parameters.json` | The four Studio settings |
| `kit/train-snippets.py` | The code added to `train.py`: flags, `export_qnn_pte()`, and the call for classifiers and for FOMO |
| `patches/*.patch` | The same result as `git apply`-able patches against the public repositories (October 2026), for review |

`qnn_export.py` and `qnn_smoke.py` are copied from [`blocks/learning-image-classifier/`](../blocks/learning-image-classifier/), so there is one source of truth.
