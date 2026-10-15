#!/usr/bin/env python3
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Add the Qualcomm NPU (QNN) export to one of Edge Impulse's public ExecuTorch
learning blocks -- the recipe in docs/porting-to-other-blocks.md, automated.

    # from the folder that holds both this repository and the block
    git clone https://github.com/edgeimpulse/executorch-pytorch-kws-block
    python edge-impulse-executorch-qnn/porting/apply_port.py executorch-pytorch-kws-block

Supported starting points: executorch-pytorch-timeseries-block,
executorch-pytorch-kws-block, executorch-pytorch-object-detection-fomo-block
(as published; ExecuTorch 0.4.0 / torch 2.5.0). It edits the folder in place --
use a clean checkout, then review with `git status && git diff`. It checks every
anchor before writing anything, and refuses to run twice.

What it changes (each step is explained in the doc):
  1. requirements.txt   -> ExecuTorch 1.5.1 / torch 2.12.1 / numpy 2.0.2
  2. qnn_export.py, qnn_smoke.py copied from blocks/learning-image-classifier
  3. Dockerfile         -> x86_64, QNN SDK baked in, build-time smoke test
  4. parameters.json    -> four QNN settings appended
  5. train.py           -> QNN flags, dynamo=False for ONNX, export_qnn_pte() + its call
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "kit")
CLASSIFIER = os.path.join(HERE, "..", "blocks", "learning-image-classifier")


TRAIN_ANCHORS = {
    "export-pte flag": '    parser.add_argument("--export-pte", action="store_true", default=False)\n',
    "ONNX dynamic_axes": '        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},\n',
    "main()": "\ndef main() -> None:\n",
    "__main__ guard": '\n\nif __name__ == "__main__":',
}


def apply(s: str, old: str, new: str) -> str:
    return s.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    block = sys.argv[1].rstrip("/")
    train = os.path.join(block, "train.py")
    if not os.path.isfile(train) or not os.path.isfile(os.path.join(block, "parameters.json")):
        sys.exit(f"{block} does not look like a learning block (train.py, parameters.json)")
    if os.path.exists(os.path.join(block, "qnn_export.py")):
        sys.exit(f"{block} already has qnn_export.py -- already ported?")
    is_fomo = "FomoExport" in open(os.path.join(block, "model.py")).read()

    # Check every edit point first, so a changed upstream file stops us before
    # anything is written (no half-ported block).
    src = open(train).read()
    missing = [name for name, text in TRAIN_ANCHORS.items() if text not in src]
    if missing:
        sys.exit(f"{train}: could not find {', '.join(missing)} -- the block has changed "
                 "since this recipe was written. Nothing was modified; port it by hand "
                 "with docs/porting-to-other-blocks.md.")

    # 1-3: files copied whole
    shutil.copy(os.path.join(KIT, "requirements.txt"), os.path.join(block, "requirements.txt"))
    for f in ("qnn_export.py", "qnn_smoke.py"):
        shutil.copy(os.path.join(CLASSIFIER, f), os.path.join(block, f))
    shutil.copy(os.path.join(KIT, "Dockerfile"), os.path.join(block, "Dockerfile"))

    # 4: parameters.json
    p = os.path.join(block, "parameters.json")
    params = json.load(open(p))
    params["parameters"] += json.load(open(os.path.join(KIT, "qnn-parameters.json")))
    with open(p, "w") as f:
        json.dump(params, f, indent=4)
        f.write("\n")

    # 5: train.py
    parts = open(os.path.join(KIT, "train-snippets.py")).read().split("# ---8<---\n")
    args_snip, func_snip, call_classifier, call_fomo = parts[1], parts[2], parts[3], parts[4]
    a = TRAIN_ANCHORS
    s = apply(src, a["export-pte flag"], a["export-pte flag"] + args_snip)
    s = apply(s, a["ONNX dynamic_axes"], a["ONNX dynamic_axes"]
              + "        dynamo=False,  # torch >= 2.9 defaults to the new exporter, which needs onnxscript\n")
    s = apply(s, a["main()"], "\n" + func_snip + a["main()"])
    end = s.rindex(a["__main__ guard"])
    call = call_fomo if is_fomo else call_classifier
    open(train, "w").write(s[:end] + "\n" + call.rstrip("\n") + "\n" + s[end:])

    print(f"Ported {block} ({'FOMO' if is_fomo else 'classifier'} recipe). Next:")
    print(f"  cd {block} && git status && git diff        # review (two files are new)")
    print("  docker build -t <name>-qnn .                 # x86_64; runs the QNN smoke test")
    print("  then follow 'Test it without Studio' in docs/porting-to-other-blocks.md")


if __name__ == "__main__":
    main()
