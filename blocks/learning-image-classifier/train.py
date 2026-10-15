# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Training entrypoint for the Edge Impulse PyTorch custom learning block.

Edge Impulse passes the following arguments automatically:

    --info-file <file>        Path to train_input.json (training configuration).
    --data-directory <dir>    Directory with the NumPy train/test splits.
    --out-directory <dir>     Directory to write model artifacts to.

Custom parameters defined in parameters.json are passed as extra arguments:

    --epochs <int>
    --learning-rate <float>
    --batch-size <int>
    --export-pte <flag>
    --export-qnn <flag>
    --soc-model <str>
    --quant-scheme <str>
    --calibration-samples <int>
    --no-pretrained <flag>
    --freeze-backbone <flag>
    --no-augment <flag>

Input data (already pre-processed and pixel-scaled by Edge Impulse):

    X_split_train.npy / X_split_test.npy   float32, NHWC (batch, height, width, channels)
    Y_split_train.npy / Y_split_test.npy   one-hot label matrix, shape (batch, num_classes)

We transpose NHWC -> NCHW for PyTorch. Because we export an ONNX model, Edge
Impulse handles the equivalent transpose on-device automatically.
"""

from __future__ import annotations

import argparse
import copy
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from model import ClassifierExport, build_model

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch custom learning block")
    parser.add_argument("--info-file", type=str, required=False)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--out-directory", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="0.0001 suits fine-tuning a pretrained backbone; "
                             "raise to ~0.001 when using --no-pretrained")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--export-pte", action="store_true", default=False)
    parser.add_argument("--export-qnn", action="store_true", default=False,
                        help="also export model_qnn.pte for the Qualcomm "
                             "Hexagon NPU (needs x86_64 Linux + QNN SDK)")
    parser.add_argument("--soc-model", type=str, default="QCS9100",
                        help="QcomChipset name to compile for (IQ-9075: QCS9100)")
    parser.add_argument("--quant-scheme", type=str, default="8a8w",
                        choices=["8a8w", "16a8w", "16a16w", "16a4w"])
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--no-pretrained", action="store_true", default=False,
                        help="train the small from-scratch CNN instead of "
                             "fine-tuning MobileNetV2")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="train only the classifier head (pretrained only)")
    parser.add_argument("--no-augment", action="store_true", default=False,
                        help="disable training-time data augmentation")
    # Ignore any unknown args Edge Impulse may add in the future.
    args, _ = parser.parse_known_args()
    return args


def load_split(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(os.path.join(data_dir, f"X_split_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"Y_split_{split}.npy"))
    return x, y


def load_class_names(info_file, n_classes: int) -> list:
    """Best-effort label names from train_input.json; falls back to indices."""
    import json

    try:
        with open(info_file) as f:
            info = json.load(f)
        names = info.get("classes") or info.get("classNames") or []
        if len(names) == n_classes:
            return [str(n) for n in names]
    except Exception:  # noqa: BLE001
        pass
    return [str(i) for i in range(n_classes)]


def to_labels(y: np.ndarray) -> np.ndarray:
    """Convert Edge Impulse's one-hot Y split to integer class indices.

    Edge Impulse encodes classification labels as a one-hot matrix of shape
    (n_samples, n_classes). PyTorch's CrossEntropyLoss expects integer targets,
    so we take the argmax across the class dimension.
    """
    y = np.asarray(y)
    if not y.size:
        return np.array([], dtype=np.int64)
    return y.argmax(axis=1).astype(np.int64)


def to_nchw(x: np.ndarray) -> np.ndarray:
    """Convert NHWC image data to contiguous NCHW for PyTorch.

    np.transpose returns a strided view (channels-last memory). We force a
    C-contiguous copy so torch.export captures dim_order (0, 1, 2, 3); the
    XNNPACK backend rejects channels-last inputs during .pte lowering.
    """
    if x.ndim != 4:
        raise ValueError(
            f"Expected 4D NHWC image data, got shape {x.shape}. "
            "This block operates on image data."
        )
    return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))


def build_augmentation(in_channels: int) -> v2.Compose:
    """Training-time augmentation.

    Small datasets are the normal case for this block, and augmentation is
    typically worth several accuracy points on a few hundred images. The
    horizontal flip matters for hand gestures specifically -- without it a model
    trained on right hands fails on left-handed users.

    Uses transforms.v2; the v1 namespace is in maintenance mode upstream.
    """
    transforms = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ]
    if in_channels == 3:
        # Saturation/hue are undefined for single-channel input.
        transforms.append(
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        )
    else:
        transforms.append(v2.ColorJitter(brightness=0.2, contrast=0.2))
    return v2.Compose(transforms)


class ImageDataset(Dataset):
    """Holds NCHW float tensors and applies per-sample transforms."""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        image = self.x[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.y[index]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    transform=None,
) -> DataLoader:
    dataset = ImageDataset(x, y, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += (preds == targets).sum().item()
    return correct / len(loader.dataset)


def export_onnx(
    model: nn.Module, sample_input: torch.Tensor, out_path: str
) -> None:
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,  # Follow up on opset and onnx requirements needed
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    print(f"Saved ONNX model to {out_path}")


def export_pte(
    model: nn.Module, sample_input: torch.Tensor, out_path: str
) -> None:
    """Export an ExecuTorch program (.pte) lowered to the XNNPACK backend.

    ExecuTorch is optional at runtime; if it isn't installed we warn and skip
    rather than failing the whole training job.
    """
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower
    except ImportError:
        print(
            "WARNING: executorch is not installed; skipping .pte export. "
            "Add 'executorch' to requirements.txt to enable it."
        )
        return

    model.eval()
    # XNNPACK only accepts contiguous (channels-first) inputs during lowering.
    # np.transpose earlier produced a channels-last view; force NCHW dim_order
    # (0, 1, 2, 3) here so lowering doesn't reject the placeholder.
    sample_input = sample_input.contiguous(memory_format=torch.contiguous_format)
    try:
        exported = torch.export.export(model, (sample_input,))
        program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()
    except Exception as exc:  # noqa: BLE001 - .pte is an optional artifact
        print(
            "WARNING: ExecuTorch lowering failed; skipping .pte export "
            f"(the ONNX model is unaffected). Reason: {exc}"
        )
        return
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    print(f"Saved ExecuTorch program to {out_path}")


def export_qnn_pte(
    model: nn.Module,
    x_train: np.ndarray,
    x_test: np.ndarray,
    label_test: np.ndarray,
    class_names: list,
    args: argparse.Namespace,
) -> dict:
    """Export model_qnn.pte + export_report.json. Never fails the job."""
    from qnn_export import export_qnn

    rng = np.random.default_rng(SEED)
    n = min(args.calibration_samples, len(x_train))
    idx = rng.choice(len(x_train), size=n, replace=False)
    calibration = np.ascontiguousarray(x_train[idx], dtype=np.float32)
    eval_x = x_test if len(x_test) else None
    eval_y = label_test if len(x_test) else None

    # Logits, not softmax: see ClassifierExport.
    logits_model = ClassifierExport(model, softmax=False).eval()
    sample = torch.from_numpy(np.ascontiguousarray(x_train[:1], dtype=np.float32))
    report = export_qnn(
        logits_model,
        sample,
        calibration,
        os.path.join(args.out_directory, "model_qnn.pte"),
        soc_model=args.soc_model,
        quant_scheme=args.quant_scheme,
        eval_x=eval_x,
        eval_y=eval_y,
        report_path=os.path.join(args.out_directory, "export_report.json"),
        extra_report={
            "labels": class_names,
            "input_layout": "NCHW",
            "input_dtype": "float32",
            "input_scaling": "0..1",
            "output_semantics": "logits (apply softmax on host)",
        },
    )
    return report


def main() -> None:
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train, y_train = load_split(args.data_directory, "train")
    x_test, y_test = load_split(args.data_directory, "test")

    x_train = to_nchw(x_train)
    x_test = to_nchw(x_test) if x_test.size else x_test.reshape(0, *x_train.shape[1:])

    class_names = load_class_names(args.info_file, int(y_train.shape[1]))

    in_channels = x_train.shape[1]
    num_classes = int(y_train.shape[1])
    label_train = to_labels(y_train)
    label_test = to_labels(y_test)
    print(f"Channels: {in_channels}, classes: {num_classes}, "
          f"train: {len(x_train)}, test: {len(x_test)}")

    pretrained = not args.no_pretrained
    augment = None if args.no_augment else build_augmentation(in_channels)
    print(f"Backbone: {'MobileNetV2 (ImageNet)' if pretrained else 'SmallCNN (scratch)'}"
          f", augmentation: {'off' if augment is None else 'on'}"
          f", lr: {args.learning_rate}")

    train_loader = make_loader(
        x_train, label_train, args.batch_size, shuffle=True, transform=augment
    )
    # No augmentation on the evaluation split.
    test_loader = make_loader(x_test, label_test, args.batch_size, shuffle=False)

    model = build_model(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only optimise parameters that require grad, so --freeze-backbone works.
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )

    # Keep the best-scoring weights rather than whatever the last epoch produced.
    # With augmentation the final epoch is often not the best one.
    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - val_acc: {acc:.4f}")
        if not np.isnan(acc) and acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights (val_acc: {best_acc:.4f})")

    os.makedirs(args.out_directory, exist_ok=True)
    model = model.to("cpu")
    export_model = ClassifierExport(model)
    # ascontiguousarray guarantees dim_order (0, 1, 2, 3) for the XNNPACK export.
    sample_input = torch.from_numpy(
        np.ascontiguousarray(x_train[:1], dtype=np.float32)
    )

    export_onnx(export_model, sample_input, os.path.join(args.out_directory, "model.onnx"))
    if args.export_pte:
        export_pte(export_model, sample_input, os.path.join(args.out_directory, "model.pte"))
    if args.export_qnn:
        report = export_qnn_pte(
            model, x_train, x_test, label_test, class_names, args
        )
        print(f"QNN export status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
