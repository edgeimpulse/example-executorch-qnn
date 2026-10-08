"""Model definitions for the PyTorch custom learning block.

Two architectures are available:

- ``PretrainedClassifier`` (default): MobileNetV2 with ImageNet weights, with a
  fresh classifier head sized to the dataset. Fine-tuning a pretrained backbone
  is what makes small datasets viable -- a few hundred images cannot learn
  useful visual features from scratch.
- ``SmallCNN``: the original compact network, trained from random
  initialisation. Kept for targets where a ~3.5M parameter backbone will not
  fit, and for cases with enough data to train from scratch.

Note on weights: ``MobileNet_V2_Weights.IMAGENET1K_V1`` will download on first
use. The Dockerfile bakes the checkpoint into the image at build time and sets
``TORCH_HOME``, so the load below resolves from local disk and needs no network
at training time.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class SmallCNN(nn.Module):
    """A compact CNN for image classification.

    Trains from scratch. Adaptive average pooling makes it independent of input
    resolution, but that same pooling discards spatial layout -- which limits it
    on tasks where the arrangement of features matters more than their presence.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Adaptive pooling makes the classifier independent of input resolution.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class PretrainedClassifier(nn.Module):
    """MobileNetV2 with ImageNet weights and a replaced classifier head.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
            Grayscale input is repeated across three channels so the pretrained
            first conv layer can be reused unchanged.
        num_classes: Number of output classes.
        freeze_backbone: If True, only the new head is trained. Faster and
            useful for very small datasets, but usually a point or two worse
            than fine-tuning the whole network.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace the 1000-class ImageNet head with one sized to this dataset.
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.last_channel, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 1:
            # MobileNetV2's first conv expects 3 channels.
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class ClassifierExport(nn.Module):
    """Wraps a trained classifier with a softmax for export.

    Edge Impulse's classifier evaluator expects the model to output per-class
    probabilities (it normalises predictions by their sum, which blows up on
    raw logits). We keep training in logit space (CrossEntropyLoss) and only
    add the softmax at export time.
    """

    def __init__(self, model: nn.Module, softmax: bool = True) -> None:
        super().__init__()
        self.model = model
        # The QNN/NPU export keeps raw logits: an int8-quantized softmax output
        # has very coarse probabilities. Apply softmax on the host instead.
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.softmax(logits, dim=1) if self.softmax else logits


def build_model(
    in_channels: int,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Factory used by the training script.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of output classes.
        pretrained: Use the ImageNet-pretrained MobileNetV2 backbone. Set False
            to fall back to the from-scratch SmallCNN.
        freeze_backbone: Only meaningful when ``pretrained`` is True.
    """
    if pretrained:
        return PretrainedClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        )
    return SmallCNN(in_channels=in_channels, num_classes=num_classes)
