# src/models/efficientnet.py
import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str = "efficientnet_b0", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Builds an EfficientNet model (B0, B1, or B3) using torchvision.
    Automatically replaces the final classifier layer for CIFAR-10 (10 classes).
    Compatible with PyTorch >= 2.0 and torchvision >= 0.13.

    Args:
        name (str): 'efficientnet_b0', 'efficientnet_b1', or 'efficientnet_b3'
        num_classes (int): number of output classes
        pretrained (bool): whether to use pretrained ImageNet weights

    Returns:
        model (nn.Module): configured EfficientNet model
    """

    name = name.lower()
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
    elif name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {name}")

    # Modify the classifier for CIFAR-10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    print(f"🧠 Loaded {name.upper()} | Pretrained: {pretrained} | Classes: {num_classes}", flush=True)
    return model
