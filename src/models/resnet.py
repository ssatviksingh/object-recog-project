# src/models/resnet.py
import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str = "resnet18", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Builds a ResNet model (or similar) from torchvision with updated weights API.

    Args:
        name (str): model architecture name, e.g., 'resnet18', 'resnet34', 'resnet50'
        num_classes (int): number of output classes
        pretrained (bool): whether to use pretrained ImageNet weights

    Returns:
        model (nn.Module): configured PyTorch model
    """

    # Map model names to torchvision constructors and weight enums
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
    elif name == "resnet152":
        weights = models.ResNet152_Weights.DEFAULT if pretrained else None
        model = models.resnet152(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {name}")

    # Modify the final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    print(f"🧠 Loaded {name.upper()} model | Pretrained: {pretrained} | Classes: {num_classes}", flush=True)
    return model
