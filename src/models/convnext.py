# src/models/convnext.py
import torch.nn as nn
from torchvision import models

def build_model(name: str = "convnext_tiny", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Builds a ConvNeXt model (tiny, small, base, large) using torchvision.
    Automatically replaces the final classifier layer for CIFAR-10 (10 classes).
    """
    name = name.lower()
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
    elif name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
    elif name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
    elif name == "convnext_large":
        weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
        model = models.convnext_large(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {name}")

    # Modify classifier for CIFAR-10
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    print(f"🧠 Loaded {name.upper()} | Pretrained: {pretrained} | Classes: {num_classes}", flush=True)
    return model
