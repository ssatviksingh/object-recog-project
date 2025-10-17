# src/models/vit.py
import torch.nn as nn
from torchvision import models


def build_model(name: str = "vit_b_16", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Builds a Vision Transformer (ViT) model using torchvision.
    Supports variants: vit_b_16, vit_b_32, vit_l_16, vit_l_32
    """
    name = name.lower()
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
    elif name == "vit_b_32":
        weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
        model = models.vit_b_32(weights=weights)
    elif name == "vit_l_16":
        weights = models.ViT_L_16_Weights.DEFAULT if pretrained else None
        model = models.vit_l_16(weights=weights)
    elif name == "vit_l_32":
        weights = models.ViT_L_32_Weights.DEFAULT if pretrained else None
        model = models.vit_l_32(weights=weights)
    else:
        raise ValueError(f"Unsupported ViT model name: {name}")

    # Modify classifier for CIFAR-10 (10 classes)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    print(f"🧠 Loaded {name.upper()} | Pretrained: {pretrained} | Classes: {num_classes}", flush=True)
    return model
