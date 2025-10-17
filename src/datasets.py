# src/datasets.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_cifar10_dataloaders(
    data_dir,
    batch_size=128,
    num_workers=4,
    augment=True,
    seed=42,
    model_name="resnet18"
):
    """
    Returns CIFAR-10 train/val/test dataloaders.
    ✅ Preserves augmentations
    ✅ Dynamically resizes to 224x224 for Vision Transformers (ViT)
    """

    # 🔹 Check if using Vision Transformer
    if model_name.lower().startswith("vit"):
        image_size = 224
        print("🖼️ Resizing CIFAR-10 images to 224×224 for ViT...", flush=True)
    else:
        image_size = 32

    # 🔹 Define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4 if image_size == 32 else 0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.247, 0.243, 0.261])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.247, 0.243, 0.261])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.243, 0.261])
    ])

    # 🔹 Load datasets
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # 🔹 Split train -> train/val (10% validation)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_set))
    val_size = int(0.1 * len(train_set))
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(train_set, val_idx)

    # 🔹 Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
