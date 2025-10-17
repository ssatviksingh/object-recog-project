# src/eval.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Import all supported model types
from src.models import resnet, efficientnet, convnext, vit


# ======================
# Utility Functions
# ======================
def get_cifar10_loader(data_dir="data", batch_size=128, num_workers=0):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # ensure compatibility with modern nets
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


def build_model_auto(model_name: str, num_classes: int, pretrained: bool = False):
    """Automatically selects the correct model file."""
    model_name = model_name.lower()
    if model_name.startswith("resnet"):
        return resnet.build_model(model_name, num_classes, pretrained)
    elif model_name.startswith("efficientnet"):
        return efficientnet.build_model(model_name, num_classes, pretrained)
    elif model_name.startswith("convnext"):
        return convnext.build_model(model_name, num_classes, pretrained)
    elif model_name.startswith("vit"):
        return vit.build_model(model_name, num_classes, pretrained)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Confusion matrix saved to: {save_path}")


# ======================
# Main Evaluation
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    print(f"🔍 Loading checkpoint: {args.checkpoint}")

    # Load checkpoint
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt["args"]["model"] if "args" in ckpt and "model" in ckpt["args"] else "resnet18"

    print(f"🧠 Loaded model: {model_name.upper()} | Pretrained: False | Classes: 10")

    # Build the model dynamically
    model = build_model_auto(model_name, num_classes=10, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load dataset
    print("📥 Loading CIFAR-10 test data...")
    test_loader = get_cifar10_loader("data", batch_size=args.batch_size)
    classes = test_loader.dataset.classes

    # Evaluate
    print("🧪 Evaluating model...")
    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * correct / total
    avg_loss = test_loss / total

    print(f"\n✅ Test Accuracy: {acc:.2f}%")
    print(f"Loss: {avg_loss:.4f}\n")

    print("Detailed Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Save confusion matrix
    os.makedirs("results", exist_ok=True)
    model_short = model_name.replace("_", "").lower()
    save_path = os.path.join("results", f"confusion_matrix_{model_short}_cifar10.png")
    plot_confusion_matrix(all_labels, all_preds, classes, save_path)


if __name__ == "__main__":
    main()
