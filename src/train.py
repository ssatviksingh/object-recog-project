# src/train.py
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from src.datasets import get_cifar10_dataloaders
from src.models import resnet, efficientnet, convnext, vit
from src.utils import set_seed, accuracy, save_checkpoint
from src.config import CHECKPOINT_DIR

def build_model(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()
    if name.startswith("resnet"):
        return resnet.build_model(name, num_classes, pretrained)
    elif name.startswith("efficientnet"):
        return efficientnet.build_model(name, num_classes, pretrained)
    elif name.startswith("convnext"):
        return convnext.build_model(name, num_classes, pretrained)
    elif name.startswith("vit"):
        return vit.build_model(name, num_classes, pretrained)
    else:
        raise ValueError(f"❌ Unsupported model name: {name}")
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=os.path.join(CHECKPOINT_DIR, "run"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast("cuda", enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        running_acc += acc1 * batch_size / 100.0
        n += batch_size

        # print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  [Batch {batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}", flush=True)

    return running_loss / n, (running_acc / n) * 100.0


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            running_acc += acc1 * batch_size / 100.0
            n += batch_size
    return running_loss / n, (running_acc / n) * 100.0


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print("🚀 Starting training script...", flush=True)

    # ✅ Load CIFAR-10
    if args.dataset.lower() == "cifar10":
        print("📥 Loading CIFAR-10 dataset...", flush=True)
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, model_name=args.model)
        num_classes = 10
    else:
        raise NotImplementedError("Only CIFAR-10 is implemented in this scaffold.")

    # ✅ Disable pin_memory for CPU (Windows fix)
    if args.device == "cpu":
        for loader in [train_loader, val_loader, test_loader]:
            loader.pin_memory = False

    print("✅ Dataloaders ready.", flush=True)
    print(f"   Train batches: {len(train_loader)}", flush=True)
    print(f"   Val batches: {len(val_loader)}", flush=True)
    print(f"   Test batches: {len(test_loader)}", flush=True)

    # ✅ Model setup
    print(f"🧠 Building model: {args.model} ...", flush=True)
    device = torch.device(args.device)
    model = build_model(name=args.model, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0

    print("🏁 Beginning training loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"\n📘 Epoch {epoch}/{args.epochs} - time: {elapsed:.1f}s", flush=True)
        print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%", flush=True)
        print(f"  Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.2f}%", flush=True)

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "args": vars(args)
        }
        save_checkpoint(ckpt, is_best, args.save_dir, filename=f"epoch_{epoch}.pt")
        if is_best:
            save_checkpoint(ckpt, True, args.save_dir, filename="best.pt")

    # ✅ Final test evaluation
    print("🧪 Evaluating on test set...", flush=True)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%", flush=True)

    # Save final model
    final_ckpt_path = os.path.join(args.save_dir, "final.pt")
    torch.save({"model_state_dict": model.state_dict(), "test_acc": test_acc}, final_ckpt_path)
    print(f"✅ Training complete. Final model saved to {final_ckpt_path}", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
