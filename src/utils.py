# src/utils.py
import os
import random
import torch
import numpy as np


# -----------------------------------------------------------
# ✅ 1. Reproducibility helper
# -----------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU & CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to {seed}", flush=True)


# -----------------------------------------------------------
# ✅ 2. Accuracy metric helper
# -----------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy for the specified values of k.
    Args:
        output: model predictions (logits) of shape [batch_size, num_classes]
        target: ground truth labels of shape [batch_size]
        topk: tuple with k values (default=(1,))
    Returns:
        list of accuracies for each k in topk
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc.item())
        return res


# -----------------------------------------------------------
# ✅ 3. Checkpoint saving helper
# -----------------------------------------------------------
def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pt"):
    """
    Saves model and optimizer state dictionaries to disk.
    Args:
        state: dict containing model state, optimizer state, epoch, etc.
        is_best: whether this checkpoint has the best validation accuracy
        save_dir: directory to store checkpoints
        filename: filename for the saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}", flush=True)

    # Save a copy as best.pt if it's the best model so far
    if is_best:
        best_path = os.path.join(save_dir, "best.pt")
        torch.save(state, best_path)
        print(f"🏆 New best model saved at: {best_path}", flush=True)
