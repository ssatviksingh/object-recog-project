# src/plot_results.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_model_comparison(save_path=None):
    """
    Plots accuracy and loss trend for all trained models.
    """

    # ✅ Model performance data
    model_data = {
        "Model": ["ResNet18", "EfficientNet-B0", "ConvNeXt-Tiny", "ViT-B/16"],
        "Accuracy (%)": [83.26, 87.45, 88.97, 96.15],
        "Loss": [0.5051, 0.41, 0.33, 0.1491],
        "Type": ["Traditional CNN", "Modern CNN", "Next-Gen CNN", "Transformer"]
    }

    df = pd.DataFrame(model_data)
    x = np.arange(len(df))
    y_acc = df["Accuracy (%)"].astype(float).to_numpy()
    y_loss = df["Loss"].astype(float).to_numpy()

    # ✅ Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy line
    ax1.plot(x, y_acc, marker='o', linestyle='-', color="#1f77b4", linewidth=3, markersize=8, label="Accuracy (%)")
    ax1.fill_between(x, y_acc, np.min(y_acc), color="#1f77b4", alpha=0.15)
    ax1.set_xlabel("Model Architecture", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=0, fontsize=11)
    ax1.set_ylim(80, 100)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ✅ Annotate accuracy values
    for i, val in enumerate(y_acc):
        ax1.text(i, val + 0.7, f"{val:.2f}%", ha='center', fontsize=10, color='black')

    # ✅ Add secondary axis for loss
    ax2 = ax1.twinx()
    ax2.plot(x, y_loss, marker='s', linestyle='--', color="#d62728", linewidth=2, markersize=6, label="Loss")
    ax2.set_ylabel("Loss", fontsize=12, color="#d62728")
    ax2.tick_params(axis='y', labelcolor="#d62728")
    ax2.set_ylim(0, 0.6)

    # ✅ Title and layout
    plt.title("📊 Accuracy & Loss Improvement Trend: CNNs → Transformer", fontsize=14, pad=15)
    fig.tight_layout()

    # ✅ Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # ✅ Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved at: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Save figure to project folder
    plot_model_comparison(save_path="results/model_comparison.png")
