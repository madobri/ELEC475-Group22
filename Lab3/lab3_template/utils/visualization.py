"""
ELEC 475: Lab 3 - Plotting & Visualization Utilities
"""
import os
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

MARGIN = 20  # pixels of space around a prediction that lands outside the image


def show_prediction(ax, image, gt, pred, title: str = ""):
    """
    Draws one image on a matplotlib axis with its nose positions.

    image    : path to an image file, or a PIL image
    gt, pred : (u, v) in the image's pixel coordinates
    Ground truth: green circle. Prediction: red cross. If the prediction lands
    outside the image, the axis expands to show it.
    """
    img = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    w, h = img.size
    gu, gv = float(gt[0]), float(gt[1])
    pu, pv = float(pred[0]), float(pred[1])

    ax.imshow(img)
    ax.plot([gu, pu], [gv, pv], "-", color="yellow", linewidth=1)
    ax.plot(gu, gv, "o", markersize=10, markerfacecolor="none", markeredgecolor="lime", markeredgewidth=2)
    ax.plot(pu, pv, "x", markersize=10, color="red", markeredgewidth=2)

    ax.set_xlim(min(-0.5, pu - MARGIN), max(w - 0.5, pu + MARGIN))
    ax.set_ylim(max(h - 0.5, pv + MARGIN), min(-0.5, pv - MARGIN))   # image y-axis points down
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_ced(curves: Dict[str, Tuple[torch.Tensor, torch.Tensor]], save_path: str = "./plots/ced.png"):
    """
    curves : dict mapping each model name to (x, y) from ced_curve():
             x = sorted errors (pixels), y = percentage of predictions
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(11, 6))
    colors = plt.cm.tab20.colors[::2] + plt.cm.tab20.colors[1::2]    # 10 dark, then 10 light colours
    for i, (name, (x, y)) in enumerate(curves.items()):
        plt.plot(x.numpy(), y.numpy(), label=name, linewidth=2, color=colors[i % len(colors)],
                 linestyle="--" if "corr." in name else "-")
    plt.xlim(0, 200)
    plt.ylim(0, 100)
    plt.title("ELEC 475: Cumulative Error Distribution", fontsize=13)
    plt.xlabel("Error (pixels)", fontsize=11)
    plt.ylabel("Percentage of predictions", fontsize=11)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"CED plot saved to: {save_path}")


def plot_best_worst(model_name: str, best: torch.Tensor, worst: torch.Tensor, errors: torch.Tensor,
                    data: Dict, img_dir: str, save_path: str):
    """
    Saves a 2-row figure: the best predictions (top) and the worst predictions (bottom).

    best, worst : index tensors from best_worst_indices()
    errors      : (N,) tensor of this model's errors
    data        : this model's predictions dictionary (names, pred, gt, ...)
    img_dir     : folder containing the original images
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, len(best), figsize=(3.5 * len(best), 7), squeeze=False)
    for r, (label, idxs) in enumerate([("Best", best), ("Worst", worst)]):
        for c, i in enumerate(idxs.tolist()):
            name = data["names"][i]
            show_prediction(axes[r, c], os.path.join(img_dir, name), data["gt"][i], data["pred"][i],
                            f"{label}: {name}\nerror {float(errors[i]):.1f} px")
    fig.suptitle(f"{model_name}: best (top) and worst (bottom) predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Best/worst figure saved to: {save_path}")


def plot_training_loss(losses: List[float], save_path: str = "./plots/loss_correction.png", title: str = "CorrectionNet"):
    """Post-Lab: training loss curve of the correction model."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, "b-o", label="Train Loss")
    plt.title(f"{title} - MSE Loss (normalized offsets)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training loss curve saved to: {save_path}")
