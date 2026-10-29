import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset import PetNoseDataset, denormalize, IMG_SIZE

if __name__ == "__main__":
    # 1. Check every sample in both splits
    for split in ["train", "test"]:
        loader = DataLoader(PetNoseDataset(split=split), batch_size=64)
        n = 0
        for images, labels, _ in loader:
            assert images.shape[1:] == (3, IMG_SIZE, IMG_SIZE), images.shape
            assert ((labels >= 0) & (labels < IMG_SIZE)).all(), labels
            n += images.shape[0]
        print(f"{split}: {n} samples OK")

    # 2. Plot 8 training images with their nose labels
    ds = PetNoseDataset(split="train")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        image, label, _ = ds[i]
        ax.imshow(denormalize(image).permute(1, 2, 0))  # (3,H,W) -> (H,W,3)
        ax.scatter(label[0], label[1], c="red", s=40)    # u = x, v = y
        ax.set_title(f"({label[0]:.0f}, {label[1]:.0f})")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/dataset_check.png")
    print("Saved plots/dataset_check.png")