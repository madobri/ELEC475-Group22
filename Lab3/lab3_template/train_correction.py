"""
ELEC 475: Lab 3 - Correction Model Training Engine
Trains CorrectionNet (or CorrectionNetModified, with --modified) on random patches
from the training set (CorrectionDataset).
The test set is never used here.

Features:
  - Random patch offsets with optional flip & colour jitter augmentation (--no-augment to disable)
  - MSE loss on normalized offsets, Adam with a cosine learning-rate schedule
  - Per-epoch loss and mean offset error in original-image pixels
"""
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from models.correction import CorrectionNet, CorrectionNetModified
from utils.correction_dataset import CorrectionDataset, HALF_PATCH
from utils.visualization import plot_training_loss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_err, n = 0.0, 0.0, 0

    for patches, targets in loader:
        patches = patches.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * patches.size(0)
        total_err += ((outputs.detach() - targets).norm(dim=1) * HALF_PATCH).sum().item()
        n += patches.size(0)

    return total_loss / n, total_err / n


def main():
    parser = argparse.ArgumentParser(description="ELEC 475: Lab 3 Correction Model Trainer")
    parser.add_argument("--data-dir", type=str, default="data/oxford-iiit-pet-noses",
                        help="Directory where the dataset is stored")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--no-augment", action="store_true", help="Disable flip and colour jitter augmentation")
    parser.add_argument("--modified", action="store_true", help="Train CorrectionNetModified (Step 5)")
    parser.add_argument("--save-model", type=str, required=True, help="Path to save the model weights (.pth)")
    parser.add_argument("--plot-file", type=str, default=None, help="Path to save the loss curve plot")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("ELEC 475: Lab 3 Correction Model Training Engine")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model         : {'CorrectionNetModified' if args.modified else 'CorrectionNet'}")
    print(f"Augmentation  : {not args.no_augment}")
    print(f"Hyperparams   : Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Seed={args.seed}")
    print("=" * 72)

    # 1. Load Data
    dataset = CorrectionDataset(root=args.data_dir, augment=not args.no_augment)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, worker_init_fn=seed_worker,
                        persistent_workers=args.num_workers > 0,
                        generator=torch.Generator().manual_seed(args.seed))
    print(f"\nTraining patches per epoch: {len(dataset)}")

    # 2. Build Model
    model = (CorrectionNetModified() if args.modified else CorrectionNet()).to(device)
    print("\nDetailed Layer Specification:")
    summary(model, input_size=(1, 3, 64, 64), device=str(device))

    # 3. Criterion, Optimizer & Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    losses = []
    print("\n" + "=" * 52)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Offset Error (px)':^17} | {'Time (s)':^8}")
    print("-" * 52)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, err = train_one_epoch(model, loader, criterion, optimizer, device)
        scheduler.step()
        losses.append(loss)
        print(f"{epoch:^7d} | {loss:^11.5f} | {err:^17.1f} | {time.time() - t0:^8.1f}")
    print("=" * 52)
    print(f"Training completed in {time.time() - start:.1f}s.")

    # 5. Save Model & Plot
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_model)
    print(f"Model weights saved to: {args.save_model}")
    if args.plot_file:
        plot_training_loss(losses, save_path=args.plot_file,
                           title=f"{type(model).__name__} ({'no augmentation' if args.no_augment else 'with augmentation'})")


if __name__ == "__main__":
    main()
