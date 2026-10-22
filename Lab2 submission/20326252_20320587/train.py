"""
ELEC 475: Lab 2 - Main Model Training Engine
Supports LeNet-5, Modernized LeNet, AlexNet-64 (None/LRN/BN), and PyTorch Model Zoo AlexNet
across CIFAR-10 and Tiny ImageNet-200.

Features:
  - Precise GPU-synchronized epoch timing via Timer
  - Model complexity profiling (FLOPs & Parameters) via ModelProfiler
  - Multi-metric evaluation: Cross-Entropy Loss, Top-1 Accuracy, and Macro F1 via ConfusionMatrixTracker
  - Early Stopping mechanism monitoring Validation Loss, Accuracy, or Macro F1
  - Real-time throughput (samples/second) logging
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from models import get_model
from utils import (
    get_dataloaders,
    AverageMeter,
    compute_accuracy,
    ConfusionMatrixTracker,
    EarlyStopping,
    plot_training_curves,
    Timer,
    ModelProfiler,
    seed_everything
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(outputs, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), imgs.size(0))
        acc_meter.update(acc, imgs.size(0))

    return loss_meter.avg, acc_meter.avg


def evaluate(model, loader, criterion, device, num_classes=10):
    model.eval()
    loss_meter = AverageMeter("Loss")
    tracker = ConfusionMatrixTracker(num_classes=num_classes)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), imgs.size(0))
            tracker.update(outputs, targets)

    metrics = tracker.compute_metrics()
    return loss_meter.avg, metrics["top1_accuracy"], metrics["macro_f1"]


def main():
    parser = argparse.ArgumentParser(description="ELEC 475: Lab 2 Model Trainer")
    parser.add_argument("--model", type=str, default="lenet",
                        choices=["lenet", "lenet_modern", "alexnet", "zoo_alexnet"],
                        help="Model architecture: 'lenet', 'lenet_modern', 'alexnet', 'zoo_alexnet'")
    parser.add_argument("--norm", type=str, default="bn", choices=["none", "lrn", "bn"],
                        help="Normalization for AlexNet: 'none', 'lrn', 'bn'")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for Model Zoo architectures")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny_imagenet"],
                        help="Dataset: 'cifar10' or 'tiny_imagenet'")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory where dataset is stored")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across CPU/GPU/NumPy")
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentation")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save best model checkpoint (.pth)")
    parser.add_argument("--plot-file", type=str, default=None, help="Path to save loss and accuracy curve plot")
    # Early Stopping Options
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping mechanism")
    parser.add_argument("--early-stopping-metric", type=str, default="val_loss",
                        choices=["val_loss", "val_acc", "val_f1"],
                        help="Validation metric to monitor for early stopping: 'val_loss', 'val_acc', or 'val_f1'")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum delta threshold to qualify as improvement")
    args = parser.parse_args()

    # Enforce global reproducibility across all RNG subsystems
    seed_everything(args.seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("ELEC 475: Lab 2 Training Engine")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model         : {args.model.upper()} (Norm: {args.norm.upper() if args.model == 'alexnet' else 'N/A'})")
    print(f"Dataset       : {args.dataset.upper()} (Augmentation: {not args.no_augment})")
    print(f"Hyperparams   : Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Seed={args.seed}")
    print(f"Early Stopping: {'Enabled (Monitor: ' + args.early_stopping_metric + ', Patience: ' + str(args.patience) + ')' if args.early_stopping else 'Disabled'}")
    print("=" * 88)

    # 1. Load Data
    print("\nInitializing DataLoaders...")
    train_loader, val_loader, num_classes, classes = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augment,
        target_size=64
    )
    num_train_samples = len(train_loader.dataset)
    print(f"Train batches: {len(train_loader)} ({num_train_samples} samples), Val batches: {len(val_loader)}, Classes: {num_classes}")

    # 2. Build Model
    model = get_model(
        args.model,
        num_classes=num_classes,
        in_channels=3,
        norm_type=args.norm,
        pretrained=args.pretrained
    )
    model = model.to(device)

    # 3. Model Profiler (FLOPs, MACs, Parameters)
    profiler = ModelProfiler(model, input_size=(1, 3, 64, 64), device=str(device))
    profiler.print_summary(model_name=f"{args.model.upper()} ({args.norm})")

    # Print summary
    print("\nDetailed Layer Specification:")
    summary(model, input_size=(1, 3, 64, 64), device=str(device))

    # 4. Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 5. Early Stopping Initialization
    early_stopper = None
    if args.early_stopping:
        mode = "min" if args.early_stopping_metric == "val_loss" else "max"
        early_stopper = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            mode=mode,
            monitor=args.early_stopping_metric,
            save_path=args.save_model,
            verbose=True
        )

    # 6. Training Loop with High-Precision CUDA Synchronization Timer
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_val_f1 = 0.0

    print("\n" + "=" * 94)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc (%)':^14} | {'Val Loss':^10} | {'Val Acc (%)':^12} | {'Val F1 (%)':^11} | {'Time (s)':^9} | {'Throughput':^10}")
    print("-" * 94)

    with Timer("Full Training Run") as total_timer:
        for epoch in range(1, args.epochs + 1):
            with Timer(f"Epoch {epoch}") as epoch_timer:
                tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                v_loss, v_acc, v_f1 = evaluate(model, val_loader, criterion, device, num_classes=num_classes)

            elapsed = epoch_timer.elapsed
            throughput = num_train_samples / elapsed if elapsed > 0 else 0.0

            train_losses.append(tr_loss)
            val_losses.append(v_loss)
            train_accs.append(tr_acc)
            val_accs.append(v_acc)

            print(f"{epoch:^7d} | {tr_loss:^11.4f} | {tr_acc:^14.2f} | {v_loss:^10.4f} | {v_acc:^12.2f} | {v_f1:^11.2f} | {elapsed:^9.2f} | {throughput:^8.1f} s/s")

            # Check and update best metrics
            if v_acc > best_val_acc:
                best_val_acc = v_acc
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1

            # Standard model checkpoint save (if early stopping is not active)
            if args.save_model and not args.early_stopping:
                if v_acc == best_val_acc:
                    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
                    torch.save(model.state_dict(), args.save_model)

            # Early Stopping Evaluation
            if early_stopper is not None:
                current_monitored = v_loss if args.early_stopping_metric == "val_loss" else (v_acc if args.early_stopping_metric == "val_acc" else v_f1)
                should_stop = early_stopper.step(current_monitored, model=model)
                if should_stop:
                    print(f"\n[EarlyStopping] Halting training loop early at Epoch {epoch} due to plateau in {args.early_stopping_metric}.")
                    early_stopper.restore_best_weights(model)
                    break

    print("=" * 94)
    print(f"Training completed in {total_timer.elapsed:.2f}s.")
    print(f"Best Val Accuracy: {best_val_acc:.2f}% | Best Val Macro F1: {best_val_f1:.2f}%")
    print(f"Model Complexity : {profiler.get_params_millions():.3f}M Parameters | {profiler.get_flops_millions():.2f}M FLOPs")
    if args.save_model:
        print(f"Best model weights saved to: {args.save_model}")

    # 7. Save Plots
    if args.plot_file:
        model_display = f"{args.model} ({args.norm})" if args.model == "alexnet" else args.model
        plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                             save_path=args.plot_file, model_name=model_display)


if __name__ == "__main__":
    main()
