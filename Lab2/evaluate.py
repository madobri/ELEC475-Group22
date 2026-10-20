"""
ELEC 475: Lab 2 - Comprehensive Evaluation & Inspection Engine
Evaluates trained checkpoint weights and reports Top-1 accuracy, Macro/Micro/Weighted F1,
and full confusion matrix & classification report matching modern machine learning conference standards.
"""
import os
import argparse
import torch
import torch.nn as nn
from models import get_model
from utils import get_dataloaders, AverageMeter, ConfusionMatrixTracker


def evaluate_checkpoint(checkpoint_path: str, model_name: str, norm_type: str = "bn",
                        dataset: str = "cifar10", data_dir: str = "./data", batch_size: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 82)
    print("ELEC 475: Model Checkpoint Multi-Metric Evaluation")
    print(f"Loading checkpoint : {checkpoint_path}")
    print(f"Model Architecture : {model_name.upper()} (Norm: {norm_type.upper() if model_name == 'alexnet' else 'N/A'})")
    print(f"Dataset            : {dataset.upper()}")
    print("=" * 82)

    # 1. DataLoader
    _, val_loader, num_classes, class_names = get_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        use_augmentation=False,
        target_size=64
    )

    # 2. Model & Checkpoint Loading
    model = get_model(model_name, num_classes=num_classes, in_channels=3, norm_type=norm_type)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found! Evaluating with uninitialized weights.")

    model = model.to(device)
    model.eval()

    # 3. Evaluation Loop with Confusion Matrix & Multi-Metric Tracking
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter("Loss")
    tracker = ConfusionMatrixTracker(num_classes=num_classes, class_names=class_names)

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), imgs.size(0))
            tracker.update(outputs, targets)

    # 4. Generate Classification Report & Metrics
    metrics = tracker.compute_metrics()

    print("\n" + "=" * 60)
    print("Overall Evaluation Summary:")
    print(f"  Test Loss            : {loss_meter.avg:.4f}")
    print(f"  Top-1 Accuracy       : {metrics['top1_accuracy']:.2f}%")
    print(f"  Macro F1-Score       : {metrics['macro_f1']:.2f}%")
    print(f"  Micro F1-Score       : {metrics['micro_f1']:.2f}%")
    print(f"  Weighted F1-Score    : {metrics['weighted_f1']:.2f}%")
    print("=" * 60)

    print("\nDetailed Academic Classification Report:")
    print(tracker.classification_report())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELEC 475: Checkpoint Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--model", type=str, default="lenet", choices=["lenet", "lenet_modern", "alexnet"])
    parser.add_argument("--norm", type=str, default="bn", choices=["none", "lrn", "bn"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny_imagenet"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        norm_type=args.norm,
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
