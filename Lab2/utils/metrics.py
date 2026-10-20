"""
ELEC 475: Lab 2 - Performance Metrics, Classification Evaluation & Early Stopping
Provides:
  1. AverageMeter: Running arithmetic mean accumulator.
  2. compute_accuracy: Top-k classification accuracy calculation.
  3. ConfusionMatrixTracker: Confusion matrix, per-class metrics (TP/FP/FN/TN, Precision, Recall, Specificity, F1),
     and global aggregations (Macro F1, Micro F1, Weighted F1).
  4. EarlyStopping: Metric-driven training termination with patience, min_delta, and model restoration.
"""
import os
import math
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn


class AverageMeter:
    """
    Computes and stores the running average, sum, and current value.
    """
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Computes precision@k (Top-k accuracy) for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class ConfusionMatrixTracker:
    """
    Tracks and computes a confusion matrix across multiple batches,
    generating per-class Precision, Recall, Specificity, F1-scores,
    as well as Macro, Micro, and Weighted F1 aggregations.

    Confusion Matrix Layout:
      Rows: Ground Truth Classes
      Columns: Predicted Classes
      Element (r, c): Number of samples with true class r predicted as class c.
    """
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)
        self.total_samples = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates confusion matrix with predictions and ground truth targets.
        Args:
            preds: Tensor of predicted class indices (B,) or logits (B, C).
            targets: Tensor of ground truth class indices (B,).
        """
        with torch.no_grad():
            if preds.dim() > 1:
                preds = torch.argmax(preds, dim=1)
            preds = preds.view(-1).cpu()
            targets = targets.view(-1).cpu()

            for t, p in zip(targets, preds):
                t_idx = int(t.item()) if hasattr(t, "item") else int(t)
                p_idx = int(p.item()) if hasattr(p, "item") else int(p)
                if 0 <= t_idx < self.num_classes and 0 <= p_idx < self.num_classes:
                    self.matrix[t_idx, p_idx] += 1
                    self.total_samples += 1

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Calculates comprehensive classification metrics from the accumulated confusion matrix.
        Returns:
            Dictionary containing:
              - per_class: Dict with TP, FP, FN, TN, precision, recall, specificity, f1, support
              - top1_accuracy: Global classification accuracy (%)
              - macro_precision, macro_recall, macro_f1: Unweighted averages (%)
              - micro_precision, micro_recall, micro_f1: Globally pooled metrics (%)
              - weighted_f1: Prevalence-weighted F1 (%)
        """
        mat = self.matrix.float()
        total = self.total_samples

        tp = torch.zeros(self.num_classes)
        fp = torch.zeros(self.num_classes)
        fn = torch.zeros(self.num_classes)
        tn = torch.zeros(self.num_classes)
        support = torch.zeros(self.num_classes)

        for c in range(self.num_classes):
            tp[c] = mat[c, c]
            fp[c] = mat[:, c].sum() - tp[c]
            fn[c] = mat[c, :].sum() - tp[c]
            tn[c] = total - (tp[c] + fp[c] + fn[c])
            support[c] = mat[c, :].sum()

        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        f1 = 2.0 * (precision * recall) / (precision + recall + eps)

        # Handle classes with 0 support
        for c in range(self.num_classes):
            if support[c] == 0:
                precision[c] = 0.0
                recall[c] = 0.0
                f1[c] = 0.0

        # Global and Macro Aggregations
        top1_acc = (tp.sum() / max(float(total), 1.0)).item() * 100.0
        macro_prec = precision.mean().item() * 100.0
        macro_rec = recall.mean().item() * 100.0
        macro_f1 = f1.mean().item() * 100.0

        # Micro Aggregations (Micro Precision = Micro Recall = Micro F1 = Top-1 Accuracy in single-label multi-class)
        micro_prec = (tp.sum() / (tp.sum() + fp.sum() + eps)).item() * 100.0
        micro_rec = (tp.sum() / (tp.sum() + fn.sum() + eps)).item() * 100.0
        micro_f1 = 2.0 * (micro_prec * micro_rec) / (micro_prec + micro_rec + eps)

        # Weighted F1
        weights = support / max(float(total), 1.0)
        weighted_f1 = (f1 * weights).sum().item() * 100.0

        per_class_data = {}
        for c in range(self.num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"Class_{c}"
            per_class_data[name] = {
                "tp": int(tp[c].item()),
                "fp": int(fp[c].item()),
                "fn": int(fn[c].item()),
                "tn": int(tn[c].item()),
                "support": int(support[c].item()),
                "precision": precision[c].item() * 100.0,
                "recall": recall[c].item() * 100.0,
                "specificity": specificity[c].item() * 100.0,
                "f1": f1[c].item() * 100.0
            }

        return {
            "top1_accuracy": top1_acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class_data,
            "total_samples": total
        }

    def classification_report(self) -> str:
        """
        Formats a formatted ASCII classification report table matching academic standards.
        """
        metrics = self.compute_metrics()
        lines = []
        lines.append("=" * 82)
        lines.append(f"{'Class Category':<18} | {'Precision (%)':^14} | {'Recall (%)':^12} | {'F1-Score (%)':^14} | {'Support':^8}")
        lines.append("-" * 82)

        for name, data in metrics["per_class"].items():
            lines.append(f"{name:<18} | {data['precision']:^14.2f} | {data['recall']:^12.2f} | {data['f1']:^14.2f} | {data['support']:^8d}")

        lines.append("-" * 82)
        lines.append(f"{'Top-1 Accuracy':<18} | {'':^14} | {'':^12} | {metrics['top1_accuracy']:^14.2f} | {metrics['total_samples']:^8d}")
        lines.append(f"{'Macro Average':<18} | {metrics['macro_precision']:^14.2f} | {metrics['macro_recall']:^12.2f} | {metrics['macro_f1']:^14.2f} | {metrics['total_samples']:^8d}")
        lines.append(f"{'Micro Average':<18} | {metrics['micro_precision']:^14.2f} | {metrics['micro_recall']:^12.2f} | {metrics['micro_f1']:^14.2f} | {metrics['total_samples']:^8d}")
        lines.append(f"{'Weighted Average':<18} | {'--':^14} | {'--':^12} | {metrics['weighted_f1']:^14.2f} | {metrics['total_samples']:^8d}")
        lines.append("=" * 82)
        return "\n".join(lines)


class EarlyStopping:
    """
    Early Stopping mechanism to halt training when a monitored validation metric ceases to improve.
    Supports monitoring validation loss ('min' mode) or validation accuracy/F1 ('max' mode).
    
    Why Monitor Validation Loss by Default?
      Cross-Entropy Loss is continuous, differentiable, and reflects output confidence. When a model
      begins to overfit, validation loss increases immediately due to penalized confident mistakes,
      acting as an early-warning 'canary' before discrete classification accuracy degrades.
      
    When to Monitor F1 or Recall?
      In class-imbalanced datasets or safety-critical domains (e.g., medical diagnostics), overall loss
      can be dominated by easy majority classes while performance on rare/critical classes collapses.
      Monitoring Macro F1 or Recall directly protects against this failure mode.
    """
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        monitor: str = "val_loss",
        save_path: Optional[str] = None,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.monitor = monitor
        self.save_path = save_path
        self.verbose = verbose

        if self.mode not in ["min", "max"]:
            raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {self.mode}")

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def step(self, current_value: float, model: Optional[nn.Module] = None) -> bool:
        """
        Evaluates the current epoch's metric against the best observed score.
        Args:
            current_value: Monitored metric value for the current epoch.
            model: PyTorch model instance whose state dict will be cached if an improvement occurs.
        Returns:
            True if early stopping patience has expired and training should halt; False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_value
            self._save_checkpoint(current_value, model)
            return False

        if self.mode == "min":
            improvement = (self.best_score - current_value) >= self.min_delta
        else:
            improvement = (current_value - self.best_score) >= self.min_delta

        if improvement:
            if self.verbose:
                print(f"[EarlyStopping] {self.monitor} improved from {self.best_score:.4f} to {current_value:.4f}. Resetting patience counter.")
            self.best_score = current_value
            self.counter = 0
            self._save_checkpoint(current_value, model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] {self.monitor} did not improve ({current_value:.4f} vs best {self.best_score:.4f}). Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Patience limit reached ({self.patience} epochs). Triggering early termination!")

        return self.early_stop

    def _save_checkpoint(self, score: float, model: Optional[nn.Module]):
        if model is not None:
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"[EarlyStopping] Best model checkpoint saved to: {self.save_path}")

    def restore_best_weights(self, model: nn.Module):
        """
        Restores the model's weights to the state corresponding to the optimal monitored score.
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"[EarlyStopping] Successfully restored optimal model weights ({self.monitor}: {self.best_score:.4f}).")
        elif self.save_path and os.path.exists(self.save_path):
            state_dict = torch.load(self.save_path, map_location="cpu")
            model.load_state_dict(state_dict)
            if self.verbose:
                print(f"[EarlyStopping] Restored model weights from file: {self.save_path}")
