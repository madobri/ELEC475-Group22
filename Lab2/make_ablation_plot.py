"""
ELEC 475 Lab 2 - Master ablation plot (plots/ablation_summary.png)

The template's `python -m utils.visualization --plot-ablation` has no entry point and
train.py never saves per-epoch history, so this script rebuilds the plot from the
"Val Acc (%)" column of each run's printed epoch table.

Values were read off the saved loss_*.png plots (about +/-0.3%); final/best epochs
use the exact printed numbers. Fill in each list with that run's Val Acc values, epoch 1 first. Runs left empty are skipped.
Usage (from Lab2/):  python make_ablation_plot.py
"""
from utils.visualization import plot_ablation_summary

CIFAR10_VAL_ACCS = {
    "LeNet-5 (base)":            [],  # 5 epochs, from the lenet run
    "Modernized LeNet":          [],  # 5 epochs, from the lenet_modern run
    "AlexNet-64 (no norm)":      [46.4, 57.1, 60.2, 63.9, 66.2, 67.0, 70.1, 71.3, 71.3, 73.52],  # 10 epochs, alexnet --norm none
    "AlexNet-64 + LRN":          [45.4, 56.5, 61.3, 66.1, 69.1, 71.2, 72.5, 72.9, 75.4, 75.15],  # 10 epochs, alexnet --norm lrn
    "AlexNet-64 + BN":           [52.5, 60.8, 64.3, 71.0, 73.8, 68.8, 76.4, 79.7, 79.5, 80.0, 80.5, 79.2, 78.5, 81.18, 78.59],  # 15 epochs, alexnet --norm bn --no-augment
    "AlexNet-64 + BN + Aug":     [51.0, 59.0, 66.7, 68.6, 69.0, 69.6, 76.9, 76.9, 75.9, 78.8, 80.0, 80.8, 82.83, 79.7, 81.65],  # 15 epochs, alexnet --norm bn (augmented)
    "PyTorch Zoo AlexNet":       [40.85, 49.37, 53.64, 59.59, 59.89, 60.68, 63.69, 65.03, 64.72, 65.24],  # zoo_alexnet run
}

if __name__ == "__main__":
    runs = {name: {"val_accs": accs} for name, accs in CIFAR10_VAL_ACCS.items() if accs}
    if not runs:
        raise SystemExit("No runs filled in yet: paste the Val Acc values into CIFAR10_VAL_ACCS.")
    plot_ablation_summary(runs, save_path="plots/ablation_summary.png")
