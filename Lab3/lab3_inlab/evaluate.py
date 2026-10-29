"""
ELEC 475: Lab 3 - Localization Evaluation Engine
Evaluates the saved test-set predictions of SnoutNet, SnoutNet-A and SnoutNet-V
against a mean-location baseline, and combines the three models into ensembles.

Uses your functions in utils/metrics.py and utils/ensemble.py. Any function you
have not implemented yet is skipped, so run this script after each In-Lab step.

Post-Lab: once correct.py has saved corrected predictions, they are added to the
tables and plots automatically.
"""
import os
import argparse
import torch
from utils.metrics import (euclidean_errors, baseline_predictions, ced_curve,
                           best_worst_indices, print_results)
from utils.ensemble import ensemble_mean, ensemble_median
from utils.visualization import plot_ced, plot_best_worst

MODEL_NAMES = {"snoutnet": "SnoutNet", "alexnet": "SnoutNet-A", "vgg16": "SnoutNet-V"}
# Post-Lab: first-stage predictions that correct.py corrects, and the correction model checkpoints
CORRECTED_MODELS = {"snoutnet": "SnoutNet", "vgg16": "SnoutNet-V", "ensemble_mean": "Ensemble (mean)"}
CORRECTIONS = {"correction_noaug": "+ corr.", "correction_aug": "+ corr. (aug)",
               "correction_mod_noaug": "+ mod. corr.", "correction_mod_aug": "+ mod. corr. (aug)"}

_skipped = set()


def run(fn, *args):
    """Calls fn, or returns None (with a one-time message) if it is not implemented yet."""
    result = fn(*args)
    if result is None and fn.__name__ not in _skipped:
        print(f"[TODO] {fn.__name__}() is not implemented yet: skipped")
        _skipped.add(fn.__name__)
    return result


def load_predictions(pred_dir: str):
    data = {}
    for key in MODEL_NAMES:
        path = os.path.join(pred_dir, f"{key}.pt")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}: run  python predict.py --model all  first.")
        data[key] = torch.load(path, weights_only=True)
    names = data["snoutnet"]["names"]
    assert all(d["names"] == names for d in data.values()), "Predictions files disagree on image order."
    return data


def evaluate_predictions(data_dir: str = "data/oxford-iiit-pet-noses", pred_dir: str = "predictions",
                         plot_dir: str = "plots"):
    img_dir = os.path.join(data_dir, "images-original", "images")
    train_file = os.path.join(data_dir, "train_noses.txt")

    print("=" * 72)
    print("ELEC 475: Lab 3 Localization Evaluation")
    print(f"Predictions   : {pred_dir}")
    print("=" * 72)

    # 1. Load Predictions
    data = load_predictions(pred_dir)
    gt = data["snoutnet"]["gt"]
    orig_size = data["snoutnet"]["orig_size"]

    # 2. Errors of the Three Models (Step 3)
    model_errors = {key: run(euclidean_errors, d["pred"], gt) for key, d in data.items()}
    if any(e is None for e in model_errors.values()):
        return  # every later step needs euclidean_errors()

    rows = {}
    baseline = run(baseline_predictions, orig_size, train_file, img_dir)
    if baseline is not None:
        rows["Baseline (mean)"] = dict(errors=euclidean_errors(baseline, gt), params=0, ms=0.0)
    for key, d in data.items():
        rows[MODEL_NAMES[key]] = dict(errors=model_errors[key], params=d["params"], ms=d["ms_per_image"])

    # 3. Ensembles (Step 6)
    preds = [d["pred"] for d in data.values()]
    total_params = sum(d["params"] for d in data.values())
    total_ms = sum(d["ms_per_image"] for d in data.values())
    for name, fn in [("Ensemble (mean)", ensemble_mean), ("Ensemble (median)", ensemble_median)]:
        ens = run(fn, preds)
        if ens is not None:
            rows[name] = dict(errors=euclidean_errors(ens, gt), params=total_params, ms=total_ms)

    # 4. Corrected Predictions (Post-Lab), included once correct.py has saved them
    corrected = {}
    for key, base_name in CORRECTED_MODELS.items():
        for tag, suffix in CORRECTIONS.items():
            path = os.path.join(pred_dir, f"{key}_{tag}.pt")
            if os.path.exists(path):
                d = torch.load(path, weights_only=True)
                name = f"{base_name} {suffix}"
                corrected[name] = (f"{key}_{tag}", d)
                rows[name] = dict(errors=euclidean_errors(d["pred"], gt), params=d["params"], ms=d["ms_per_image"])

    # 5. Accuracy & Cost Tables (Steps 3 and 6)
    print_results(rows)

    # 6. Cumulative Error Distribution (Step 4)
    curves = {name: run(ced_curve, row["errors"]) for name, row in rows.items()}
    if all(c is not None for c in curves.values()):
        plot_ced(curves, save_path=os.path.join(plot_dir, "ced.png"))

    # 7. Best & Worst Predictions (Step 5, and Post-Lab corrected predictions)
    figures = [(MODEL_NAMES[key], key, d) for key, d in data.items()]
    figures += [(name, file_key, d) for name, (file_key, d) in corrected.items()]
    for name, file_key, d in figures:
        errors = rows[name]["errors"]
        result = run(best_worst_indices, errors)
        if result is not None:
            best, worst = result
            plot_best_worst(name, best, worst, errors, d, img_dir,
                            save_path=os.path.join(plot_dir, f"best_worst_{file_key}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELEC 475: Lab 3 Localization Evaluation")
    parser.add_argument("--data-dir", type=str, default="data/oxford-iiit-pet-noses",
                        help="Directory where the dataset is stored")
    parser.add_argument("--pred-dir", type=str, default="predictions",
                        help="Directory containing the predictions files from predict.py")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save the plots")
    args = parser.parse_args()

    evaluate_predictions(data_dir=args.data_dir, pred_dir=args.pred_dir, plot_dir=args.plot_dir)
