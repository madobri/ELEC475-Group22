"""
ELEC 475: Lab 3 - Localization Metrics & Mean-Location Baseline (Starter Template)
Provides:
  1. euclidean_errors: Per-image localization error in original-image pixels.
  2. error_stats: Min, mean, median, max and standard deviation of the errors.
  3. baseline_predictions: Mean-location baseline built from the training labels.
  4. ced_curve: Points of the cumulative error distribution (CED) curve.
  5. best_worst_indices: Indices of the best and worst predictions.
  6. print_results: Accuracy and cost tables (provided).
"""
import os
from typing import Dict, Tuple
import torch
from PIL import Image


def euclidean_errors(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean localization error for every image.

    Args:
        pred : torch.Tensor, shape (N, 2), float. Predicted (u, v) in original-image pixels
        gt   : torch.Tensor, shape (N, 2), float. Ground-truth (u, v) in original-image pixels

    Returns:
        torch.Tensor, shape (N,), float. The Euclidean distance between pred[i] and gt[i],
        in pixels, for each image i
    """
    # ======================================================================
    # TODO [IN-LAB STEP 3]: Implement the Euclidean Error
    #
    # Requirements:
    #   1. For each image, compute the distance between the predicted (u, v)
    #      and the ground-truth (u, v)
    # ======================================================================
    pass


def error_stats(errors: torch.Tensor) -> Dict[str, float]:
    """
    Summarizes a set of errors.

    Args:
        errors : torch.Tensor, shape (N,), float. Errors in pixels

    Returns:
        dict with exactly the keys "min", "mean", "median", "max" and "std".
        Each value is a Python float (not a tensor).
    """
    # ======================================================================
    # TODO [IN-LAB STEP 3]: Implement the Summary Statistics
    #
    # Requirements:
    #   1. Compute each statistic with errors.min(), .mean(), .median(), .max() and .std()
    #   2. Convert each result to a Python float with .item()
    # ======================================================================
    pass


def baseline_predictions(orig_size: torch.Tensor, train_file: str, img_dir: str) -> torch.Tensor:
    """
    The baseline uses the average snout position from the training set as its
    prediction for every test image.

    Args:
        orig_size  : torch.Tensor, shape (N, 2), float. Each test image's size (W, H)
        train_file : path to train_noses.txt
        img_dir    : folder containing the original images

    Returns:
        torch.Tensor, shape (N, 2), float. Baseline prediction (u, v) for each test
        image, in original-image pixels: column 0 is u, column 1 is v
    """
    # ======================================================================
    # TODO [IN-LAB STEP 3]: Implement the Mean-Location Baseline
    #
    # Requirements:
    #   1. For each training image in train_file, compute the normalized snout
    #      position (u / W, v / H): the snout position as a fraction of the image
    #      width and height. Get (W, H) with Image.open(os.path.join(img_dir, name)).size
    #   2. Average the normalized snout positions over the whole training set.
    #      This gives one average normalized snout position, (u_avg, v_avg)
    #   3. Convert the average normalized snout position into a pixel position for
    #      each test image: (u_avg * W, v_avg * H), using that test image's (W, H)
    #      from orig_size. This is the baseline's prediction for that image
    # ======================================================================
    pass


def ced_curve(errors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the points of the cumulative error distribution (CED) curve.

    Args:
        errors : torch.Tensor, shape (N,), float. Errors in pixels

    Returns:
        tuple (x, y) of two torch.Tensors, each shape (N,), float:
          x : the errors sorted in increasing order, in pixels
          y : y[k-1] = 100 * k / N for k = 1, ..., N, the percentage of predictions
              with an error of at most x[k-1]. The last value is 100
    """
    # ======================================================================
    # TODO [IN-LAB STEP 4]: Implement the CED Curve
    #
    # Hint: use torch.sort
    # ======================================================================
    pass


def best_worst_indices(errors: torch.Tensor, k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the best and worst predictions.

    Args:
        errors : torch.Tensor, shape (N,), float. Errors in pixels
        k      : int. Number of best and worst images to return

    Returns:
        tuple (best, worst) of two torch.Tensors, each shape (k,), int64 (torch.long),
        holding indices into the test set (0 to N-1):
          best  : indices of the k smallest errors, smallest error first
          worst : indices of the k largest errors, largest error first
    """
    # ======================================================================
    # TODO [IN-LAB STEP 5]: Implement the Best and Worst Indices
    #
    # Hint: use torch.argsort
    # ======================================================================
    pass


def print_results(rows: Dict[str, Dict]):
    """
    Provided. Prints the accuracy and cost tables.

    rows : dict mapping each model name to a dict with the keys
           "errors" (tensor of errors), "params" (int) and "ms" (float)
    """
    stats = {name: error_stats(row["errors"]) for name, row in rows.items()}
    w = max([19] + [len(name) + 2 for name in rows])     # name column width
    print()
    if all(s is not None for s in stats.values()):
        print(f"{'Model':<{w}}" + "".join(f"{h:>8}" for h in ["Min", "Mean", "Median", "Max", "Std"]))
        for name, s in stats.items():
            print(f"{name:<{w}}" + "".join(f"{s[k]:>8.1f}" for k in ["min", "mean", "median", "max", "std"]))
        print()
    else:
        print("[TODO] error_stats() is not implemented yet: accuracy table skipped\n")
    print(f"{'Model':<{w}}{'Params':>13}{'ms/img':>9}")
    for name, row in rows.items():
        print(f"{name:<{w}}{row['params']:>13,}{row['ms']:>9.1f}")
    print()
