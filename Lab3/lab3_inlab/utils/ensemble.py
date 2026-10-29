"""
ELEC 475: Lab 3 - Prediction Ensembles (Starter Template)
Combines the test-set predictions of several models into one prediction per image.
"""
from typing import List
import torch


def ensemble_mean(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        preds : list of 3 torch.Tensors, each shape (N, 2), float.
                Each model's predicted (u, v) in original-image pixels

    Returns:
        torch.Tensor, shape (N, 2), float. For each test image, the average of the
        three predictions, in original-image pixels (column 0 is u, column 1 is v)
    """
    # ======================================================================
    # TODO [IN-LAB STEP 6]: Implement the Mean Ensemble
    #
    # Requirements:
    #   1. For each image, average the three models' predicted (u, v)
    # ======================================================================
    pass


def ensemble_median(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        preds : list of 3 torch.Tensors, each shape (N, 2), float.
                Each model's predicted (u, v) in original-image pixels

    Returns:
        torch.Tensor, shape (N, 2), float. For each test image, the median of the
        three predictions, taken separately for u (column 0) and v (column 1),
        in original-image pixels
    """
    # ======================================================================
    # TODO [IN-LAB STEP 6]: Implement the Median Ensemble
    #
    # Requirements:
    #   1. For each image, take the median of the three models' predictions,
    #      separately for u and for v
    # ======================================================================
    pass
