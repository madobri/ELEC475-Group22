"""
ELEC 475: Lab 2 - Reproducibility, Seed Management & Statistical Aggregation
Provides:
  1. seed_everything: Universal random seed initialization across Python, NumPy, PyTorch, and CUDA.
  2. seed_worker: DataLoader worker seed initialization function for multiprocessing hygiene.
  3. calculate_mean_std: Unbiased sample mean and standard deviation (ddof=1) calculator.
  4. format_mean_std: String formatter for rigorous experimental tables (Mean ± Std).
"""
import os
import random
from typing import Sequence, Tuple, Union
import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets random seeds globally across all execution environments to guarantee reproducibility.
    
    Args:
        seed (int): The integer seed value (e.g. 42, 123, 999).
        deterministic (bool): If True, configures cuDNN to use deterministic algorithms
                              and disables the benchmarking autotuner.
    """
    # 1. Python built-in hash seed and random module
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    
    # 2. NumPy random state
    np.random.seed(seed)
    
    # 3. PyTorch CPU and GPU random states
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Covers multi-GPU setups
        
    # 4. cuDNN backend determinism
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function to ensure each subprocess worker
    receives a distinct, reproducible random seed.
    
    Usage:
        loader = DataLoader(dataset, batch_size=128, num_workers=2,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(42))
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_mean_std(values: Sequence[Union[int, float]]) -> Tuple[float, float]:
    """
    Calculates the sample mean and sample standard deviation (Bessel's correction ddof=1).
    
    Args:
        values (Sequence): List or array of numerical scores across multiple seeds or folds.
        
    Returns:
        Tuple[float, float]: (sample_mean, sample_std)
    """
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))  # Unbiased sample standard deviation (N-1)
    return mean_val, std_val

def format_mean_std(values: Sequence[Union[int, float]], unit: str = "%", decimals: int = 2) -> str:
    """
    Formats a sequence of numbers into standard academic notation: 'Mean ± Std%'.
    
    Example:
        format_mean_std([78.2, 78.9, 78.5]) -> "78.53 ± 0.35%"
    """
    mean_val, std_val = calculate_mean_std(values)
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}{unit}"
