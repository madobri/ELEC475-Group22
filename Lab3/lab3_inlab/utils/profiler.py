"""
ELEC 475: Lab 3 - Inference Timing Utility
Provides:
  1. benchmark_inference: Average inference time per image (ms) at batch size 1,
     with warm-up runs and CUDA synchronization.
"""
import time
from typing import List
import torch
import torch.nn as nn


def benchmark_inference(model: nn.Module, inputs: List[torch.Tensor], device: torch.device,
                        warmup: int = 5) -> float:
    """
    Runs the model on each input tensor, one at a time, and returns the average
    time per input in milliseconds. The first `warmup` inputs are run once beforehand
    as untimed warm-up passes; then every input is run and timed.
    """
    model.eval()
    with torch.no_grad():
        for x in inputs[:warmup]:
            model(x.unsqueeze(0).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for x in inputs:
            model(x.unsqueeze(0).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0 / len(inputs)
