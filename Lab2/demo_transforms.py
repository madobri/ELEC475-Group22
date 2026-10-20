"""
ELEC 475: Lab 2 - Educational Transform Paradigm Diagnostic
Compares PyTorch GPU Tensors vs OpenCV vs NumPy across:
    1. Python Object Types
    2. Data Types (dtypes)
    3. Dimensionality & Memory Layouts (CHW vs HWC)
    4. Memory Residency (Host RAM vs GPU VRAM)
    5. Execution Latency & PCIe Host-to-Device Transfer Costs
"""
import time
import numpy as np
import torch
import torchvision.transforms as T

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

def run_diagnostic():
    print("=" * 80)
    print("ELEC 475: Transform Paradigm & Variable Type Diagnostic")
    print("=" * 80)

    np_img_orig = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # 1. OpenCV Pipeline
    if HAS_OPENCV:
        cv_img_bgr = cv2.cvtColor(np_img_orig, cv2.COLOR_RGB2BGR)
        cv_resized = cv2.resize(cv_img_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
        cv_flipped = cv2.flip(cv_resized, 1)
        cv_rgb = cv2.cvtColor(cv_flipped, cv2.COLOR_BGR2RGB)
        cv_normalized = cv_rgb.astype(np.float32) / 255.0

        print("\n[1] OpenCV (cv2) Output:")
        print(f"    - Object Type      : {type(cv_normalized)}")
        print(f"    - Data Type (dtype): {cv_normalized.dtype}")
        print(f"    - Shape (Layout)   : {cv_normalized.shape}  --> [Height, Width, Channels] (Channels-Last)")
        print(f"    - Value Range      : Min = {cv_normalized.min():.4f}, Max = {cv_normalized.max():.4f}")
        print(f"    - Color Space      : Converted BGR -> RGB")
        print(f"    - Memory Location  : Host System RAM (CPU)")
    else:
        print("\n[1] OpenCV (cv2): Not installed, skipping.")

    # 2. NumPy Pipeline
    np_flipped = np_img_orig[:, ::-1, :]
    np_normalized = np_flipped.astype(np.float32) / 255.0

    print("\n[2] NumPy Output:")
    print(f"    - Object Type      : {type(np_normalized)}")
    print(f"    - Data Type (dtype): {np_normalized.dtype}")
    print(f"    - Shape (Layout)   : {np_normalized.shape}  --> [Height, Width, Channels] (Channels-Last)")
    print(f"    - Value Range      : Min = {np_normalized.min():.4f}, Max = {np_normalized.max():.4f}")
    print(f"    - Memory Location  : Host System RAM (CPU)")

    # 3. PyTorch Tensor Pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_input = torch.from_numpy(np_img_orig).permute(2, 0, 1).float().to(device) / 255.0

    torch_pipeline = T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    tensor_output = torch_pipeline(tensor_input)

    print("\n[3] PyTorch (torchvision) Tensor Output:")
    print(f"    - Object Type      : {type(tensor_output)}")
    print(f"    - Data Type (dtype): {tensor_output.dtype}")
    print(f"    - Shape (Layout)   : {tensor_output.shape}  --> [Channels, Height, Width] (Channels-First)")
    print(f"    - Value Range      : Min = {tensor_output.min().item():.4f}, Max = {tensor_output.max().item():.4f} (Standardized)")
    print(f"    - Memory Location  : {tensor_output.device} ({'NVIDIA GPU VRAM' if 'cuda' in str(device) else 'Host CPU RAM'})")

    # 4. Latency Benchmark
    N = 1000
    print("\n" + "-" * 80)
    print(f"Benchmark: Processing {N:,} Images (Resize + Flip + Normalize)")
    print("-" * 80)

    if HAS_OPENCV:
        t0 = time.perf_counter()
        for _ in range(N):
            _ = cv2.flip(cv_img_bgr, 1).astype(np.float32) / 255.0
        t_cv = (time.perf_counter() - t0) * 1000
        print(f"- OpenCV (CPU)            : {t_cv:>7.2f} ms  ({N / (t_cv/1000):>7,.0f} img/s) + Host-to-Device Copy required")
    
    t0 = time.perf_counter()
    for _ in range(N):
        _ = np_img_orig[:, ::-1, :].astype(np.float32) / 255.0
    t_np = (time.perf_counter() - t0) * 1000
    print(f"- NumPy (CPU)             : {t_np:>7.2f} ms  ({N / (t_np/1000):>7,.0f} img/s) + Host-to-Device Copy required")

    if torch.cuda.is_available():
        for _ in range(50):
            _ = torch_pipeline(tensor_input)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            _ = torch_pipeline(tensor_input)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) * 1000
        print(f"- PyTorch (GPU Tensors)   : {t_torch:>7.2f} ms  ({N / (t_torch/1000):>7,.0f} img/s) [ZERO PCIe Transfer Overhead]")
    else:
        t0 = time.perf_counter()
        for _ in range(N):
            _ = torch_pipeline(tensor_input)
        t_torch = (time.perf_counter() - t0) * 1000
        print(f"- PyTorch (CPU Tensors)   : {t_torch:>7.2f} ms  ({N / (t_torch/1000):>7,.0f} img/s)")

    print("=" * 80)

if __name__ == "__main__":
    run_diagnostic()
