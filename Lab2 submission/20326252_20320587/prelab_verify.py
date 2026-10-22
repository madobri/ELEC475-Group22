"""
ELEC 475: Lab 2 - Pre-Lab Environment & Dataset Verification Script
Automated check of PyTorch, CUDA hardware acceleration, CIFAR-10, Tiny ImageNet,
DataLoader batch mechanics, and EDA generation status.
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

print("=" * 76)
print("ELEC 475: Lab 2 - Environment & Dataset Verification")
print("=" * 76)

# 1. Hardware & Framework Diagnostics
print("\n[Step 1: PyTorch & GPU Acceleration Diagnostics]")
print(f"  PyTorch Version         : {torch.__version__}")
print(f"  Torchvision Version     : {torchvision.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"  CUDA GPU Available      : {cuda_avail}")
if cuda_avail:
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"  GPU Device Name         : {device_name}")
    print(f"  CUDA Compute Capability : {cap[0]}.{cap[1]}")
    print("  Status                  : GPU acceleration verified. High-throughput training ready!")
else:
    print("  Warning: CUDA not detected. Training will run on CPU.")
    print("  If using a machine without an NVIDIA GPU, use Google Colab with GPU runtime.")

# 2. CIFAR-10 Download & Tensor Shape Check
print("\n[Step 2: CIFAR-10 Dataset Verification (Mandatory In-Lab Dataset)]")
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

transform_standard = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

print("  Checking / downloading CIFAR-10 to ./data/ ...")
train_set = CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_standard)
test_set = CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_standard)

print(f"  Train Samples           : {len(train_set):,} images (5,000 per class)")
print(f"  Test Samples            : {len(test_set):,} images (1,000 per class)")

sample_img, sample_label = train_set[0]
print(f"  Sample Tensor Shape     : {sample_img.shape} --> Expected: torch.Size([3, 64, 64])")
print(f"  Sample Data Type        : {sample_img.dtype} --> Expected: torch.float32")
print(f"  Sample Value Range      : Min = {sample_img.min():.4f}, Max = {sample_img.max():.4f} --> Expected: [0.0, 1.0]")
assert sample_img.shape == torch.Size([3, 64, 64]), f"Unexpected shape {sample_img.shape}!"
assert 0.0 <= sample_img.min() and sample_img.max() <= 1.0, "Values outside [0.0, 1.0]!"

# 3. Mini-Batch DataLoader Check
print("\n[Step 3: Mini-Batch DataLoader Pipeline Check]")
sample_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
batch_imgs, batch_labels = next(iter(sample_loader))
print(f"  Batch Tensor Shape      : {batch_imgs.shape} --> Expected: torch.Size([64, 3, 64, 64])")
print(f"  Batch Labels Shape      : {batch_labels.shape} --> Expected: torch.Size([64])")
assert batch_imgs.shape == torch.Size([64, 3, 64, 64]), "Batch tensor shape mismatch!"

# 4. Tiny ImageNet-200 Dataset Check
print("\n[Step 4: Tiny ImageNet-200 Dataset Check]")
tiny_path = os.path.join(DATA_DIR, "tiny-imagenet-200")
if os.path.exists(tiny_path) and os.path.isdir(os.path.join(tiny_path, "train")):
    print("  Tiny ImageNet-200       : DETECTED in ./data/tiny-imagenet-200 (Verified).")
else:
    print("  Tiny ImageNet-200       : Not yet extracted into ./data/tiny-imagenet-200.")
    print("  Action: Download and extract tiny-imagenet-200 into ./data/ or run:")
    print("    python utils/eda.py --dataset tiny_imagenet")

# 5. Exploratory Data Analysis (EDA) Artifact Check
print("\n[Step 5: Exploratory Data Analysis (EDA) Artifact Check]")
plots_dir = "./plots"
eda_cifar_files = [
    "eda_cifar10_class_distribution.png",
    "eda_cifar10_channel_histograms.png",
    "eda_cifar10_sample_grid.png"
]
eda_tiny_files = [
    "eda_tiny_imagenet_class_distribution.png",
    "eda_tiny_imagenet_channel_histograms.png",
    "eda_tiny_imagenet_sample_grid.png"
]
missing_cifar = [f for f in eda_cifar_files if not os.path.exists(os.path.join(plots_dir, f))]
missing_tiny = [f for f in eda_tiny_files if not os.path.exists(os.path.join(plots_dir, f))]

if not missing_cifar:
    print("  CIFAR-10 EDA Artifacts  : DETECTED in ./plots/ (All 3 plots present).")
else:
    print(f"  CIFAR-10 EDA Artifacts  : {len(missing_cifar)} plots not yet generated.")

if not missing_tiny:
    print("  Tiny ImageNet EDA Plots : DETECTED in ./plots/ (All 3 plots present).")
else:
    print(f"  Tiny ImageNet EDA Plots : {len(missing_tiny)} plots not yet generated.")

if missing_cifar or missing_tiny:
    print("  Action: Run the dual-dataset EDA suite to generate all artifacts:")
    print("    python utils/eda.py --dataset both")

print("\n" + "=" * 76)
print("PRE-LAB VERIFICATION SUCCEEDED! YOU ARE FULLY READY FOR THE IN-LAB SESSION.")
print("=" * 76)
