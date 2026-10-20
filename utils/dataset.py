"""
ELEC 475: Lab 2 - Unified Dataset & DataLoader Factory
Supports CIFAR-10 (primary) and Tiny ImageNet-200 (secondary benchmark).
"""
import os
import urllib.request
import zipfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from .transforms import get_train_transforms, get_test_transforms

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def download_and_extract_tiny_imagenet(data_dir: str = "./data"):
    target_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(target_dir):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet-200 from {TINY_IMAGENET_URL}...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")
    return target_dir

def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    target_size: int = 64
):
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        train_tf = get_train_transforms("cifar10", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("cifar10", target_size=target_size)

        train_set = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
        num_classes = 10
        classes = CIFAR10_CLASSES

    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        train_dir = os.path.join(tiny_dir, "train")

        train_tf = get_train_transforms("tiny_imagenet", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("tiny_imagenet", target_size=target_size)

        train_set = ImageFolder(root=train_dir, transform=train_tf)
        val_set   = ImageFolder(root=train_dir, transform=val_tf)
        num_classes = 200
        classes = train_set.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader, num_classes, classes
