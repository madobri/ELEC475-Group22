"""
ELEC 475: Lab 3 - Pet Nose Localization Dataset (Starter Template)
Custom PyTorch Dataset for the oxford-iiit-pet-noses re-annotation of the
Oxford-IIIT Pet Dataset.

Each sample returns (image, label, orig_size):
    image     : (3, 227, 227) float tensor, ImageNet-normalized
    label     : (2,) float tensor, nose position (u, v) in the 227x227 image (pixels)
                u = column (x), v = row (y)
    orig_size : (2,) float tensor, original image (W, H)
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_SIZE = 227

# ImageNet channel statistics: required by the pretrained AlexNet/VGG16 backbones,
# and used by the provided SnoutNet weights as well.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetNoseDataset(Dataset):
    """
    Loads pet images and nose labels, resized to (3, 227, 227).

    Label file format (one sample per line):
        beagle_145.jpg,"(198, 304)"
    The ordered pair is (u, v) in pixels of the ORIGINAL image.
    """
    def __init__(self, root: str = "data/oxford-iiit-pet-noses", split: str = "train"):
        super(PetNoseDataset, self).__init__()
        self.img_dir = Path(root) / "images-original" / "images"
        label_file = Path(root) / f"{split}_noses.txt"

        self.samples = []
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, coords = line.split(",", 1)
                coords = coords.replace('"', "").replace("(", "").replace(")", "")
                u_str, v_str = coords.split(",")
                self.samples.append((name, float(u_str), float(v_str)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, u, v = self.samples[idx]
        img = Image.open(self.img_dir / name).convert("RGB")
        w, h = img.size
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        u = u * IMG_SIZE / w
        v = v * IMG_SIZE / h
        image = TF.to_tensor(img)
        image = TF.normalize(image, MEAN, STD)
        label = torch.tensor([u, v], dtype=torch.float32)
        orig_size = torch.tensor([w, h], dtype=torch.float32)
        return image, label, orig_size
    

def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display (returns values in [0, 1])."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


if __name__ == "__main__":
    print("Testing PetNoseDataset:")
    try:
        for split in ["train", "test"]:
            ds = PetNoseDataset(split=split)
            image, label, orig_size = ds[0]
            print(f"  {split:5s}: {len(ds):,} samples | image {tuple(image.shape)} "
                  f"| label {label.tolist()} | orig_size {orig_size.tolist()}")
    except Exception as e:
        print(f"Dataset not fully implemented yet: {e}")
