import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import VOCSegmentation

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SegTransform:
    """Resizes an (image, mask) pair to a fixed size and converts both to tensors."""

    def __init__(self, size=320):
        self.size = [size, size]

    def __call__(self, img, mask):
        img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


def get_voc(data_dir, image_set, size=320):
    return VOCSegmentation(data_dir, year='2012', image_set=image_set, download=False,
                           transforms=SegTransform(size))
