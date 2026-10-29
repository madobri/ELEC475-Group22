"""
ELEC 475: Lab 3 - Correction Model Dataset (Starter Template)
Builds training patches for the correction model from the training set only.

CorrectionDataset is a subclass of your Pre-Lab PetNoseDataset: it reuses its
parsed labels (self.samples) and image folder (self.img_dir), and replaces only
__getitem__.

Each sample is a 64 x 64 patch cut from a 128 x 128 region of the original image,
centred on a point near the true nose. The target is the offset from the patch
centre to the true nose, divided by HALF_PATCH so it lies roughly in [-1, 1]:

    corrected (u, v) = patch centre + target * HALF_PATCH
"""
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from utils.dataset import PetNoseDataset, MEAN, STD

PATCH_SIZE = 128              # side of the cropped region, in original-image pixels
HALF_PATCH = PATCH_SIZE // 2  # offsets are divided by this to normalize them
INPUT_SIZE = 64               # the patch is resized to INPUT_SIZE x INPUT_SIZE
MAX_OFFSET = 40               # largest training offset in u and v, in original-image pixels


def extract_patch(img: Image.Image, center) -> Image.Image:
    """
    Args:
        img    : PIL RGB image, original size
        center : (u, v) integer pixel coordinates of the patch centre.
                 The centre may lie near or outside the image border.

    Returns:
        PIL RGB image, INPUT_SIZE x INPUT_SIZE: the PATCH_SIZE x PATCH_SIZE region
        of img centred on center, resized with Image.BILINEAR. Any part of the
        region outside the image is black.
    """
    cu, cv = center
    left, top = cu - HALF_PATCH, cv - HALF_PATCH
    region = img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
    return region.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)


class CorrectionDataset(PetNoseDataset):
    """
    One random patch per training image per epoch. Inherits self.samples
    (a list of (name, u, v)), self.img_dir and __len__ from PetNoseDataset.
    """
    def __init__(self, root: str = "data/oxford-iiit-pet-noses", augment: bool = True):
        super(CorrectionDataset, self).__init__(root=root, split="train")
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    def __getitem__(self, idx: int):
        """
        Returns (patch, target):
            patch  : torch.Tensor, shape (3, INPUT_SIZE, INPUT_SIZE), float,
                     TF.normalize(TF.to_tensor(patch), MEAN, STD)
            target : torch.Tensor, shape (2,), float. (du, dv) / HALF_PATCH, where (du, dv)
                     is the offset from the patch centre to the true nose, in pixels
        """
        name, u, v = self.samples[idx]
        img = Image.open(self.img_dir / name).convert("RGB")

        cu = int(round(u + random.uniform(-MAX_OFFSET, MAX_OFFSET)))
        cv = int(round(v + random.uniform(-MAX_OFFSET, MAX_OFFSET)))
        patch = extract_patch(img, (cu, cv))
        du, dv = u - cu, v - cv

        if self.augment:
            if random.random() < 0.5:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
                du = -du
            patch = self.color_jitter(patch)

        x = TF.normalize(TF.to_tensor(patch), MEAN, STD)
        target = torch.tensor([du, dv], dtype=torch.float32) / HALF_PATCH
        return x, target


if __name__ == "__main__":
    # Reality check: saves 8 patches with their centre (red cross) and
    # target nose position (green circle) drawn on.
    import os
    import matplotlib.pyplot as plt
    from utils.dataset import denormalize
    from utils.visualization import show_prediction

    centre = INPUT_SIZE / 2
    scale = INPUT_SIZE / PATCH_SIZE               # original-image pixels -> patch pixels
    os.makedirs("plots", exist_ok=True)
    try:
        for augment in [False, True]:
            ds = CorrectionDataset(augment=augment)
            random.seed(0)
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for ax, i in zip(axes.flat, range(8)):
                patch, target = ds[i]
                offset = target * HALF_PATCH
                nose = (centre + offset[0] * scale, centre + offset[1] * scale)
                show_prediction(ax, TF.to_pil_image(denormalize(patch)), nose, (centre, centre),
                                f"offset ({offset[0]:.0f}, {offset[1]:.0f}) px")
            save_path = os.path.join("plots", f"correction_patches{'_aug' if augment else ''}.png")
            fig.tight_layout()
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
            print(f"{len(ds):,} training images | augment={augment} | saved to: {save_path}")
    except Exception as e:
        print(f"CorrectionDataset not fully implemented yet: {e}")
