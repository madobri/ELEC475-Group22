"""Prelab: evaluate pretrained FCN-ResNet50 on PASCAL VOC 2012 (val) using mIoU."""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

NUM_CLASSES = 21      # 20 VOC object classes + background
IGNORE_INDEX = 255    # VOC "void" boundary pixels, excluded from the metric
IMG_SIZE = 520        # resolution used by the torchvision pretrained weights
BATCH_SIZE = 16
ROOT = "C:/datasets"

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class SegTransform:
    """Resize image (bilinear) and mask (nearest) to a fixed size, then normalize the image."""

    def __init__(self, size):
        self.img_resize = T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR)
        self.mask_resize = T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __call__(self, img, mask):
        img = self.normalize(T.functional.to_tensor(self.img_resize(img)))
        mask = torch.as_tensor(
            T.functional.pil_to_tensor(self.mask_resize(mask)), dtype=torch.long
        ).squeeze(0)  # (H, W), values in [0, 20] or 255
        return img, mask


def update_confusion(conf, preds, targets):
    """Accumulate an (C x C) confusion matrix; rows = ground truth, cols = prediction."""
    valid = targets != IGNORE_INDEX
    t, p = targets[valid], preds[valid]
    idx = t * NUM_CLASSES + p
    conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return conf


def compute_miou(conf):
    conf = conf.double()
    intersection = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - intersection
    iou = intersection / union          # NaN if a class never appears (0/0)
    return iou, torch.nanmean(iou).item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)["out"]            # (B, 21, H, W)
        preds = logits.argmax(dim=1)             # (B, H, W)
        assert preds.shape == masks.shape, (preds.shape, masks.shape)
        conf = update_confusion(conf, preds, masks)
        if (i + 1) % 10 == 0:
            print(f"  batch {i + 1}/{len(loader)}")
    return compute_miou(conf.cpu())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Weights trained on a COCO subset restricted to the 20 VOC categories (21 outputs).
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).to(device)
    dataset = VOCSegmentation(ROOT, year="2012", image_set="val", download=True, transforms=SegTransform(IMG_SIZE))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    print(f"Validation images: {len(dataset)}")

    iou, miou = evaluate(model, loader, device)
    for name, v in zip(VOC_CLASSES, iou):
        print(f"{name:>12s}: {v.item() * 100:5.1f}")
    print(f"\nmIoU: {miou * 100:.2f}%")


if __name__ == "__main__":
    main()