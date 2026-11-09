"""
ELEC 475 Lab 4 Pre-lab, Step 1
Evaluate torchvision's pretrained FCN-ResNet50 on the PASCAL VOC 2012
segmentation validation set using mean Intersection over Union (mIoU).

Usage (from the folder that holds this file):
    python test_fcn_resnet50_voc.py --download          # first run: fetches VOC 2012 (~2 GB)
    python test_fcn_resnet50_voc.py                     # later runs
    python test_fcn_resnet50_voc.py --limit 50          # quick smoke test on 50 images
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torchvision.transforms.functional as TF

NUM_CLASSES = 21          # 20 object classes + background
IGNORE_INDEX = 255        # VOC marks object borders / "difficult" pixels with 255
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class EvalTransform:
    """Applies the same resize to image and mask.

    The image is resized with bilinear interpolation and normalized with ImageNet
    statistics. The mask MUST be resized with nearest-neighbour interpolation,
    otherwise class ids get blended into meaningless in-between values.
    """

    def __init__(self, resize=520):
        self.resize = resize

    def __call__(self, img, mask):
        if self.resize:
            img = TF.resize(img, self.resize, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.resize, interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)                 # (3, H, W) float
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)        # (H, W) class ids, 255 = ignore
        return img, mask


def update_confusion(conf, pred, target):
    """Accumulate a NUM_CLASSES x NUM_CLASSES confusion matrix (rows = truth, cols = prediction)."""
    valid = target != IGNORE_INDEX
    idx = target[valid] * NUM_CLASSES + pred[valid]
    conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return conf


def compute_metrics(conf):
    conf = conf.double()
    tp = conf.diag()
    iou = tp / (conf.sum(0) + conf.sum(1) - tp)    # TP / (TP + FP + FN); NaN if class never appears
    pixel_acc = tp.sum() / conf.sum()
    return iou, torch.nanmean(iou), pixel_acc


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--download", action="store_true", help="download VOC 2012 trainval if missing")
    p.add_argument("--image-set", default="val", choices=["train", "val", "trainval"])
    p.add_argument("--resize", type=int, default=520, help="short-side resize (0 = native resolution)")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    # ---- Data ----
    dataset = VOCSegmentation(root=args.data_root, year="2012", image_set=args.image_set,
                              download=args.download, transforms=EvalTransform(args.resize))
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    # batch_size=1 because VOC images have different sizes, so they can't be stacked without padding
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    print(f"VOC 2012 {args.image_set}: {len(dataset)} images")

    # ---- Model ----
    weights = FCN_ResNet50_Weights.DEFAULT   # COCO_WITH_VOC_LABELS_V1: trained on COCO subset with the 20 VOC classes
    model = fcn_resnet50(weights=weights).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: FCN-ResNet50 ({weights}), {n_params / 1e6:.1f} M parameters")

    # ---- Evaluate ----
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    start = time.time()
    with torch.inference_mode():
        for i, (img, target) in enumerate(loader):
            img = img.to(device)                       # (1, 3, H, W)
            out = model(img)["out"]                    # (1, 21, H, W), already upsampled to input size
            pred = out.argmax(dim=1).cpu()             # (1, H, W)
            conf = update_confusion(conf, pred.flatten(), target.flatten())
            if (i + 1) % 100 == 0:
                _, miou, _ = compute_metrics(conf)
                print(f"  [{i + 1}/{len(dataset)}] running mIoU = {miou * 100:.2f}%")
    elapsed = time.time() - start

    iou, miou, pixel_acc = compute_metrics(conf)
    print("\nPer-class IoU:")
    for name, v in zip(VOC_CLASSES, iou.tolist()):
        print(f"  {name:<12s} {v * 100:6.2f}%")
    print(f"\nmIoU:           {miou * 100:.2f}%")
    print(f"Pixel accuracy: {pixel_acc * 100:.2f}%")
    print(f"Time: {elapsed:.1f} s ({elapsed / len(dataset) * 1000:.1f} ms/image on {device})")


if __name__ == "__main__":   # needed on macOS/Windows when num_workers > 0
    main()
