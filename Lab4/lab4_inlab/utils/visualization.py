import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .dataset import MEAN, STD


def voc_palette():
    """Standard PASCAL VOC colour map, (256, 3) uint8."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = [r, g, b]
    return pal


PALETTE = voc_palette()


def to_rgb(img):
    img = img.cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def save_comparison(images, masks, preds, path):
    """One row per image: image, ground truth, then one overlay per model in preds (dict)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    n, cols = len(images), 2 + len(preds)
    fig, ax = plt.subplots(n, cols, figsize=(3 * cols, 3 * n), squeeze=False)
    for i in range(n):
        rgb = to_rgb(images[i])
        panels = [(rgb, 'Image'), (PALETTE[masks[i].cpu().numpy()] / 255, 'Ground truth')]
        for name, p in preds.items():
            panels.append((0.5 * rgb + 0.5 * PALETTE[p[i].cpu().numpy()] / 255, name))
        for a, (im, t) in zip(ax[i], panels):
            a.imshow(im)
            a.set_title(t, fontsize=9)
            a.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
