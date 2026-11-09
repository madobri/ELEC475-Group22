import torch


def update_confusion(conf, pred, target, num_classes=21, ignore_index=255):
    """Adds one batch to the (C, C) confusion matrix. Rows = true class, columns = predicted."""
    valid = target != ignore_index
    idx = num_classes * target[valid] + pred[valid]
    conf += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf


def iou_per_class(conf):
    """Returns a (C,) tensor of IoU = TP / (TP + FP + FN), NaN where the class never appears."""
    conf = conf.double()
    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    iou = tp / denom
    iou[denom == 0] = float('nan')
    return iou


def mean_iou(conf):
    """Returns the mean IoU over classes, ignoring NaN entries."""
    return torch.nanmean(iou_per_class(conf)).item()
