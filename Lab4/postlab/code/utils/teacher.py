import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


def load_teacher(device):
    """Pretrained FCN-ResNet50, frozen and in eval mode."""
    teacher = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def teacher_forward(teacher, x):
    """Returns the teacher's logits (N, 21, H, W) and its layer2 and layer4 features."""
    feats = {}
    h = x
    for name, module in teacher.backbone.items():
        h = module(h)
        if name in ('layer2', 'layer4'):
            feats[name] = h
    logits = F.interpolate(teacher.classifier(h), size=x.shape[-2:], mode='bilinear',
                           align_corners=False)
    return logits, feats
