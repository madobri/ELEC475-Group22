"""Step 3: our own compact student, LiteSegNet.

Design changes compared with the baseline StudentNet (3.27 M parameters):

1. Encoder at output stride 16 instead of 32. The last MobileNetV3-Small stage keeps
   stride 1 and uses dilation 2 instead (torchvision's `dilated=True`), so the deepest
   features are 20x20 instead of 10x10 for a 320x320 input. Same pretrained weights,
   no extra parameters (DeepLabv3, Chen et al. 2017).
2. Lightweight ASPP. The baseline ASPP uses full 3x3 convolutions on 576 channels and
   holds about 2.1 M of the baseline's 3.27 M parameters. Here every atrous branch is
   depthwise-separable (depthwise 3x3 + pointwise 1x1, as in MobileNet / DeepLabv3+),
   the width is 96 instead of 128, and a global image-pooling branch adds image-level
   context. Rates (3, 6, 9) are scaled down from DeepLab's (6, 12, 18) because our
   stride-16 map is only 20x20.
3. Two-stage decoder. The ASPP output is upsampled and fused first with the stride-8
   features ("mid"), then with the stride-4 features ("low"), using depthwise-separable
   3x3 convolutions, so fine boundaries come from higher-resolution skips.
4. Dropout before the 1x1 classifier, then bilinear upsampling to the input size.

Feature taps for feature-based KD (see KD_TAPS): low (16 ch, stride 4),
mid (24 ch, stride 8) and high (576 ch, stride 16).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def conv_bn_relu(in_ch, out_ch, k=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))


def ds_conv(in_ch, out_ch, dilation=1):
    """Depthwise-separable 3x3 conv: depthwise 3x3 (+BN+ReLU), then pointwise 1x1 (+BN+ReLU)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        conv_bn_relu(in_ch, out_ch, 1))


class LiteASPP(nn.Module):
    def __init__(self, in_ch, out_ch=96, rates=(3, 6, 9)):
        super().__init__()
        self.branches = nn.ModuleList(
            [conv_bn_relu(in_ch, out_ch, 1)] + [ds_conv(in_ch, out_ch, r) for r in rates])
        # Image pooling branch (no BN: after global pooling it is 1x1 per image)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_ch, out_ch, 1),
                                  nn.ReLU(inplace=True))
        self.project = conv_bn_relu(out_ch * (len(rates) + 2), out_ch, 1)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(self.pool(x).expand(-1, -1, *x.shape[-2:]))
        return self.project(torch.cat(outs, dim=1))


class LiteSegNet(nn.Module):
    # student tap -> (teacher layer, student channels); used by train.py for feature KD
    KD_TAPS = {'mid': ('layer2', 24), 'high': ('layer4', 576)}

    def __init__(self, num_classes=21, pretrained=True, aspp_ch=96, dec_ch=64, dropout=0.5):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = mobilenet_v3_small(weights=weights, dilated=True).features
        self.low = features[:2]     # 16 channels, stride 4
        self.mid = features[2:4]    # 24 channels, stride 8
        self.high = features[4:]    # 576 channels, stride 16 (dilated last stage)
        self.aspp = LiteASPP(576, aspp_ch)
        self.mid_proj = conv_bn_relu(24, 32, 1)
        self.fuse_mid = ds_conv(aspp_ch + 32, dec_ch)
        self.low_proj = conv_bn_relu(16, 16, 1)
        self.fuse_low = ds_conv(dec_ch + 16, dec_ch)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(dec_ch, num_classes, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        mid = self.mid(low)
        high = self.high(mid)
        y = self.aspp(high)
        y = self.fuse_mid(torch.cat([self._up(y, mid), self.mid_proj(mid)], dim=1))
        y = self.fuse_low(torch.cat([self._up(y, low), self.low_proj(low)], dim=1))
        y = self.classifier(self.dropout(y))
        out = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
        return {'out': out, 'low': low, 'mid': mid, 'high': high}


if __name__ == '__main__':
    # Run from the code folder: python -m models.custom_student
    from torchinfo import summary
    from utils.profiler import ms_per_image

    model = LiteSegNet().eval()
    summary(model, input_size=(1, 3, 320, 320), depth=1)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    for k, v in out.items():
        print(f'{k:<5} {tuple(v.shape)}')
    print(f'ms/img (CPU): {ms_per_image(model, torch.device("cpu")):.1f}')
