import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def conv_bn_relu(in_ch, out_ch, k, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=128, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [conv_bn_relu(in_ch, out_ch, 1)] +
            [conv_bn_relu(in_ch, out_ch, 3, dilation=r) for r in rates])
        self.project = conv_bn_relu(out_ch * (1 + len(rates)), out_ch, 1)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class StudentNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = mobilenet_v3_small(weights=weights).features
        self.low = features[:4]     # 24 channels, stride 8
        self.mid = features[4:7]    # 40 channels, stride 16
        self.high = features[7:]    # 576 channels, stride 32
        self.aspp = ASPP(576, 128)
        self.low_proj = conv_bn_relu(24, 48, 1)
        self.fuse = conv_bn_relu(128 + 48, 128, 3)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        mid = self.mid(low)
        high = self.high(mid)
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode='bilinear', align_corners=False)
        y = self.fuse(torch.cat([y, self.low_proj(low)], dim=1))
        y = self.classifier(self.dropout(y))
        out = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
        return {'out': out, 'low': low, 'mid': mid, 'high': high}


if __name__ == '__main__':
    from torchinfo import summary
    from utils.profiler import ms_per_image

    model = StudentNet().eval()
    summary(model, input_size=(1, 3, 320, 320), depth=1)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    for k, v in out.items():
        print(f'{k:<5} {tuple(v.shape)}')
    print(f'ms/img (CPU): {ms_per_image(model, torch.device("cpu")):.1f}')
