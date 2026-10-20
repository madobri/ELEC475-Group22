import torch
import torch.nn as nn

class AlexNet64(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, norm_type: str = "bn"):
        super(AlexNet64, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.norm_type = norm_type.lower()

        def get_norm_layer(num_features: int) -> nn.Module:
            if self.norm_type == "bn":
                return nn.BatchNorm2d(num_features)
            elif self.norm_type == "lrn":
                return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
            elif self.norm_type == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm_type '{norm_type}'. Choose from 'none', 'lrn', 'bn'.")

        # BN goes before ReLU, LRN goes after ReLU (before pooling).
        def conv_block(in_c, out_c, k, s, p, pool):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)]
            if self.norm_type == "bn":
                layers.append(get_norm_layer(out_c))
            layers.append(nn.ReLU(inplace=True))
            if self.norm_type == "lrn":
                layers.append(get_norm_layer(out_c))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            return layers

        def bn_or_identity(num_features: int) -> nn.Module:
            return nn.BatchNorm2d(num_features) if self.norm_type == "bn" else nn.Identity()

        self.features = nn.Sequential(
            *conv_block(in_channels, 64, 5, 2, 2, True),
            *conv_block(64, 192, 5, 1, 2, True),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            bn_or_identity(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            bn_or_identity(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            bn_or_identity(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

if __name__ == "__main__":
    from torchinfo import summary
    model = AlexNet64(num_classes=10, in_channels=3, norm_type="bn")
    summary(model, input_size=(1, 3, 64, 64))
