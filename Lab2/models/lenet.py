"""
ELEC 475: Lab 2 - Base LeNet-5 Architecture (Starter Template)
Adapted for RGB 64x64 input images.

Reference:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner,
    "Gradient-Based Learning Applied to Document Recognition,"
    Proceedings of the IEEE, Nov. 1998.
"""
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    Classical LeNet-5 Architecture adapted for (3, 64, 64) RGB images.
    
    Spatial dimension arithmetic (Lecture 05 Slide 13 formula):
        Input  : (B, 3, 64, 64)
        Conv1  : (64 - 5)/1 + 1 = 60  --> (B, 6, 60, 60)
        AvgPool1: (60 - 2)/2 + 1 = 30 --> (B, 6, 30, 30)
        Conv2  : (30 - 5)/1 + 1 = 26  --> (B, 16, 26, 26)
        AvgPool2: (26 - 2)/2 + 1 = 13 --> (B, 16, 13, 13)
        Flatten : 16 * 13 * 13 = 2,704 features
        FC1    : 2,704 -> 120
        FC2    : 120 -> 84
        FC3    : 84 -> num_classes
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ======================================================================
        # TODO [IN-LAB STEP 2]: Implement the LeNet-5 Feature Extractor
        #
        # Requirements:
        #   1. Conv1: in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0
        #   2. Activation: nn.Tanh()
        #   3. AvgPool1: kernel_size=2, stride=2
        #   4. Conv2: in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        #   5. Activation: nn.Tanh()
        #   6. AvgPool2: kernel_size=2, stride=2
        # ======================================================================
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # ======================================================================
        # TODO [IN-LAB STEP 2]: Implement the LeNet-5 Dense Classifier Head
        #
        # Requirements:
        #   1. FC1: in_features = 16 * 13 * 13 (2,704), out_features = 120
        #   2. Activation: nn.Tanh()
        #   3. FC2: in_features = 120, out_features = 84
        #   4. Activation: nn.Tanh()
        #   5. FC3 (Output): in_features = 84, out_features = num_classes
        # ======================================================================
        self.classifier = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [IN-LAB STEP 2]: Implement the Forward Pass
        #
        # 1. Pass input x through self.feature_extractor
        # 2. Flatten spatial features using torch.flatten(x, start_dim=1)
        # 3. Pass flattened vector through self.classifier
        # 4. Return the computed logits
        # ======================================================================
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    model = LeNet5(num_classes=10, in_channels=3)
    dummy_input = torch.randn(1, 3, 64, 64)
    print("Testing LeNet-5 Architecture:")
    try:
        summary(model, input_size=(1, 3, 64, 64))
    except Exception as e:
        print(f"Model not fully implemented yet: {e}")