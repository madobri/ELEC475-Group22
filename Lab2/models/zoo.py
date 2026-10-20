"""
ELEC 475: Lab 2 - PyTorch Model Zoo Adapter
Provides access to torchvision official reference architectures (e.g. AlexNet, ResNet-18)
adapted for 64x64 resolution on CIFAR-10 (10 classes) and Tiny ImageNet-200 (200 classes).

This allows students to benchmark their manual from-scratch PyTorch implementations
against industry reference implementations from the PyTorch Model Zoo.
"""
import torch
import torch.nn as nn
import torchvision.models as models

def get_torchvision_alexnet(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Loads official torchvision AlexNet and adapts its final projection head
    for the target dataset class count (10 for CIFAR-10, 200 for Tiny ImageNet).
    
    Architectural notes:
      - Uses AdaptiveAvgPool2d((6, 6)), allowing 64x64 inputs to flow into the dense classifier.
      - Default torchvision classifier:
          (0): Dropout(p=0.5)
          (1): Linear(in_features=9216, out_features=4096)
          (2): ReLU(inplace=True)
          (3): Dropout(p=0.5)
          (4): Linear(in_features=4096, out_features=4096)
          (5): ReLU(inplace=True)
          (6): Linear(in_features=4096, out_features=1000) -> Replaced with Linear(4096, num_classes)
    """
    try:
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        model = models.alexnet(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.alexnet(pretrained=pretrained)

    # Adapt classifier head for target classes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def get_torchvision_model(model_name: str = "alexnet", num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Factory for torchvision model zoo reference baselines.
    Supports: 'alexnet', 'resnet18', 'vgg11'
    """
    name = model_name.lower()
    if "alexnet" in name:
        return get_torchvision_alexnet(num_classes=num_classes, pretrained=pretrained)
    elif "resnet18" in name:
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported torchvision model zoo name: '{model_name}'. Choose from 'alexnet', 'resnet18'.")
