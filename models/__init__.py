from .lenet import LeNet5
from .lenet_modern import LeNetModern
from .alexnet import AlexNet64
from .zoo import get_torchvision_alexnet, get_torchvision_model

def get_model(model_name: str, num_classes: int = 10, in_channels: int = 3, norm_type: str = "bn", pretrained: bool = False):
    name = model_name.lower()
    if name == "lenet":
        return LeNet5(num_classes=num_classes, in_channels=in_channels)
    elif name in ["lenet_modern", "modern_lenet", "lenetmodern"]:
        return LeNetModern(num_classes=num_classes, in_channels=in_channels)
    elif name == "alexnet":
        return AlexNet64(num_classes=num_classes, in_channels=in_channels, norm_type=norm_type)
    elif name in ["zoo_alexnet", "torchvision_alexnet", "alexnet_zoo"]:
        return get_torchvision_alexnet(num_classes=num_classes, pretrained=pretrained)
    elif name in ["zoo_resnet18", "torchvision_resnet18"]:
        return get_torchvision_model("resnet18", num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from 'lenet', 'lenet_modern', 'alexnet', 'zoo_alexnet'."
        )

__all__ = ["LeNet5", "LeNetModern", "AlexNet64", "get_torchvision_alexnet", "get_torchvision_model", "get_model"]
