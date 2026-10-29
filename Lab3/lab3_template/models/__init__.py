from .snoutnet import SnoutNet, SnoutNetAlexNet, SnoutNetVGG16

def get_model(model_name: str, pretrained: bool = True):
    name = model_name.lower()
    if name == "snoutnet":
        return SnoutNet()
    elif name in ["alexnet", "snoutnet_a", "snoutnet-a"]:
        return SnoutNetAlexNet(pretrained=pretrained)
    elif name in ["vgg16", "snoutnet_v", "snoutnet-v"]:
        return SnoutNetVGG16(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from 'snoutnet', 'alexnet', 'vgg16'."
        )

__all__ = ["SnoutNet", "SnoutNetAlexNet", "SnoutNetVGG16", "get_model"]
