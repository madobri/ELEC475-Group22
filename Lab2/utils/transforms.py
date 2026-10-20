import torchvision.transforms as transforms

DATASET_STATS = {
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std":  [0.2470, 0.2435, 0.2616]
    },
    "tiny_imagenet": {
        "mean": [0.4802, 0.4481, 0.3975],
        "std":  [0.2764, 0.2689, 0.2821]
    }
}

# Constructs the training transform pipeline.
def get_train_transforms(dataset: str = "cifar10", augment: bool = True, target_size: int = 64):
    stats = DATASET_STATS.get(dataset.lower(), DATASET_STATS["cifar10"])

    transform_list = [
        transforms.Resize((target_size, target_size))
    ]

    if augment:
        transform_list.extend([
            transforms.RandomCrop(target_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"])
    ])

    return transforms.Compose(transform_list)

# Deterministic evaluation pipeline (no random augmentations).
def get_test_transforms(dataset: str = "cifar10", target_size: int = 64):
    stats = DATASET_STATS.get(dataset.lower(), DATASET_STATS["cifar10"])
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"])
    ])