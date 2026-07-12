from __future__ import annotations

from typing import Dict

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_classifier_transforms(image_size: int = 128) -> Dict[str, transforms.Compose]:
    """Return train, validation, and test transforms for the classifier."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(size=image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {
        "train": train_transform,
        "validation": eval_transform,
        "test": eval_transform,
    }


def get_srgan_transforms(high_resolution_size: int = 128) -> Dict[str, transforms.Compose]:
    """Return deterministic transforms for SRGAN target images."""
    high_resolution_transform = transforms.Compose(
        [
            transforms.Resize((high_resolution_size, high_resolution_size)),
            transforms.ToTensor(),
        ]
    )
    low_resolution_transform = transforms.Compose(
        [
            transforms.Resize((high_resolution_size, high_resolution_size)),
            transforms.ToTensor(),
        ]
    )
    return {
        "high_resolution": high_resolution_transform,
        "low_resolution": low_resolution_transform,
    }
