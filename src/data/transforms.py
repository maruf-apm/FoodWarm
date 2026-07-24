"""Image transforms and augmentation pipeline for Foodwarm."""
from typing import Tuple
from torchvision import transforms
from torchvision.transforms import autoaugment


class FoodTransform:
    """Transform pipeline for Food-101 dataset."""

    def __init__(self, image_size: int = 224, is_training: bool = True):
        self.image_size = image_size
        self.is_training = is_training

        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                autoaugment.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),  # 256 for 224
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __call__(self, image):
        return self.transform(image)


def get_transforms(image_size: int = 224) -> Tuple:
    """Get train and validation transforms."""
    train_transform = FoodTransform(image_size=image_size, is_training=True)
    val_transform = FoodTransform(image_size=image_size, is_training=False)
    return train_transform, val_transform
