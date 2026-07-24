"""Food-101 dataset and LightningDataModule implementation."""
import os
from typing import Optional, Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101
from torchvision import transforms
import pytorch_lightning as pl

from src.data.transforms import get_transforms
from src.utils.config import DataConfig


class Food101DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Food-101 dataset.

    Handles download, splitting, and dataloader creation.
    """

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.image_size = config.image_size
        self.pin_memory = config.pin_memory and config.num_workers > 0
        self.persistent_workers = config.persistent_workers and config.num_workers > 0

        self.train_transform = None
        self.val_transform = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 101
        self.class_names = None

    def prepare_data(self):
        """Download dataset if not present."""
        Food101(root=self.data_dir, split="train", download=True)
        Food101(root=self.data_dir, split="test", download=True)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with transforms."""
        self.train_transform, self.val_transform = get_transforms(self.image_size)

        if stage == "fit" or stage is None:
            self.train_dataset = Food101Wrapper(
                root=self.data_dir,
                split="train",
                transform=self.train_transform,
            )
            self.val_dataset = Food101Wrapper(
                root=self.data_dir,
                split="test",
                transform=self.val_transform,
            )
            self.class_names = self.train_dataset.classes
            self.num_classes = len(self.class_names)

        if stage == "test" or stage is None:
            self.test_dataset = Food101Wrapper(
                root=self.data_dir,
                split="test",
                transform=self.val_transform,
            )
            if self.class_names is None:
                self.class_names = self.test_dataset.classes
                self.num_classes = len(self.class_names)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class Food101Wrapper(Dataset):
    """Wrapper around torchvision Food101 to apply transforms."""

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        self.dataset = Food101(root=root, split=split, download=False)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
