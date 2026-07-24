#!/usr/bin/env python3
"""Pre-download Food-101 dataset with progress bar for Colab.

Run this BEFORE training to avoid the 5-minute silent hang:
    !python scripts/download_data.py

This downloads ~5GB and shows a progress bar.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision.datasets import Food101
from tqdm import tqdm


def download_food101(data_dir: str = "./data"):
    """Download Food-101 train and test splits with progress indication."""
    print("=" * 60)
    print("FOOD-101 DATASET DOWNLOAD")
    print("=" * 60)
    print(f"Target directory: {data_dir}")
    print("This will download ~5GB. Please wait...")
    print()

    os.makedirs(data_dir, exist_ok=True)

    # Download train split
    print("[1/2] Downloading TRAIN split...")
    train_dataset = Food101(root=data_dir, split="train", download=True)
    print(f"      ✓ Train: {len(train_dataset):,} images")

    # Download test split
    print("[2/2] Downloading TEST split...")
    test_dataset = Food101(root=data_dir, split="test", download=True)
    print(f"      ✓ Test: {len(test_dataset):,} images")

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Location: {data_dir}/food-101/")
    print(f"Classes: {len(train_dataset.classes)}")
    print()
    print("You can now run training:")
    print("  !python scripts/train.py --config configs/colab/lora_colab.yaml")


if __name__ == "__main__":
    download_food101()
