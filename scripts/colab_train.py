#!/usr/bin/env python3
"""One-shot Colab training script with all fixes built-in.

This script handles Colab-specific issues automatically:
- Downloads dataset with progress bar
- Verifies GPU is available
- Sets num_workers=0 automatically
- Shows clear status messages

Usage in Colab:
    !python scripts/colab_train.py --config configs/colab/lora_colab.yaml
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from src.utils.config import load_config, save_config
from src.utils.logging_utils import setup_logger
from src.data.food101_dataset import Food101DataModule
from src.models.vit_classifier import ViTClassifier
from src.training.callbacks import SpeedMonitor, GradientNormMonitor, EpochTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Foodwarm Colab Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def check_colab_environment():
    """Verify Colab environment is properly configured."""
    print("=" * 60)
    print("COLAB ENVIRONMENT CHECK")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram:.1f} GB")
    else:
        print("❌ NO GPU DETECTED!")
        print("   Go to Runtime → Change runtime type → Hardware accelerator: GPU")
        print("   Then restart the runtime.")
        raise RuntimeError("GPU required for training. Enable GPU in Colab runtime settings.")

    # Check PyTorch CUDA
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.version.cuda}")
    print()


def download_with_progress(config):
    """Download dataset with visible progress."""
    from torchvision.datasets import Food101
    import os

    print("=" * 60)
    print("DATASET DOWNLOAD")
    print("=" * 60)
    print("Food-101 is ~5GB. This may take 5-10 minutes...")
    print()

    data_dir = config.data.data_dir
    os.makedirs(data_dir, exist_ok=True)

    # Check if already downloaded
    food101_dir = Path(data_dir) / "food-101"
    if food101_dir.exists():
        print("✅ Dataset already downloaded. Skipping.")
        return

    print("[1/2] Downloading TRAIN split (~2.5GB)...")
    Food101(root=data_dir, split="train", download=True)
    print("      ✓ Done")

    print("[2/2] Downloading TEST split (~2.5GB)...")
    Food101(root=data_dir, split="test", download=True)
    print("      ✓ Done")
    print()


def main():
    args = parse_args()

    # Environment check
    check_colab_environment()

    # Load config
    config = load_config(args.config)

    # Auto-fix Colab settings
    if config.data.num_workers > 0:
        print("⚠️  Auto-fixing num_workers: 0 (Colab requirement)")
        config.data.num_workers = 0
        config.data.pin_memory = False
        config.data.persistent_workers = False

    # Download data first (with progress)
    download_with_progress(config)

    # Setup
    logger = setup_logger("foodwarm-colab")
    pl.seed_everything(config.training.seed)

    # DataModule
    datamodule = Food101DataModule(config.data)
    datamodule.prepare_data()  # Already downloaded, this is fast
    datamodule.setup("fit")

    print(f"Train samples: {len(datamodule.train_dataset):,}")
    print(f"Val samples: {len(datamodule.val_dataset):,}")
    print()

    # Model
    model = ViTClassifier(config=config.model, num_classes=datamodule.num_classes)
    info = model.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"PEFT: {info['peft_method']}")
    print(f"Trainable params: {info['trainable']:,} ({info['trainable_pct']:.4f}%)")
    print()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.logging.checkpoint_dir,
            filename="{epoch:02d}-{val/acc:.4f}",
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode,
            save_top_k=config.training.save_top_k,
            save_last=True,
        ),
        EarlyStopping(
            monitor=config.training.monitor_metric,
            patience=config.training.early_stopping_patience,
            mode=config.training.monitor_mode,
        ),
        SpeedMonitor(),
        GradientNormMonitor(),
        EpochTimer(),
        RichProgressBar(),
    ]

    csv_logger = CSVLogger(
        save_dir=config.logging.log_dir,
        name=config.logging.experiment_name,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    # Test
    print()
    print("=" * 60)
    print("TESTING")
    print("=" * 60)
    trainer.test(model, datamodule=datamodule)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
