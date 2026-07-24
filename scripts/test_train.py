#!/usr/bin/env python3
"""Quick test script to verify the Foodwarm pipeline works.

This runs a minimal training loop (2 epochs, tiny batch) to ensure:
- Data loading works
- Model forward/backward pass works
- PEFT injection works
- Checkpointing works

Usage:
    python scripts/test_train.py --config configs/lora/food101.yaml
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.data.food101_dataset import Food101DataModule
from src.models.vit_classifier import ViTClassifier
from src.training.callbacks import SpeedMonitor, GradientNormMonitor, EpochTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Test Foodwarm Pipeline")
    parser.add_argument("--config", type=str, default="configs/lora/food101.yaml",
                        help="Path to config YAML")
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Run quick test (2 epochs, small batch)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logger("foodwarm-test")
    logger.info("=" * 60)
    logger.info("FOODWARM PIPELINE TEST")
    logger.info("=" * 60)

    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override for quick test
    if args.quick:
        logger.info("Quick test mode: 2 epochs, batch_size=8, small data")
        config.training.max_epochs = 2
        config.data.batch_size = 8
        config.data.num_workers = 2
        config.training.warmup_steps = 10

    # Set seed for reproducibility
    pl.seed_everything(config.training.seed)

    # Initialize DataModule
    logger.info("Initializing Food101 DataModule...")
    datamodule = Food101DataModule(config.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    logger.info(f"  Classes: {datamodule.num_classes}")
    logger.info(f"  Train samples: {len(datamodule.train_dataset)}")
    logger.info(f"  Val samples: {len(datamodule.val_dataset)}")

    # Initialize Model
    logger.info(f"Creating ViT model with PEFT method: {config.model.peft_method}")
    model = ViTClassifier(config=config.model, num_classes=datamodule.num_classes)

    # Log model info
    model_info = model.get_model_info()
    logger.info(f"  Model: {model_info['model_name']}")
    logger.info(f"  Total params: {model_info['total']:,}")
    logger.info(f"  Trainable params: {model_info['trainable']:,}")
    logger.info(f"  Trainable %: {model_info['trainable_pct']:.2f}%")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.logging.checkpoint_dir,
            filename="test-{epoch:02d}-{val/acc:.3f}",
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode,
            save_top_k=1,
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
    ]

    # Logger
    csv_logger = CSVLogger(
        save_dir=config.logging.log_dir,
        name="test_run",
    )

    # Trainer
    logger.info("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    logger.info("Starting test training...")
    trainer.fit(model, datamodule=datamodule)

    # Quick validation
    logger.info("Running validation...")
    trainer.validate(model, datamodule=datamodule)

    logger.info("=" * 60)
    logger.info("PIPELINE TEST PASSED!")
    logger.info("All components working correctly.")
    logger.info("You can now run the full training with scripts/train.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
