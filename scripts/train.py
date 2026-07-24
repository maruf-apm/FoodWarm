#!/usr/bin/env python3
"""Foodwarm - Final Training Pipeline.

Professional training script for fine-tuning ViT on Food-101 with PEFT.

Usage:
    # Full fine-tuning
    python scripts/train.py --config configs/full/food101.yaml

    # LoRA fine-tuning
    python scripts/train.py --config configs/lora/food101.yaml

    # Linear probing
    python scripts/train.py --config configs/linear/food101.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/lora/food101.yaml --resume ./outputs/checkpoints/lora/last.ckpt
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.utils.config import load_config, save_config
from src.utils.logging_utils import setup_logger
from src.data.food101_dataset import Food101DataModule
from src.models.vit_classifier import ViTClassifier
from src.models.model_factory import ModelFactory
from src.training.callbacks import SpeedMonitor, GradientNormMonitor, EpochTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Foodwarm Training Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run test evaluation, skip training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logger("foodwarm")
    logger.info("=" * 70)
    logger.info("  FOODWARM - ViT Fine-Tuning Pipeline")
    logger.info("  Food-101 Dataset | PEFT Methods | Production Ready")
    logger.info("=" * 70)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Save config to output dir for reproducibility
    config_save_path = Path(config.logging.checkpoint_dir) / "config.yaml"
    save_config(config, str(config_save_path))
    logger.info(f"Configuration saved to: {config_save_path}")

    # Set random seed
    pl.seed_everything(config.training.seed)
    logger.info(f"Random seed set to: {config.training.seed}")

    # Initialize DataModule
    logger.info("Setting up Food-101 dataset...")
    datamodule = Food101DataModule(config.data)
    datamodule.prepare_data()

    if not args.test_only:
        datamodule.setup("fit")
    else:
        datamodule.setup("test")

    logger.info(f"  Dataset: Food-101")
    logger.info(f"  Classes: {datamodule.num_classes}")
    if hasattr(datamodule, "train_dataset") and datamodule.train_dataset:
        logger.info(f"  Train samples: {len(datamodule.train_dataset):,}")
    if hasattr(datamodule, "val_dataset") and datamodule.val_dataset:
        logger.info(f"  Val samples: {len(datamodule.val_dataset):,}")

    # Initialize Model
    logger.info(f"Creating model: {config.model.model_name}")
    logger.info(f"  PEFT method: {config.model.peft_method}")

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = ModelFactory.create_model(
            config=config.model,
            num_classes=datamodule.num_classes,
            checkpoint_path=args.resume,
        )
    else:
        model = ViTClassifier(config=config.model, num_classes=datamodule.num_classes)

    # Log model statistics
    model_info = model.get_model_info()
    logger.info(f"  Total parameters: {model_info['total']:,}")
    logger.info(f"  Trainable parameters: {model_info['trainable']:,}")
    logger.info(f"  Trainable percentage: {model_info['trainable_pct']:.4f}%")

    # Setup callbacks
    logger.info("Setting up training callbacks...")
    callbacks = [
        ModelCheckpoint(
            dirpath=config.logging.checkpoint_dir,
            filename="{epoch:02d}-{val/acc:.4f}",
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode,
            save_top_k=config.training.save_top_k,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor=config.training.monitor_metric,
            patience=config.training.early_stopping_patience,
            mode=config.training.monitor_mode,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        SpeedMonitor(),
        GradientNormMonitor(),
        EpochTimer(),
        RichProgressBar(),
    ]

    # Setup loggers
    loggers = []

    # CSV Logger (always enabled)
    csv_logger = CSVLogger(
        save_dir=config.logging.log_dir,
        name=config.logging.experiment_name,
    )
    loggers.append(csv_logger)
    logger.info(f"  CSV Logger: {config.logging.log_dir}/{config.logging.experiment_name}")

    # WandB Logger (optional)
    if config.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=config.logging.project_name,
            name=config.logging.experiment_name,
            entity=config.logging.wandb_entity,
        )
        loggers.append(wandb_logger)
        logger.info(f"  WandB Logger: {config.logging.project_name}/{config.logging.experiment_name}")

    # Initialize Trainer
    logger.info("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.logging.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        benchmark=True,  # Enable cudnn benchmark for consistent input sizes
    )

    # Training
    if not args.test_only:
        logger.info("=" * 70)
        logger.info("  STARTING TRAINING")
        logger.info("=" * 70)

        trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

        logger.info("=" * 70)
        logger.info("  TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        logger.info(f"Best score: {trainer.checkpoint_callback.best_model_score:.4f}")

    # Testing
    logger.info("=" * 70)
    logger.info("  RUNNING TEST EVALUATION")
    logger.info("=" * 70)

    # Load best checkpoint for testing
    best_ckpt = args.resume if args.resume else trainer.checkpoint_callback.best_model_path
    if best_ckpt and Path(best_ckpt).exists():
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        test_model = ModelFactory.create_model(
            config=config.model,
            num_classes=datamodule.num_classes,
            checkpoint_path=best_ckpt,
        )
    else:
        test_model = model

    trainer.test(test_model, datamodule=datamodule)

    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
