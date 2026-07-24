#!/usr/bin/env python3
"""One-shot Colab training script with GPU verification."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.data.food101_dataset import Food101DataModule
from src.models.vit_classifier import ViTClassifier
from src.training.callbacks import SpeedMonitor, GradientNormMonitor, EpochTimer
from src.training.gpu_monitor import GPUMemoryMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="Foodwarm Colab Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def check_gpu():
    """Verify GPU and print VRAM info."""
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ NO GPU! Enable it: Runtime → Change runtime type → GPU")
        raise RuntimeError("GPU required")

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ Total VRAM: {vram_total:.1f} GB")
    print()

    # Quick VRAM test
    test_tensor = torch.randn(5000, 5000, device="cuda")
    _ = test_tensor @ test_tensor.T
    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"✅ VRAM test passed. Currently using: {vram_used:.2f} GB")
    del test_tensor
    torch.cuda.empty_cache()
    print()


def download_data(config):
    """Download Food-101 with progress."""
    from torchvision.datasets import Food101
    import os

    print("=" * 60)
    print("DATASET DOWNLOAD")
    print("=" * 60)

    data_dir = config.data.data_dir
    os.makedirs(data_dir, exist_ok=True)

    if (Path(data_dir) / "food-101").exists():
        print("✅ Dataset already present.\n")
        return

    print("Downloading Food-101 (~5GB)... This takes 5-10 minutes.\n")
    Food101(root=data_dir, split="train", download=True)
    Food101(root=data_dir, split="test", download=True)
    print("✅ Download complete!\n")


def main():
    args = parse_args()

    # 1. Verify GPU
    check_gpu()

    # 2. Load config
    config = load_config(args.config)

    # 3. Auto-fix Colab settings
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.persistent_workers = False

    # 4. Download data
    download_data(config)

    # 5. Setup
    logger = setup_logger("foodwarm-colab")
    pl.seed_everything(config.training.seed)

    datamodule = Food101DataModule(config.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    print(
        f"📊 Train: {len(datamodule.train_dataset):,} | Val: {len(datamodule.val_dataset):,}"
    )
    print(
        f"📦 Batch size: {config.data.batch_size} (effective: {config.data.batch_size * config.training.accumulate_grad_batches})"
    )
    print()

    # 6. Create model
    model = ViTClassifier(config=config.model, num_classes=datamodule.num_classes)
    info = model.get_model_info()
    print(f"🧠 Model: {info['model_name']} | PEFT: {info['peft_method']}")
    print(f"🔧 Trainable: {info['trainable']:,} params ({info['trainable_pct']:.4f}%)")
    print()

    # 7. Callbacks
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
        GPUMemoryMonitor(),
        RichProgressBar(),
    ]

    csv_logger = CSVLogger(
        save_dir=config.logging.log_dir,
        name=config.logging.experiment_name,
    )

    # 8. Trainer
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

    # 9. Train
    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    # 10. Test
    print()
    print("=" * 60)
    print("🧪 TESTING")
    print("=" * 60)
    trainer.test(model, datamodule=datamodule)

    print()
    print("=" * 60)
    print("✅ DONE!")
    print(f"🏆 Best: {trainer.checkpoint_callback.best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
