"""Custom PyTorch Lightning callbacks for Foodwarm."""
import time
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SpeedMonitor(Callback):
    """Monitor and log training speed (samples/sec)."""

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.batch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        batch_size = trainer.datamodule.batch_size
        num_batches = len(trainer.train_dataloader)
        samples_per_sec = (batch_size * num_batches) / elapsed

        pl_module.log("train/samples_per_sec", samples_per_sec, prog_bar=False, logger=True)
        pl_module.log("train/epoch_time_sec", elapsed, prog_bar=False, logger=True)


class GradientNormMonitor(Callback):
    """Monitor gradient norms during training."""

    def on_after_backward(self, trainer, pl_module):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5

        pl_module.log("train/grad_norm", total_norm, prog_bar=False, logger=True)


class LearningRateMonitor(Callback):
    """Log current learning rate."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            pl_module.log("train/lr", lr, prog_bar=False, logger=True)


class EpochTimer(Callback):
    """Time each epoch and log it."""

    def __init__(self):
        self.train_start = None
        self.val_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.train_start
        pl_module.log("time/train_epoch_sec", elapsed, prog_bar=False, logger=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.val_start
        pl_module.log("time/val_epoch_sec", elapsed, prog_bar=False, logger=True)
