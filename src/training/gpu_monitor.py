"""GPU memory monitoring callback."""

import torch
import pytorch_lightning as pl


class GPUMemoryMonitor(pl.Callback):
    """Monitor and log GPU memory usage during training."""

    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            pl_module.log("gpu/allocated_gb", allocated, prog_bar=False, logger=True)
            pl_module.log("gpu/reserved_gb", reserved, prog_bar=False, logger=True)
            pl_module.log("gpu/peak_gb", peak, prog_bar=False, logger=True)

            print(f"\n[GPU Memory] Allocated: {allocated:.2f}GB | Peak: {peak:.2f}GB")
