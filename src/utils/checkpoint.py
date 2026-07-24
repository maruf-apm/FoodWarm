"""Checkpoint management utilities."""
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manages model checkpoint saving and loading."""

    def __init__(self, checkpoint_dir: str = "./outputs/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None,
    ) -> str:
        """Save a training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
        }

        torch.save(checkpoint, filepath)
        return str(filepath)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint from disk."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint in the directory."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)

    def cleanup_old_checkpoints(self, keep_top_k: int = 3) -> None:
        """Remove old checkpoints, keeping only the top k by metric."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) <= keep_top_k:
            return

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for ckpt in checkpoints[keep_top_k:]:
            ckpt.unlink()
