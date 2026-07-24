"""ViT-based image classifier with PEFT support."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import ViTForImageClassification, ViTConfig
from torchmetrics import Accuracy, F1Score, Precision, Recall

from src.models.peft_modules import (
    inject_lora,
    freeze_base_model,
    unfreeze_classifier,
    get_trainable_parameters,
)
from src.utils.config import ModelConfig


class ViTClassifier(pl.LightningModule):
    """LightningModule for ViT image classification with PEFT.

    Supports:
    - Full fine-tuning
    - Linear probing
    - LoRA adaptation
    """

    def __init__(self, config: ModelConfig, num_classes: int = 101):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.num_classes = num_classes

        # Load pretrained ViT
        self.model = ViTForImageClassification.from_pretrained(
            config.model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        )

        # Apply PEFT method
        self._apply_peft()

        # FIX: Ensure model is in train mode (kills Lightning eval-mode warning)
        # Frozen layers stay frozen via requires_grad=False, but must be in train mode
        self.model.train()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Log parameter stats
        param_stats = get_trainable_parameters(self.model)
        self.log_dict(
            {
                "params/total": float(param_stats["total"]),
                "params/trainable": float(param_stats["trainable"]),
                "params/trainable_pct": param_stats["trainable_pct"],
            }
        )

    def _apply_peft(self):
        """Apply the selected PEFT method."""
        method = self.config.peft_method.lower()

        if method == "full":
            pass  # All trainable by default

        elif method == "linear":
            freeze_base_model(self.model)
            unfreeze_classifier(self.model, "classifier")

        elif method == "lora":
            freeze_base_model(self.model)
            inject_lora(
                self.model,
                target_modules=self.config.lora_target_modules,
                r=self.config.lora_r,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )
            unfreeze_classifier(self.model, "classifier")

        else:
            raise ValueError(f"Unknown PEFT method: {method}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        """Single training step."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Log LR safely
        current_lr = (
            self.trainer.optimizers[0].param_groups[0]["lr"]
            if self.trainer.optimizers
            else self.config.lr
        )
        self.log("train/lr", current_lr, on_step=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, labels)
        self.val_acc_top5.update(logits, labels)
        self.val_f1.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end."""
        self.log("val/acc", self.val_acc.compute(), prog_bar=True, logger=True)
        self.log(
            "val/acc_top5", self.val_acc_top5.compute(), prog_bar=True, logger=True
        )
        self.log("val/f1", self.val_f1.compute(), prog_bar=True, logger=True)
        self.log(
            "val/precision", self.val_precision.compute(), prog_bar=True, logger=True
        )
        self.log("val/recall", self.val_recall.compute(), prog_bar=True, logger=True)

        self.val_acc.reset()
        self.val_acc_top5.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        """Single test step."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, labels)

        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_test_epoch_end(self):
        self.log("test/acc", self.val_acc.compute(), prog_bar=True, logger=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler."""
        lr = self.hparams.get("lr", 5e-5)
        weight_decay = self.hparams.get("weight_decay", 0.01)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture and parameter info."""
        return {
            "model_name": self.config.model_name,
            "peft_method": self.config.peft_method,
            "num_classes": self.num_classes,
            **get_trainable_parameters(self.model),
        }
