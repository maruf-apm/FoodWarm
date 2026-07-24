"""Parameter-Efficient Fine-Tuning (PEFT) modules for ViT.

Implements LoRA and other PEFT methods for Vision Transformers.
"""
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.

    Adds trainable low-rank matrices to existing linear layers.
    W_eff = W + (alpha/r) * B * A
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: (alpha/r) * x @ A @ B"""
        x = self.dropout(x)
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """Linear layer wrapped with LoRA adaptation."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora(x)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


class BlockExpansion(nn.Module):
    """Block Expansion PEFT method.

    Adds parallel expansion blocks to transformer layers.
    Reference: peft-vit paper (CVPR eLVM 2024)
    """

    def __init__(self, hidden_dim: int, expansion_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.expansion = nn.Sequential(
            nn.Linear(hidden_dim, expansion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        nn.init.xavier_uniform_(self.expansion[0].weight)
        nn.init.zeros_(self.expansion[3].weight)
        nn.init.zeros_(self.expansion[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expansion(x)


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> nn.Module:
    """Inject LoRA layers into target modules of a model.

    Args:
        model: The base model to adapt
        target_modules: List of module names to target (e.g., ['query', 'value'])
        r: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout rate for LoRA layers

    Returns:
        Model with LoRA layers injected
    """
    for name, module in model.named_modules():
        # Check if this module should be adapted
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Replace with LoRA-wrapped layer
                lora_layer = LinearWithLoRA(
                    base_layer=module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, child_name, lora_layer)

    return model


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters in the base model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_classifier(model: nn.Module, classifier_name: str = "classifier") -> None:
    """Unfreeze the classifier head for training."""
    for name, param in model.named_parameters():
        if classifier_name in name:
            param.requires_grad = True


def get_trainable_parameters(model: nn.Module) -> dict:
    """Get statistics about trainable vs total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_pct": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
