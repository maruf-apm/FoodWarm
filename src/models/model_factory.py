"""Factory for creating models with different PEFT configurations."""
from typing import Optional

from src.models.vit_classifier import ViTClassifier
from src.utils.config import ModelConfig


class ModelFactory:
    """Factory class to create ViT models with various PEFT methods."""

    @staticmethod
    def create_model(
        config: ModelConfig,
        num_classes: int = 101,
        checkpoint_path: Optional[str] = None,
    ) -> ViTClassifier:
        """Create a ViT model with specified PEFT configuration.

        Args:
            config: Model configuration
            num_classes: Number of output classes
            checkpoint_path: Optional path to load checkpoint from

        Returns:
            Configured ViTClassifier model
        """
        model = ViTClassifier(config=config, num_classes=num_classes)

        if checkpoint_path is not None:
            checkpoint = ModelFactory.load_checkpoint(checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"], strict=False)

        return model

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> dict:
        """Load a Lightning checkpoint."""
        import torch
        return torch.load(checkpoint_path, map_location="cpu")

    @staticmethod
    def get_available_methods() -> list:
        """Return list of available PEFT methods."""
        return ["full", "linear", "lora"]
