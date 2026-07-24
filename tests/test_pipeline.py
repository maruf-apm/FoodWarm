"""Unit tests for the Foodwarm training pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from omegaconf import OmegaConf

from src.utils.config import Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig
from src.data.transforms import get_transforms
from src.models.peft_modules import LoRALayer, LinearWithLoRA, inject_lora, get_trainable_parameters
from src.models.vit_classifier import ViTClassifier


class TestConfig:
    """Test configuration loading."""

    def test_dataclass_creation(self):
        cfg = Config()
        assert cfg.data.batch_size == 32
        assert cfg.model.peft_method == "full"
        assert cfg.training.max_epochs == 20

    def test_config_from_dict(self):
        d = {"data": {"batch_size": 64}, "model": {"peft_method": "lora"}}
        cfg = OmegaConf.create(d)
        assert cfg.data.batch_size == 64
        assert cfg.model.peft_method == "lora"


class TestTransforms:
    """Test image transforms."""

    def test_transforms_return_tensors(self):
        from PIL import Image
        import numpy as np

        train_t, val_t = get_transforms(224)

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        train_tensor = train_t(img)
        val_tensor = val_t(img)

        assert isinstance(train_tensor, torch.Tensor)
        assert isinstance(val_tensor, torch.Tensor)
        assert train_tensor.shape == (3, 224, 224)
        assert val_tensor.shape == (3, 224, 224)


class TestPEFTModules:
    """Test PEFT module implementations."""

    def test_lora_layer_shape(self):
        lora = LoRALayer(in_features=768, out_features=768, r=8, alpha=16)
        x = torch.randn(2, 768)
        out = lora(x)
        assert out.shape == (2, 768)

    def test_lora_scaling(self):
        lora = LoRALayer(in_features=768, out_features=768, r=8, alpha=16)
        assert lora.scaling == 2.0  # alpha / r = 16 / 8

    def test_linear_with_lora(self):
        base = torch.nn.Linear(768, 768)
        wrapped = LinearWithLoRA(base, r=4, alpha=8, dropout=0.1)

        x = torch.randn(2, 768)
        out = wrapped(x)
        assert out.shape == (2, 768)

        # Base should be frozen
        for param in wrapped.base_layer.parameters():
            assert not param.requires_grad

        # LoRA should be trainable
        assert wrapped.lora.lora_A.requires_grad
        assert wrapped.lora.lora_B.requires_grad

    def test_parameter_stats(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 5),
        )
        stats = get_trainable_parameters(model)
        assert stats["total"] == 10*10 + 10 + 10*5 + 5  # weights + biases
        assert stats["trainable"] == stats["total"]
        assert stats["trainable_pct"] == 100.0


class TestViTClassifier:
    """Test ViT classifier model."""

    def test_model_creation_full(self):
        model_cfg = ModelConfig(peft_method="full")
        model = ViTClassifier(config=model_cfg, num_classes=10)
        assert model is not None
        assert model.num_classes == 10

    def test_model_creation_lora(self):
        model_cfg = ModelConfig(peft_method="lora", lora_r=4, lora_alpha=8)
        model = ViTClassifier(config=model_cfg, num_classes=10)
        info = model.get_model_info()
        assert info["trainable_pct"] < 10.0  # LoRA should be parameter-efficient

    def test_model_creation_linear(self):
        model_cfg = ModelConfig(peft_method="linear")
        model = ViTClassifier(config=model_cfg, num_classes=10)
        info = model.get_model_info()
        assert info["trainable_pct"] < 5.0  # Linear probe should be very small

    def test_forward_pass(self):
        model_cfg = ModelConfig(peft_method="linear")
        model = ViTClassifier(config=model_cfg, num_classes=10)
        model.eval()

        # Dummy input (batch_size=2, channels=3, 224x224)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (2, 10)

    def test_training_step(self):
        model_cfg = ModelConfig(peft_method="linear")
        model = ViTClassifier(config=model_cfg, num_classes=10)

        # Dummy batch
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        batch = (images, labels)

        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_forward(self):
        """Test a complete forward pass through the model."""
        from PIL import Image
        import numpy as np

        # Create model
        model_cfg = ModelConfig(peft_method="linear")
        model = ViTClassifier(config=model_cfg, num_classes=101)
        model.eval()

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # Apply transform
        _, val_transform = get_transforms(224)
        tensor = val_transform(img).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            logits = model(tensor)

        assert logits.shape == (1, 101)
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
