"""Configuration management for Foodwarm training pipeline."""
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import OmegaConf
import yaml
import os


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "food101"
    data_dir: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "google/vit-base-patch16-224"
    num_classes: int = 101
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.0
    peft_method: str = "full"  # full, lora, linear
    # LoRA specific
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    # Linear probing specific
    linear_probe_layers: List[str] = field(default_factory=lambda: ["classifier"])


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    max_epochs: int = 20
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    scheduler: str = "cosine"  # cosine, linear, plateau
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 42
    early_stopping_patience: int = 5
    save_top_k: int = 3
    monitor_metric: str = "val_acc"
    monitor_mode: str = "max"


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    project_name: str = "foodwarm"
    experiment_name: str = "vit-food101"
    log_dir: str = "./outputs/logs"
    checkpoint_dir: str = "./outputs/checkpoints"
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 50


@dataclass
class Config:
    """Master configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    # Convert dict to OmegaConf then to dataclass
    cfg = OmegaConf.create(yaml_dict)

    # Build nested dataclass
    config = Config(
        data=DataConfig(**cfg.data),
        model=ModelConfig(**cfg.model),
        training=TrainingConfig(**cfg.training),
        logging=LoggingConfig(**cfg.logging),
    )
    return config


def save_config(config: Config, save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cfg_dict = {
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "logging": config.logging.__dict__,
    }
    with open(save_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
