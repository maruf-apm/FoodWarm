# Foodwarm - ViT Fine-Tuning Pipeline for Food-101

A professional, production-ready training pipeline for fine-tuning Vision Transformers (ViT) on the Food-101 dataset using Parameter-Efficient Fine-Tuning (PEFT) methods.

## Architecture

```
Foodwarm/
├── configs/              # YAML configuration files
│   ├── full/            # Full fine-tuning configs
│   ├── lora/            # LoRA fine-tuning configs
│   └── linear/          # Linear probing configs
├── src/
│   ├── data/            # Dataset & transforms
│   ├── models/          # ViT model + PEFT modules
│   ├── training/        # Callbacks & metrics
│   ├── evaluation/      # Evaluation utilities
│   └── utils/           # Config, logging, checkpoints
├── scripts/
│   ├── test_train.py    # Quick pipeline verification
│   ├── train.py         # Full training pipeline
│   └── evaluate.py      # Model evaluation
└── tests/               # Unit & integration tests
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Pipeline (Quick Verification)

Run a minimal 2-epoch test to verify everything works:

```bash
python scripts/test_train.py --config configs/lora/food101.yaml
```

This will:
- Download Food-101 (if not present)
- Create a ViT model with LoRA
- Run 2 quick training epochs
- Validate the pipeline end-to-end

### 3. Full Training

```bash
# Full fine-tuning
python scripts/train.py --config configs/full/food101.yaml

# LoRA fine-tuning (recommended - memory efficient)
python scripts/train.py --config configs/lora/food101.yaml

# Linear probing (fastest, least parameters)
python scripts/train.py --config configs/linear/food101.yaml

# Resume from checkpoint
python scripts/train.py --config configs/lora/food101.yaml --resume ./outputs/checkpoints/lora/last.ckpt
```

### 4. Evaluate Trained Model

```bash
python scripts/evaluate.py --config configs/lora/food101.yaml --checkpoint ./outputs/checkpoints/lora/best.ckpt
```

### 5. Run Tests

```bash
pytest tests/test_pipeline.py -v
```

## PEFT Methods

| Method | Trainable Params | Speed | Accuracy | Use Case |
|--------|-----------------|-------|----------|----------|
| **Full** | 100% | Slow | Highest | Maximum accuracy, GPU memory available |
| **LoRA** | ~1-2% | Fast | High | Best efficiency/accuracy tradeoff |
| **Linear** | <1% | Fastest | Good | Quick baseline, limited data |

## Configuration

All hyperparameters are controlled via YAML configs. Key sections:

```yaml
# Data
data:
  batch_size: 32
  image_size: 224
  num_workers: 4

# Model
model:
  model_name: "google/vit-base-patch16-224"
  peft_method: "lora"  # full | lora | linear
  lora_r: 8
  lora_alpha: 16

# Training
training:
  max_epochs: 20
  lr: 0.0001
  precision: "16-mixed"  # Mixed precision training
```

## Outputs

Training outputs are organized as:

```
outputs/
├── checkpoints/       # Saved model checkpoints
│   ├── lora/
│   ├── full/
│   └── linear/
└── logs/              # Training logs & metrics
    ├── csv/
    └── wandb/         # If WandB enabled
```

## Integrating with Your App

Once training completes, export the model for your FoodLens app:

```python
from src.models.model_factory import ModelFactory
from src.utils.config import load_config

config = load_config("configs/lora/food101.yaml")
model = ModelFactory.create_model(config.model, num_classes=101)

# Save for inference
torch.save(model.state_dict(), "foodlens_model.pt")
```

Then place `foodlens_model.pt` in your app's `models/` folder and update `app/model.py` to load it.

## License

MIT License
