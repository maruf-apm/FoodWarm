#!/usr/bin/env python3
"""Evaluate a trained Foodwarm model on the test set.

Usage:
    python scripts/evaluate.py --config configs/lora/food101.yaml --checkpoint ./outputs/checkpoints/lora/best.ckpt
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.data.food101_dataset import Food101DataModule
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Foodwarm Evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cpu, cuda, auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logger("foodwarm-eval")
    logger.info("=" * 60)
    logger.info("FOODWARM EVALUATION")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)

    # Setup data
    logger.info("Loading Food-101 test set...")
    datamodule = Food101DataModule(config.data)
    datamodule.prepare_data()
    datamodule.setup("test")

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = ModelFactory.create_model(
        config=config.model,
        num_classes=datamodule.num_classes,
        checkpoint_path=args.checkpoint,
    )

    # Evaluate
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    logger.info(f"Running evaluation on device: {device}")

    evaluator = Evaluator(model, device=device)
    results = evaluator.evaluate(datamodule.test_dataloader())

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test Loss: {results['loss']:.4f}")
    logger.info(f"Total Samples: {results['num_samples']:,}")
    logger.info("=" * 60)

    # Save results
    save_path = Path(config.logging.checkpoint_dir) / "evaluation_results.json"
    evaluator.save_results(results, str(save_path))
    logger.info(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
