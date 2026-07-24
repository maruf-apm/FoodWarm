"""Model evaluation utilities."""
from typing import Dict, List, Optional
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vit_classifier import ViTClassifier
from src.data.food101_dataset import Food101DataModule


class Evaluator:
    """Evaluate a trained model on test data."""

    def __init__(self, model: ViTClassifier, device: str = "auto"):
        self.model = model
        self.device = device
        if device != "auto":
            self.model = self.model.to(device)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Run full evaluation and return metrics."""
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        criterion = torch.nn.CrossEntropyLoss()

        for batch in tqdm(dataloader, desc="Evaluating"):
            images, labels = batch
            if self.device != "auto":
                images = images.to(self.device)
                labels = labels.to(self.device)

            logits = self.model(images)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total_loss += loss.item()
            num_batches += 1

        # Compute metrics
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        avg_loss = total_loss / num_batches

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "num_samples": len(all_labels),
            "predictions": all_preds,
            "labels": all_labels,
        }

    def save_results(self, results: Dict, save_path: str) -> None:
        """Save evaluation results to JSON."""
        # Remove large arrays before saving
        save_dict = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=2)

        # Save predictions separately if needed
        pred_path = save_path.replace(".json", "_predictions.json")
        with open(pred_path, "w") as f:
            json.dump({
                "predictions": results["predictions"],
                "labels": results["labels"],
            }, f)
