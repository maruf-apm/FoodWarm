"""Custom metrics and evaluation utilities."""
import torch
from torchmetrics import Metric


class TopKAccuracy(Metric):
    """Custom top-k accuracy metric."""

    def __init__(self, num_classes: int, k: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state."""
        _, top_k_preds = preds.topk(self.k, dim=1)
        correct = top_k_preds.eq(target.unsqueeze(1).expand_as(top_k_preds))
        self.correct += correct.any(dim=1).sum()
        self.total += target.numel()

    def compute(self):
        """Compute final metric value."""
        return self.correct.float() / self.total


def compute_confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """Compute confusion matrix."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm
