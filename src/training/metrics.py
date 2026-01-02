"""Training metrics for solar panel fault detection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    class_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average: Averaging method for multi-class metrics.
        class_names: Names of classes for per-class metrics.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        f"precision_{average}": precision_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        f"recall_{average}": recall_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        f"f1_{average}": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Add per-class metrics if class names provided
    if class_names:
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f"precision_{class_name}"] = precision_per_class[i]
                metrics[f"recall_{class_name}"] = recall_per_class[i]
                metrics[f"f1_{class_name}"] = f1_per_class[i]

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    normalize: bool = False,
) -> tuple[np.ndarray, Optional[list[str]]]:
    """Compute confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names of classes.
        normalize: Whether to normalize the confusion matrix.

    Returns:
        Tuple of confusion matrix and class names.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    return cm, class_names


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> str:
    """Get sklearn classification report as string.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names of classes.

    Returns:
        Classification report string.
    """
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )


# Segmentation metrics
def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute segmentation metrics.

    Args:
        predictions: Model predictions (logits or probabilities).
        targets: Ground truth masks.
        threshold: Threshold for binarizing predictions.

    Returns:
        Dictionary of segmentation metrics.
    """
    # Apply sigmoid if needed and threshold
    if predictions.min() < 0:
        predictions = torch.sigmoid(predictions)

    pred_binary = (predictions > threshold).float()
    targets = targets.float()

    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)

    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    # IoU (Jaccard Index)
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Dice coefficient
    dice = (2.0 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    # Pixel accuracy
    correct = (pred_flat == target_flat).sum()
    pixel_accuracy = correct / pred_flat.numel()

    # Precision and Recall
    true_positive = intersection
    false_positive = pred_flat.sum() - true_positive
    false_negative = target_flat.sum() - true_positive

    precision = (true_positive + 1e-6) / (true_positive + false_positive + 1e-6)
    recall = (true_positive + 1e-6) / (true_positive + false_negative + 1e-6)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "pixel_accuracy": pixel_accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


class MetricTracker:
    """Track metrics during training."""

    def __init__(self, metric_names: list[str]) -> None:
        self.metric_names = metric_names
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self._data = {name: [] for name in self.metric_names}
        self._count = 0

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        """Update metrics with new values.

        Args:
            metrics: Dictionary of metric values.
            n: Number of samples (for weighted averaging).
        """
        for name, value in metrics.items():
            if name in self._data:
                self._data[name].append(value * n)
        self._count += n

    def compute(self) -> dict[str, float]:
        """Compute average metrics."""
        return {
            name: sum(values) / max(self._count, 1)
            for name, values in self._data.items()
        }

    def get_value(self, name: str) -> float:
        """Get current average value of a metric."""
        if name in self._data:
            return sum(self._data[name]) / max(self._count, 1)
        return 0.0
