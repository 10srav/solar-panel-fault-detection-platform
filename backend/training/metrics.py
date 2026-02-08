"""
=============================================================================
 METRICS MODULE - Training Evaluation & Visualization
 Comprehensive metrics calculation and plotting for model evaluation
=============================================================================
"""

import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from pathlib import Path


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        class_names: List of class names

    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics['per_class'] = {}
    for class_name in class_names:
        if class_name in report:
            metrics['per_class'][class_name] = {
                'precision': float(report[class_name]['precision']),
                'recall': float(report[class_name]['recall']),
                'f1_score': float(report[class_name]['f1-score']),
                'support': int(report[class_name]['support']),
            }

    return metrics


def compute_f1(y_true, y_pred, average='macro'):
    """
    Compute F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro', 'weighted', or 'micro'

    Returns:
        float: F1 score
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix'):
    """
    Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Confusion matrix saved: {save_path}")


def plot_class_accuracy_bar(metrics, save_path, title='Per-Class Accuracy'):
    """
    Plot per-class accuracy/F1 as bar chart.

    Args:
        metrics: Metrics dictionary from compute_metrics()
        save_path: Path to save the plot
        title: Plot title
    """
    if 'per_class' not in metrics:
        print("[WARNING] No per-class metrics found")
        return

    class_names = list(metrics['per_class'].keys())
    precisions = [metrics['per_class'][c]['precision'] * 100 for c in class_names]
    recalls = [metrics['per_class'][c]['recall'] * 100 for c in class_names]
    f1_scores = [metrics['per_class'][c]['f1_score'] * 100 for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Class accuracy bar chart saved: {save_path}")


def plot_training_curves(history, save_path, title='Training History'):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys: train_loss, train_acc, val_loss, val_acc
                 Each value is a list of epoch values
        save_path: Path to save the plot
        title: Plot title
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s')
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy', marker='s')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Training curves saved: {save_path}")


def save_metrics_json(metrics, save_path):
    """Save metrics dictionary to JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[METRICS] JSON saved: {save_path}")


def save_metrics_csv(metrics, save_path):
    """Save metrics to CSV file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Overall metrics
    rows.append(['Metric', 'Value'])
    rows.append(['Accuracy', f"{metrics['accuracy']:.4f}"])
    rows.append(['Precision (Macro)', f"{metrics['precision_macro']:.4f}"])
    rows.append(['Precision (Weighted)', f"{metrics['precision_weighted']:.4f}"])
    rows.append(['Recall (Macro)', f"{metrics['recall_macro']:.4f}"])
    rows.append(['Recall (Weighted)', f"{metrics['recall_weighted']:.4f}"])
    rows.append(['F1-Score (Macro)', f"{metrics['f1_macro']:.4f}"])
    rows.append(['F1-Score (Weighted)', f"{metrics['f1_weighted']:.4f}"])
    rows.append([])

    # Per-class metrics
    rows.append(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    for class_name, class_metrics in metrics.get('per_class', {}).items():
        rows.append([
            class_name,
            f"{class_metrics['precision']:.4f}",
            f"{class_metrics['recall']:.4f}",
            f"{class_metrics['f1_score']:.4f}",
            str(class_metrics['support'])
        ])

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[METRICS] CSV saved: {save_path}")


def save_training_log(epoch_logs, save_path):
    """
    Save epoch-by-epoch training logs to CSV.

    Args:
        epoch_logs: List of dicts, each containing epoch, train_loss, train_acc, val_loss, val_acc
        save_path: Path to save CSV
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if not epoch_logs:
        print("[WARNING] No epoch logs to save")
        return

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"[LOG] Training log saved: {save_path}")
