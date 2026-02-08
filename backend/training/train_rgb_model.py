"""
=============================================================================
 MODEL TRAINING — train_rgb_model.py
 ResNet18 transfer learning with early stopping, weighted loss, live logging.
=============================================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEVICE, MODEL_PATH, PLOTS_DIR, NUM_CLASSES, CLASS_NAMES,
    LEARNING_RATE, EPOCHS, STEP_SIZE, GAMMA, EARLY_STOPPING_PATIENCE,
    FREEZE_LAYERS,
)
from training.preprocessing import build_dataloaders


# ============================================================================
# MODEL BUILDER
# ============================================================================

def build_model():
    """
    Load pretrained ResNet18, freeze early layers, replace FC -> 6 outputs.
    Returns model on configured device.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze specified layers
    layers_map = {
        "conv1": model.conv1,
        "bn1": model.bn1,
        "layer1": model.layer1,
        "layer2": model.layer2,
    }
    for name in FREEZE_LAYERS:
        if name in layers_map:
            for param in layers_map[name].parameters():
                param.requires_grad = False

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    model = model.to(DEVICE)

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\n[MODEL] ResNet18 -> {NUM_CLASSES} classes on {DEVICE}")
    print(f"  Total params:     {total:>10,}")
    print(f"  Trainable params: {trainable:>10,}")
    print(f"  Frozen params:    {frozen:>10,}")

    return model


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n[EARLY STOP] No improvement for {self.patience} epochs.")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, class_weights, class_names):
    """
    Full training loop with:
    - Weighted CrossEntropyLoss
    - Adam optimizer + StepLR scheduler
    - Early stopping
    - Best model checkpoint
    - Live per-epoch logging
    """

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []}

    header = (
        f"{'Epoch':>7} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>10} | {'Val Acc':>9} | {'LR':>10} | {'Time':>6} | Status"
    )

    print(f"\n{'=' * 90}")
    print(f" TRAINING — {EPOCHS} epochs, Early Stop patience={EARLY_STOPPING_PATIENCE}")
    print(f"{'=' * 90}")
    print(header)
    print("-" * 90)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # ---- VALIDATE ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        elapsed = time.time() - epoch_start

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, MODEL_PATH)
            status = "* BEST"

        lr = scheduler.get_last_lr()[0]
        print(
            f"  {epoch:2d}/{EPOCHS:2d}  | "
            f"{train_loss:10.4f} | {train_acc:8.1f}% | "
            f"{val_loss:10.4f} | {val_acc:8.1f}% | "
            f"{lr:10.6f} | {elapsed:5.1f}s | {status}"
        )

        scheduler.step()

        # Early stopping check
        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            break

    print(f"\n{'=' * 90}")
    print(f" TRAINING COMPLETE — Best Val Accuracy: {best_val_acc:.2f}%")
    print(f" Model saved: {MODEL_PATH}")
    print(f"{'=' * 90}")

    # ---- CONFUSION MATRIX ----
    print(f"\n[CLASSIFICATION REPORT]\n")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, digits=3
    ))

    _plot_confusion_matrix(all_labels, all_preds, class_names)
    _plot_training_curves(history)

    return history, best_val_acc


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_confusion_matrix(labels, preds, class_names):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix — Final Epoch", fontsize=14)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix: {path}")


def _plot_training_curves(history):
    """Generate and save loss + accuracy curves."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(epochs_range, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"], "o-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_range, history["train_acc"], "o-", label="Train Acc", color="blue")
    axes[1].plot(epochs_range, history["val_acc"], "o-", label="Val Acc", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Training curves: {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 90)
    print(" SOLAR PANEL FAULT DETECTION — RGB MODEL TRAINING")
    print("=" * 90)
    print(f"Device: {DEVICE}")

    # Data pipeline
    train_loader, val_loader, class_weights, class_names = build_dataloaders()

    # Build model
    model = build_model()

    # Train
    history, best_acc = train_model(
        model, train_loader, val_loader, class_weights, class_names
    )

    print("\nTraining complete. Run the API server or inference script next.")
