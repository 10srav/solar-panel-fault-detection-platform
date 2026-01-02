"""Training loop implementations for SparkNet and U-Net."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.metrics import (
    calculate_metrics,
    compute_segmentation_metrics,
    MetricTracker,
)
from src.config import Config, get_config


class Trainer:
    """Base trainer class.

    Args:
        model: PyTorch model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        config: Configuration object.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or get_config()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Mixed precision training
        self.use_amp = self.config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping.patience,
            min_delta=self.config.training.early_stopping.min_delta,
            mode="max",
        )

        self.checkpoint = ModelCheckpoint(
            save_dir=self.config.checkpoints.save_dir,
            monitor="val_f1_macro",
            mode="max",
            save_best_only=self.config.checkpoints.save_best_only,
        )

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_f1_macro": [],
            "learning_rate": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        raise NotImplementedError

    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Validate the model."""
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs (overrides config).

        Returns:
            Training history.
        """
        epochs = epochs or self.config.training.epochs

        print(f"Training on {self.device}")
        print(f"Mixed precision: {self.use_amp}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics.get("accuracy", 0))
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0))
            self.history["val_f1_macro"].append(val_metrics.get("f1_macro", 0))
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Print progress
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f} - "
                f"Val F1: {val_metrics.get('f1_macro', 0):.4f}"
            )

            # Checkpoint
            self.checkpoint(
                self.model,
                epoch,
                val_metrics,
                self.optimizer,
                self.scheduler,
            )

            # Early stopping
            if self.early_stopping(val_metrics.get("f1_macro", 0)):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history

    def save_history(self, path: str | Path) -> None:
        """Save training history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


class SparkNetTrainer(Trainer):
    """Trainer for SparkNet classification model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[Config] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        config = config or get_config()

        # Default criterion
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Default optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                betas=tuple(config.training.optimizer.betas),
                weight_decay=config.training.optimizer.weight_decay,
            )

        # Default scheduler
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.lr_scheduler.step_size,
                gamma=config.training.lr_scheduler.gamma,
            )

        super().__init__(model, criterion, optimizer, scheduler, device, config)

        self.class_names = class_names or config.classes

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = calculate_metrics(
            np.array(all_labels), np.array(all_preds), class_names=self.class_names
        )
        metrics["loss"] = epoch_loss

        return metrics

    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Validate the model.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = calculate_metrics(
            np.array(all_labels), np.array(all_preds), class_names=self.class_names
        )
        metrics["loss"] = epoch_loss

        return metrics

    def predict(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions on a dataset.

        Args:
            dataloader: Data loader.

        Returns:
            Tuple of (predictions, probabilities, labels).
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_preds), np.array(all_probs), np.array(all_labels)


class UNetTrainer(Trainer):
    """Trainer for U-Net segmentation model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[Config] = None,
    ) -> None:
        from src.models.unet import DiceBCELoss

        config = config or get_config()

        # Default criterion (combined BCE + Dice)
        if criterion is None:
            criterion = DiceBCELoss()

        # Default optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                betas=tuple(config.training.optimizer.betas),
                weight_decay=config.training.optimizer.weight_decay,
            )

        # Default scheduler
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )

        super().__init__(model, criterion, optimizer, scheduler, device, config)

        # Override early stopping for segmentation
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping.patience,
            min_delta=self.config.training.early_stopping.min_delta,
            mode="max",  # Monitor IoU
        )

        # Override checkpoint
        self.checkpoint = ModelCheckpoint(
            save_dir=self.config.checkpoints.save_dir,
            filename="unet_{epoch:03d}_{metric:.4f}.pth",
            monitor="iou",
            mode="max",
            save_best_only=self.config.checkpoints.save_best_only,
        )

        # Override history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_iou": [],
            "val_iou": [],
            "train_dice": [],
            "val_dice": [],
            "learning_rate": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        metric_tracker = MetricTracker(["iou", "dice", "pixel_accuracy"])

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Compute metrics
            with torch.no_grad():
                batch_metrics = compute_segmentation_metrics(outputs, masks)
                metric_tracker.update(batch_metrics, images.size(0))

            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = metric_tracker.compute()
        metrics["loss"] = epoch_loss

        return metrics

    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        metric_tracker = MetricTracker(["iou", "dice", "pixel_accuracy"])

        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validating", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                running_loss += loss.item() * images.size(0)

                batch_metrics = compute_segmentation_metrics(outputs, masks)
                metric_tracker.update(batch_metrics, images.size(0))

        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = metric_tracker.compute()
        metrics["loss"] = epoch_loss

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> dict[str, list[float]]:
        """Train the model."""
        epochs = epochs or self.config.training.epochs

        print(f"Training on {self.device}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler (ReduceLROnPlateau)
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_iou"].append(train_metrics.get("iou", 0))
            self.history["val_iou"].append(val_metrics.get("iou", 0))
            self.history["train_dice"].append(train_metrics.get("dice", 0))
            self.history["val_dice"].append(val_metrics.get("dice", 0))
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Print progress
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val IoU: {val_metrics.get('iou', 0):.4f} - "
                f"Val Dice: {val_metrics.get('dice', 0):.4f}"
            )

            # Checkpoint
            self.checkpoint(
                self.model,
                epoch,
                val_metrics,
                self.optimizer,
                self.scheduler,
            )

            # Early stopping
            if self.early_stopping(val_metrics.get("iou", 0)):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history

    def predict_mask(
        self, image: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Predict segmentation mask for a single image.

        Args:
            image: Input image tensor.
            threshold: Threshold for binarization.

        Returns:
            Binary mask tensor.
        """
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

            output = self.model(image)
            prob = torch.sigmoid(output)
            mask = (prob > threshold).float()

        return mask.squeeze()
