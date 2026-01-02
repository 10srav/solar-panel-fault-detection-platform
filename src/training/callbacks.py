"""Training callbacks for early stopping and model checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for accuracy/f1.
        verbose: Whether to print messages.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints.
        filename: Filename template (can include {epoch} and {metric}).
        monitor: Metric to monitor for best model.
        mode: 'min' for loss, 'max' for accuracy/f1.
        save_best_only: Only save when metric improves.
        verbose: Whether to print messages.
    """

    def __init__(
        self,
        save_dir: str | Path,
        filename: str = "model_{epoch:03d}_{metric:.4f}.pth",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.best_path: Optional[Path] = None

        if mode == "min":
            self.is_better = lambda current, best: current < best
            self.best_score = float("inf")
        else:
            self.is_better = lambda current, best: current > best
            self.best_score = float("-inf")

    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Optional[Path]:
        """Save checkpoint if conditions are met.

        Args:
            model: Model to save.
            epoch: Current epoch number.
            metrics: Dictionary of current metrics.
            optimizer: Optional optimizer to save state.
            scheduler: Optional scheduler to save state.

        Returns:
            Path to saved checkpoint if saved, None otherwise.
        """
        current_score = metrics.get(self.monitor, 0.0)

        should_save = False
        if self.save_best_only:
            if self.is_better(current_score, self.best_score):
                self.best_score = current_score
                should_save = True
        else:
            should_save = True

        if should_save:
            # Create filename
            filename = self.filename.format(epoch=epoch, metric=current_score)
            filepath = self.save_dir / filename

            # Create checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }

            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            if scheduler:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            # Save
            torch.save(checkpoint, filepath)

            if self.verbose:
                print(f"Saved checkpoint: {filepath}")

            # Save best model separately
            if self.save_best_only:
                best_path = self.save_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                self.best_path = best_path

            return filepath

        return None

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best saved model.

        Args:
            model: Model to load weights into.

        Returns:
            Model with loaded weights.
        """
        if self.best_path and self.best_path.exists():
            checkpoint = torch.load(self.best_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
        return model


class LearningRateSchedulerCallback:
    """Wrapper for learning rate scheduler with logging."""

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        verbose: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.verbose = verbose

    def step(self, metrics: Optional[dict[str, float]] = None) -> None:
        """Step the scheduler.

        Args:
            metrics: Current metrics (for ReduceLROnPlateau).
        """
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            if metrics and "val_loss" in metrics:
                self.scheduler.step(metrics["val_loss"])
        else:
            self.scheduler.step()

        if self.verbose:
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]
