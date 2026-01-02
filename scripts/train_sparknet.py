#!/usr/bin/env python
"""Training script for SparkNet RGB classification model with ablation study support."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, Config
from src.data.dataloader import create_rgb_dataloaders
from src.models.sparknet import SparkNet
from src.training.trainer import SparkNetTrainer

# Optional imports for ablation study
try:
    from src.models.ablation_classifiers import (
        AblationStudy,
        SKLEARN_AVAILABLE,
        XGBOOST_AVAILABLE,
    )
    ABLATION_AVAILABLE = SKLEARN_AVAILABLE or XGBOOST_AVAILABLE
except ImportError:
    ABLATION_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_features(
    model: SparkNet,
    dataloader,
    device: torch.device,
) -> tuple:
    """Extract features from all images using SparkNet.

    Returns:
        Tuple of (features, labels)
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.get_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels)


def run_ablation_study(
    model: SparkNet,
    dataloaders: dict,
    device: torch.device,
    output_dir: Path,
    sparknet_test_accuracy: float = None,
) -> dict:
    """Run ablation study comparing SparkNet with RF/XGBoost.

    Args:
        model: Trained SparkNet model
        dataloaders: Dictionary with train/val/test dataloaders
        device: PyTorch device
        output_dir: Directory to save results
        sparknet_test_accuracy: SparkNet test accuracy for comparison

    Returns:
        Dictionary of ablation study results
    """
    logger.info("Extracting features for ablation study...")

    # Extract features
    train_features, train_labels = extract_features(model, dataloaders["train"], device)
    val_features, val_labels = extract_features(model, dataloaders["val"], device)

    test_features, test_labels = None, None
    if "test" in dataloaders:
        test_features, test_labels = extract_features(model, dataloaders["test"], device)
    else:
        # Use validation as test if no test set
        test_features, test_labels = val_features, val_labels

    logger.info(f"Feature dimensions: {train_features.shape}")

    # Run ablation study
    study = AblationStudy()
    results = study.run_study(
        train_features,
        train_labels,
        test_features,
        test_labels,
        validation_data=(val_features, val_labels),
    )

    # Generate comparison report
    report = study.compare_results(results, sparknet_accuracy=sparknet_test_accuracy)
    logger.info("\n" + report)

    # Save results
    study.save_results(results, output_dir / "ablation_results.json")

    # Save report
    with open(output_dir / "ablation_report.txt", "w") as f:
        f.write(report)

    # Save trained classifiers
    for name, classifier in study.classifiers.items():
        classifier.save(output_dir / f"{name.lower()}_classifier.pkl")
        logger.info(f"Saved {name} classifier to {output_dir / f'{name.lower()}_classifier.pkl'}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SparkNet for solar panel fault classification"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Path to validation data directory",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to test data directory (optional)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run ablation study comparing SparkNet with RF/XGBoost",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging (default: timestamp)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.no_augment:
        config.augmentation.enabled = False

    config.checkpoints.save_dir = args.output_dir

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_rgb_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        config=config,
    )

    # Print dataset info
    train_size = len(dataloaders["train"].dataset)
    val_size = len(dataloaders["val"].dataset)
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Create model
    print("Creating SparkNet model...")
    model = SparkNet(
        num_classes=len(config.classes),
        input_channels=3,
        dropout_rate=config.sparknet.dropout_rate,
    )

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = SparkNetTrainer(
        model=model,
        device=device,
        config=config,
        class_names=config.classes,
    )

    # Train
    print(f"\nStarting training for {config.training.epochs} epochs...")
    history = trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=config.training.epochs,
    )

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Evaluate on test set if provided
    test_accuracy = None
    if args.test_dir and "test" in dataloaders:
        print("\nEvaluating on test set...")
        test_metrics = trainer.validate(dataloaders["test"])
        print(f"Test Results:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")
        test_accuracy = test_metrics.get("accuracy", test_metrics.get("val_accuracy"))

        # Save test results
        test_path = output_dir / "test_results.json"
        with open(test_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

    # Run ablation study if requested
    if args.run_ablation:
        if ABLATION_AVAILABLE:
            logger.info("\nRunning ablation study...")
            ablation_results = run_ablation_study(
                model=model,
                dataloaders=dataloaders,
                device=device,
                output_dir=output_dir,
                sparknet_test_accuracy=test_accuracy,
            )
        else:
            logger.warning(
                "Ablation study requires scikit-learn or xgboost. "
                "Install with: pip install scikit-learn xgboost"
            )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
