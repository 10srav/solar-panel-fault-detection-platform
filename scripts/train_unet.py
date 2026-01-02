#!/usr/bin/env python
"""Training script for U-Net thermal segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import load_config
from src.data.dataloader import create_thermal_dataloaders
from src.models.unet import UNet
from src.training.trainer import UNetTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train U-Net for thermal image segmentation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        required=True,
        help="Path to training images directory",
    )
    parser.add_argument(
        "--train-masks",
        type=str,
        required=True,
        help="Path to training masks directory",
    )
    parser.add_argument(
        "--val-images",
        type=str,
        required=True,
        help="Path to validation images directory",
    )
    parser.add_argument(
        "--val-masks",
        type=str,
        required=True,
        help="Path to validation masks directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size

    config.checkpoints.save_dir = args.output_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_thermal_dataloaders(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        config=config,
    )

    print(f"Training samples: {len(dataloaders['train'].dataset)}")
    print(f"Validation samples: {len(dataloaders['val'].dataset)}")

    # Create model
    print("Creating U-Net model...")
    model = UNet(
        in_channels=config.unet.input_channels,
        out_channels=config.unet.output_channels,
        features=config.unet.features,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = UNetTrainer(
        model=model,
        device=device,
        config=config,
    )

    # Train
    print(f"\nStarting training for {config.training.epochs} epochs...")
    history = trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=config.training.epochs,
    )

    # Save history
    history_path = output_dir / "unet_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
