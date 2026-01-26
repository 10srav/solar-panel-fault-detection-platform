"""DataLoader creation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.datasets import SolarPanelRGBDataset, ThermalSegmentationDataset
from src.data.transforms import get_rgb_transforms, get_thermal_transforms
from src.config import Config, get_config


def create_rgb_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    test_dir: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_weighted_sampler: bool = True,
) -> dict[str, DataLoader]:
    """Create DataLoaders for RGB classification.

    Args:
        train_dir: Training data directory.
        val_dir: Validation data directory.
        test_dir: Test data directory (optional).
        config: Configuration object.
        batch_size: Batch size (overrides config).
        num_workers: Number of workers (overrides config).
        use_weighted_sampler: Use weighted sampler for class imbalance.

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders.
    """
    if config is None:
        config = get_config()

    batch_size = batch_size or config.training.batch_size
    num_workers = num_workers or config.training.num_workers

    # Get transforms
    transforms = get_rgb_transforms(
        input_size=tuple(config.sparknet.input_size),
        augment=config.augmentation.enabled,
        rotation_range=config.augmentation.rotation_range,
        brightness_range=tuple(config.augmentation.brightness_range),
        contrast_range=tuple(config.augmentation.contrast_range),
        horizontal_flip=config.augmentation.horizontal_flip,
        vertical_flip=config.augmentation.vertical_flip,
        affine_scale=tuple(config.augmentation.affine_scale),
    )

    # Create datasets
    train_dataset = SolarPanelRGBDataset(
        root_dir=train_dir,
        transform=transforms["train"],
        classes=config.classes,
    )

    val_dataset = SolarPanelRGBDataset(
        root_dir=val_dir,
        transform=transforms["val"],
        classes=config.classes,
    )

    # Create samplers
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory,
        ),
    }

    if test_dir:
        test_dataset = SolarPanelRGBDataset(
            root_dir=test_dir,
            transform=transforms["test"],
            classes=config.classes,
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory,
        )

    return dataloaders


def create_thermal_dataloaders(
    train_images_dir: Union[str, Path],
    train_masks_dir: Union[str, Path],
    val_images_dir: Union[str, Path],
    val_masks_dir: Union[str, Path],
    config: Optional[Config] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> dict[str, DataLoader]:
    """Create DataLoaders for thermal segmentation.

    Args:
        train_images_dir: Training images directory.
        train_masks_dir: Training masks directory.
        val_images_dir: Validation images directory.
        val_masks_dir: Validation masks directory.
        config: Configuration object.
        batch_size: Batch size (overrides config).
        num_workers: Number of workers (overrides config).

    Returns:
        Dictionary with 'train' and 'val' DataLoaders.
    """
    if config is None:
        config = get_config()

    batch_size = batch_size or config.training.batch_size
    num_workers = num_workers or config.training.num_workers

    # Get transforms
    transforms = get_thermal_transforms(
        input_size=tuple(config.unet.input_size),
        augment=config.augmentation.enabled,
        rotation_range=config.augmentation.rotation_range,
        horizontal_flip=config.augmentation.horizontal_flip,
        vertical_flip=config.augmentation.vertical_flip,
    )

    # Create datasets
    train_dataset = ThermalSegmentationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=transforms["train"],
    )

    val_dataset = ThermalSegmentationDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        transform=transforms["val"],
    )

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory,
        ),
    }

    return dataloaders


def create_dataloaders(
    data_type: str,
    config: Optional[Config] = None,
    **kwargs,
) -> dict[str, DataLoader]:
    """Create dataloaders based on data type.

    Args:
        data_type: Either 'rgb' or 'thermal'.
        config: Configuration object.
        **kwargs: Additional arguments for specific dataloader creator.

    Returns:
        Dictionary of DataLoaders.
    """
    if config is None:
        config = get_config()

    if data_type == "rgb":
        return create_rgb_dataloaders(
            train_dir=kwargs.get("train_dir", config.data.rgb.get("train_dir")),
            val_dir=kwargs.get("val_dir", config.data.rgb.get("val_dir")),
            test_dir=kwargs.get("test_dir", config.data.rgb.get("test_dir")),
            config=config,
            **{k: v for k, v in kwargs.items() if k not in ["train_dir", "val_dir", "test_dir"]},
        )
    elif data_type == "thermal":
        return create_thermal_dataloaders(
            train_images_dir=kwargs.get("train_images_dir", config.data.thermal.get("train_images")),
            train_masks_dir=kwargs.get("train_masks_dir", config.data.thermal.get("train_masks")),
            val_images_dir=kwargs.get("val_images_dir", config.data.thermal.get("val_images")),
            val_masks_dir=kwargs.get("val_masks_dir", config.data.thermal.get("val_masks")),
            config=config,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "train_images_dir",
                    "train_masks_dir",
                    "val_images_dir",
                    "val_masks_dir",
                ]
            },
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}. Must be 'rgb' or 'thermal'.")
