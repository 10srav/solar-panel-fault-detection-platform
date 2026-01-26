"""Image transforms for solar panel fault detection."""

from __future__ import annotations

from typing import Any, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_rgb_transforms(
    input_size: tuple[int, int] = (227, 227),
    augment: bool = True,
    rotation_range: int = 15,
    brightness_range: tuple[float, float] = (0.8, 1.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    affine_scale: tuple[float, float] = (0.9, 1.1),
) -> dict[str, A.Compose]:
    """Get transforms for RGB classification.

    Args:
        input_size: Target image size (height, width).
        augment: Whether to apply augmentation.
        rotation_range: Maximum rotation angle in degrees.
        brightness_range: Brightness adjustment range.
        contrast_range: Contrast adjustment range.
        horizontal_flip: Apply horizontal flip.
        vertical_flip: Apply vertical flip.
        affine_scale: Scale range for affine transform.

    Returns:
        Dictionary with 'train' and 'val' transforms.
    """
    # Normalization values (ImageNet statistics)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation (improved for better accuracy)
    if augment:
        train_transform = A.Compose(
            [
                A.Resize(input_size[0], input_size[1]),
                A.HorizontalFlip(p=0.5 if horizontal_flip else 0.0),
                A.VerticalFlip(p=0.5 if vertical_flip else 0.0),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=rotation_range,
                    border_mode=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(brightness_range[0] - 1, brightness_range[1] - 1),
                    contrast_limit=(contrast_range[0] - 1, contrast_range[1] - 1),
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=0.3,
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(input_size[0] * 0.1),
                    max_width=int(input_size[1] * 0.1),
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        train_transform = A.Compose(
            [
                A.Resize(input_size[0], input_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    # Validation/Test transforms (no augmentation)
    val_transform = A.Compose(
        [
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return {"train": train_transform, "val": val_transform, "test": val_transform}


def get_thermal_transforms(
    input_size: tuple[int, int] = (256, 256),
    augment: bool = True,
    rotation_range: int = 15,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
) -> dict[str, A.Compose]:
    """Get transforms for thermal segmentation.

    Args:
        input_size: Target image size (height, width).
        augment: Whether to apply augmentation.
        rotation_range: Maximum rotation angle.
        horizontal_flip: Apply horizontal flip.
        vertical_flip: Apply vertical flip.

    Returns:
        Dictionary with 'train' and 'val' transforms.
    """
    # Training transforms
    if augment:
        train_transform = A.Compose(
            [
                A.Resize(input_size[0], input_size[1]),
                A.HorizontalFlip(p=0.5 if horizontal_flip else 0.0),
                A.VerticalFlip(p=0.5 if vertical_flip else 0.0),
                A.Rotate(limit=rotation_range, p=0.5, border_mode=0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ]
        )
    else:
        train_transform = A.Compose(
            [
                A.Resize(input_size[0], input_size[1]),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ]
        )

    # Validation transforms
    val_transform = A.Compose(
        [
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ]
    )

    return {"train": train_transform, "val": val_transform, "test": val_transform}


def denormalize_rgb(
    image: np.ndarray,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize an RGB image.

    Args:
        image: Normalized image array (C, H, W) or (H, W, C).
        mean: Normalization mean values.
        std: Normalization std values.

    Returns:
        Denormalized image in range [0, 255].
    """
    if image.shape[0] == 3:  # (C, H, W)
        image = image.transpose(1, 2, 0)

    mean = np.array(mean)
    std = np.array(std)

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def denormalize_thermal(
    image: np.ndarray,
    mean: float = 0.5,
    std: float = 0.5,
) -> np.ndarray:
    """Denormalize a thermal image.

    Args:
        image: Normalized image array.
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Denormalized image in range [0, 255].
    """
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image
