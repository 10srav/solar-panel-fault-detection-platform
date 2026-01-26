"""Dataset classes for solar panel fault detection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SolarPanelRGBDataset(Dataset):
    """Dataset for RGB solar panel classification.

    Expects directory structure:
        root_dir/
            class_name_1/
                image1.jpg
                image2.jpg
            class_name_2/
                image3.jpg
                ...

    Args:
        root_dir: Root directory containing class subdirectories.
        transform: Transform to apply to images.
        classes: List of class names (if None, inferred from directory structure).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Get class names
        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    def get_class_counts(self) -> dict[str, int]:
        """Get the count of samples per class."""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts

    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling."""
        counts = self.get_class_counts()
        total = len(self.samples)

        weights = []
        for _, label in self.samples:
            class_name = self.classes[label]
            weight = total / (len(self.classes) * counts[class_name])
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float)


class ThermalSegmentationDataset(Dataset):
    """Dataset for thermal image segmentation.

    Expects directory structure:
        images_dir/
            image1.png
            image2.png
        masks_dir/
            image1.png  (same name as corresponding image)
            image2.png

    Args:
        images_dir: Directory containing thermal images.
        masks_dir: Directory containing segmentation masks.
        transform: Transform to apply to both images and masks.
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Get all image files
        self.image_files = sorted(
            [
                f
                for f in self.images_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Load corresponding mask
        mask_path = self.masks_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Try different extensions
            mask_found = False
            for ext in [".png", ".jpg", ".bmp"]:
                mask_path = self.masks_dir / (img_path.stem + ext)
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask_found = True
                    break
            if not mask_found:
                # Create empty mask if not found and log warning
                logger.warning(
                    f"Mask not found for image {img_path.name}, "
                    f"creating empty mask. Expected at: {self.masks_dir / img_path.name}"
                )
                mask = np.zeros_like(image)

        # Ensure mask is binary
        mask = (mask > 127).astype(np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Ensure mask has correct shape
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask


class InferenceDataset(Dataset):
    """Dataset for inference on single images.

    Args:
        image_paths: List of image paths or single directory.
        transform: Transform to apply to images.
        is_thermal: Whether images are thermal (grayscale).
    """

    def __init__(
        self,
        image_paths: Union[str, Path, list[Union[str, Path]]],
        transform: Optional[Callable] = None,
        is_thermal: bool = False,
    ) -> None:
        if isinstance(image_paths, (str, Path)):
            path = Path(image_paths)
            if path.is_dir():
                self.image_paths = sorted(
                    [
                        f
                        for f in path.iterdir()
                        if f.suffix.lower()
                        in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
                    ]
                )
            else:
                self.image_paths = [path]
        else:
            self.image_paths = [Path(p) for p in image_paths]

        self.transform = transform
        self.is_thermal = is_thermal

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]

        if self.is_thermal:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, str(img_path)


class CombinedDataset(Dataset):
    """Dataset for combined RGB and thermal inference.

    Args:
        rgb_dir: Directory containing RGB images.
        thermal_dir: Directory containing thermal images.
        rgb_transform: Transform for RGB images.
        thermal_transform: Transform for thermal images.
    """

    def __init__(
        self,
        rgb_dir: Union[str, Path],
        thermal_dir: Union[str, Path],
        rgb_transform: Optional[Callable] = None,
        thermal_transform: Optional[Callable] = None,
    ) -> None:
        self.rgb_dir = Path(rgb_dir)
        self.thermal_dir = Path(thermal_dir)
        self.rgb_transform = rgb_transform
        self.thermal_transform = thermal_transform

        # Get matched pairs
        rgb_files = {f.stem: f for f in self.rgb_dir.iterdir() if f.is_file()}
        thermal_files = {f.stem: f for f in self.thermal_dir.iterdir() if f.is_file()}

        self.pairs = [
            (rgb_files[stem], thermal_files[stem])
            for stem in rgb_files
            if stem in thermal_files
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        rgb_path, thermal_path = self.pairs[idx]

        # Load RGB
        rgb_image = cv2.imread(str(rgb_path))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Load thermal
        thermal_image = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.rgb_transform:
            transformed = self.rgb_transform(image=rgb_image)
            rgb_image = transformed["image"]

        if self.thermal_transform:
            transformed = self.thermal_transform(image=thermal_image)
            thermal_image = transformed["image"]

        return rgb_image, thermal_image, rgb_path.stem
