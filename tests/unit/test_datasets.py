"""Unit tests for dataset classes."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.datasets import (
    SolarPanelRGBDataset,
    ThermalSegmentationDataset,
    InferenceDataset,
)
from src.data.transforms import get_rgb_transforms, get_thermal_transforms


class TestSolarPanelRGBDataset:
    """Tests for RGB solar panel dataset."""

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path):
        """Create a temporary directory with sample RGB images."""
        classes = ["Clean", "Dusty", "Bird-drop"]

        for class_name in classes:
            class_dir = tmp_path / class_name
            class_dir.mkdir()

            # Create 5 sample images per class
            for i in range(5):
                img = np.random.randint(0, 255, (227, 227, 3), dtype=np.uint8)
                cv2.imwrite(str(class_dir / f"image_{i}.jpg"), img)

        return tmp_path

    def test_dataset_length(self, sample_dataset_dir):
        """Test dataset returns correct length."""
        dataset = SolarPanelRGBDataset(sample_dataset_dir)

        # 3 classes * 5 images = 15 total
        assert len(dataset) == 15

    def test_dataset_getitem(self, sample_dataset_dir):
        """Test dataset returns image and label."""
        transforms = get_rgb_transforms(augment=False)["val"]
        dataset = SolarPanelRGBDataset(sample_dataset_dir, transform=transforms)

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 227, 227)
        assert isinstance(label, int)
        assert 0 <= label < 3

    def test_dataset_class_names(self, sample_dataset_dir):
        """Test dataset correctly identifies class names."""
        dataset = SolarPanelRGBDataset(sample_dataset_dir)

        assert len(dataset.classes) == 3
        assert "Clean" in dataset.classes
        assert "Dusty" in dataset.classes
        assert "Bird-drop" in dataset.classes

    def test_dataset_class_to_idx(self, sample_dataset_dir):
        """Test dataset class to index mapping."""
        dataset = SolarPanelRGBDataset(sample_dataset_dir)

        assert len(dataset.class_to_idx) == 3
        for class_name in dataset.classes:
            assert class_name in dataset.class_to_idx
            assert isinstance(dataset.class_to_idx[class_name], int)

    def test_dataset_get_class_counts(self, sample_dataset_dir):
        """Test dataset class count method."""
        dataset = SolarPanelRGBDataset(sample_dataset_dir)
        counts = dataset.get_class_counts()

        assert len(counts) == 3
        for class_name, count in counts.items():
            assert count == 5

    def test_dataset_get_sample_weights(self, sample_dataset_dir):
        """Test dataset sample weights for balanced sampling."""
        dataset = SolarPanelRGBDataset(sample_dataset_dir)
        weights = dataset.get_sample_weights()

        assert isinstance(weights, torch.Tensor)
        assert len(weights) == len(dataset)
        assert (weights > 0).all()

    def test_dataset_with_custom_classes(self, sample_dataset_dir):
        """Test dataset with custom class list."""
        custom_classes = ["Clean", "Dusty"]
        dataset = SolarPanelRGBDataset(
            sample_dataset_dir, classes=custom_classes
        )

        # Should only include specified classes
        assert dataset.classes == custom_classes
        assert len(dataset) == 10  # 2 classes * 5 images


class TestThermalSegmentationDataset:
    """Tests for thermal segmentation dataset."""

    @pytest.fixture
    def sample_thermal_dir(self, tmp_path):
        """Create a temporary directory with sample thermal images and masks."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        # Create 5 sample thermal images and masks
        for i in range(5):
            # Thermal image (grayscale)
            img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"thermal_{i}.png"), img)

            # Binary mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[100:150, 100:150] = 255  # Add a fault region
            cv2.imwrite(str(masks_dir / f"thermal_{i}.png"), mask)

        return images_dir, masks_dir

    def test_dataset_length(self, sample_thermal_dir):
        """Test thermal dataset returns correct length."""
        images_dir, masks_dir = sample_thermal_dir
        dataset = ThermalSegmentationDataset(images_dir, masks_dir)

        assert len(dataset) == 5

    def test_dataset_getitem(self, sample_thermal_dir):
        """Test thermal dataset returns image and mask."""
        images_dir, masks_dir = sample_thermal_dir
        transforms = get_thermal_transforms(augment=False)["val"]
        dataset = ThermalSegmentationDataset(
            images_dir, masks_dir, transform=transforms
        )

        image, mask = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (1, 256, 256)
        assert mask.shape == (1, 256, 256)

    def test_dataset_mask_binary(self, sample_thermal_dir):
        """Test that masks are binary."""
        images_dir, masks_dir = sample_thermal_dir
        transforms = get_thermal_transforms(augment=False)["val"]
        dataset = ThermalSegmentationDataset(
            images_dir, masks_dir, transform=transforms
        )

        _, mask = dataset[0]

        # Mask values should be 0 or 1
        unique_values = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())

    def test_dataset_without_transform(self, sample_thermal_dir):
        """Test thermal dataset without transforms."""
        images_dir, masks_dir = sample_thermal_dir
        dataset = ThermalSegmentationDataset(images_dir, masks_dir)

        image, mask = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)


class TestInferenceDataset:
    """Tests for inference dataset."""

    @pytest.fixture
    def sample_inference_dir(self, tmp_path):
        """Create a temporary directory with sample images for inference."""
        for i in range(3):
            img = np.random.randint(0, 255, (227, 227, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"test_{i}.jpg"), img)
        return tmp_path

    def test_inference_dataset_from_directory(self, sample_inference_dir):
        """Test inference dataset from directory."""
        dataset = InferenceDataset(sample_inference_dir)

        assert len(dataset) == 3

    def test_inference_dataset_from_file_list(self, sample_inference_dir):
        """Test inference dataset from file list."""
        files = list(sample_inference_dir.glob("*.jpg"))
        dataset = InferenceDataset(files)

        assert len(dataset) == 3

    def test_inference_dataset_getitem(self, sample_inference_dir):
        """Test inference dataset returns image and path."""
        transforms = get_rgb_transforms(augment=False)["val"]
        dataset = InferenceDataset(
            sample_inference_dir, transform=transforms, is_thermal=False
        )

        image, path = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 227, 227)
        assert isinstance(path, str)

    def test_inference_dataset_thermal_mode(self, tmp_path):
        """Test inference dataset in thermal mode."""
        # Create grayscale thermal images
        for i in range(2):
            img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"thermal_{i}.png"), img)

        transforms = get_thermal_transforms(augment=False)["val"]
        dataset = InferenceDataset(
            tmp_path, transform=transforms, is_thermal=True
        )

        image, path = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 256, 256)


class TestTransforms:
    """Tests for data transforms."""

    def test_rgb_transforms_train(self):
        """Test RGB training transforms."""
        transforms = get_rgb_transforms(augment=True)

        assert "train" in transforms
        assert "val" in transforms

    def test_rgb_transforms_output_shape(self):
        """Test RGB transforms produce correct output shape."""
        transforms = get_rgb_transforms(augment=False)["val"]
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

        result = transforms(image=image)
        tensor = result["image"]

        assert tensor.shape == (3, 227, 227)

    def test_thermal_transforms_train(self):
        """Test thermal training transforms."""
        transforms = get_thermal_transforms(augment=True)

        assert "train" in transforms
        assert "val" in transforms

    def test_thermal_transforms_output_shape(self):
        """Test thermal transforms produce correct output shape."""
        transforms = get_thermal_transforms(augment=False)["val"]
        image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)

        result = transforms(image=image)
        tensor = result["image"]

        assert tensor.shape == (1, 256, 256)

    def test_thermal_transforms_with_mask(self):
        """Test thermal transforms with mask."""
        transforms = get_thermal_transforms(augment=False)["val"]
        image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        mask = np.zeros((300, 300), dtype=np.float32)
        mask[100:200, 100:200] = 1.0

        result = transforms(image=image, mask=mask)

        assert "image" in result
        assert "mask" in result
