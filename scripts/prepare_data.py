#!/usr/bin/env python
"""Data preparation script for solar panel fault detection training.

This script:
1. Splits RGB solar panel images into train/val/test sets (60/20/20)
2. Processes thermal infrared images and generates binary segmentation masks
3. Organizes data into the expected directory structure
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Target image sizes
RGB_SIZE = (227, 227)
THERMAL_SIZE = (256, 256)


def prepare_rgb_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> dict:
    """Prepare RGB dataset by splitting into train/val/test sets.

    Args:
        source_dir: Source directory containing class folders
        output_dir: Output directory for organized data
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility

    Returns:
        Dictionary with dataset statistics
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create output directories
    splits = ["train", "val", "test"]
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": {}, "val": {}, "test": {}}
    total_images = 0

    # Get all class directories
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        class_name = class_dir.name

        # Get all images in class
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                 list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPG"))

        if not images:
            logger.warning(f"No images found in {class_dir}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Create class directories in each split
        for split in splits:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        # Copy and resize images
        split_data = [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]

        for split_name, split_images in split_data:
            for img_path in tqdm(split_images, desc=f"{class_name}/{split_name}"):
                # Read and resize image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                img_resized = cv2.resize(img, RGB_SIZE, interpolation=cv2.INTER_AREA)

                # Save to output directory
                output_path = output_dir / split_name / class_name / img_path.name
                cv2.imwrite(str(output_path), img_resized)

            stats[split_name][class_name] = len(split_images)
            total_images += len(split_images)

    logger.info(f"RGB Dataset prepared: {total_images} total images")
    for split in splits:
        split_total = sum(stats[split].values())
        logger.info(f"  {split}: {split_total} images")
        for class_name, count in stats[split].items():
            logger.info(f"    {class_name}: {count}")

    return stats


def prepare_thermal_dataset(
    source_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_images: Optional[int] = None,
) -> dict:
    """Prepare thermal dataset with binary segmentation masks.

    For thermal segmentation, we create binary masks where:
    - Anomaly pixels = 1 (white)
    - Normal pixels = 0 (black)

    Since the infrared dataset has image-level labels (not pixel-level),
    we generate approximate masks by:
    - For "No-Anomaly": mask is all zeros
    - For anomaly classes: use thresholding to highlight hotspots

    Args:
        source_dir: Source directory containing thermal images
        metadata_path: Path to metadata JSON file
        output_dir: Output directory for organized data
        train_ratio: Ratio of training data (rest goes to val)
        seed: Random seed for reproducibility
        max_images: Maximum number of images to process (for faster testing)

    Returns:
        Dictionary with dataset statistics
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata for {len(metadata)} images")

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    # Separate anomaly and no-anomaly images
    anomaly_images = []
    normal_images = []

    for img_id, info in metadata.items():
        if info["anomaly_class"] == "No-Anomaly":
            normal_images.append((img_id, info))
        else:
            anomaly_images.append((img_id, info))

    logger.info(f"Anomaly images: {len(anomaly_images)}")
    logger.info(f"Normal images: {len(normal_images)}")

    # Shuffle and limit if needed
    random.shuffle(anomaly_images)
    random.shuffle(normal_images)

    if max_images:
        # Take equal samples from each class
        n_each = max_images // 2
        anomaly_images = anomaly_images[:n_each]
        normal_images = normal_images[:n_each]

    # Combine and shuffle
    all_images = anomaly_images + normal_images
    random.shuffle(all_images)

    # Split into train/val
    n_train = int(len(all_images) * train_ratio)
    train_data = all_images[:n_train]
    val_data = all_images[n_train:]

    stats = {"train": {"anomaly": 0, "normal": 0}, "val": {"anomaly": 0, "normal": 0}}

    def process_images(data_list, split_name):
        """Process images for a given split."""
        for img_id, info in tqdm(data_list, desc=f"Processing {split_name}"):
            img_path = source_dir / info["image_filepath"]
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            # Read image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue

            # Resize image
            img_resized = cv2.resize(img, THERMAL_SIZE, interpolation=cv2.INTER_AREA)

            # Generate mask
            if info["anomaly_class"] == "No-Anomaly":
                # Empty mask for normal images
                mask = np.zeros(THERMAL_SIZE, dtype=np.uint8)
                stats[split_name]["normal"] += 1
            else:
                # For anomaly images, use adaptive thresholding to find hotspots
                # Thermal anomalies typically appear as brighter regions
                blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

                # Use Otsu's thresholding to find anomaly regions
                _, mask = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # Clean up mask with morphological operations
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                stats[split_name]["anomaly"] += 1

            # Save image and mask
            output_img_path = output_dir / split_name / "images" / f"{img_id}.png"
            output_mask_path = output_dir / split_name / "masks" / f"{img_id}.png"

            cv2.imwrite(str(output_img_path), img_resized)
            cv2.imwrite(str(output_mask_path), mask)

    process_images(train_data, "train")
    process_images(val_data, "val")

    logger.info(f"Thermal Dataset prepared:")
    logger.info(f"  train: {stats['train']['anomaly']} anomaly, {stats['train']['normal']} normal")
    logger.info(f"  val: {stats['val']['anomaly']} anomaly, {stats['val']['normal']} normal")

    return stats


def create_sample_test_images(output_dir: Path):
    """Create sample test images for API testing."""
    # Create a sample RGB test image
    rgb_test = np.random.randint(50, 200, (RGB_SIZE[0], RGB_SIZE[1], 3), dtype=np.uint8)
    # Add some grid pattern to simulate solar panel
    for i in range(0, RGB_SIZE[0], 25):
        rgb_test[i:i+2, :] = [30, 30, 50]
        rgb_test[:, i:i+2] = [30, 30, 50]
    cv2.imwrite(str(output_dir / "test_rgb_real.png"), rgb_test)

    # Create a sample thermal test image
    thermal_test = np.random.randint(100, 180, THERMAL_SIZE, dtype=np.uint8)
    # Add a hotspot
    cv2.circle(thermal_test, (128, 128), 30, 255, -1)
    cv2.imwrite(str(output_dir / "test_thermal_real.png"), thermal_test)

    logger.info(f"Created sample test images in {output_dir}")


def main():
    """Main function to prepare all datasets."""
    parser = argparse.ArgumentParser(
        description="Prepare solar panel fault detection datasets"
    )
    parser.add_argument(
        "--rgb-source",
        type=str,
        default="data/raw/rgb/Faulty_solar_panel",
        help="Source directory for RGB dataset",
    )
    parser.add_argument(
        "--thermal-source",
        type=str,
        default="data/raw/thermal/InfraredSolarModules/images",
        help="Source directory for thermal images",
    )
    parser.add_argument(
        "--thermal-metadata",
        type=str,
        default="data/raw/thermal/InfraredSolarModules/module_metadata.json",
        help="Path to thermal metadata JSON",
    )
    parser.add_argument(
        "--rgb-output",
        type=str,
        default="data/rgb",
        help="Output directory for RGB dataset",
    )
    parser.add_argument(
        "--thermal-output",
        type=str,
        default="data/thermal",
        help="Output directory for thermal dataset",
    )
    parser.add_argument(
        "--max-thermal",
        type=int,
        default=4000,
        help="Maximum thermal images to process (for faster training)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-rgb",
        action="store_true",
        help="Skip RGB dataset preparation",
    )
    parser.add_argument(
        "--skip-thermal",
        action="store_true",
        help="Skip thermal dataset preparation",
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent

    # Prepare RGB dataset
    if not args.skip_rgb:
        rgb_source = project_root / args.rgb_source
        rgb_output = project_root / args.rgb_output

        if rgb_source.exists():
            logger.info("Preparing RGB dataset...")
            prepare_rgb_dataset(
                source_dir=rgb_source,
                output_dir=rgb_output,
                seed=args.seed,
            )
        else:
            logger.error(f"RGB source directory not found: {rgb_source}")

    # Prepare thermal dataset
    if not args.skip_thermal:
        thermal_source = project_root / args.thermal_source
        thermal_metadata = project_root / args.thermal_metadata
        thermal_output = project_root / args.thermal_output

        if thermal_source.exists() and thermal_metadata.exists():
            logger.info("Preparing thermal dataset...")
            prepare_thermal_dataset(
                source_dir=thermal_source.parent,
                metadata_path=thermal_metadata,
                output_dir=thermal_output,
                seed=args.seed,
                max_images=args.max_thermal,
            )
        else:
            logger.error(f"Thermal source not found: {thermal_source}")

    # Create sample test images
    create_sample_test_images(project_root)

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
