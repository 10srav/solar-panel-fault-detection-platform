"""
=============================================================================
 DATA PIPELINE — preprocessing.py
 Loads dataset, validates files, applies transforms, splits, computes weights.
 Zero data leakage guaranteed: transforms applied AFTER splitting.
=============================================================================
"""

import os
import sys
import json
import copy
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, SPLIT_RATIO, RANDOM_SEED,
    VALID_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD, DEVICE, MODEL_DIR,
)


# ============================================================================
# IMAGE VALIDATION
# ============================================================================

def is_valid_image(path: str) -> bool:
    """Reject non-image files (desktop.ini, thumbs.db, etc.)."""
    return path.lower().endswith(VALID_EXTENSIONS)


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transform():
    """Training: augmentation + normalization."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    """Validation/Inference: resize + normalization only."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================================
# SAFE IMAGE FOLDER (handles corrupted images)
# ============================================================================

class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips non-image files and handles corrupted images."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, is_valid_file=is_valid_image)

    def __getitem__(self, index):
        """Return image,label. If corrupted, return a black placeholder."""
        try:
            return super().__getitem__(index)
        except Exception as e:
            path, label = self.samples[index]
            print(f"[WARNING] Corrupted image skipped: {path} — {e}")
            # Return a black image as placeholder
            placeholder = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            return placeholder, label


# ============================================================================
# DATA SPLITTING — NO LEAKAGE
# ============================================================================

def create_stratified_split(dataset):
    """
    Stratified 80/20 split ensuring proportional class representation.
    Uses indices only — no data is transformed yet.
    """
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    train_indices = []
    val_indices = []

    rng = np.random.RandomState(RANDOM_SEED)

    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)

        split_point = int(len(cls_indices) * SPLIT_RATIO)
        train_indices.extend(cls_indices[:split_point].tolist())
        val_indices.extend(cls_indices[split_point:].tolist())

    # Shuffle within splits
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


# ============================================================================
# CLASS WEIGHT COMPUTATION
# ============================================================================

def compute_class_weights(dataset, train_indices):
    """
    Compute inverse-frequency weights from TRAINING set only.
    This prevents val distribution from influencing training.
    """
    train_targets = [dataset.targets[i] for i in train_indices]
    class_counts = Counter(train_targets)
    total = sum(class_counts.values())
    num_classes = len(class_counts)

    weights = []
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 1)
        w = total / (num_classes * count)
        weights.append(w)

    return torch.FloatTensor(weights).to(DEVICE)


# ============================================================================
# SAVE CLASS MAPPING
# ============================================================================

def save_class_mapping(dataset):
    """Save class index → name mapping as JSON for inference."""
    mapping = {str(i): name for i, name in enumerate(dataset.classes)}
    path = os.path.join(MODEL_DIR, "class_mapping.json")
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[DATA] Class mapping saved: {path}")
    return mapping


# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def build_dataloaders(dataset_path=None):
    """
    Complete data pipeline:
      1. Load dataset with validation
      2. Stratified split (no leakage)
      3. Apply transforms per-split
      4. Build DataLoaders
      5. Compute class weights from training set

    Returns:
        train_loader, val_loader, class_weights, class_names
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"[DATA] Loading dataset from: {dataset_path}")

    # Step 1: Load full dataset (no transform — just for indexing)
    full_dataset = SafeImageFolder(dataset_path)
    class_names = full_dataset.classes
    total = len(full_dataset)

    print(f"[DATA] Found {total} valid images across {len(class_names)} classes")
    for i, name in enumerate(class_names):
        count = sum(1 for t in full_dataset.targets if t == i)
        print(f"  [{i}] {name}: {count} images")

    # Step 2: Stratified split
    train_indices, val_indices = create_stratified_split(full_dataset)
    print(f"\n[SPLIT] Train: {len(train_indices)} | Val: {len(val_indices)}")

    # Verify no leakage
    train_set = set(train_indices)
    val_set = set(val_indices)
    assert len(train_set & val_set) == 0, "DATA LEAKAGE DETECTED!"
    print("[SPLIT] No data leakage confirmed - OK")

    # Step 3: Create subsets with SEPARATE transforms
    train_dataset = SafeImageFolder(dataset_path, transform=get_train_transform())
    val_dataset = SafeImageFolder(dataset_path, transform=get_val_transform())

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Step 4: DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Step 5: Class weights (from training indices only)
    class_weights = compute_class_weights(full_dataset, train_indices)
    print("\n[WEIGHTS] Class weights (inverse frequency):")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i].item():.4f}")

    # Step 6: Save class mapping
    save_class_mapping(full_dataset)

    return train_loader, val_loader, class_weights, class_names


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    train_loader, val_loader, weights, names = build_dataloaders()

    # Verify a batch
    images, labels = next(iter(train_loader))
    print(f"\n[TEST] Batch shape: {images.shape}")
    print(f"[TEST] Labels: {labels[:8].tolist()}")
    print(f"[TEST] Label names: {[names[l] for l in labels[:8].tolist()]}")
    print("[DATA PIPELINE] All checks passed.")
