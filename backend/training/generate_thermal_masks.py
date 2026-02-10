"""
=============================================================================
 THERMAL MASK GENERATION — generate_thermal_masks.py
 Converts YOLO bounding boxes to binary segmentation masks for U-Net training.
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
THERMAL_DATASET = PROJECT_ROOT / "dataset" / "Thermal Imaging.v6i.yolov8"

# Output directory for masks
MASKS_DIR = THERMAL_DATASET / "masks"
MASKS_DIR.mkdir(exist_ok=True)

# ============================================================================
# YOLO TO MASK CONVERSION
# ============================================================================

def yolo_box_to_mask(image_path, label_path, output_path):
    """
    Convert YOLO bounding box annotations to binary segmentation mask.

    Args:
        image_path: Path to thermal image
        label_path: Path to YOLO .txt label file
        output_path: Path to save binary mask

    Returns:
        mask: numpy array [H, W] with 0 (background) and 255 (fault)
    """
    # Read image to get dimensions
    img = Image.open(image_path)
    width, height = img.size

    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Read YOLO labels
    if not label_path.exists():
        # No annotations → all background
        Image.fromarray(mask).save(output_path)
        return mask

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Parse each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        class_id, x_center, y_center, box_width, box_height = map(float, parts[:5])

        # Convert normalized coords to pixels
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        box_width_px = int(box_width * width)
        box_height_px = int(box_height * height)

        # Calculate top-left and bottom-right corners
        x1 = int(x_center_px - box_width_px / 2)
        y1 = int(y_center_px - box_height_px / 2)
        x2 = int(x_center_px + box_width_px / 2)
        y2 = int(y_center_px + box_height_px / 2)

        # Clamp to image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Fill bounding box region with 255 (fault)
        mask[y1:y2, x1:x2] = 255

    # Save mask as grayscale image
    Image.fromarray(mask).save(output_path)

    return mask

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def generate_masks_for_split(split='train'):
    """
    Generate masks for all images in a dataset split.

    Args:
        split: 'train', 'valid', or 'test'
    """
    print(f"\n[{split.upper()}] Generating masks...")

    img_dir = THERMAL_DATASET / split / 'images'
    lbl_dir = THERMAL_DATASET / split / 'labels'
    mask_dir = MASKS_DIR / split
    mask_dir.mkdir(exist_ok=True)

    # Get all images
    images = list(img_dir.glob('*.jpg'))

    if not images:
        print(f"   No images found in {img_dir}")
        return

    # Process each image
    fault_count = 0
    total_boxes = 0

    for img_path in tqdm(images, desc=f"Processing {split}"):
        # Corresponding label file
        lbl_path = lbl_dir / (img_path.stem + '.txt')

        # Output mask path
        mask_path = mask_dir / (img_path.stem + '.png')

        # Generate mask
        mask = yolo_box_to_mask(img_path, lbl_path, mask_path)

        # Count faults
        if mask.max() > 0:
            fault_count += 1

        # Count boxes
        if lbl_path.exists():
            with open(lbl_path) as f:
                total_boxes += len(f.readlines())

    print(f"   Generated {len(images)} masks")
    print(f"   Images with faults: {fault_count} ({fault_count/len(images):.1%})")
    print(f"   Total bounding boxes: {total_boxes}")
    print(f"   Saved to: {mask_dir}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" THERMAL MASK GENERATION - YOLO Boxes to Segmentation Masks")
    print("="*80)

    # Generate for all splits
    for split in ['train', 'valid', 'test']:
        generate_masks_for_split(split)

    print(f"\n{'='*80}")
    print(" MASK GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nMasks saved to: {MASKS_DIR}")
    print("\nNext step: python train_thermal_unet.py")
