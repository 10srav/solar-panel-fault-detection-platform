"""
=============================================================================
 SEGMENTATION MODULE — segmentation.py
 U-Net fault segmentation for pixel-level fault area estimation.
 Currently returns placeholder — ready for U-Net model integration.
=============================================================================
"""

import os
import sys
import io
import base64
import torch
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMG_SIZE, DEVICE


def generate_segmentation_mask(image_input, model=None):
    """
    Generate pixel-level segmentation mask.

    Args:
        image_input: PIL Image or file path
        model: U-Net segmentation model (None = use Grad-CAM fallback)

    Returns:
        dict:
            mask: np.array [H, W] binary 0-1
            overlay: np.array [H, W, 3] colorized mask overlay
            fault_area_percent: float
            mask_base64: str (base64 encoded PNG)
    """

    # Load image
    if isinstance(image_input, str):
        original = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        original = image_input.convert("RGB")
    else:
        original = Image.open(image_input).convert("RGB")

    original_resized = original.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original_resized).astype(np.float32) / 255.0

    # TODO: Replace with U-Net inference when model is trained
    # For now, return a placeholder mask (will be replaced)
    if model is None:
        # Placeholder: empty mask (all zeros)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        fault_area_percent = 0.0
    else:
        # U-Net inference logic goes here
        # model_output = model(preprocessed_image)
        # mask = (model_output > 0.5).float()
        pass

    # Create colored overlay
    mask_colored = cv2.applyColorMap(
        (mask * 255).astype(np.uint8), cv2.COLORMAP_HOT
    )
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Blend with original
    overlay = 0.7 * original_np + 0.3 * mask_colored
    overlay = np.clip(overlay, 0, 1)

    # Encode mask overlay to base64
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    img = Image.fromarray(overlay_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    mask_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Compute fault area
    if fault_area_percent == 0.0:
        fault_area_percent = float((mask > 0.5).sum() / (IMG_SIZE * IMG_SIZE) * 100.0)

    return {
        "mask": mask,
        "overlay": overlay,
        "fault_area_percent": fault_area_percent,
        "mask_base64": mask_base64,
        "segmentation_available": model is not None,
    }


def estimate_fault_area_from_mask(mask, threshold=0.5):
    """
    Calculate fault area percentage from segmentation mask.

    Args:
        mask: np.array [H, W] with values 0-1
        threshold: binarization threshold

    Returns:
        fault_area_percent: float 0-100
    """
    binary_mask = (mask > threshold).astype(np.float32)
    total_pixels = mask.shape[0] * mask.shape[1]
    fault_pixels = binary_mask.sum()

    return float((fault_pixels / total_pixels) * 100.0)
