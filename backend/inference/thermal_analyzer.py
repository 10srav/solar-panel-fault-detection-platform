"""
=============================================================================
 THERMAL ANALYZER — thermal_analyzer.py
 U-Net segmentation pipeline for thermal fault detection.
 ARCHITECTURE: Pixel-level segmentation (NOT object detection)
=============================================================================
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEVICE

# Lazy import for U-Net
_unet_model = None

MODEL_DIR = Path(__file__).parent.parent / "models"
UNET_PATH = MODEL_DIR / "thermal_segmentation_unet_v2.pth"

# ============================================================================
# MODEL LOADER
# ============================================================================

def load_thermal_unet():
    """Load U-Net thermal segmentation model (singleton)."""
    global _unet_model

    if _unet_model is not None:
        return _unet_model

    if not UNET_PATH.exists():
        raise FileNotFoundError(
            f"Thermal U-Net model not found: {UNET_PATH}\n"
            f"Train first: python -m training.train_thermal_unet"
        )

    # Import U-Net architecture
    from training.train_thermal_unet import UNet

    _unet_model = UNet().to(DEVICE)
    checkpoint = torch.load(UNET_PATH, map_location=DEVICE, weights_only=True)
    _unet_model.load_state_dict(checkpoint['model_state_dict'])
    _unet_model.eval()

    print(f"[THERMAL] U-Net loaded: {UNET_PATH}")

    return _unet_model

# ============================================================================
# U-NET SEGMENTATION (MAIN INFERENCE)
# ============================================================================

def segment_thermal_image(image_input, img_size=256):
    """
    Run U-Net segmentation on thermal image.

    Args:
        image_input: str (path) or PIL Image or file-like object

    Returns:
        dict:
            mask:               np.array [H, W] float 0-1 (probability map)
            binary_mask:        np.array [H, W] uint8 0/255 (thresholded)
            overlay:            np.array [H, W, 3] RGB visualization
            fault_area_percent: float (% of image classified as fault)
            mask_base64:        str (base64 PNG for API)
    """
    model = load_thermal_unet()

    # Load image
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        img = Image.open(image_input).convert('RGB')

    original_size = img.size
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)

    # Preprocess: normalize to [0, 1]
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # U-Net inference (model outputs logits, need sigmoid)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.sigmoid(logits)  # Convert logits to probabilities
        pred_mask = pred_mask.squeeze().cpu().numpy()  # [H, W] float 0-1

    # Binary mask (threshold at 0.5)
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Calculate fault area percentage
    fault_pixels = (pred_mask > 0.5).sum()
    total_pixels = img_size * img_size
    fault_area_percent = float((fault_pixels / total_pixels) * 100.0)

    # Relative heat check: compare mask region intensity vs whole image
    gray = np.mean(img_np, axis=2)  # img_np is already [0, 1]
    image_mean_intensity = float(gray.mean() * 255.0)
    mask_bool = pred_mask > 0.5
    if mask_bool.sum() > 0:
        mask_mean_intensity = float(gray[mask_bool].mean() * 255.0)
    else:
        mask_mean_intensity = image_mean_intensity
    heat_difference = mask_mean_intensity - image_mean_intensity

    # Create overlay visualization
    img_np_viz = np.array(img_resized).astype(np.float32) / 255.0

    # Colorize mask: Red for faults
    mask_colored = np.zeros_like(img_np_viz)
    mask_colored[:, :, 0] = pred_mask  # Red channel
    mask_colored[:, :, 1] = pred_mask * 0.3  # Slight green for orange effect

    # Blend: 70% original + 30% red mask
    overlay = 0.7 * img_np_viz + 0.3 * mask_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    # Encode overlay to base64
    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format='PNG')
    buffer.seek(0)
    mask_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {
        "mask": pred_mask,
        "binary_mask": binary_mask,
        "overlay": overlay,
        "fault_area_percent": round(fault_area_percent, 2),
        "mask_base64": mask_base64,
        "image_mean_intensity": round(image_mean_intensity, 1),
        "mask_mean_intensity": round(mask_mean_intensity, 1),
        "heat_difference": round(heat_difference, 1),
    }

# ============================================================================
# COMPLETE THERMAL ANALYSIS
# ============================================================================

def analyze_thermal_image(image_input, return_base64=False):
    """
    Complete thermal image analysis pipeline.

    Pipeline:
        Thermal Image → U-Net → Binary Mask → Fault Area % → Risk Level

    Returns:
        dict:
            input_type: "thermal"
            fault_area_percent: float
            severity_score: float
            risk_level: str
            maintenance_suggestion: str
            alert: dict
            mask_base64: str (if return_base64=True)
            overlay: np.array
            segmentation_available: True
    """
    from risk_engine.severity_analysis import thermal_risk_analysis

    # Step 1: U-Net segmentation
    seg_result = segment_thermal_image(image_input)

    # Step 2: Risk analysis (area + heat difference)
    risk_result = thermal_risk_analysis(
        fault_area_percent=seg_result["fault_area_percent"],
        heat_difference=seg_result["heat_difference"],
    )

    # Build response
    result = {
        "input_type": "thermal",
        "fault_area_percent": seg_result["fault_area_percent"],
        "severity_score": risk_result["severity_score"],
        "risk_level": risk_result["risk_level"],
        "maintenance_suggestion": risk_result["maintenance_suggestion"],
        "alert": risk_result["alert"],
        "mask": seg_result["mask"],
        "overlay": seg_result["overlay"],
        "binary_mask": seg_result["binary_mask"],
        "segmentation_available": True,
        "fault_area_source": "unet_segmentation",
        "heat_difference": seg_result["heat_difference"],
        "image_mean_intensity": seg_result["image_mean_intensity"],
        "mask_mean_intensity": seg_result["mask_mean_intensity"],
    }

    if return_base64:
        result["segmentation_mask_base64"] = seg_result["mask_base64"]

    return result

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_segmentation_grid(image_input, ground_truth_mask=None, save_path=None):
    """
    Create 4-panel visualization: Image | Ground Truth | Predicted Mask | Overlay

    Args:
        image_input: original image
        ground_truth_mask: optional ground truth mask (if available)
        save_path: path to save visualization

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    result = segment_thermal_image(image_input)

    # Load original
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        img = image_input

    img_resized = img.resize((256, 256))

    # Create figure
    num_panels = 4 if ground_truth_mask is not None else 3
    fig, axes = plt.subplots(1, num_panels, figsize=(num_panels * 5, 5))

    # Panel 1: Original image
    axes[0].imshow(img_resized)
    axes[0].set_title("Thermal Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground truth (if available)
    if ground_truth_mask is not None:
        axes[1].imshow(ground_truth_mask, cmap='hot')
        axes[1].set_title("Ground Truth Mask", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        pred_idx = 2
        overlay_idx = 3
    else:
        pred_idx = 1
        overlay_idx = 2

    # Panel 3: Predicted mask
    axes[pred_idx].imshow(result["mask"], cmap='hot', vmin=0, vmax=1)
    axes[pred_idx].set_title(
        f"Predicted Mask ({result['fault_area_percent']:.1f}%)",
        fontsize=12, fontweight='bold'
    )
    axes[pred_idx].axis('off')

    # Panel 4: Overlay
    axes[overlay_idx].imshow(result["overlay"])
    axes[overlay_idx].set_title("Overlay Visualization", fontsize=12, fontweight='bold')
    axes[overlay_idx].axis('off')

    plt.suptitle("Thermal Fault Segmentation", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] Visualization saved: {save_path}")

    return fig

# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path

    # Test image
    dataset_path = Path(__file__).parent.parent.parent / "dataset" / "Thermal Imaging.v6i.yolov8"
    test_img_dir = dataset_path / 'test' / 'images'
    test_images = list(test_img_dir.glob('*.jpg'))

    if not test_images:
        print("No test images found")
        sys.exit(1)

    test_img = test_images[0]
    print(f"Testing on: {test_img.name}")

    result = analyze_thermal_image(test_img, return_base64=True)

    print(f"\n[THERMAL ANALYSIS]")
    print(f"   Fault area: {result['fault_area_percent']:.1f}%")
    print(f"   Severity:   {result['severity_score']:.1f}")
    print(f"   Risk:       {result['risk_level']}")
    print(f"   Suggestion: {result['maintenance_suggestion'][:60]}...")
    print(f"\n   Mask base64 length: {len(result.get('segmentation_mask_base64', '')) // 1024}KB")

    # Create visualization
    save_path = Path(__file__).parent.parent / "plots" / "thermal_segmentation_test.png"
    create_segmentation_grid(test_img, save_path=save_path)
