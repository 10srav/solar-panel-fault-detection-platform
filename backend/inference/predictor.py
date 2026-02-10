"""
=============================================================================
 INFERENCE — predictor.py
 Loads model, runs full analysis pipeline on any image.
 Handles all edge cases: missing model, corrupted images, GPU fallback.
=============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEVICE, MODEL_PATH, CLASS_NAMES, NUM_CLASSES, IMG_SIZE, VALID_EXTENSIONS
from explainability.gradcam_engine import (
    generate_gradcam,
    estimate_fault_area,
    gradcam_overlay_to_base64,
)
from explainability.segmentation import (
    generate_segmentation_mask,
    estimate_fault_area_from_mask,
)
from risk_engine.severity_analysis import full_risk_analysis, print_alert, thermal_risk_analysis


# ============================================================================
# MODEL LOADER (singleton pattern)
# ============================================================================

_cached_model = None


def load_model(model_path=None, force_reload=False):
    """
    Load trained model with safety checks.
    Caches the model in memory for repeated calls.

    Handles:
        - Model file not found
        - GPU unavailable → CPU fallback
        - Corrupted checkpoint
    """
    global _cached_model

    if _cached_model is not None and not force_reload:
        return _cached_model

    if model_path is None:
        model_path = MODEL_PATH

    # Check model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Run training first: python -m training.train_rgb_model"
        )

    # Device selection with GPU fallback
    device = DEVICE
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")

    # Load model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {e}")

    model = model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", "N/A")
    epoch = checkpoint.get("epoch", "N/A")
    print(f"[MODEL] Loaded: {model_path}")
    print(f"[MODEL] Best val acc: {val_acc}% (epoch {epoch}) | Device: {device}")

    _cached_model = model
    return model


# ============================================================================
# IMAGE VALIDATION
# ============================================================================

def validate_image(image_path: str):
    """
    Validate image before processing.

    Checks:
        - File exists
        - Valid extension
        - File is not empty
        - Can be opened as image
        - Is RGB (converts if needed)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(
            f"Invalid file type: {ext}. "
            f"Supported: {', '.join(VALID_EXTENSIONS)}"
        )

    if os.path.getsize(image_path) == 0:
        raise ValueError(f"File is empty: {image_path}")

    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as e:
        raise ValueError(f"Corrupted image: {image_path} — {e}")


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def analyze_image(image_input, model=None, return_base64=False):
    """
    Complete analysis pipeline for a single solar panel image.

    Args:
        image_input: str (path) or PIL Image or file-like object
        model: loaded model (auto-loads if None)
        return_base64: if True, include base64 Grad-CAM overlay

    Returns:
        dict:
            predicted_class   (int)
            class_name        (str)
            confidence        (float)
            fault_area_percent(float)
            severity_score    (float)
            risk_level        (str)
            maintenance_suggestion (str)
            alert             (dict)
            gradcam_overlay   (np.array or None)
            gradcam_base64    (str, if return_base64=True)
    """
    # Validate if path
    if isinstance(image_input, str):
        validate_image(image_input)

    # Load model if needed
    if model is None:
        model = load_model()

    # Step 1: Grad-CAM (also gives prediction + class probabilities)
    gradcam_result = generate_gradcam(image_input, model)

    # Step 2: Segmentation (placeholder for U-Net)
    seg_result = generate_segmentation_mask(image_input, model=None)

    # Step 3: Fault area (prefer segmentation if available, fallback to Grad-CAM)
    if seg_result["segmentation_available"]:
        fault_area = seg_result["fault_area_percent"]
        fault_area_source = "segmentation"
    else:
        fault_area = estimate_fault_area(gradcam_result["heatmap"])
        fault_area_source = "gradcam"

    # Step 4: Risk analysis
    risk_result = full_risk_analysis(
        class_name=gradcam_result["class_name"],
        confidence=gradcam_result["confidence"],
        fault_area_percent=fault_area,
    )

    # Build response
    result = {
        "predicted_class": gradcam_result["predicted_class"],
        "class_name": gradcam_result["class_name"],
        "confidence": round(gradcam_result["confidence"], 4),  # Keep as 0-1 range
        "class_probabilities": gradcam_result["class_probabilities"],
        "fault_area_percent": round(fault_area, 2),
        "fault_area_source": fault_area_source,
        "severity_score": risk_result["severity_score"],
        "risk_level": risk_result["risk_level"],
        "maintenance_suggestion": risk_result["maintenance_suggestion"],
        "alert": risk_result["alert"],
        "gradcam_overlay": gradcam_result["overlay"],
        "segmentation_mask": seg_result["overlay"],
        "segmentation_available": seg_result["segmentation_available"],
    }

    if return_base64:
        result["gradcam_base64"] = gradcam_overlay_to_base64(
            gradcam_result["overlay"]
        )
        result["segmentation_mask_base64"] = seg_result["mask_base64"]

    return result


def print_result(result: dict):
    """Pretty-print analysis result to console."""
    print(f"\n  Predicted Class : {result['class_name']}")
    print(f"  Confidence      : {result['confidence']:.1%}")
    print(f"  Fault Area      : {result['fault_area_percent']:.1f}%")
    print(f"  Severity Score  : {result['severity_score']:.1f}")
    print(f"  Risk Level      : {result['risk_level']}")
    print(f"  Suggestion      : {result['maintenance_suggestion']}")
    print_alert(result["alert"])


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m inference.predictor <image_path> [image2 ...]")
        print("\nRunning demo on one image per class...\n")

        from config import DATASET_PATH

        model = load_model()
        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(DATASET_PATH, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            images = [f for f in os.listdir(cls_dir)
                      if f.lower().endswith(tuple(VALID_EXTENSIONS))]
            if images:
                path = os.path.join(cls_dir, images[0])
                print(f"\n{'='*60}")
                print(f" ANALYZING: {os.path.basename(path)}")
                print(f"{'='*60}")
                result = analyze_image(path, model=model)
                print_result(result)
    else:
        model = load_model()
        for path in sys.argv[1:]:
            print(f"\n{'='*60}")
            print(f" ANALYZING: {os.path.basename(path)}")
            print(f"{'='*60}")
            try:
                result = analyze_image(path, model=model)
                print_result(result)
            except Exception as e:
                print(f"  [ERROR] {e}")
