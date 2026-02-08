"""
=============================================================================
 EXPLAINABILITY ENGINE — gradcam_engine.py
 Grad-CAM for ResNet18 + fault area estimation from heatmaps.
=============================================================================
"""

import os
import sys
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    GRADCAM_THRESHOLD, GRADCAM_OVERLAY_ALPHA, CLASS_NAMES,
)
from training.preprocessing import get_val_transform


# ============================================================================
# GRAD-CAM CORE
# ============================================================================

class GradCAM:
    """
    Grad-CAM implementation for ResNet architectures.
    Hooks into the last convolutional layer (layer4) to produce
    class-discriminative heatmaps.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._hooks = []

        # Default: last conv layer in ResNet18's layer4
        if target_layer is None:
            target_layer = self.model.layer4[-1].conv2

        # Register hooks
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            target_layer.register_backward_hook(self._save_gradients)
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: [1, 3, 224, 224] preprocessed tensor
            target_class: int or None (uses predicted class)

        Returns:
            heatmap: numpy [H, W] normalized 0-1
            predicted_class: int
            confidence: float (softmax probability)
        """
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        all_probs = probs[0].cpu().detach().numpy()  # All class probabilities

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Grad-CAM: global average pool gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class, confidence, all_probs

    def remove_hooks(self):
        """Clean up registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def generate_gradcam(image_input, model):
    """
    Full Grad-CAM pipeline: image → heatmap + overlay.

    Args:
        image_input: file path (str) or PIL Image or file-like object
        model: trained ResNet18

    Returns:
        dict with keys:
            heatmap:          np.array [224, 224] float 0-1
            overlay:          np.array [224, 224, 3] float 0-1
            original:         PIL Image resized to 224x224
            predicted_class:  int
            class_name:       str
            confidence:       float
    """
    # Load image from various input types
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        original = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        original = image_input.convert("RGB")
    else:
        # File-like object (e.g., from FastAPI upload)
        original = Image.open(image_input).convert("RGB")

    # Preprocess
    transform = get_val_transform()
    input_tensor = transform(original).unsqueeze(0)

    # Generate heatmap
    gradcam = GradCAM(model)
    try:
        heatmap, predicted_class, confidence, all_probs = gradcam.generate(input_tensor)
    finally:
        gradcam.remove_hooks()

    # Create overlay
    original_resized = original.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original_resized).astype(np.float32) / 255.0

    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    alpha = GRADCAM_OVERLAY_ALPHA
    overlay = (1 - alpha) * original_np + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    class_name = CLASS_NAMES[predicted_class]

    # Format class probabilities as dict
    class_probabilities = {
        CLASS_NAMES[i]: float(all_probs[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "original": original_resized,
        "predicted_class": predicted_class,
        "class_name": class_name,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
    }


# ============================================================================
# FAULT AREA ESTIMATION
# ============================================================================

def estimate_fault_area(heatmap, threshold=None):
    """
    Compute % of image area where Grad-CAM activation exceeds threshold.

    Args:
        heatmap: np.array [H, W] with values 0-1
        threshold: float (default from config: 0.5)

    Returns:
        fault_area_percent: float 0-100
    """
    if threshold is None:
        threshold = GRADCAM_THRESHOLD

    binary_mask = (heatmap > threshold).astype(np.float32)
    total_pixels = heatmap.shape[0] * heatmap.shape[1]
    activated = binary_mask.sum()

    return float((activated / total_pixels) * 100.0)


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_gradcam_figure(gradcam_result, fault_area=None, save_path=None):
    """
    Create a 3-panel figure: Original | Heatmap | Overlay.
    Returns the figure, optionally saves to disk.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gradcam_result["original"])
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(gradcam_result["heatmap"], cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    conf = gradcam_result["confidence"]
    name = gradcam_result["class_name"]
    title = f"{name} ({conf:.1%})"
    if fault_area is not None:
        title += f" | Fault Area: {fault_area:.1f}%"

    axes[2].imshow(gradcam_result["overlay"])
    axes[2].set_title(title, fontsize=11)
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GRAD-CAM] Saved: {save_path}")

    return fig


def gradcam_overlay_to_base64(overlay_np):
    """Convert overlay numpy array to base64 PNG for API responses."""
    overlay_uint8 = (overlay_np * 255).astype(np.uint8)
    img = Image.fromarray(overlay_uint8)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import torch.nn as nn
    from torchvision import models as tv_models
    from config import MODEL_PATH

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run training first.")
        sys.exit(1)

    # Load model
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Test on a sample image
    from config import DATASET_PATH
    test_dir = os.path.join(DATASET_PATH, "Dusty")
    images = [f for f in os.listdir(test_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if images:
        test_path = os.path.join(test_dir, images[0])
        print(f"Testing: {test_path}")

        result = generate_gradcam(test_path, model)
        fault_area = estimate_fault_area(result["heatmap"])

        print(f"Predicted: {result['class_name']} ({result['confidence']:.1%})")
        print(f"Fault area: {fault_area:.1f}%")

        save_path = os.path.join(os.path.dirname(__file__), "..", "plots", "gradcam_test.png")
        create_gradcam_figure(result, fault_area, save_path)
