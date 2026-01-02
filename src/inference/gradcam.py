"""Grad-CAM implementation for explainable predictions.

Provides visual explanations (WHY) for SparkNet predictions by highlighting
regions in the input image that contributed most to the prediction.
"""

from __future__ import annotations

from typing import Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM (Gradient-weighted Class Activation Mapping).

    Generates heatmaps showing which regions of an image were most important
    for a particular class prediction.

    Args:
        model: Trained model.
        target_layer: Layer to compute gradients for.
        device: Device to run on.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        # Get target layer (last convolutional layer if not specified)
        if target_layer is None:
            target_layer = self._find_target_layer()
        self.target_layer = target_layer

        # Storage for gradients and activations
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        # Try to use model's method if available
        if hasattr(self.model, "get_cam_target_layer"):
            return self.model.get_cam_target_layer()

        # Otherwise find last conv layer
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        return target_layer

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""

        def forward_hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(
            module: nn.Module, grad_input: tuple, grad_output: tuple
        ) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W).
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Tuple of (heatmap, predicted_class, confidence).
        """
        # Ensure input is on correct device
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Get predicted class and confidence
        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)
        pred_class = pred_class.item()
        confidence = confidence.item()

        # Use predicted class if target not specified
        if target_class is None:
            target_class = pred_class

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=0)  # (H, W)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Convert to numpy
        heatmap = cam.cpu().numpy()

        return heatmap, pred_class, confidence

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        original_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap resized to original image size.

        Args:
            input_tensor: Input image tensor.
            target_class: Target class index.
            original_size: Original image size (H, W) to resize heatmap.

        Returns:
            Heatmap array of shape (H, W) in range [0, 1].
        """
        heatmap, _, _ = self(input_tensor, target_class)

        if original_size:
            heatmap = cv2.resize(heatmap, (original_size[1], original_size[0]))

        return heatmap


def generate_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> np.ndarray:
    """Generate overlay of Grad-CAM heatmap on original image.

    Args:
        image: Original image (H, W, 3) in RGB format, range [0, 255].
        heatmap: Grad-CAM heatmap (H, W) in range [0, 1].
        colormap: OpenCV colormap to use.
        alpha: Blend factor for overlay.

    Returns:
        Overlay image (H, W, 3) in RGB format.
    """
    # Resize heatmap to image size
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert from BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ for improved saliency maps.

    Uses second-order gradients for better localization.
    """

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """Generate Grad-CAM++ heatmap."""
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)
        pred_class = pred_class.item()
        confidence = confidence.item()

        if target_class is None:
            target_class = pred_class

        # First backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        # Compute alpha weights (Grad-CAM++)
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + activations * grad_3.sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom)
        )
        alpha = alpha_num / (alpha_denom + 1e-8)

        # Weight by gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.cpu().numpy()

        return heatmap, pred_class, confidence


class ScoreCAM(GradCAM):
    """Score-CAM for gradient-free saliency maps.

    Uses activation-based importance instead of gradients.
    """

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """Generate Score-CAM heatmap."""
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Get activations
        with torch.no_grad():
            output = self.model(input_tensor)

        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)
        pred_class = pred_class.item()
        confidence = confidence.item()

        if target_class is None:
            target_class = pred_class

        activations = self.activations[0]  # (C, H, W)

        # Upsample activations to input size
        upsampled = F.interpolate(
            activations.unsqueeze(0),
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )[0]  # (C, H, W)

        # Normalize each activation map
        upsampled_norm = (upsampled - upsampled.min()) / (
            upsampled.max() - upsampled.min() + 1e-8
        )

        # Compute score for each channel
        scores = []
        with torch.no_grad():
            for i in range(activations.shape[0]):
                # Mask input with activation
                mask = upsampled_norm[i].unsqueeze(0).unsqueeze(0)
                masked_input = input_tensor * mask

                # Get prediction
                out = self.model(masked_input)
                score = F.softmax(out, dim=1)[0, target_class].item()
                scores.append(score)

        scores = torch.tensor(scores).to(self.device)

        # Weighted combination
        cam = (scores.view(-1, 1, 1) * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.cpu().numpy()

        return heatmap, pred_class, confidence
