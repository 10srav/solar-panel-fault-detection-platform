"""Unified inference pipeline for solar panel fault detection."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import Config, get_config
from src.data.transforms import get_rgb_transforms, get_thermal_transforms
from src.inference.gradcam import GradCAM, generate_gradcam_overlay
from src.inference.severity import SeverityScorer, SeverityResult, RiskLevel
from src.models.sparknet import SparkNet
from src.models.unet import UNet


@dataclass
class RGBInferenceResult:
    """Result from RGB classification inference.

    Attributes:
        predicted_class: Predicted fault class name.
        class_index: Class index.
        confidence: Prediction confidence.
        all_probabilities: Probabilities for all classes.
        gradcam_heatmap: Grad-CAM heatmap array.
        gradcam_overlay: Overlay image.
    """

    predicted_class: str
    class_index: int
    confidence: float
    all_probabilities: dict[str, float]
    gradcam_heatmap: Optional[np.ndarray] = None
    gradcam_overlay: Optional[np.ndarray] = None


@dataclass
class ThermalInferenceResult:
    """Result from thermal segmentation inference.

    Attributes:
        segmentation_mask: Binary fault mask.
        fault_area_ratio: Ratio of fault area to total area.
        mask_overlay: Overlay visualization.
    """

    segmentation_mask: np.ndarray
    fault_area_ratio: float
    mask_overlay: Optional[np.ndarray] = None


@dataclass
class CombinedInferenceResult:
    """Combined result from RGB + thermal inference.

    Attributes:
        rgb_result: RGB classification result.
        thermal_result: Thermal segmentation result.
        severity: Severity assessment result.
        panel_id: Optional panel identifier.
        timestamp: Inference timestamp.
    """

    rgb_result: RGBInferenceResult
    thermal_result: ThermalInferenceResult
    severity: SeverityResult
    panel_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class InferencePipeline:
    """Unified inference pipeline for solar panel fault detection.

    Combines SparkNet classification, U-Net segmentation, Grad-CAM
    explainability, and severity scoring.

    Args:
        sparknet_path: Path to SparkNet weights.
        unet_path: Path to U-Net weights.
        config: Configuration object.
        device: Device to run inference on.
    """

    def __init__(
        self,
        sparknet_path: Optional[str | Path] = None,
        unet_path: Optional[str | Path] = None,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or get_config()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.class_names = self.config.classes

        # Initialize models
        self.sparknet: Optional[SparkNet] = None
        self.unet: Optional[UNet] = None
        self.gradcam: Optional[GradCAM] = None

        if sparknet_path:
            self.load_sparknet(sparknet_path)

        if unet_path:
            self.load_unet(unet_path)

        # Initialize severity scorer
        self.severity_scorer = SeverityScorer(self.config)

        # Initialize transforms
        self.rgb_transforms = get_rgb_transforms(
            input_size=tuple(self.config.sparknet.input_size),
            augment=False,
        )["val"]

        self.thermal_transforms = get_thermal_transforms(
            input_size=tuple(self.config.unet.input_size),
            augment=False,
        )["val"]

    def load_sparknet(self, path: str | Path) -> None:
        """Load SparkNet model from checkpoint.

        Args:
            path: Path to model checkpoint.
        """
        self.sparknet = SparkNet(
            num_classes=len(self.class_names),
            input_channels=3,
            dropout_rate=self.config.sparknet.dropout_rate,
        )

        checkpoint = torch.load(path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.sparknet.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.sparknet.load_state_dict(checkpoint)

        self.sparknet.to(self.device)
        self.sparknet.eval()

        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.sparknet, device=self.device)

    def load_unet(self, path: str | Path) -> None:
        """Load U-Net model from checkpoint.

        Args:
            path: Path to model checkpoint.
        """
        self.unet = UNet(
            in_channels=self.config.unet.input_channels,
            out_channels=self.config.unet.output_channels,
            features=self.config.unet.features,
        )

        checkpoint = torch.load(path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.unet.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.unet.load_state_dict(checkpoint)

        self.unet.to(self.device)
        self.unet.eval()

    def preprocess_rgb(
        self, image: Union[np.ndarray, str, Path, Image.Image]
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Preprocess RGB image for inference.

        Args:
            image: Input image (array, path, or PIL Image).

        Returns:
            Tuple of (preprocessed tensor, original image array).
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        original = image.copy()

        # Apply transforms
        transformed = self.rgb_transforms(image=image)
        tensor = transformed["image"].unsqueeze(0)

        return tensor, original

    def preprocess_thermal(
        self, image: Union[np.ndarray, str, Path]
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Preprocess thermal image for inference.

        Args:
            image: Input thermal image.

        Returns:
            Tuple of (preprocessed tensor, original image array).
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)

        original = image.copy()

        # Apply transforms
        transformed = self.thermal_transforms(image=image)
        tensor = transformed["image"].unsqueeze(0)

        return tensor, original

    def infer_rgb(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        generate_gradcam: bool = True,
    ) -> RGBInferenceResult:
        """Run RGB classification inference.

        Args:
            image: Input RGB image.
            generate_gradcam: Whether to generate Grad-CAM visualization.

        Returns:
            RGBInferenceResult with predictions and explanations.
        """
        if self.sparknet is None:
            raise RuntimeError("SparkNet model not loaded")

        # Preprocess
        tensor, original = self.preprocess_rgb(image)
        tensor = tensor.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.sparknet(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        # Get prediction
        confidence, class_idx = probs.max(dim=0)
        class_idx = class_idx.item()
        confidence = confidence.item()
        predicted_class = self.class_names[class_idx]

        # All probabilities
        all_probs = {
            self.class_names[i]: probs[i].item() for i in range(len(self.class_names))
        }

        # Generate Grad-CAM
        gradcam_heatmap = None
        gradcam_overlay = None

        if generate_gradcam and self.gradcam:
            heatmap, _, _ = self.gradcam(tensor, class_idx)
            gradcam_heatmap = cv2.resize(
                heatmap, (original.shape[1], original.shape[0])
            )
            gradcam_overlay = generate_gradcam_overlay(original, gradcam_heatmap)

        return RGBInferenceResult(
            predicted_class=predicted_class,
            class_index=class_idx,
            confidence=confidence,
            all_probabilities=all_probs,
            gradcam_heatmap=gradcam_heatmap,
            gradcam_overlay=gradcam_overlay,
        )

    def infer_thermal(
        self,
        image: Union[np.ndarray, str, Path],
        threshold: float = 0.5,
        generate_overlay: bool = True,
    ) -> ThermalInferenceResult:
        """Run thermal segmentation inference.

        Args:
            image: Input thermal image.
            threshold: Threshold for binarizing mask.
            generate_overlay: Whether to generate visualization.

        Returns:
            ThermalInferenceResult with mask and metrics.
        """
        if self.unet is None:
            raise RuntimeError("U-Net model not loaded")

        # Preprocess
        tensor, original = self.preprocess_thermal(image)
        tensor = tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.unet(tensor)
            prob = torch.sigmoid(output)[0, 0]
            mask = (prob > threshold).cpu().numpy().astype(np.uint8)

        # Resize mask to original size
        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))

        # Compute fault area ratio
        fault_area_ratio = mask_resized.sum() / mask_resized.size

        # Generate overlay
        mask_overlay = None
        if generate_overlay:
            # Create colored overlay
            overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            overlay[mask_resized > 0] = [255, 0, 0]  # Red for faults
            mask_overlay = cv2.addWeighted(
                cv2.cvtColor(original, cv2.COLOR_GRAY2RGB),
                0.6,
                overlay,
                0.4,
                0,
            )

        return ThermalInferenceResult(
            segmentation_mask=mask_resized,
            fault_area_ratio=fault_area_ratio,
            mask_overlay=mask_overlay,
        )

    def infer_combined(
        self,
        rgb_image: Union[np.ndarray, str, Path, Image.Image],
        thermal_image: Union[np.ndarray, str, Path],
        panel_id: Optional[str] = None,
        previous_severity: Optional[float] = None,
        time_delta_hours: Optional[float] = None,
    ) -> CombinedInferenceResult:
        """Run combined RGB + thermal inference with severity assessment.

        Args:
            rgb_image: Input RGB image.
            thermal_image: Input thermal image.
            panel_id: Optional panel identifier.
            previous_severity: Previous severity score.
            time_delta_hours: Time since previous measurement.

        Returns:
            CombinedInferenceResult with full analysis.
        """
        # Run RGB inference
        rgb_result = self.infer_rgb(rgb_image)

        # Run thermal inference
        thermal_result = self.infer_thermal(thermal_image)

        # Get thermal image for severity calculation
        if isinstance(thermal_image, (str, Path)):
            thermal_array = cv2.imread(str(thermal_image), cv2.IMREAD_GRAYSCALE)
        else:
            thermal_array = thermal_image

        # Compute severity
        severity = self.severity_scorer.assess_from_prediction(
            segmentation_mask=thermal_result.segmentation_mask,
            thermal_image=thermal_array,
            fault_class=rgb_result.predicted_class,
            confidence=rgb_result.confidence,
            previous_severity=previous_severity,
            time_delta_hours=time_delta_hours,
        )

        return CombinedInferenceResult(
            rgb_result=rgb_result,
            thermal_result=thermal_result,
            severity=severity,
            panel_id=panel_id,
            timestamp=datetime.now(),
        )

    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        """Convert numpy image to base64 string.

        Args:
            image: Image array (H, W, 3) in RGB format.

        Returns:
            Base64 encoded string.
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Save to buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def base64_to_image(b64_string: str) -> np.ndarray:
        """Convert base64 string to numpy image.

        Args:
            b64_string: Base64 encoded image string.

        Returns:
            Image array.
        """
        # Decode
        image_data = base64.b64decode(b64_string)

        # Load image
        pil_image = Image.open(io.BytesIO(image_data))

        return np.array(pil_image)


def create_pipeline(
    sparknet_path: Optional[str | Path] = None,
    unet_path: Optional[str | Path] = None,
    config: Optional[Config] = None,
) -> InferencePipeline:
    """Factory function to create inference pipeline.

    Args:
        sparknet_path: Path to SparkNet weights.
        unet_path: Path to U-Net weights.
        config: Configuration object.

    Returns:
        Configured InferencePipeline instance.
    """
    return InferencePipeline(
        sparknet_path=sparknet_path,
        unet_path=unet_path,
        config=config,
    )
