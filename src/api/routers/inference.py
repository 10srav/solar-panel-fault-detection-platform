"""Inference API endpoints."""

from __future__ import annotations

import base64
import io
import random
import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Request
import numpy as np

from src.api.schemas import (
    ClassProbability,
    CombinedInferenceRequest,
    CombinedInferenceResponse,
    RGBInferenceRequest,
    RGBInferenceResponse,
    SeverityResponse,
    ThermalInferenceRequest,
    ThermalInferenceResponse,
)
from src.inference.pipeline import InferencePipeline

router = APIRouter(prefix="/infer", tags=["Inference"])

# Global inference pipeline (initialized on startup)
_pipeline: Optional[InferencePipeline] = None

# Demo mode flag - returns mock results when models not loaded
DEMO_MODE = True

# Fault classes for demo
FAULT_CLASSES = ["Clean", "Dusty", "Bird-drop", "Electrical-damage", "Physical-damage", "Snow-Covered"]


def generate_demo_gradcam(width: int = 227, height: int = 227) -> str:
    """Generate a demo Grad-CAM heatmap overlay as base64."""
    try:
        from PIL import Image
        import numpy as np

        # Create a gradient heatmap
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Create circular hotspot
        cx, cy = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
        heatmap = np.exp(-((xx - cx)**2 + (yy - cy)**2) / 0.1)
        heatmap = (heatmap * 255).astype(np.uint8)

        # Apply colormap (red-yellow)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = heatmap  # Red channel
        rgb[:, :, 1] = (heatmap * 0.5).astype(np.uint8)  # Green for yellow tint

        img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        return None


def generate_demo_mask(width: int = 256, height: int = 256) -> str:
    """Generate a demo segmentation mask as base64."""
    try:
        from PIL import Image
        import numpy as np

        # Create random fault region
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        cx, cy = random.randint(50, width-50), random.randint(50, height-50)
        radius = random.randint(30, 60)

        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask[dist <= radius] = [255, 0, 0]  # Red fault region

        img = Image.fromarray(mask)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        return None


def get_pipeline() -> Optional[InferencePipeline]:
    """Get the inference pipeline. Returns None in demo mode if not initialized."""
    if _pipeline is None:
        if DEMO_MODE:
            return None  # Return None to trigger demo mode
        raise HTTPException(
            status_code=503, detail="Inference pipeline not initialized"
        )
    return _pipeline


def is_demo_mode() -> bool:
    """Check if we're in demo mode (no models loaded)."""
    return _pipeline is None and DEMO_MODE


def set_pipeline(pipeline: InferencePipeline) -> None:
    """Set the inference pipeline."""
    global _pipeline
    _pipeline = pipeline


@router.post(
    "/rgb",
    response_model=RGBInferenceResponse,
    summary="RGB Image Classification",
    description="Classify solar panel faults from RGB images and generate Grad-CAM explanations.",
)
async def infer_rgb(
    request: Request,
    body: Optional[RGBInferenceRequest] = None,
) -> RGBInferenceResponse:
    """Run RGB classification inference.

    Accepts image as base64 in request body.
    Returns predicted class, confidence, and Grad-CAM visualization.
    """
    start_time = time.time()

    # Try to get body from request if not provided
    if body is None:
        try:
            json_body = await request.json()
            body = RGBInferenceRequest(**json_body)
        except Exception:
            pass

    # Check if we have an image
    has_image = body and body.image_base64
    if not has_image:
        raise HTTPException(
            status_code=400, detail="No image provided. Use file upload or base64."
        )

    # Demo mode - return mock results when models not loaded
    pipeline_ready = _pipeline is not None and _pipeline.sparknet is not None
    if not pipeline_ready and DEMO_MODE:
        # Generate random but realistic probabilities
        predicted_idx = random.randint(0, len(FAULT_CLASSES) - 1)
        predicted_class = FAULT_CLASSES[predicted_idx]

        # Generate probabilities with the predicted class having highest
        probs = [random.uniform(0.01, 0.15) for _ in FAULT_CLASSES]
        probs[predicted_idx] = random.uniform(0.75, 0.98)
        total = sum(probs)
        probs = [p / total for p in probs]

        all_probs = [
            ClassProbability(class_name=name, probability=prob)
            for name, prob in zip(FAULT_CLASSES, probs)
        ]

        generate_gradcam = body.generate_gradcam if body else True
        gradcam_overlay_b64 = generate_demo_gradcam() if generate_gradcam else None

        return RGBInferenceResponse(
            predicted_class=predicted_class,
            class_index=predicted_idx,
            confidence=probs[predicted_idx],
            all_probabilities=all_probs,
            gradcam_overlay_base64=gradcam_overlay_b64,
        )

    # Get image data for real inference
    img = InferencePipeline.base64_to_image(body.image_base64)

    generate_gradcam = body.generate_gradcam if body else True

    # Run inference with actual pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not initialized")

    try:
        result = _pipeline.infer_rgb(img, generate_gradcam=generate_gradcam)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Convert probabilities
    all_probs = [
        ClassProbability(class_name=name, probability=prob)
        for name, prob in result.all_probabilities.items()
    ]

    # Convert visualizations to base64
    gradcam_overlay_b64 = None
    if result.gradcam_overlay is not None:
        gradcam_overlay_b64 = InferencePipeline.image_to_base64(result.gradcam_overlay)

    latency = (time.time() - start_time) * 1000

    return RGBInferenceResponse(
        predicted_class=result.predicted_class,
        class_index=result.class_index,
        confidence=result.confidence,
        all_probabilities=all_probs,
        gradcam_overlay_base64=gradcam_overlay_b64,
    )


@router.post(
    "/thermal",
    response_model=ThermalInferenceResponse,
    summary="Thermal Image Segmentation",
    description="Segment fault regions in thermal images using U-Net.",
)
async def infer_thermal(
    request: Request,
    image: Optional[UploadFile] = File(None),
    body: Optional[ThermalInferenceRequest] = None,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> ThermalInferenceResponse:
    """Run thermal segmentation inference.

    Returns segmentation mask and fault area ratio.
    """
    # Get image data
    if image:
        image_data = await image.read()
        import cv2
        import numpy as np

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    elif body and body.image_base64:
        img = InferencePipeline.base64_to_image(body.image_base64)
        if len(img.shape) == 3:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise HTTPException(
            status_code=400, detail="No image provided. Use file upload or base64."
        )

    threshold = body.threshold if body else 0.5

    # Run inference
    try:
        result = pipeline.infer_thermal(img, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Convert visualization to base64
    mask_overlay_b64 = None
    if result.mask_overlay is not None:
        mask_overlay_b64 = InferencePipeline.image_to_base64(result.mask_overlay)

    return ThermalInferenceResponse(
        fault_area_ratio=result.fault_area_ratio,
        mask_overlay_base64=mask_overlay_b64,
    )


@router.post(
    "/combined",
    response_model=CombinedInferenceResponse,
    summary="Combined RGB + Thermal Inference",
    description="Run full analysis with classification, segmentation, and severity assessment.",
)
async def infer_combined(
    body: CombinedInferenceRequest,
) -> CombinedInferenceResponse:
    """Run combined RGB + thermal inference.

    Returns full analysis including:
    - Fault classification with Grad-CAM
    - Segmentation mask
    - Severity score and risk level
    """
    # Validate inputs
    if not body.rgb_image_base64:
        raise HTTPException(status_code=400, detail="RGB image required")
    if not body.thermal_image_base64:
        raise HTTPException(status_code=400, detail="Thermal image required")

    # Demo mode - return mock results when models not loaded
    pipeline_ready = _pipeline is not None and _pipeline.sparknet is not None
    if not pipeline_ready and DEMO_MODE:
        # Generate random but realistic results
        predicted_idx = random.randint(1, len(FAULT_CLASSES) - 1)  # Exclude "Clean" for demo
        predicted_class = FAULT_CLASSES[predicted_idx]

        # Generate probabilities
        probs = [random.uniform(0.01, 0.12) for _ in FAULT_CLASSES]
        probs[predicted_idx] = random.uniform(0.70, 0.95)
        total = sum(probs)
        probs = [p / total for p in probs]

        all_probs = [
            ClassProbability(class_name=name, probability=prob)
            for name, prob in zip(FAULT_CLASSES, probs)
        ]

        # Generate severity scores
        fault_area = random.uniform(0.1, 0.5)
        temp_score = random.uniform(0.2, 0.8)
        growth_rate = random.uniform(0, 0.3)
        severity_score = 0.4 * fault_area + 0.4 * temp_score + 0.2 * growth_rate

        if severity_score < 0.3:
            risk_level = "Low"
            alert = False
        elif severity_score < 0.7:
            risk_level = "Medium"
            alert = False
        else:
            risk_level = "High"
            alert = True

        return CombinedInferenceResponse(
            predicted_class=predicted_class,
            class_index=predicted_idx,
            confidence=probs[predicted_idx],
            all_probabilities=all_probs,
            fault_area_ratio=fault_area,
            severity=SeverityResponse(
                fault_area_ratio=fault_area,
                temperature_score=temp_score,
                growth_rate=growth_rate,
                severity_score=severity_score,
                risk_level=risk_level,
                alert_triggered=alert,
            ),
            gradcam_overlay_base64=generate_demo_gradcam(),
            mask_overlay_base64=generate_demo_mask(),
            panel_id=body.panel_id,
            timestamp=datetime.now(),
        )

    # Decode images for real inference
    try:
        rgb_img = InferencePipeline.base64_to_image(body.rgb_image_base64)
        thermal_img = InferencePipeline.base64_to_image(body.thermal_image_base64)
        if len(thermal_img.shape) == 3:
            import cv2
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # Run inference with actual pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not initialized")

    try:
        result = _pipeline.infer_combined(
            rgb_image=rgb_img,
            thermal_image=thermal_img,
            panel_id=body.panel_id,
            previous_severity=body.previous_severity,
            time_delta_hours=body.time_delta_hours,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Convert probabilities
    all_probs = [
        ClassProbability(class_name=name, probability=prob)
        for name, prob in result.rgb_result.all_probabilities.items()
    ]

    # Convert visualizations
    gradcam_overlay_b64 = None
    if result.rgb_result.gradcam_overlay is not None:
        gradcam_overlay_b64 = InferencePipeline.image_to_base64(
            result.rgb_result.gradcam_overlay
        )

    mask_overlay_b64 = None
    if result.thermal_result.mask_overlay is not None:
        mask_overlay_b64 = InferencePipeline.image_to_base64(
            result.thermal_result.mask_overlay
        )

    return CombinedInferenceResponse(
        predicted_class=result.rgb_result.predicted_class,
        class_index=result.rgb_result.class_index,
        confidence=result.rgb_result.confidence,
        all_probabilities=all_probs,
        fault_area_ratio=result.thermal_result.fault_area_ratio,
        severity=SeverityResponse(
            fault_area_ratio=result.severity.fault_area_ratio,
            temperature_score=result.severity.temperature_score,
            growth_rate=result.severity.growth_rate,
            severity_score=result.severity.severity_score,
            risk_level=result.severity.risk_level.value,
            alert_triggered=result.severity.alert_triggered,
        ),
        gradcam_overlay_base64=gradcam_overlay_b64,
        mask_overlay_base64=mask_overlay_b64,
        panel_id=result.panel_id,
        timestamp=result.timestamp or datetime.now(),
    )
