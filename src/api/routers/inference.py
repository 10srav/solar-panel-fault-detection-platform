"""Inference API endpoints."""

from __future__ import annotations

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


def get_pipeline() -> InferencePipeline:
    """Get the inference pipeline."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503, detail="Inference pipeline not initialized"
        )
    return _pipeline


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
    image: Optional[UploadFile] = File(None),
    body: Optional[RGBInferenceRequest] = None,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> RGBInferenceResponse:
    """Run RGB classification inference.

    Accepts image either as file upload or base64 in request body.
    Returns predicted class, confidence, and Grad-CAM visualization.
    """
    start_time = time.time()

    # Get image data
    if image:
        image_data = await image.read()
        import cv2
        import numpy as np

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif body and body.image_base64:
        img = InferencePipeline.base64_to_image(body.image_base64)
    else:
        raise HTTPException(
            status_code=400, detail="No image provided. Use file upload or base64."
        )

    generate_gradcam = body.generate_gradcam if body else True

    # Run inference
    try:
        result = pipeline.infer_rgb(img, generate_gradcam=generate_gradcam)
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
    pipeline: InferencePipeline = Depends(get_pipeline),
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

    # Decode images
    try:
        rgb_img = InferencePipeline.base64_to_image(body.rgb_image_base64)
        thermal_img = InferencePipeline.base64_to_image(body.thermal_image_base64)
        if len(thermal_img.shape) == 3:
            import cv2
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # Run inference
    try:
        result = pipeline.infer_combined(
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
