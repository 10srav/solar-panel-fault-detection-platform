"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============ Request Schemas ============


class RGBInferenceRequest(BaseModel):
    """Request for RGB classification inference."""

    image_base64: Optional[str] = Field(None, description="Base64 encoded RGB image")
    image_url: Optional[str] = Field(None, description="URL to RGB image")
    generate_gradcam: bool = Field(True, description="Generate Grad-CAM visualization")


class ThermalInferenceRequest(BaseModel):
    """Request for thermal segmentation inference."""

    image_base64: Optional[str] = Field(None, description="Base64 encoded thermal image")
    image_url: Optional[str] = Field(None, description="URL to thermal image")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Segmentation threshold")


class CombinedInferenceRequest(BaseModel):
    """Request for combined RGB + thermal inference."""

    rgb_image_base64: Optional[str] = Field(None, description="Base64 encoded RGB image")
    thermal_image_base64: Optional[str] = Field(
        None, description="Base64 encoded thermal image"
    )
    panel_id: Optional[str] = Field(None, description="Panel identifier")
    previous_severity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Previous severity score"
    )
    time_delta_hours: Optional[float] = Field(
        None, ge=0.0, description="Hours since previous measurement"
    )


class PanelCreate(BaseModel):
    """Request to create a new panel."""

    name: str = Field(..., min_length=1, max_length=255)
    location: Optional[str] = Field(None, max_length=500)
    installation_date: Optional[datetime] = None
    capacity_kw: Optional[float] = Field(None, ge=0.0)


class PanelUpdate(BaseModel):
    """Request to update a panel."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    location: Optional[str] = Field(None, max_length=500)
    installation_date: Optional[datetime] = None
    capacity_kw: Optional[float] = Field(None, ge=0.0)


# ============ Response Schemas ============


class ClassProbability(BaseModel):
    """Probability for a single class."""

    class_name: str
    probability: float


class RGBInferenceResponse(BaseModel):
    """Response from RGB classification inference."""

    predicted_class: str = Field(..., description="Predicted fault class")
    class_index: int = Field(..., description="Class index")
    confidence: float = Field(..., description="Prediction confidence")
    all_probabilities: list[ClassProbability] = Field(
        ..., description="Probabilities for all classes"
    )
    gradcam_overlay_base64: Optional[str] = Field(
        None, description="Base64 encoded Grad-CAM overlay"
    )
    gradcam_heatmap_base64: Optional[str] = Field(
        None, description="Base64 encoded Grad-CAM heatmap"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "Dusty",
                "class_index": 1,
                "confidence": 0.92,
                "all_probabilities": [
                    {"class_name": "Clean", "probability": 0.05},
                    {"class_name": "Dusty", "probability": 0.92},
                ],
                "gradcam_overlay_base64": "iVBORw0KGgo...",
            }
        }


class ThermalInferenceResponse(BaseModel):
    """Response from thermal segmentation inference."""

    fault_area_ratio: float = Field(..., description="Ratio of fault area to total area")
    mask_overlay_base64: Optional[str] = Field(
        None, description="Base64 encoded mask overlay"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "fault_area_ratio": 0.15,
                "mask_overlay_base64": "iVBORw0KGgo...",
            }
        }


class SeverityResponse(BaseModel):
    """Severity assessment response."""

    fault_area_ratio: float
    temperature_score: float
    growth_rate: float
    severity_score: float
    risk_level: str
    alert_triggered: bool


class CombinedInferenceResponse(BaseModel):
    """Response from combined RGB + thermal inference."""

    # Classification
    predicted_class: str
    class_index: int
    confidence: float
    all_probabilities: list[ClassProbability]

    # Segmentation
    fault_area_ratio: float

    # Severity
    severity: SeverityResponse

    # Visualizations
    gradcam_overlay_base64: Optional[str] = None
    mask_overlay_base64: Optional[str] = None

    # Metadata
    panel_id: Optional[str] = None
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "Electrical-damage",
                "class_index": 3,
                "confidence": 0.87,
                "all_probabilities": [],
                "fault_area_ratio": 0.23,
                "severity": {
                    "fault_area_ratio": 0.23,
                    "temperature_score": 0.65,
                    "growth_rate": 0.1,
                    "severity_score": 0.72,
                    "risk_level": "High",
                    "alert_triggered": True,
                },
                "panel_id": "panel-001",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class PanelResponse(BaseModel):
    """Panel response schema."""

    id: str
    name: str
    location: Optional[str]
    installation_date: Optional[datetime]
    capacity_kw: Optional[float]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FaultEventResponse(BaseModel):
    """Fault event response schema."""

    id: str
    panel_id: str
    fault_class: str
    confidence: float
    severity_score: float
    risk_level: str
    fault_area_ratio: Optional[float]
    temperature_score: Optional[float]
    growth_rate: Optional[float]
    alert_triggered: bool
    alert_acknowledged: bool
    detected_at: datetime

    class Config:
        from_attributes = True


class PanelHistoryResponse(BaseModel):
    """Response for panel fault history."""

    panel: PanelResponse
    fault_events: list[FaultEventResponse]
    total_events: int
    high_risk_events: int
    latest_severity: Optional[float]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    models_loaded: dict[str, bool]
    database_connected: bool


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
