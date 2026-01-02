"""Severity scoring and risk assessment for solar panel faults.

Combines fault area, temperature intensity, and growth rate to compute
a severity score and map it to risk levels (Low, Medium, High).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np

from src.config import Config, get_config


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class SeverityResult:
    """Result of severity scoring.

    Attributes:
        fault_area_ratio: Ratio of fault area to total panel area.
        temperature_score: Normalized temperature-based score.
        growth_rate: Rate of fault area growth over time.
        severity_score: Combined severity score (0-1).
        risk_level: Categorical risk level.
        timestamp: Time of assessment.
        alert_triggered: Whether a high-risk alert should be triggered.
    """

    fault_area_ratio: float
    temperature_score: float
    growth_rate: float
    severity_score: float
    risk_level: RiskLevel
    timestamp: datetime
    alert_triggered: bool


class SeverityScorer:
    """Compute severity scores for solar panel faults.

    Combines multiple factors to assess fault severity:
    - Fault area: Percentage of panel covered by fault
    - Temperature: Normalized temperature statistics from thermal image
    - Growth rate: Change in severity over time (optional)

    Args:
        config: Configuration object with severity weights and thresholds.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self._load_weights()

    def _load_weights(self) -> None:
        """Load weights and thresholds from config."""
        self.w_area = self.config.severity.weights.fault_area
        self.w_temp = self.config.severity.weights.temperature
        self.w_growth = self.config.severity.weights.growth_rate

        self.threshold_low = self.config.severity.thresholds.low
        self.threshold_high = self.config.severity.thresholds.high

        self.temp_min = self.config.severity.temperature.normalize_min
        self.temp_max = self.config.severity.temperature.normalize_max

    def compute_fault_area_ratio(
        self,
        mask: np.ndarray,
        panel_mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the ratio of fault area to panel area.

        Args:
            mask: Binary fault mask (H, W) where 1 indicates fault.
            panel_mask: Optional binary mask for panel region.

        Returns:
            Fault area ratio in range [0, 1].
        """
        if panel_mask is not None:
            panel_pixels = panel_mask.sum()
            fault_pixels = (mask * panel_mask).sum()
        else:
            panel_pixels = mask.size
            fault_pixels = mask.sum()

        if panel_pixels == 0:
            return 0.0

        return float(fault_pixels / panel_pixels)

    def compute_temperature_score(
        self,
        thermal_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute temperature-based severity score.

        Uses statistics from thermal image (mean, max) in fault regions.

        Args:
            thermal_image: Thermal image array (H, W).
            mask: Optional fault mask to focus on fault regions.

        Returns:
            Temperature score in range [0, 1].
        """
        if mask is not None and mask.sum() > 0:
            # Focus on fault regions
            fault_temps = thermal_image[mask > 0]
        else:
            fault_temps = thermal_image.flatten()

        if len(fault_temps) == 0:
            return 0.0

        # Compute statistics
        mean_temp = fault_temps.mean()
        max_temp = fault_temps.max()

        # Weighted combination of mean and max
        temp_value = 0.6 * max_temp + 0.4 * mean_temp

        # Normalize to [0, 1]
        normalized = (temp_value - self.temp_min) / (self.temp_max - self.temp_min)
        normalized = np.clip(normalized, 0.0, 1.0)

        return float(normalized)

    def compute_growth_rate(
        self,
        current_severity: float,
        previous_severity: Optional[float] = None,
        time_delta_hours: Optional[float] = None,
    ) -> float:
        """Compute fault growth rate over time.

        Args:
            current_severity: Current severity score.
            previous_severity: Previous severity score.
            time_delta_hours: Time between measurements in hours.

        Returns:
            Growth rate score in range [0, 1].
        """
        if previous_severity is None or time_delta_hours is None:
            return 0.0

        if time_delta_hours <= 0:
            return 0.0

        # Rate of change per hour
        rate = (current_severity - previous_severity) / time_delta_hours

        # Normalize (assume max growth rate of 0.1 per hour)
        max_rate = 0.1
        normalized = np.clip(rate / max_rate, 0.0, 1.0)

        return float(normalized)

    def compute_severity(
        self,
        fault_area_ratio: float,
        temperature_score: float,
        growth_rate: float = 0.0,
    ) -> float:
        """Compute combined severity score.

        Args:
            fault_area_ratio: Ratio of fault area to panel area.
            temperature_score: Temperature-based score.
            growth_rate: Growth rate score.

        Returns:
            Combined severity score in range [0, 1].
        """
        severity = (
            self.w_area * fault_area_ratio
            + self.w_temp * temperature_score
            + self.w_growth * growth_rate
        )

        return float(np.clip(severity, 0.0, 1.0))

    def get_risk_level(self, severity_score: float) -> RiskLevel:
        """Map severity score to risk level.

        Args:
            severity_score: Severity score in range [0, 1].

        Returns:
            Categorical risk level.
        """
        if severity_score < self.threshold_low:
            return RiskLevel.LOW
        elif severity_score < self.threshold_high:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def assess(
        self,
        mask: np.ndarray,
        thermal_image: np.ndarray,
        panel_mask: Optional[np.ndarray] = None,
        previous_severity: Optional[float] = None,
        time_delta_hours: Optional[float] = None,
    ) -> SeverityResult:
        """Perform full severity assessment.

        Args:
            mask: Binary fault mask.
            thermal_image: Thermal image array.
            panel_mask: Optional panel region mask.
            previous_severity: Previous severity score for growth calculation.
            time_delta_hours: Time since previous measurement.

        Returns:
            SeverityResult with all computed values.
        """
        # Compute individual scores
        fault_area_ratio = self.compute_fault_area_ratio(mask, panel_mask)
        temperature_score = self.compute_temperature_score(thermal_image, mask)

        # Compute preliminary severity for growth rate
        preliminary_severity = self.compute_severity(
            fault_area_ratio, temperature_score, 0.0
        )

        growth_rate = self.compute_growth_rate(
            preliminary_severity, previous_severity, time_delta_hours
        )

        # Final severity with growth rate
        severity_score = self.compute_severity(
            fault_area_ratio, temperature_score, growth_rate
        )

        # Get risk level
        risk_level = self.get_risk_level(severity_score)

        # Determine if alert should be triggered
        alert_triggered = risk_level == RiskLevel.HIGH

        return SeverityResult(
            fault_area_ratio=fault_area_ratio,
            temperature_score=temperature_score,
            growth_rate=growth_rate,
            severity_score=severity_score,
            risk_level=risk_level,
            timestamp=datetime.now(),
            alert_triggered=alert_triggered,
        )

    def assess_from_prediction(
        self,
        segmentation_mask: np.ndarray,
        thermal_image: np.ndarray,
        fault_class: str,
        confidence: float,
        panel_mask: Optional[np.ndarray] = None,
        previous_severity: Optional[float] = None,
        time_delta_hours: Optional[float] = None,
    ) -> SeverityResult:
        """Assess severity from model predictions.

        Args:
            segmentation_mask: Predicted fault mask from U-Net.
            thermal_image: Thermal image.
            fault_class: Predicted fault class from SparkNet.
            confidence: Prediction confidence.
            panel_mask: Optional panel region mask.
            previous_severity: Previous severity score.
            time_delta_hours: Time since previous measurement.

        Returns:
            SeverityResult with all computed values.
        """
        # Adjust weights based on fault class
        class_weight_adjustments = {
            "Clean": 0.0,
            "Dusty": 0.3,
            "Bird-drop": 0.5,
            "Electrical-damage": 0.9,
            "Physical-damage": 0.8,
            "Snow-Covered": 0.4,
        }

        class_weight = class_weight_adjustments.get(fault_class, 0.5)

        # Get base assessment
        result = self.assess(
            mask=segmentation_mask,
            thermal_image=thermal_image,
            panel_mask=panel_mask,
            previous_severity=previous_severity,
            time_delta_hours=time_delta_hours,
        )

        # Adjust severity based on fault class and confidence
        adjusted_severity = result.severity_score * class_weight * confidence

        # Update risk level
        adjusted_risk = self.get_risk_level(adjusted_severity)

        return SeverityResult(
            fault_area_ratio=result.fault_area_ratio,
            temperature_score=result.temperature_score,
            growth_rate=result.growth_rate,
            severity_score=adjusted_severity,
            risk_level=adjusted_risk,
            timestamp=result.timestamp,
            alert_triggered=adjusted_risk == RiskLevel.HIGH,
        )


def create_severity_scorer(config: Optional[Config] = None) -> SeverityScorer:
    """Factory function to create a severity scorer.

    Args:
        config: Configuration object.

    Returns:
        Configured SeverityScorer instance.
    """
    return SeverityScorer(config)
