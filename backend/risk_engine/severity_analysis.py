"""
=============================================================================
 RISK & MAINTENANCE ENGINE — severity_analysis.py
 Severity scoring, risk classification, maintenance suggestions, alerts.
=============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_THRESHOLDS, MAINTENANCE_SUGGESTIONS


# ============================================================================
# SEVERITY SCORE
# ============================================================================

def compute_severity(fault_area_percent: float, confidence: float) -> float:
    """
    Severity = fault_area_percent * confidence

    Args:
        fault_area_percent: 0-100 from Grad-CAM analysis
        confidence: 0.0-1.0 from model softmax output

    Returns:
        severity_score: float
    """
    return fault_area_percent * confidence


# ============================================================================
# RISK CLASSIFICATION — FAULT-TYPE AWARE
# ============================================================================

def classify_risk_intelligent(
    class_name: str,
    confidence: float,
    fault_area_percent: float
) -> str:
    """
    Intelligent risk classification based on fault type characteristics.

    Critical faults (Electrical/Physical damage):
        - High confidence (>80%) → High risk (needs immediate action)
        - Lower confidence → Medium risk

    Maintenance faults (Dusty/Bird droppings/Snow):
        - Based on fault area coverage
        - <30% → Low, 30-60% → Medium, >60% → High

    Clean panels:
        - Always Low risk
    """
    # Clean panels are always low risk
    if class_name == "Clean":
        return "Low"

    # Critical structural/electrical damage — confidence-driven
    if class_name in ["Electrical_damage_generated", "Physcial_damage_generated"]:
        if confidence > 0.8:
            return "High"
        else:
            return "Medium"

    # Maintenance issues — area-driven
    elif class_name in ["Dusty", "Bird_drop_generateds", "Snow_covered_generated"]:
        if fault_area_percent < 30:
            return "Low"
        elif fault_area_percent < 60:
            return "Medium"
        else:
            return "High"

    # Fallback for any unknown classes
    else:
        # Use old severity-based logic
        severity = compute_severity(fault_area_percent, confidence)
        if severity < RISK_THRESHOLDS["low_max"]:
            return "Low"
        elif severity <= RISK_THRESHOLDS["medium_max"]:
            return "Medium"
        else:
            return "High"


def classify_risk(severity_score: float) -> str:
    """
    Legacy severity-based classification (kept for backwards compatibility).
    Use classify_risk_intelligent() for better results.

    < 30  → Low
    30-60 → Medium
    > 60  → High
    """
    if severity_score < RISK_THRESHOLDS["low_max"]:
        return "Low"
    elif severity_score <= RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    else:
        return "High"


# ============================================================================
# MAINTENANCE SUGGESTION
# ============================================================================

def get_maintenance_suggestion(class_name: str) -> str:
    """Get the recommended maintenance action for a fault type."""
    return MAINTENANCE_SUGGESTIONS.get(class_name, "Manual inspection required")


# ============================================================================
# ALERT SYSTEM
# ============================================================================

def generate_alert(risk_level: str, class_name: str) -> dict:
    """
    Generate alert payload based on risk level.

    Returns:
        dict with: triggered (bool), level, message, action
    """
    suggestion = get_maintenance_suggestion(class_name)

    if risk_level == "High":
        return {
            "triggered": True,
            "level": "CRITICAL",
            "message": "CRITICAL FAULT — Immediate Maintenance Required",
            "fault_type": class_name,
            "action": suggestion,
        }
    elif risk_level == "Medium":
        return {
            "triggered": True,
            "level": "WARNING",
            "message": f"WARNING — Schedule maintenance soon for: {class_name}",
            "fault_type": class_name,
            "action": suggestion,
        }
    else:
        return {
            "triggered": False,
            "level": "INFO",
            "message": "Panel condition acceptable",
            "fault_type": class_name,
            "action": suggestion,
        }


def print_alert(alert: dict):
    """Print alert to console with formatting."""
    if alert["level"] == "CRITICAL":
        print(f"\n{'='*60}")
        print(f"  ⚠ {alert['message']}")
        print(f"  Fault: {alert['fault_type']}")
        print(f"  Action: {alert['action']}")
        print(f"{'='*60}\n")
    elif alert["level"] == "WARNING":
        print(f"\n  ⚡ {alert['message']}")
        print(f"     Action: {alert['action']}\n")


# ============================================================================
# FULL ANALYSIS PIPELINE
# ============================================================================

def full_risk_analysis(
    class_name: str,
    confidence: float,
    fault_area_percent: float,
) -> dict:
    """
    Complete risk analysis from prediction results.

    Returns dict with all risk fields:
        severity_score, risk_level, suggestion, alert
    """
    # Compute severity score
    severity = compute_severity(fault_area_percent, confidence)

    # Use intelligent fault-type-aware risk classification
    risk = classify_risk_intelligent(class_name, confidence, fault_area_percent)

    suggestion = get_maintenance_suggestion(class_name)
    alert = generate_alert(risk, class_name)

    return {
        "severity_score": round(severity, 2),
        "risk_level": risk,
        "maintenance_suggestion": suggestion,
        "alert": alert,
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    test_cases = [
        ("Dusty", 0.92, 45.0),
        ("Electrical_damage_generated", 0.97, 72.0),
        ("Clean", 0.99, 5.0),
        ("Physcial_damage_generated", 0.85, 55.0),
        ("Snow_covered_generated", 0.78, 80.0),
        ("Bird_drop_generateds", 0.60, 30.0),
    ]

    print("=" * 70)
    print(" RISK ENGINE — Test Cases")
    print("=" * 70)

    for cls, conf, area in test_cases:
        result = full_risk_analysis(cls, conf, area)
        print(f"\n  Class: {cls}")
        print(f"  Confidence: {conf:.0%} | Fault Area: {area:.1f}%")
        print(f"  Severity: {result['severity_score']:.1f} | Risk: {result['risk_level']}")
        print(f"  Suggestion: {result['maintenance_suggestion']}")
        print_alert(result["alert"])
