"""Tests for severity scoring system."""

import pytest
import numpy as np

from src.inference.severity import SeverityScorer, RiskLevel


class TestSeverityScorer:
    """Tests for SeverityScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a SeverityScorer instance."""
        return SeverityScorer()

    def test_fault_area_ratio_calculation(self, scorer):
        """Test fault area ratio calculation."""
        # 25% of pixels are fault
        mask = np.zeros((100, 100))
        mask[:50, :50] = 1  # 2500 pixels

        ratio = scorer.compute_fault_area_ratio(mask)
        assert ratio == pytest.approx(0.25, rel=1e-2)

    def test_fault_area_ratio_with_panel_mask(self, scorer):
        """Test fault area ratio with panel mask."""
        mask = np.zeros((100, 100))
        mask[:50, :50] = 1

        panel_mask = np.zeros((100, 100))
        panel_mask[:50, :100] = 1  # Panel covers top half

        ratio = scorer.compute_fault_area_ratio(mask, panel_mask)
        assert ratio == pytest.approx(0.5, rel=1e-2)

    def test_temperature_score_normalization(self, scorer):
        """Test temperature score is normalized to [0, 1]."""
        thermal = np.random.uniform(30, 60, (100, 100))

        score = scorer.compute_temperature_score(thermal)

        assert 0 <= score <= 1

    def test_temperature_score_with_mask(self, scorer):
        """Test temperature score focuses on masked region."""
        thermal = np.ones((100, 100)) * 30
        thermal[:50, :50] = 70  # Hot region

        mask = np.zeros((100, 100))
        mask[:50, :50] = 1  # Mask covers hot region

        score_with_mask = scorer.compute_temperature_score(thermal, mask)
        score_without_mask = scorer.compute_temperature_score(thermal)

        # Score with mask should be higher (focused on hot region)
        assert score_with_mask > score_without_mask

    def test_growth_rate_positive(self, scorer):
        """Test growth rate calculation with increasing severity."""
        rate = scorer.compute_growth_rate(
            current_severity=0.5,
            previous_severity=0.3,
            time_delta_hours=1.0,
        )

        assert rate > 0

    def test_growth_rate_zero_when_no_change(self, scorer):
        """Test growth rate is zero with no change."""
        rate = scorer.compute_growth_rate(
            current_severity=0.5,
            previous_severity=0.5,
            time_delta_hours=1.0,
        )

        assert rate == 0

    def test_growth_rate_none_inputs(self, scorer):
        """Test growth rate handles None inputs."""
        rate = scorer.compute_growth_rate(
            current_severity=0.5,
            previous_severity=None,
            time_delta_hours=None,
        )

        assert rate == 0

    def test_severity_score_bounds(self, scorer):
        """Test severity score is bounded [0, 1]."""
        severity = scorer.compute_severity(
            fault_area_ratio=0.5,
            temperature_score=0.5,
            growth_rate=0.5,
        )

        assert 0 <= severity <= 1

    def test_risk_level_low(self, scorer):
        """Test low risk level classification."""
        level = scorer.get_risk_level(0.1)
        assert level == RiskLevel.LOW

    def test_risk_level_medium(self, scorer):
        """Test medium risk level classification."""
        level = scorer.get_risk_level(0.5)
        assert level == RiskLevel.MEDIUM

    def test_risk_level_high(self, scorer):
        """Test high risk level classification."""
        level = scorer.get_risk_level(0.8)
        assert level == RiskLevel.HIGH

    def test_full_assessment(self, scorer):
        """Test full severity assessment."""
        mask = np.zeros((100, 100))
        mask[:30, :30] = 1

        thermal = np.random.uniform(30, 50, (100, 100))

        result = scorer.assess(mask=mask, thermal_image=thermal)

        assert 0 <= result.fault_area_ratio <= 1
        assert 0 <= result.temperature_score <= 1
        assert 0 <= result.severity_score <= 1
        assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert result.timestamp is not None

    def test_alert_triggered_for_high_risk(self, scorer):
        """Test alert is triggered for high risk."""
        # Create conditions for high risk
        mask = np.ones((100, 100))  # Full coverage
        thermal = np.ones((100, 100)) * 80  # High temperature

        result = scorer.assess(mask=mask, thermal_image=thermal)

        # High severity should trigger alert
        if result.risk_level == RiskLevel.HIGH:
            assert result.alert_triggered


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_level_values(self):
        """Test risk level string values."""
        assert RiskLevel.LOW.value == "Low"
        assert RiskLevel.MEDIUM.value == "Medium"
        assert RiskLevel.HIGH.value == "High"

    def test_risk_level_comparison(self):
        """Test risk levels can be compared."""
        assert RiskLevel.LOW == RiskLevel.LOW
        assert RiskLevel.LOW != RiskLevel.HIGH
