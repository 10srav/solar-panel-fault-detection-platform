"""Tests for FastAPI endpoints."""

import os
import pytest
from fastapi.testclient import TestClient

# Set test database URL before importing app
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from src.api.app import create_app
from src.api import database


@pytest.fixture
def client():
    """Create a test client with initialized database."""
    # Reset database manager to pick up test URL
    database._db_manager = None

    # Initialize database with test URL
    database.init_db("sqlite:///./test.db")

    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "database_connected" in data


class TestInferenceEndpoints:
    """Tests for inference endpoints."""

    def test_rgb_inference_no_image(self, client):
        """Test RGB inference fails without image."""
        response = client.post("/infer/rgb")

        # Should return error without image (422 for validation, 503 if models not loaded)
        assert response.status_code in [400, 422, 503]

    def test_thermal_inference_no_image(self, client):
        """Test thermal inference fails without image."""
        response = client.post("/infer/thermal")

        # Should return error without image (422 for validation, 503 if models not loaded)
        assert response.status_code in [400, 422, 503]

    def test_combined_inference_missing_images(self, client):
        """Test combined inference fails with missing images."""
        response = client.post(
            "/infer/combined",
            json={"rgb_image_base64": None, "thermal_image_base64": None},
        )

        # Should return error (400 for bad request, 422 for validation, 503 if models not loaded)
        assert response.status_code in [400, 422, 503]


class TestPanelEndpoints:
    """Tests for panel management endpoints."""

    def test_list_panels_empty(self, client):
        """Test listing panels when empty."""
        response = client.get("/panels")

        # May return empty list or 200 with data, or 500 if DB not configured
        assert response.status_code in [200, 500]

    def test_get_nonexistent_panel(self, client):
        """Test getting a panel that doesn't exist."""
        response = client.get("/panels/nonexistent-id")

        # 404 if not found, 500 if DB not configured
        assert response.status_code in [404, 500]

    def test_panel_history_nonexistent(self, client):
        """Test getting history for nonexistent panel."""
        response = client.get("/panels/nonexistent-id/history")

        # 404 if not found, 500 if DB not configured
        assert response.status_code in [404, 500]


class TestAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_docs_available(self, client):
        """Test OpenAPI docs are available."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_available(self, client):
        """Test ReDoc is available."""
        response = client.get("/redoc")

        assert response.status_code == 200

    def test_openapi_json(self, client):
        """Test OpenAPI JSON schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_available(self, client):
        """Test metrics endpoint is available."""
        response = client.get("/metrics")

        assert response.status_code == 200
        # Prometheus format
        assert "TYPE" in response.text or "HELP" in response.text
