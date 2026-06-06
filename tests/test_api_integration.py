

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import io

# We import the app after patching dependencies
# to prevent real DB connections during tests


class TestSystemEndpoints:
    """Tests for system-level endpoints that don't need DB."""

    def setup_method(self):
        """
        Create test client with mocked database.
        Called before each test method.
        """
        # Patch db_manager to prevent real MongoDB connection
        with patch("backend.database.db_manager") as mock_db:
            mock_db.connect.return_value = None
            mock_db.ping = AsyncMock(return_value=True)
            mock_db.get_db.return_value = MagicMock()

            from backend.main import app
            self.client = TestClient(app, raise_server_exceptions=False)

    def test_root_returns_200(self):
        """Root endpoint should return 200 with service info."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "docs" in data

    def test_health_returns_alive(self):
        """Health probe should return alive status."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "uptime_s" in data

    def test_request_id_header_present(self):
        """Every response should have X-Request-ID header."""
        response = self.client.get("/health")
        assert "x-request-id" in response.headers

    def test_process_time_header_present(self):
        """Every response should have X-Process-Time header."""
        response = self.client.get("/health")
        assert "x-process-time" in response.headers

    def test_metrics_endpoint(self):
        """Metrics endpoint should return environment info."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "environment" in data


class TestErrorHandling:
    """Tests for global error handler behavior."""

    def setup_method(self):
        with patch("backend.database.db_manager") as mock_db:
            mock_db.connect.return_value = None
            mock_db.ping = AsyncMock(return_value=True)
            mock_db.get_db.return_value = MagicMock()

            from backend.main import app
            self.client = TestClient(app, raise_server_exceptions=False)

    def test_404_returns_structured_json(self):
        """404 errors should return JSON, not HTML."""
        response = self.client.get("/api/v1/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        # Should be structured JSON, not HTML
        assert isinstance(data, dict)

    def test_cors_headers_present(self):
        """CORS preflight should return appropriate headers."""
        response = self.client.options(
            "/api/v1/jobs/",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            }
        )
        # OPTIONS preflight should succeed
        assert response.status_code in (200, 204)


class TestPaginationDependency:
    """Tests for the pagination dependency."""

    def test_default_pagination(self):
        from backend.dependencies.common import get_pagination, PaginationParams
        params = get_pagination(skip=0, limit=20)
        assert params.skip == 0
        assert params.limit == 20

    def test_custom_pagination(self):
        from backend.dependencies.common import get_pagination
        params = get_pagination(skip=40, limit=10)
        assert params.skip == 40
        assert params.limit == 10

    def test_pipeline_stage_validator_passes(self):
        from backend.dependencies.common import validate_pipeline_stage
        # Should not raise — status meets requirement
        validate_pipeline_stage("nlp_processed", "nlp_processed", "test-id")
        validate_pipeline_stage("nlp_processed", "skills_extracted", "test-id")

    def test_pipeline_stage_validator_fails(self):
        from backend.dependencies.common import validate_pipeline_stage
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_pipeline_stage("skills_extracted", "uploaded", "test-id")
        assert exc_info.value.status_code == 422

    def test_http_status_names(self):
        from backend.middleware.error_handler import http_status_name
        assert http_status_name(404) == "Not Found"
        assert http_status_name(422) == "Unprocessable Entity"
        assert http_status_name(500) == "Internal Server Error"
        assert "HTTP Error" in http_status_name(999)  # Unknown code


class TestMiddlewareOrder:
    """Tests that middleware is applied in correct order."""

    def test_request_id_generated_when_not_provided(self):
        with patch("backend.database.db_manager") as mock_db:
            mock_db.connect.return_value = None
            mock_db.ping = AsyncMock(return_value=True)
            mock_db.get_db.return_value = MagicMock()

            from backend.main import app
            client = TestClient(app)

            response = client.get("/health")
            request_id = response.headers.get("x-request-id")
            assert request_id is not None
            assert len(request_id) == 36  # UUID4 length

    def test_client_request_id_preserved(self):
        """Client-provided X-Request-ID should be echoed back."""
        with patch("backend.database.db_manager") as mock_db:
            mock_db.connect.return_value = None
            mock_db.ping = AsyncMock(return_value=True)
            mock_db.get_db.return_value = MagicMock()

            from backend.main import app
            client = TestClient(app)

            custom_id = "my-trace-id-12345"
            response  = client.get(
                "/health",
                headers={"X-Request-ID": custom_id}
            )
            assert response.headers.get("x-request-id") == custom_id
