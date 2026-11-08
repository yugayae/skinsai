"""Unit and integration tests for service health probes."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_health_probe(client: TestClient):
    """Test GET /health returns 200, valid engine, and active profiles."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "0.1.0" in data["version"]
    assert "smartphone_clinical" in data["loaded_profiles"]
    assert data["execution_engine"] == "onnx"
    assert data["uptime_seconds"] >= 0.0


def test_v1_health_probe(client: TestClient):
    """Test GET /v1/health returns matching payload."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "smartphone_clinical" in data["loaded_profiles"]
