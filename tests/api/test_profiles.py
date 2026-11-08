"""Integration tests for model profile registry discovery."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_list_profiles_endpoint(client: TestClient):
    """Test GET /v1/profiles returns smartphone_clinical and dermoscopy metadata."""
    response = client.get("/v1/profiles")
    assert response.status_code == 200
    data = response.json()
    assert "profiles" in data
    profiles_by_id = {p["profile_id"]: p for p in data["profiles"]}

    # Verify smartphone_clinical profile
    assert "smartphone_clinical" in profiles_by_id
    sc = profiles_by_id["smartphone_clinical"]
    assert sc["is_active"] is True
    assert sc["license_scope"] == "commercial_ok"
    assert set(sc["classes"]) == {"bcc", "scc", "ack", "sek", "mel", "nev"}
    assert sc["tau_abstain"] == 0.40
    assert sc["tau_referral_melanoma"] == 0.20
    assert "bcc" in sc["class_names_en"]
    assert "bcc" in sc["class_names_ru"]

    # Verify dermoscopy profile entry exists
    assert "dermoscopy" in profiles_by_id
