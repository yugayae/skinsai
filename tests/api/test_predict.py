"""Integration tests for the inference prediction endpoint."""

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.db.database import SessionLocal
from backend.app.db.models import PredictionAuditLog
from backend.app.main import app

SAMPLE_IMG_PATH = Path("data/raw/pad_ufes_20/ISIC_0076337.jpg")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_image_bytes():
    assert SAMPLE_IMG_PATH.exists(), f"Sample test image missing at {SAMPLE_IMG_PATH}"
    with open(SAMPLE_IMG_PATH, "rb") as f:
        return f.read()


def test_predict_valid_lesion_image(client: TestClient, real_image_bytes: bytes):
    """Test successful image inference returns structured classification response."""
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("lesion.jpg", real_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_class_id"] in ["bcc", "scc", "ack", "sek", "mel", "nev"]
    assert data["predicted_class_name_en"] is not None
    assert data["predicted_class_name_ru"] is not None
    assert data["predicted_class_name_ko"] is not None
    assert 0.0 < data["confidence"] <= 1.0
    assert len(data["class_probabilities"]) == 6
    assert "disclaimer_ru" in data
    assert "disclaimer_en" in data
    assert "disclaimer_ko" in data
    assert data["latency_ms"] > 0.0


def test_predict_with_glasgow_anamnesis_form(client: TestClient, real_image_bytes: bytes):
    """Test multi-modal Bayesian prior adjustment via Glasgow checklist form parameters."""
    response = client.post(
        "/v1/predictions",
        data={
            "profile": "smartphone_clinical",
            "change_in_size": "true",
            "irregular_shape": "true",
            "irregular_color": "true",
        },
        files={"file": ("lesion.jpg", real_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["anamnesis_evaluation"] is not None
    assert data["anamnesis_evaluation"]["score"] == 6
    assert data["anamnesis_evaluation"]["is_high_risk"] is True
    assert data["referral_recommended"] is True
    assert "Glasgow score (6/10)" in data["referral_reason"]


def test_predict_with_glasgow_json(client: TestClient, real_image_bytes: bytes):
    """Test Glasgow anamnesis passed as JSON string payload."""
    anamnesis_json = '{"change_in_size": true, "diameter_ge_7mm": true}'
    response = client.post(
        "/v1/predictions",
        data={
            "profile": "smartphone_clinical",
            "anamnesis": anamnesis_json,
        },
        files={"file": ("lesion.jpg", real_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["anamnesis_evaluation"] is not None
    # 2 (change_in_size) + 1 (diameter_ge_7mm) = 3 (NICE threshold)
    assert data["anamnesis_evaluation"]["score"] == 3
    assert data["anamnesis_evaluation"]["is_high_risk"] is True
    assert data["referral_recommended"] is True


def test_predict_quality_rejection_blurry(client: TestClient):
    """Test image quality filter rejects flat / blurry images."""
    # Create smooth solid image
    img = Image.new("RGB", (160, 160), color=(150, 150, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("flat.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "excessive_blur" in data["rejection_reason"]


def test_predict_rejection_invalid_magic_bytes(client: TestClient):
    """Test non-image payload rejection."""
    fake_payload = b"NOT_A_VALID_IMAGE_HEADER_PADDING_BYTES_123456"
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("fake.jpg", fake_payload, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "Unsupported or corrupt" in data["rejection_reason"]


def test_predict_unknown_profile_error(client: TestClient, real_image_bytes: bytes):
    """Test requesting unsupported profile returns 400 Bad Request."""
    response = client.post(
        "/v1/predictions",
        data={"profile": "non_existent_profile"},
        files={"file": ("lesion.jpg", real_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "not active or supported" in response.json()["detail"]


def test_audit_log_written_to_database(client: TestClient, real_image_bytes: bytes):
    """Verify inference request creates an audit record in the database."""
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("audit_test.jpg", real_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    req_id = response.json()["request_id"]

    db = SessionLocal()
    try:
        record = db.query(PredictionAuditLog).filter_by(request_id=req_id).first()
        assert record is not None
        assert record.profile == "smartphone_clinical"
        assert record.status == "success"
        assert record.client_hash is not None
        assert record.predicted_class is not None
        assert record.confidence > 0.0
    finally:
        db.close()
