"""End-to-end integration tests simulating the full client -> API -> inference -> telemetry loop."""

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


def test_end_to_end_bot_api_database_pipeline(client: TestClient):
    """Simulate complete user lifecycle: image upload + Glasgow anamnesis + DB audit."""
    assert SAMPLE_IMG_PATH.exists(), f"Image missing: {SAMPLE_IMG_PATH}"
    with open(SAMPLE_IMG_PATH, "rb") as f:
        img_bytes = f.read()

    # Step 1: Client sends image with 3 major Glasgow flags
    response = client.post(
        "/v1/predictions",
        data={
            "profile": "smartphone_clinical",
            "change_in_size": "true",
            "irregular_shape": "true",
            "irregular_color": "true",
            "diameter_ge_7mm": "true",
        },
        files={"file": ("lesion.jpg", img_bytes, "image/jpeg")},
        headers={"X-User-ID": "telegram_user_999888"},
    )
    assert response.status_code == 200
    data = response.json()

    # Step 2: Validate inference outcome
    assert data["status"] == "success"
    assert data["predicted_class_id"] in ["bcc", "scc", "ack", "sek", "mel", "nev"]
    assert data["anamnesis_evaluation"]["score"] == 7  # 2 + 2 + 2 + 1 = 7
    assert data["anamnesis_evaluation"]["is_high_risk"] is True
    assert data["referral_recommended"] is True
    assert "Glasgow score" in data["referral_reason"]

    # Step 3: Verify Disclaimers
    assert "не заменяет" in data["disclaimer_ru"]
    assert "does not replace" in data["disclaimer_en"]

    # Step 4: Verify Database Audit Logging
    req_id = data["request_id"]
    db = SessionLocal()
    try:
        record = db.query(PredictionAuditLog).filter_by(request_id=req_id).first()
        assert record is not None
        assert record.profile == "smartphone_clinical"
        assert record.status == "success"
        assert record.has_anamnesis is True
        assert record.glasgow_score == 7
        assert record.referral_recommended is True
        assert record.latency_ms > 0
        assert len(record.client_hash) == 32
    finally:
        db.close()


def test_end_to_end_blurry_rejection_flow(client: TestClient):
    """Simulate user uploading a blurry / blank image."""
    img = Image.new("RGB", (150, 150), color=(180, 120, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("blurry.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "rejected_quality"
    assert "excessive_blur" in data["rejection_reason"]
    assert data["predicted_class_id"] is None
    assert data["confidence"] == 0.0

    # Ensure rejection is audited
    db = SessionLocal()
    try:
        record = db.query(PredictionAuditLog).filter_by(request_id=data["request_id"]).first()
        assert record is not None
        assert record.status == "rejected_quality"
        assert "excessive_blur" in record.rejection_reason
    finally:
        db.close()
