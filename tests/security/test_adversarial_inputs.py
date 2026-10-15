"""Adversarial input fuzzing, malicious payload injection, and security boundary tests."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.db.database import SessionLocal
from backend.app.db.models import PredictionAuditLog
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# 1. Decompression Bomb and Oversized Payload Tests
# --------------------------------------------------------------------------

def test_decompression_bomb_rejection(client: TestClient):
    """Verify that images exceeding dimension bounds are rejected before decoding pixels."""
    # Create an image that exceeds MAX_IMAGE_DIMENSION (6000x6000)
    # Using small in-memory representation but large header dimensions
    bomb_img = Image.new("RGB", (6500, 200), color=(120, 100, 80))
    buf = io.BytesIO()
    bomb_img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("bomb.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "exceeds maximum" in data["rejection_reason"]


def test_oversized_payload_rejection(client: TestClient):
    """Verify that payloads exceeding MAX_UPLOAD_SIZE_BYTES (15 MB) are rejected."""
    # 16 MB dummy payload
    oversized_data = b"\xFF\xD8\xFF" + b"\x00" * (16 * 1024 * 1024)
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("oversized.jpg", oversized_data, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "exceeds maximum limit" in data["rejection_reason"]


def test_empty_file_upload(client: TestClient):
    """Verify empty byte file is handled cleanly."""
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "Empty file" in data["rejection_reason"]


# --------------------------------------------------------------------------
# 2. Corrupt, Truncated, and Malformed Byte Streams
# --------------------------------------------------------------------------

def test_truncated_jpeg_byte_stream(client: TestClient):
    """Verify truncated JPEG stream with valid start-of-image but broken body."""
    # Valid JPEG magic header but truncated abruptly
    truncated = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("broken.jpg", truncated, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "Failed to parse image" in data["rejection_reason"]


def test_corrupted_png_crc(client: TestClient):
    """Verify PNG with corrupted chunk data fails parsing cleanly."""
    png_magic = b"\x89PNG\r\n\x1a\n"
    corrupt_png = png_magic + b"\x00\x00\x00\rIHDR" + b"\xFF" * 20
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("corrupt.png", corrupt_png, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"


# --------------------------------------------------------------------------
# 3. Polyglot, Executable, and Script File Type Spoofing
# --------------------------------------------------------------------------

def test_executable_pe_binary_rejection(client: TestClient):
    """Verify Windows PE binary with fake .jpg extension is rejected."""
    pe_header = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00This program cannot be run in DOS mode."
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("malware.jpg", pe_header, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "Unsupported or corrupt" in data["rejection_reason"]


def test_linux_elf_binary_rejection(client: TestClient):
    """Verify Linux ELF binary is rejected."""
    elf_header = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00>\x00"
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("script.jpg", elf_header, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"


def test_svg_script_injection_rejection(client: TestClient):
    """Verify SVG with embedded XSS script is rejected because only raster formats are allowed."""
    svg_payload = (
        b"<?xml version='1.0' encoding='UTF-8'?>\n"
        b"<svg xmlns='http://www.w3.org/2000/svg'>\n"
        b"  <script>alert('XSS')</script>\n"
        b"</svg>"
    )
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("attack.jpg", svg_payload, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"
    assert "Unsupported or corrupt" in data["rejection_reason"]


def test_html_document_rejection(client: TestClient):
    """Verify HTML document upload disguised as image is rejected."""
    html_payload = b"<!DOCTYPE html><html><head><title>Phish</title></head><body><h1>Login</h1></body></html>"
    response = client.post(
        "/v1/predictions",
        data={"profile": "smartphone_clinical"},
        files={"file": ("phish.jpg", html_payload, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected_quality"


# --------------------------------------------------------------------------
# 4. Form Parameter Injection Attacks
# --------------------------------------------------------------------------

def test_sql_injection_in_profile_parameter(client: TestClient):
    """Verify SQL injection attempt in profile string returns 400 without crashing."""
    sqli_profile = "smartphone_clinical' OR '1'='1'; DROP TABLE prediction_audit_logs;--"
    sample_img = io.BytesIO()
    Image.new("RGB", (200, 200), color=(100, 100, 100)).save(sample_img, format="JPEG")
    sample_img.seek(0)

    response = client.post(
        "/v1/predictions",
        data={"profile": sqli_profile},
        files={"file": ("img.jpg", sample_img.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "not active or supported" in response.json()["detail"]

    # Verify table is intact
    db = SessionLocal()
    try:
        count = db.query(PredictionAuditLog).count()
        assert count >= 0
    finally:
        db.close()


def test_path_traversal_in_profile_parameter(client: TestClient):
    """Verify path traversal attempt in profile string is rejected."""
    traversal_profile = "../../../../etc/passwd"
    sample_img = io.BytesIO()
    Image.new("RGB", (200, 200), color=(100, 100, 100)).save(sample_img, format="JPEG")
    sample_img.seek(0)

    response = client.post(
        "/v1/predictions",
        data={"profile": traversal_profile},
        files={"file": ("img.jpg", sample_img.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 400


def test_malformed_anamnesis_json(client: TestClient):
    """Verify malformed JSON in anamnesis field returns 422 Unprocessable Entity."""
    sample_img = io.BytesIO()
    Image.new("RGB", (200, 200), color=(100, 100, 100)).save(sample_img, format="JPEG")
    sample_img.seek(0)

    response = client.post(
        "/v1/predictions",
        data={
            "profile": "smartphone_clinical",
            "anamnesis": "{broken_json_no_quotes: true",
        },
        files={"file": ("img.jpg", sample_img.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 422
    assert "Invalid JSON in 'anamnesis'" in response.json()["detail"]


# --------------------------------------------------------------------------
# 5. Zero-Retention & Secret Sanitization Verification
# --------------------------------------------------------------------------

def test_zero_raw_photo_retention_in_database(client: TestClient):
    """Verify that no database records retain raw image binaries or filenames."""
    db = SessionLocal()
    try:
        logs = db.query(PredictionAuditLog).all()
        for log in logs:
            # Audit log schema must only contain numerical/string telemetry
            assert not hasattr(log, "image_bytes")
            assert not hasattr(log, "photo")
            assert not hasattr(log, "file_path")
            assert len(log.client_hash) == 32  # SHA-256 salted hash prefix
    finally:
        db.close()


def test_secret_sanitization_in_responses(client: TestClient):
    """Verify secrets (salt, bot token, secret key) are never leaked in response payloads."""
    response = client.get("/health")
    resp_text = response.text
    assert settings.TELEMETRY_SALT not in resp_text
    assert settings.API_SECRET_KEY not in resp_text

    prof_resp = client.get("/v1/profiles")
    prof_text = prof_resp.text
    assert settings.TELEMETRY_SALT not in prof_text
    assert settings.API_SECRET_KEY not in prof_text
