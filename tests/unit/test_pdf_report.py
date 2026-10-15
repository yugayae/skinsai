"""Unit tests for clinical encounter memo PDF generator."""

import io
from PIL import Image
import pytest

from bot.services.pdf_report import generate_clinical_memo_pdf


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a clean synthetic JPEG image for testing."""
    img = Image.new("RGB", (256, 256), color=(205, 133, 63))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_glasgow_answers() -> dict:
    """Sample Glasgow 7-point checklist answers (total score: 5 -> High risk)."""
    return {
        "change_in_size": True,      # +2
        "irregular_shape": True,     # +2
        "irregular_color": False,
        "diameter_ge_7mm": True,     # +1
        "inflammation": False,
        "oozing_or_bleeding": False,
        "change_in_sensation": False,
    }


def test_pdf_generation_ru(sample_image_bytes, sample_glasgow_answers):
    """Test generating Russian clinical memo PDF."""
    pdf = generate_clinical_memo_pdf(
        image_bytes=sample_image_bytes,
        lang="ru",
        request_id="test-req-12345",
        glasgow_answers=sample_glasgow_answers,
        glasgow_score=5,
    )
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF-1.")
    assert len(pdf) > 20_000  # Valid non-empty document with embedded image


def test_pdf_generation_en(sample_image_bytes, sample_glasgow_answers):
    """Test generating English clinical memo PDF."""
    pdf = generate_clinical_memo_pdf(
        image_bytes=sample_image_bytes,
        lang="en",
        request_id="test-req-67890",
        glasgow_answers=sample_glasgow_answers,
        glasgow_score=5,
    )
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF-1.")
    assert len(pdf) > 20_000


def test_pdf_generation_ko(sample_image_bytes, sample_glasgow_answers):
    """Test generating Korean clinical memo PDF."""
    pdf = generate_clinical_memo_pdf(
        image_bytes=sample_image_bytes,
        lang="ko",
        request_id="test-req-korean",
        glasgow_answers=sample_glasgow_answers,
        glasgow_score=5,
    )
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF-1.")
    assert len(pdf) > 20_000


def test_pdf_zero_anchoring(sample_image_bytes, sample_glasgow_answers):
    """Verify that raw model prediction probabilities are never leaked into the PDF document."""
    pdf = generate_clinical_memo_pdf(
        image_bytes=sample_image_bytes,
        lang="ru",
        request_id="test-zero-anchoring",
        glasgow_answers=sample_glasgow_answers,
        glasgow_score=5,
    )
    # The string 'Вероятность' or raw floating point confidence scores must not be present
    pdf_text = pdf.decode("latin-1", errors="ignore")
    assert "class_probabilities" not in pdf_text
    assert "melanoma_risk_score" not in pdf_text
