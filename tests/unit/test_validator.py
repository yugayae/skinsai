"""Unit tests for ImageValidator and QualityMetrics."""

import io
from PIL import Image
import numpy as np

from ml.datasets.validator import ImageValidator


def make_test_jpeg(width: int = 200, height: int = 200, is_blurry: bool = False) -> bytes:
    if is_blurry:
        # Uniform solid gray (zero variance, highly blurry)
        arr = np.full((height, width, 3), 128, dtype=np.uint8)
    else:
        # Sharp high-frequency checkerboard / noise pattern
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_validator_valid_image():
    raw = make_test_jpeg(width=224, height=224, is_blurry=False)
    validator = ImageValidator(min_dimension=128, min_blur_score=50.0)
    metrics, pil_img = validator.validate_bytes(raw)

    assert metrics.is_valid is True
    assert metrics.rejection_reason is None
    assert metrics.width == 224
    assert metrics.height == 224
    assert metrics.magic_format == "jpeg"
    assert metrics.blur_score > 50.0
    assert pil_img is not None


def test_validator_rejects_corrupted_magic_bytes():
    bad_bytes = b"NOT_AN_IMAGE_HEADER_AT_ALL_JUST_RANDOM_TEXT"
    validator = ImageValidator()
    metrics, pil_img = validator.validate_bytes(bad_bytes)

    assert metrics.is_valid is False
    assert metrics.rejection_reason == "invalid_magic_bytes"
    assert pil_img is None


def test_validator_rejects_small_dimension():
    # 64x64 is below min_dimension=128
    raw = make_test_jpeg(width=64, height=64, is_blurry=False)
    validator = ImageValidator(min_dimension=128)
    metrics, _ = validator.validate_bytes(raw)

    assert metrics.is_valid is False
    assert "dimension_too_small" in (metrics.rejection_reason or "")


def test_validator_rejects_blurry_image():
    # Solid gray image has zero Laplacian variance
    raw = make_test_jpeg(width=224, height=224, is_blurry=True)
    validator = ImageValidator(min_dimension=128, min_blur_score=60.0)
    metrics, _ = validator.validate_bytes(raw)

    assert metrics.is_valid is False
    assert "excessive_blur" in (metrics.rejection_reason or "")
    assert metrics.blur_score < 10.0
