"""Unit tests for production OnnxPredictor engine."""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from ml.inference.predictor import OnnxPredictor


def make_sharp_jpeg(width: int = 224, height: int = 224) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_blurry_jpeg(width: int = 224, height: int = 224) -> bytes:
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_onnx_predictor_with_bundle():
    bundle_dir = Path("artifacts/models/smartphone_clinical")
    if not (bundle_dir / "model.onnx").exists():
        pytest.skip("Model bundle not trained yet")

    predictor = OnnxPredictor(bundle_dir=bundle_dir)
    assert predictor.profile == "smartphone_clinical"
    assert len(predictor.classes) == 6

    # Test valid image
    raw = make_sharp_jpeg()
    result = predictor.predict_bytes(raw)

    assert result.status in ("success", "abstained")
    assert result.latency_ms > 0
    assert len(result.class_probabilities) == 6

    prob_sum = sum(result.class_probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-2

    # Check risk and referral flags
    assert 0.0 <= result.melanoma_risk_score <= 1.0
    if result.status == "success":
        assert result.predicted_class_id in predictor.classes
        assert result.predicted_class_name_en is not None
        assert result.predicted_class_name_ru is not None
        assert result.predicted_class_name_ko is not None


def test_onnx_predictor_rejects_corrupted_data():
    bundle_dir = Path("artifacts/models/smartphone_clinical")
    if not (bundle_dir / "model.onnx").exists():
        pytest.skip("Model bundle not trained yet")

    predictor = OnnxPredictor(bundle_dir=bundle_dir)
    bad_bytes = b"CORRUPTED_NON_IMAGE_DATA_12345"
    result = predictor.predict_bytes(bad_bytes)

    assert result.status == "rejected_quality"
    assert result.rejection_reason == "invalid_magic_bytes"
    assert result.predicted_class_id is None


def test_onnx_predictor_rejects_blurry_image():
    bundle_dir = Path("artifacts/models/smartphone_clinical")
    if not (bundle_dir / "model.onnx").exists():
        pytest.skip("Model bundle not trained yet")

    predictor = OnnxPredictor(bundle_dir=bundle_dir)
    blurry_bytes = make_blurry_jpeg()
    result = predictor.predict_bytes(blurry_bytes)

    assert result.status == "rejected_quality"
    assert "excessive_blur" in (result.rejection_reason or "")
