"""Unit tests for production OnnxPredictor engine."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


def test_onnx_predictor_abcde_and_ood_gate():
    """Verify DS-5 ABCDE morphometry and FS-5 CIE L*a*b* OOD non-skin gate."""
    bundle_dir = Path("artifacts/models/smartphone_clinical")
    if not (bundle_dir / "model.onnx").exists():
        pytest.skip("Model bundle not trained yet")

    predictor = OnnxPredictor(bundle_dir=bundle_dir)
    # Random RGB noise is sharp enough to pass blur filter, but fails CIE LAB skin OOD gate
    raw = make_sharp_jpeg()
    result = predictor.predict_bytes(raw)

    assert result.abcde_metrics is not None
    assert "asymmetry_shape" in result.abcde_metrics
    assert "border_irregularity" in result.abcde_metrics
    assert "color_entropy" in result.abcde_metrics
    assert result.ood_metrics is not None
    assert result.ood_metrics["is_ood_non_skin"] is True
    assert result.status == "abstained"
    assert "out_of_distribution_non_skin" in (result.rejection_reason or "")
    assert result.triage is not None
    assert result.triage["triage_zone"] == "yellow"
    assert result.similar_biopsy_cases is not None
    assert len(result.similar_biopsy_cases) == 3
    assert "subtype_id" in result.similar_biopsy_cases[0]


def test_onnx_predictor_fusion_head_and_atlas():
    """Verify exported fusion_head.onnx (896D+7D) and Biopsy Atlas k-NN retrieval."""
    bundle_dir = Path("artifacts/models/smartphone_clinical")
    if not (bundle_dir / "fusion_head.onnx").exists():
        pytest.skip("Fusion head not exported yet")

    predictor = OnnxPredictor(bundle_dir=bundle_dir)
    assert predictor.fusion_session is not None
    assert predictor.biopsy_atlas is not None

    dummy_vis = np.random.randn(896).astype(np.float32)
    t_probs, s_probs = predictor.predict_fusion_embeddings(dummy_vis)
    assert len(t_probs) == 2
    assert len(s_probs) == 6
    assert abs(float(np.sum(s_probs)) - 1.0) < 1e-4

    twins = predictor.retrieve_similar_biopsy_cases(s_probs, vis_896d=dummy_vis, k=3)
    assert len(twins) == 3
    assert twins[0]["rank"] == 1
    assert "cosine_similarity" in twins[0]


