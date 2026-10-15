"""Unit tests for ONNX export and parity verification."""

from pathlib import Path
import yaml

from ml.export.exporter import ModelExporter
from ml.models.classifier import SkinClassifier


def test_model_export_and_parity(tmp_path: Path):
    # Lightweight test model
    model = SkinClassifier(
        backbone_name="efficientnet_b0",
        num_classes=6,
        pretrained=False,
    )
    class_names = ["bcc", "scc", "ack", "sek", "mel", "nev"]
    exporter = ModelExporter(model=model, class_names=class_names)

    bundle_dir = tmp_path / "bundle"
    manifest = exporter.create_artifact_bundle(
        bundle_dir=bundle_dir,
        profile_name="smartphone_clinical",
        metrics={"macro_f1": 0.85, "accuracy": 0.88},
        temperature=1.2,
    )

    onnx_file = bundle_dir / "model.onnx"
    manifest_file = bundle_dir / "manifest.yaml"

    assert onnx_file.exists()
    assert manifest_file.exists()
    assert manifest["classes"] == class_names
    assert manifest["onnx_parity_max_diff"] < 1e-4

    # Verify manifest content on disk
    with open(manifest_file, "r", encoding="utf-8") as f:
        loaded_yaml = yaml.safe_load(f)
    assert loaded_yaml["profile"] == "smartphone_clinical"
    assert loaded_yaml["sha256"] == manifest["sha256"]
