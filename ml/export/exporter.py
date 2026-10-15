"""ONNX export and numerical parity verification engine for SkinAI."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

from ml.models.classifier import SkinClassifier
from ml.models.dataset import CANONICAL_MEAN, CANONICAL_STD, DEFAULT_INPUT_SIZE


class ModelExporter:
    """Exports PyTorch model checkpoints to ONNX with strict parity testing."""

    def __init__(
        self,
        model: SkinClassifier,
        class_names: List[str],
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        mean: List[float] = CANONICAL_MEAN,
        std: List[float] = CANONICAL_STD,
    ):
        self.model = model.eval().cpu()
        self.class_names = class_names
        self.input_size = input_size
        self.mean = mean
        self.std = std

    def export_onnx(
        self,
        output_path: Path,
        opset_version: int = 17,
    ) -> Path:
        """Export PyTorch model to ONNX format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(1, 3, self.input_size[0], self.input_size[1], dtype=torch.float32)

        # Dynamic batch size
        dynamic_axes = {
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        # Check model validity
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        return output_path

    def verify_parity(
        self,
        onnx_path: Path,
        num_samples: int = 5,
        tolerance: float = 1e-4,
    ) -> float:
        """Verify numerical parity between PyTorch and ONNX Runtime outputs."""
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        max_diff = 0.0
        for _ in range(num_samples):
            dummy = np.random.randn(1, 3, self.input_size[0], self.input_size[1]).astype(np.float32)

            # PyTorch forward
            with torch.no_grad():
                pt_out = self.model(torch.from_numpy(dummy)).numpy()

            # ONNX forward
            ort_out = session.run(None, {input_name: dummy})[0]

            diff = float(np.max(np.abs(pt_out - ort_out)))
            if diff > max_diff:
                max_diff = diff

        if max_diff > tolerance:
            raise ValueError(
                f"ONNX parity check failed! Max diff {max_diff:.6f} exceeds tolerance {tolerance:.6f}"
            )

        return max_diff

    def create_artifact_bundle(
        self,
        bundle_dir: Path,
        profile_name: str,
        metrics: Dict[str, Any],
        temperature: float = 1.0,
        tau_abstain: float = 0.40,
        tau_referral_melanoma: float = 0.20,
    ) -> Dict[str, Any]:
        """Produce versioned model bundle with weights, ONNX, and manifest.yaml."""
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = bundle_dir / "model.onnx"
        self.export_onnx(onnx_path)
        max_diff = self.verify_parity(onnx_path)

        # Compute SHA-256
        hasher = hashlib.sha256()
        hasher.update(onnx_path.read_bytes())
        onnx_sha256 = hasher.hexdigest()

        manifest = {
            "profile": profile_name,
            "version": "1.0.0",
            "backbone": self.model.backbone_name,
            "classes": self.class_names,
            "input_size": [3, self.input_size[0], self.input_size[1]],
            "normalization": {
                "mean": self.mean,
                "std": self.std,
            },
            "temperature": float(temperature),
            "thresholds": {
                "tau_abstain": float(tau_abstain),
                "tau_referral_melanoma": float(tau_referral_melanoma),
            },
            "onnx_parity_max_diff": float(max_diff),
            "sha256": onnx_sha256,
            "metrics": metrics,
        }

        manifest_path = bundle_dir / "manifest.yaml"
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest, f, sort_keys=False)

        return manifest
