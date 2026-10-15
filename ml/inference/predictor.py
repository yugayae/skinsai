"""Production ONNX Runtime inference engine with quality validation and abstention."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image
from pydantic import BaseModel, Field
import yaml

from ml.anamnesis.questionnaire import (
    AnamnesisEngine,
    GlasgowChecklistInput,
)
from ml.datasets.validator import ImageValidator

PredictionStatus = Literal["success", "abstained", "rejected_quality"]

# Russian, English, and Korean class display names
CLASS_DISPLAY_NAMES = {
    "smartphone_clinical": {
        "bcc": {"en": "Basal cell carcinoma", "ru": "Базальноклеточная карцинома", "ko": "기저세포암"},
        "scc": {"en": "Squamous cell carcinoma", "ru": "Плоскоклеточная карцинома", "ko": "편평세포암"},
        "ack": {"en": "Actinic keratosis", "ru": "Актинический кератоз", "ko": "광선각화증"},
        "sek": {"en": "Seborrheic keratosis", "ru": "Себорейный кератоз", "ko": "지루각화증"},
        "mel": {"en": "Melanoma", "ru": "Меланома", "ko": "흑색종"},
        "nev": {"en": "Melanocytic nevus", "ru": "Меланоцитарный невус", "ko": "멜라닌세포 모반"},
    },
    "dermoscopy": {
        "mel": {"en": "Melanoma", "ru": "Меланома", "ko": "흑색종"},
        "nv": {"en": "Melanocytic nevus", "ru": "Меланоцитарный невус", "ko": "멜라닌세포 모반"},
        "bcc": {"en": "Basal cell carcinoma", "ru": "Базальноклеточная карцинома", "ko": "기저세포암"},
        "akiec": {"en": "Actinic keratosis", "ru": "Актинический кератоз", "ko": "광선각화증"},
        "bkl": {"en": "Benign keratosis", "ru": "Доброкачественный кератоз", "ko": "양성 각화증"},
        "df": {"en": "Dermatofibroma", "ru": "Дерматофиброма", "ko": "피부섬유종"},
        "vasc": {"en": "Vascular lesion", "ru": "Сосудистое поражение", "ko": "혈관 병변"},
    },
}


class PredictionResult(BaseModel):
    """Structured inference response returned by OnnxPredictor."""

    profile: str
    status: PredictionStatus
    predicted_class_id: Optional[str] = None
    predicted_class_name_en: Optional[str] = None
    predicted_class_name_ru: Optional[str] = None
    predicted_class_name_ko: Optional[str] = None
    confidence: float = 0.0
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    melanoma_risk_score: float = 0.0
    referral_recommended: bool = False
    rejection_reason: Optional[str] = None
    latency_ms: float = 0.0
    quality_metrics: Optional[Dict[str, Any]] = None
    anamnesis_evaluation: Optional[Dict[str, Any]] = None
    triage: Optional[Dict[str, Any]] = None


class OnnxPredictor:
    """Canonical inference pipeline executing quality gating, preprocessing, and ONNX prediction."""

    def __init__(self, bundle_dir: Union[str, Path]):
        self.bundle_dir = Path(bundle_dir)
        manifest_path = self.bundle_dir / "manifest.yaml"
        onnx_path = self.bundle_dir / "model.onnx"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

        self.profile = self.manifest["profile"]
        self.classes: List[str] = self.manifest["classes"]
        self.input_size: Tuple[int, int] = (
            self.manifest["input_size"][1],
            self.manifest["input_size"][2],
        )
        self.mean = np.array(self.manifest["normalization"]["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(self.manifest["normalization"]["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.temperature = float(self.manifest.get("temperature", 1.0))
        self.tau_abstain = float(self.manifest.get("thresholds", {}).get("tau_abstain", 0.40))
        self.tau_referral_mel = float(self.manifest.get("thresholds", {}).get("tau_referral_melanoma", 0.20))

        # Setup ONNX Runtime session on CPU for VPS / serverless portability
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Image quality validator
        self.validator = ImageValidator(min_dimension=128, min_blur_score=8.0)

    def preprocess_image(self, pil_img: Image.Image) -> np.ndarray:
        """Apply canonical resize and normalization to produce (1, 3, 224, 224) float32 array."""
        resized = pil_img.resize(self.input_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0  # shape (H, W, 3)
        # Transpose to (3, H, W) and add batch dim -> (1, 3, H, W)
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        # Normalize
        normalized = (tensor - self.mean) / self.std
        return normalized.astype(np.float32)

    def predict_bytes(
        self,
        raw_bytes: bytes,
        anamnesis: Optional[GlasgowChecklistInput] = None,
    ) -> PredictionResult:
        """Run complete inference pipeline on untrusted raw image bytes with optional clinical anamnesis."""
        t0 = time.time()

        # 1. Quality & Safety Validation
        metrics, pil_img = self.validator.validate_bytes(raw_bytes)
        if not metrics.is_valid or pil_img is None:
            latency = (time.time() - t0) * 1000.0
            return PredictionResult(
                profile=self.profile,
                status="rejected_quality",
                rejection_reason=metrics.rejection_reason,
                latency_ms=round(latency, 2),
                quality_metrics=metrics.model_dump(),
            )

        # 2. Canonical Preprocessing
        input_tensor = self.preprocess_image(pil_img)

        # 3. ONNX Runtime Forward Pass
        ort_outputs = self.session.run(None, {self.input_name: input_tensor})
        raw_logits = ort_outputs[0][0]  # shape (num_classes,)

        # 4. Temperature Scaling Calibration
        scaled_logits = raw_logits / max(0.01, self.temperature)

        # 5. Stable Softmax Computation
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Map to class dictionary
        prob_dict = {
            cls_name: round(float(probabilities[i]), 4)
            for i, cls_name in enumerate(self.classes)
        }

        # 6. Decoupled Vision Analysis & Clinical Triage
        top_idx = int(np.argmax(probabilities))
        top_prob = float(probabilities[top_idx])
        top_class = self.classes[top_idx]
        mel_idx = self.classes.index("mel") if "mel" in self.classes else None
        mel_risk = float(probabilities[mel_idx]) if mel_idx is not None else 0.0

        # Compute Shannon entropy of calibrated prediction
        probs_clipped = np.clip(probabilities, 1e-12, 1.0)
        entropy = float(-np.sum(probs_clipped * np.log(probs_clipped)))

        # Evaluate anamnesis independently
        anamnesis_eval = None
        if anamnesis is not None:
            has_answers = any(
                v is not None
                for v in [
                    anamnesis.change_in_size,
                    anamnesis.irregular_shape,
                    anamnesis.irregular_color,
                    anamnesis.diameter_ge_7mm,
                    anamnesis.inflammation,
                    anamnesis.oozing_or_bleeding,
                    anamnesis.change_in_sensation,
                ]
            )
            if has_answers:
                anamnesis_eval = AnamnesisEngine.evaluate_glasgow(anamnesis).model_dump()

        # Clinical Triage Decision without multiplying probabilities
        triage_decision = AnamnesisEngine.determine_triage(
            image_probabilities=prob_dict,
            anamnesis=anamnesis,
            is_blurry=False,
            entropy=entropy,
        )
        referral = triage_decision.referral_recommended

        # Check Abstention Engine (uncertainty rejection)
        if top_prob < self.tau_abstain:
            latency = (time.time() - t0) * 1000.0
            return PredictionResult(
                profile=self.profile,
                status="abstained",
                confidence=round(top_prob, 4),
                class_probabilities=prob_dict,
                melanoma_risk_score=round(mel_risk, 4),
                referral_recommended=referral,
                rejection_reason=f"confidence_below_threshold ({top_prob:.2f} < {self.tau_abstain:.2f})",
                latency_ms=round(latency, 2),
                quality_metrics=metrics.model_dump(),
                anamnesis_evaluation=anamnesis_eval,
                triage=triage_decision.model_dump(),
            )

        # Success
        display_names = CLASS_DISPLAY_NAMES.get(self.profile, {}).get(top_class, {})
        name_en = display_names.get("en", top_class)
        name_ru = display_names.get("ru", top_class)
        name_ko = display_names.get("ko", top_class)
        latency = (time.time() - t0) * 1000.0

        return PredictionResult(
            profile=self.profile,
            status="success",
            predicted_class_id=top_class,
            predicted_class_name_en=name_en,
            predicted_class_name_ru=name_ru,
            predicted_class_name_ko=name_ko,
            confidence=round(top_prob, 4),
            class_probabilities=prob_dict,
            melanoma_risk_score=round(mel_risk, 4),
            referral_recommended=referral,
            latency_ms=round(latency, 2),
            quality_metrics=metrics.model_dump(),
            anamnesis_evaluation=anamnesis_eval,
            triage=triage_decision.model_dump(),
        )
