"""Production ONNX Runtime inference engine with quality validation and abstention."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image
from pydantic import BaseModel, Field

from ml.anamnesis.questionnaire import (
    AnamnesisEngine,
    GlasgowChecklistInput,
)
from ml.datasets.validator import ImageValidator
from ml.explainability.deterministic_xai import ABCDEMorphometryEngine

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
    abcde_metrics: Optional[Dict[str, Any]] = None
    ood_metrics: Optional[Dict[str, Any]] = None
    similar_biopsy_cases: Optional[List[Dict[str, Any]]] = None
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

        # Load De-Confounded Dual-Backbone Fusion ONNX Head (896D DINOv2+BiomedCLIP + 7D clinical -> 2D triage + 6D subtype) if exported
        fusion_onnx_path = self.bundle_dir / "fusion_head.onnx"
        self.fusion_session: Optional[ort.InferenceSession] = None
        if fusion_onnx_path.exists():
            self.fusion_session = ort.InferenceSession(
                str(fusion_onnx_path),
                providers=["CPUExecutionProvider"],
            )

        # Load Biopsy Atlas k-NN Index (Deterministic XAI #2) if exported
        atlas_path = self.bundle_dir / "biopsy_atlas_index.pt"
        self.biopsy_atlas: Optional[Dict[str, Any]] = None
        if atlas_path.exists():
            import torch
            self.biopsy_atlas = torch.load(atlas_path, map_location="cpu", weights_only=False)

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

    @staticmethod
    def _evaluate_skin_ood(pil_img: Image.Image) -> Dict[str, Any]:
        """Evaluate CIE L*a*b* human skin/lesion chromaticity & spatial coherence (FS-5 OOD Gate)."""
        resized = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        rgb = np.array(resized, dtype=np.uint8)
        lab = ABCDEMorphometryEngine.rgb_to_cielab(rgb)
        l_star, a_star, b_star = lab[..., 0], lab[..., 1], lab[..., 2]

        # Physiological human skin (Fitzpatrick I-VI) + pigmented/erythematous lesion envelope in CIE LAB
        in_skin_envelope = (
            (l_star >= 8.0)
            & (l_star <= 95.0)
            & (a_star >= -5.0)
            & (a_star <= 55.0)
            & (b_star >= -6.0)
            & (b_star <= 60.0)
            & ((a_star + b_star) >= 1.5)
        )
        skin_pixel_ratio = float(np.mean(in_skin_envelope))

        # Local spatial coherence (detects uncorrelated random noise or synthetic high-frequency textures)
        diff_x = np.abs(rgb[:, 1:, :].astype(np.float32) - rgb[:, :-1, :].astype(np.float32))
        local_noise_mad = float(np.mean(diff_x))

        is_ood_non_skin = (skin_pixel_ratio < 0.25) or (local_noise_mad > 55.0)
        return {
            "skin_pixel_ratio": round(skin_pixel_ratio, 4),
            "local_noise_mad": round(local_noise_mad, 2),
            "is_ood_non_skin": bool(is_ood_non_skin),
        }

    def _apply_deconfounded_demographic_prior(
        self,
        logits: np.ndarray,
        anamnesis: Optional[GlasgowChecklistInput],
    ) -> np.ndarray:
        """Apply bounded De-Confounded demographic prior adjustment (FS-6, zero dataset-shortcut flags)."""
        if anamnesis is None:
            return logits

        adj = logits.copy().astype(np.float32)
        cls_map = {c: idx for idx, c in enumerate(self.classes)}

        # Age-based epidemiological prior (validated in EXP-017/018 with 50% Metadata Dropout)
        if anamnesis.patient_age is not None:
            age = int(anamnesis.patient_age)
            if age >= 60:
                for c in ("bcc", "scc", "ack", "mel", "akiec"):
                    if c in cls_map:
                        adj[cls_map[c]] += 0.20
                for c in ("nev", "nv"):
                    if c in cls_map:
                        adj[cls_map[c]] -= 0.20
            elif age < 30:
                for c in ("nev", "nv"):
                    if c in cls_map:
                        adj[cls_map[c]] += 0.25
                for c in ("bcc", "scc", "ack", "akiec"):
                    if c in cls_map:
                        adj[cls_map[c]] -= 0.20

        # Anatomical site UV exposure prior (explicitly without any 'unknown' category)
        if anamnesis.anatom_site:
            site = anamnesis.anatom_site.strip().lower()
            if site in ("head/neck", "face", "scalp", "ear", "neck", "nose"):
                for c in ("bcc", "scc", "ack", "akiec"):
                    if c in cls_map:
                        adj[cls_map[c]] += 0.18
            elif site in ("trunk", "back", "chest", "abdomen", "lower extremity", "upper extremity"):
                for c in ("mel", "nev", "nv"):
                    if c in cls_map:
                        adj[cls_map[c]] += 0.10

        return adj

    @staticmethod
    def _build_deconfounded_clinical_vector(anamnesis: Optional[GlasgowChecklistInput]) -> np.ndarray:
        """Build 7D de-confounded clinical vector [age_norm, s_head, s_trunk, s_upper, s_lower, sex_f, sex_m]."""
        if anamnesis is None:
            return np.zeros((1, 7), dtype=np.float32)

        age_norm = (float(anamnesis.patient_age) - 50.0) / 25.0 if anamnesis.patient_age is not None else 0.0
        site = (anamnesis.anatom_site or "").strip().lower()
        sex = (anamnesis.patient_sex or "").strip().lower()

        s_head = 1.0 if ("head" in site or "neck" in site or "face" in site) else 0.0
        s_trunk = 1.0 if ("trunk" in site or "torso" in site or "back" in site or "chest" in site) else 0.0
        s_upper = 1.0 if ("upper" in site or "arm" in site or "hand" in site) else 0.0
        s_lower = 1.0 if ("lower" in site or "leg" in site or "foot" in site) else 0.0
        sex_f = 1.0 if "female" in sex else 0.0
        sex_m = 1.0 if ("male" in sex and "female" not in sex) else 0.0

        return np.array([[age_norm, s_head, s_trunk, s_upper, s_lower, sex_f, sex_m]], dtype=np.float32)

    def predict_fusion_embeddings(
        self,
        vis_896d: np.ndarray,
        anamnesis: Optional[GlasgowChecklistInput] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute exported ONNX De-Confounded Dual-Backbone Fusion Head on 896D (DINOv2+BiomedCLIP) embeddings."""
        if self.fusion_session is None:
            raise RuntimeError("fusion_head.onnx is not loaded in this bundle.")
        vis_arr = np.asarray(vis_896d, dtype=np.float32).reshape(1, 896)
        clin_arr = self._build_deconfounded_clinical_vector(anamnesis)
        triage_logits, subtype_logits = self.fusion_session.run(
            None, {"vis_896d": vis_arr, "clin_7d": clin_arr}
        )
        t_exp = np.exp(triage_logits[0] - np.max(triage_logits[0]))
        t_probs = t_exp / np.sum(t_exp)
        s_exp = np.exp(subtype_logits[0] - np.max(subtype_logits[0]))
        s_probs = s_exp / np.sum(s_exp)
        return t_probs, s_probs

    def retrieve_similar_biopsy_cases(
        self,
        probabilities: np.ndarray,
        vis_896d: Optional[np.ndarray] = None,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve Top-K histologically verified twin cases from the Biopsy Atlas (Deterministic XAI #2)."""
        if self.biopsy_atlas is None:
            return []

        records = self.biopsy_atlas["records"]
        if vis_896d is not None and "vis_embeddings_896d" in self.biopsy_atlas:
            atlas_vecs = self.biopsy_atlas["vis_embeddings_896d"].float().numpy()
            q = np.asarray(vis_896d, dtype=np.float32).ravel()
            q_norm = q / max(float(np.linalg.norm(q)), 1e-8)
            sims = atlas_vecs @ q_norm
        else:
            atlas_probs = self.biopsy_atlas["prob_signatures_6d"].float().numpy()
            # Map current predictor's class probabilities to the 6-class atlas order
            atlas_order = self.biopsy_atlas.get("class_order", ["bcc", "scc", "ack", "sek", "mel", "nev"])
            prob_map = {c: float(probabilities[i]) for i, c in enumerate(self.classes)}
            q_vec = np.array([prob_map.get(c, 0.0) for c in atlas_order], dtype=np.float32)
            q_norm = q_vec / max(float(np.linalg.norm(q_vec)), 1e-8)
            sims = atlas_probs @ q_norm

        top_indices = np.argsort(-sims)[:k]
        out: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            rec = dict(records[int(idx)])
            rec["rank"] = rank
            rec["cosine_similarity"] = round(float(sims[int(idx)]), 4)
            out.append(rec)
        return out

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

        # 2. DS-5 Quantitative ABCDE Morphometry & FS-5 OOD Skin Gate
        abcde_obj, _ = ABCDEMorphometryEngine.compute_metrics(
            pil_img, target_size=224, method="attention_grabcut"
        )
        abcde_dict = abcde_obj.to_dict()
        ood_dict = self._evaluate_skin_ood(pil_img)

        # 3. Canonical Preprocessing & ONNX Runtime Forward Pass
        input_tensor = self.preprocess_image(pil_img)
        ort_outputs = self.session.run(None, {self.input_name: input_tensor})
        raw_logits = ort_outputs[0][0]  # shape (num_classes,)

        # 4. Temperature Scaling Calibration & De-Confounded Demographic Prior (FS-6)
        scaled_logits = raw_logits / max(0.01, self.temperature)
        scaled_logits = self._apply_deconfounded_demographic_prior(scaled_logits, anamnesis)

        # 5. Stable Softmax Computation
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Map to class dictionary
        prob_dict = {
            cls_name: round(float(probabilities[i]), 4)
            for i, cls_name in enumerate(self.classes)
        }
        similar_cases = self.retrieve_similar_biopsy_cases(probabilities, k=3)

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
            is_blurry=bool(ood_dict["is_ood_non_skin"]),
            entropy=entropy,
        )
        referral = triage_decision.referral_recommended

        # 7. Check Out-of-Distribution / Non-Skin Gate (FS-5) or Uncertainty Abstention
        if ood_dict["is_ood_non_skin"]:
            latency = (time.time() - t0) * 1000.0
            return PredictionResult(
                profile=self.profile,
                status="abstained",
                confidence=round(top_prob, 4),
                class_probabilities=prob_dict,
                melanoma_risk_score=round(mel_risk, 4),
                referral_recommended=referral,
                rejection_reason=(
                    f"out_of_distribution_non_skin "
                    f"(skin_ratio={ood_dict['skin_pixel_ratio']:.2f}, noise_mad={ood_dict['local_noise_mad']:.1f})"
                ),
                latency_ms=round(latency, 2),
                quality_metrics=metrics.model_dump(),
                abcde_metrics=abcde_dict,
                ood_metrics=ood_dict,
                similar_biopsy_cases=similar_cases,
                anamnesis_evaluation=anamnesis_eval,
                triage=triage_decision.model_dump(),
            )

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
                abcde_metrics=abcde_dict,
                ood_metrics=ood_dict,
                similar_biopsy_cases=similar_cases,
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
            abcde_metrics=abcde_dict,
            ood_metrics=ood_dict,
            similar_biopsy_cases=similar_cases,
            anamnesis_evaluation=anamnesis_eval,
            triage=triage_decision.model_dump(),
        )
