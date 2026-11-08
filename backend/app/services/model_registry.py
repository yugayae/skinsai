"""Model registry managing loaded inference engines and profile metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from backend.app.core.config import settings
from backend.app.schemas.profiles import ProfileDetail
from ml.inference.predictor import CLASS_DISPLAY_NAMES, OnnxPredictor

logger = logging.getLogger("skinai.registry")


class ModelRegistry:
    """Central registry caching initialized OnnxPredictors across profiles."""

    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = Path(artifacts_dir or settings.ARTIFACTS_DIR)
        self._predictors: Dict[str, OnnxPredictor] = {}
        self.initialize_models()

    def initialize_models(self) -> None:
        """Scan artifacts directory and instantiate ONNX predictor bundles."""
        models_dir = self.artifacts_dir / "models"

        # 1. Initialize smartphone_clinical profile
        smartphone_dir = models_dir / "smartphone_clinical"
        if smartphone_dir.exists() and (smartphone_dir / "model.onnx").exists():
            try:
                predictor = OnnxPredictor(bundle_dir=smartphone_dir)
                self._predictors["smartphone_clinical"] = predictor
                logger.info(f"Loaded active predictor for 'smartphone_clinical' from {smartphone_dir}")
            except Exception as e:
                logger.error(f"Failed to load 'smartphone_clinical' model: {e}")

        # 2. Check dermoscopy profile (if present)
        dermoscopy_dir = models_dir / "dermoscopy"
        if dermoscopy_dir.exists() and (dermoscopy_dir / "model.onnx").exists():
            try:
                predictor = OnnxPredictor(bundle_dir=dermoscopy_dir)
                self._predictors["dermoscopy"] = predictor
                logger.info(f"Loaded active predictor for 'dermoscopy' from {dermoscopy_dir}")
            except Exception as e:
                logger.error(f"Failed to load 'dermoscopy' model: {e}")

    def get_predictor(self, profile: str) -> Optional[OnnxPredictor]:
        """Retrieve predictor instance for the requested profile."""
        return self._predictors.get(profile)

    def list_loaded_profiles(self) -> List[str]:
        """Return list of active profile IDs."""
        return list(self._predictors.keys())

    def get_profile_details(self) -> List[ProfileDetail]:
        """Return detailed capabilities for all known profiles."""
        details: List[ProfileDetail] = []

        # Smartphone clinical profile
        if "smartphone_clinical" in self._predictors:
            pred = self._predictors["smartphone_clinical"]
            details.append(
                ProfileDetail(
                    profile_id="smartphone_clinical",
                    is_active=True,
                    model_version=pred.manifest.get("version", "0.1.0"),
                    license_scope="commercial_ok",
                    classes=pred.classes,
                    class_names_en={c: CLASS_DISPLAY_NAMES["smartphone_clinical"].get(c, {}).get("en", c) for c in pred.classes},
                    class_names_ru={c: CLASS_DISPLAY_NAMES["smartphone_clinical"].get(c, {}).get("ru", c) for c in pred.classes},
                    tau_abstain=pred.tau_abstain,
                    tau_referral_melanoma=pred.tau_referral_mel,
                )
            )
        else:
            details.append(
                ProfileDetail(
                    profile_id="smartphone_clinical",
                    is_active=False,
                    model_version="0.1.0",
                    license_scope="commercial_ok",
                    classes=["bcc", "scc", "ack", "sek", "mel", "nev"],
                    class_names_en={c: CLASS_DISPLAY_NAMES["smartphone_clinical"].get(c, {}).get("en", c) for c in ["bcc", "scc", "ack", "sek", "mel", "nev"]},
                    class_names_ru={c: CLASS_DISPLAY_NAMES["smartphone_clinical"].get(c, {}).get("ru", c) for c in ["bcc", "scc", "ack", "sek", "mel", "nev"]},
                    tau_abstain=0.40,
                    tau_referral_melanoma=0.20,
                )
            )

        # Dermoscopy profile (baseline status)
        if "dermoscopy" in self._predictors:
            pred = self._predictors["dermoscopy"]
            details.append(
                ProfileDetail(
                    profile_id="dermoscopy",
                    is_active=True,
                    model_version=pred.manifest.get("version", "0.1.0"),
                    license_scope="public_ok",
                    classes=pred.classes,
                    class_names_en={c: CLASS_DISPLAY_NAMES["dermoscopy"].get(c, {}).get("en", c) for c in pred.classes},
                    class_names_ru={c: CLASS_DISPLAY_NAMES["dermoscopy"].get(c, {}).get("ru", c) for c in pred.classes},
                    tau_abstain=pred.tau_abstain,
                    tau_referral_melanoma=pred.tau_referral_mel,
                )
            )
        else:
            details.append(
                ProfileDetail(
                    profile_id="dermoscopy",
                    is_active=False,
                    model_version="0.0.0",
                    license_scope="public_ok",
                    classes=["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"],
                    class_names_en={c: CLASS_DISPLAY_NAMES["dermoscopy"].get(c, {}).get("en", c) for c in ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]},
                    class_names_ru={c: CLASS_DISPLAY_NAMES["dermoscopy"].get(c, {}).get("ru", c) for c in ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]},
                    tau_abstain=0.40,
                    tau_referral_melanoma=0.20,
                )
            )

        return details
