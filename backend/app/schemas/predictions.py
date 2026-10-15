"""Prediction request and response schemas."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from ml.anamnesis.questionnaire import GlasgowChecklistInput

PredictionStatus = Literal["success", "abstained", "rejected_quality"]


class GlasgowAnamnesisRequest(BaseModel):
    """Evidence-based Glasgow 7-Point Checklist (NICE Guidelines NG12)."""

    # Major features (2 points each)
    change_in_size: Optional[bool] = Field(None, description="Recent growth or change in size")
    irregular_shape: Optional[bool] = Field(None, description="Irregular border or asymmetrical outline")
    irregular_color: Optional[bool] = Field(None, description="Uneven color with multiple shades")

    # Minor features (1 point each)
    diameter_ge_7mm: Optional[bool] = Field(None, description="Largest diameter 7 mm or greater")
    inflammation: Optional[bool] = Field(None, description="Visible redness or inflammation halo")
    oozing_or_bleeding: Optional[bool] = Field(None, description="Oozing, crusting, or bleeding")
    change_in_sensation: Optional[bool] = Field(None, description="Itchiness, pain, or altered sensation")
    patient_age: Optional[int] = Field(None, ge=0, le=120, description="Approximate patient age in years")

    def to_domain_input(self) -> GlasgowChecklistInput:
        """Convert schema to ML anamnesis domain model."""
        return GlasgowChecklistInput(
            change_in_size=self.change_in_size,
            irregular_shape=self.irregular_shape,
            irregular_color=self.irregular_color,
            diameter_ge_7mm=self.diameter_ge_7mm,
            inflammation=self.inflammation,
            oozing_or_bleeding=self.oozing_or_bleeding,
            change_in_sensation=self.change_in_sensation,
            patient_age=self.patient_age,
        )


class PredictionResponse(BaseModel):
    """Complete, structured prediction response from SkinAI backend."""

    request_id: str = Field(..., description="Unique UUID for this prediction request")
    profile: str = Field(..., description="Classification profile evaluated")
    status: PredictionStatus = Field(..., description="Inference outcome state")
    
    predicted_class_id: Optional[str] = Field(None, description="Most probable class ID (e.g. 'bcc', 'mel', 'nev')")
    predicted_class_name_en: Optional[str] = Field(None, description="English name of predicted condition")
    predicted_class_name_ru: Optional[str] = Field(None, description="Russian name of predicted condition")
    predicted_class_name_ko: Optional[str] = Field(None, description="Korean name of predicted condition")
    
    confidence: float = Field(0.0, description="Calibrated posterior probability for the top class (0.0 to 1.0)")
    class_probabilities: Dict[str, float] = Field(default_factory=dict, description="Normalized probability distribution")
    
    melanoma_risk_score: float = Field(0.0, description="Calibrated risk probability for melanoma")
    referral_recommended: bool = Field(False, description="Whether clinical evaluation threshold is triggered")
    referral_reason: Optional[str] = Field(None, description="Clinical rationale for referral recommendation")
    
    rejection_reason: Optional[str] = Field(None, description="Explanation if status is 'abstained' or 'rejected_quality'")
    latency_ms: float = Field(0.0, description="End-to-end inference execution time in milliseconds")
    
    quality_metrics: Optional[Dict[str, Any]] = Field(None, description="Empirical image quality evaluation scores")
    anamnesis_evaluation: Optional[Dict[str, Any]] = Field(None, description="Glasgow checklist score and risk category")
    triage: Optional[Dict[str, Any]] = Field(None, description="Clinical triage decision separating vision and anamnesis")
    
    disclaimer_ru: str = Field(..., description="Mandatory non-diagnostic legal disclaimer in Russian")
    disclaimer_en: str = Field(..., description="Mandatory non-diagnostic legal disclaimer in English")
    disclaimer_ko: Optional[str] = Field(None, description="Mandatory non-diagnostic legal disclaimer in Korean")
