"""Model profile inspection schemas."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, ConfigDict, Field


class ProfileDetail(BaseModel):
    """Detailed metadata for a specific classification profile."""

    model_config = ConfigDict(protected_namespaces=())

    profile_id: str = Field(..., description="Unique profile identifier ('smartphone_clinical' or 'dermoscopy')")
    is_active: bool = Field(..., description="Whether model weights are loaded and ready for inference")
    model_version: str = Field(..., description="Semantic version of the model bundle")
    license_scope: str = Field(..., description="Dataset governance license tier")
    classes: List[str] = Field(default_factory=list, description="List of recognized class IDs")
    class_names_en: Dict[str, str] = Field(default_factory=dict, description="Class descriptions in English")
    class_names_ru: Dict[str, str] = Field(default_factory=dict, description="Class descriptions in Russian")
    tau_abstain: float = Field(..., description="Uncertainty threshold for abstention")
    tau_referral_melanoma: float = Field(..., description="Calibrated risk threshold triggering dermatologist referral")


class ProfilesResponse(BaseModel):
    """List of all registered inference profiles."""

    profiles: List[ProfileDetail]
