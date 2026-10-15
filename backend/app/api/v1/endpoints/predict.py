"""Inference prediction endpoint supporting dual-profile image classification and Glasgow anamnesis."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.database import get_db
from backend.app.db.models import PredictionAuditLog
from backend.app.schemas.predictions import (
    GlasgowAnamnesisRequest,
    PredictionResponse,
)
from backend.app.services.image_validator import validate_image_upload
from backend.app.services.model_registry import ModelRegistry

logger = logging.getLogger("skinai.predict")
router = APIRouter()


def _compute_client_hash(client_ip: str, user_header: Optional[str] = None) -> str:
    """Compute anonymous salted hash of the requesting client."""
    raw_identifier = f"{user_header or client_ip}:{settings.TELEMETRY_SALT}"
    return hashlib.sha256(raw_identifier.encode("utf-8")).hexdigest()[:32]


def _parse_anamnesis(
    anamnesis_json: Optional[str],
    change_in_size: Optional[bool],
    irregular_shape: Optional[bool],
    irregular_color: Optional[bool],
    diameter_ge_7mm: Optional[bool],
    inflammation: Optional[bool],
    oozing_or_bleeding: Optional[bool],
    change_in_sensation: Optional[bool],
    patient_age: Optional[int],
) -> Optional[GlasgowAnamnesisRequest]:
    """Parse Glasgow checklist from JSON string or explicit form arguments."""
    if anamnesis_json:
        try:
            data = json.loads(anamnesis_json)
            return GlasgowAnamnesisRequest(**data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON in 'anamnesis': {str(exc)}",
            ) from exc

    # Check if any explicit form boolean/age is set
    has_any = any(
        x is not None
        for x in [
            change_in_size,
            irregular_shape,
            irregular_color,
            diameter_ge_7mm,
            inflammation,
            oozing_or_bleeding,
            change_in_sensation,
            patient_age,
        ]
    )
    if has_any:
        return GlasgowAnamnesisRequest(
            change_in_size=change_in_size,
            irregular_shape=irregular_shape,
            irregular_color=irregular_color,
            diameter_ge_7mm=diameter_ge_7mm,
            inflammation=inflammation,
            oozing_or_bleeding=oozing_or_bleeding,
            change_in_sensation=change_in_sensation,
            patient_age=patient_age,
        )

    return None


@router.post("/predictions", response_model=PredictionResponse, tags=["Inference"])
async def predict_image(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP, up to 15MB)"),
    profile: str = Form("smartphone_clinical", description="Classification profile identifier"),
    anamnesis: Optional[str] = Form(None, description="Optional JSON string of Glasgow 7-Point Checklist"),
    change_in_size: Optional[bool] = Form(None, description="Major: recent growth or change in size"),
    irregular_shape: Optional[bool] = Form(None, description="Major: irregular border / asymmetrical outline"),
    irregular_color: Optional[bool] = Form(None, description="Major: uneven colour / multiple shades"),
    diameter_ge_7mm: Optional[bool] = Form(None, description="Minor: largest diameter >= 7 mm"),
    inflammation: Optional[bool] = Form(None, description="Minor: visible redness or inflammatory halo"),
    oozing_or_bleeding: Optional[bool] = Form(None, description="Minor: oozing, bleeding, or crusting"),
    change_in_sensation: Optional[bool] = Form(None, description="Minor: itch, pain, or altered sensation"),
    patient_age: Optional[int] = Form(None, ge=0, le=120, description="Approximate patient age"),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Analyze skin lesion image with calibrated neural network and evidence-based anamnesis."""
    t0 = time.time()
    request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "127.0.0.1"
    client_hash = _compute_client_hash(client_ip, request.headers.get("X-User-ID"))

    # 1. Retrieve Predictor for Profile
    registry: Optional[ModelRegistry] = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not initialized.",
        )

    predictor = registry.get_predictor(profile)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested profile '{profile}' is not active or supported. Available: {registry.list_loaded_profiles()}",
        )

    # 2. Read Image Bytes
    try:
        raw_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        ) from exc

    # 3. Security & Format Validation
    is_valid, validation_error, pil_img = validate_image_upload(raw_bytes)
    if not is_valid:
        latency = (time.time() - t0) * 1000.0
        # Audit rejected input
        log_record = PredictionAuditLog(
            request_id=request_id,
            client_hash=client_hash,
            profile=profile,
            status="rejected_quality",
            rejection_reason=validation_error,
            latency_ms=round(latency, 2),
        )
        db.add(log_record)
        db.commit()

        return PredictionResponse(
            request_id=request_id,
            profile=profile,
            status="rejected_quality",
            rejection_reason=validation_error,
            latency_ms=round(latency, 2),
            disclaimer_ru=settings.STATIC_DISCLAIMER_RU,
            disclaimer_en=settings.STATIC_DISCLAIMER_EN,
            disclaimer_ko=settings.STATIC_DISCLAIMER_KO,
        )

    # 4. Parse Anamnesis (Glasgow 7-Point Checklist)
    anamnesis_req = _parse_anamnesis(
        anamnesis_json=anamnesis,
        change_in_size=change_in_size,
        irregular_shape=irregular_shape,
        irregular_color=irregular_color,
        diameter_ge_7mm=diameter_ge_7mm,
        inflammation=inflammation,
        oozing_or_bleeding=oozing_or_bleeding,
        change_in_sensation=change_in_sensation,
        patient_age=patient_age,
    )
    domain_anamnesis = anamnesis_req.to_domain_input() if anamnesis_req else None

    # 5. Execute Canonical ML Inference
    result = predictor.predict_bytes(raw_bytes, anamnesis=domain_anamnesis)

    # 6. Referral Rationale Construction
    referral_reason = None
    if result.referral_recommended:
        reasons = []
        if result.anamnesis_evaluation:
            score = result.anamnesis_evaluation.get("score", result.anamnesis_evaluation.get("glasgow_score", 0))
            if score >= 3:
                reasons.append(f"Glasgow score ({score}/10) indicates elevated clinical risk")
        if result.melanoma_risk_score >= predictor.tau_referral_mel:
            reasons.append(f"Model melanoma risk score ({result.melanoma_risk_score:.2f}) exceeds threshold ({predictor.tau_referral_mel:.2f})")
        referral_reason = "; ".join(reasons) if reasons else "Lesion characteristics warrant clinical dermatologist examination"

    # 7. Persist Telemetry Log to Database (Zero Raw Photo Retention)
    total_latency = (time.time() - t0) * 1000.0
    glasgow_score = None
    if result.anamnesis_evaluation:
        glasgow_score = result.anamnesis_evaluation.get("score", result.anamnesis_evaluation.get("glasgow_score"))

    log_record = PredictionAuditLog(
        request_id=request_id,
        client_hash=client_hash,
        profile=profile,
        status=result.status,
        predicted_class=result.predicted_class_id,
        confidence=result.confidence,
        melanoma_risk_score=result.melanoma_risk_score,
        referral_recommended=result.referral_recommended,
        rejection_reason=result.rejection_reason,
        has_anamnesis=domain_anamnesis is not None,
        glasgow_score=glasgow_score,
        latency_ms=round(total_latency, 2),
    )
    db.add(log_record)
    db.commit()

    return PredictionResponse(
        request_id=request_id,
        profile=profile,
        status=result.status,
        predicted_class_id=result.predicted_class_id,
        predicted_class_name_en=result.predicted_class_name_en,
        predicted_class_name_ru=result.predicted_class_name_ru,
        predicted_class_name_ko=result.predicted_class_name_ko,
        confidence=result.confidence,
        class_probabilities=result.class_probabilities,
        melanoma_risk_score=result.melanoma_risk_score,
        referral_recommended=result.referral_recommended,
        referral_reason=referral_reason,
        rejection_reason=result.rejection_reason,
        latency_ms=round(total_latency, 2),
        quality_metrics=result.quality_metrics,
        anamnesis_evaluation=result.anamnesis_evaluation,
        triage=result.triage,
        disclaimer_ru=settings.STATIC_DISCLAIMER_RU,
        disclaimer_en=settings.STATIC_DISCLAIMER_EN,
        disclaimer_ko=settings.STATIC_DISCLAIMER_KO,
    )
