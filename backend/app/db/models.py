"""SQLAlchemy ORM models for telemetry and audit logging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base


class PredictionAuditLog(Base):
    """Audit log recording prediction metadata for quality assurance and telemetry.
    
    CRITICAL PRIVACY RULE: Zero raw image bytes or patient photos are stored.
    Only anonymized numerical and categorical telemetry is retained.
    """

    __tablename__ = "prediction_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    client_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    profile: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # success, abstained, rejected_quality
    predicted_class: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    melanoma_risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    referral_recommended: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    rejection_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    has_anamnesis: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    glasgow_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
