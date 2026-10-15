"""Health and readiness check endpoints."""

from __future__ import annotations

import time
from fastapi import APIRouter, Request

from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse

router = APIRouter()
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health(request: Request) -> HealthResponse:
    """Return health status, active profile models, and service uptime."""
    registry = getattr(request.app.state, "registry", None)
    loaded = registry.list_loaded_profiles() if registry else []

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        loaded_profiles=loaded,
        uptime_seconds=round(time.time() - START_TIME, 2),
        execution_engine="onnx",
    )
