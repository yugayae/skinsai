"""Health check schemas."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness and readiness probe response schema."""

    status: str = Field("healthy", description="Service health state")
    version: str = Field(..., description="API version")
    loaded_profiles: List[str] = Field(default_factory=list, description="List of initialized model profiles")
    uptime_seconds: float = Field(..., description="Process uptime in seconds")
    execution_engine: str = Field("onnx", description="Active inference engine backend")
