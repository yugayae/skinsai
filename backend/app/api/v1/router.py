"""API v1 Router aggregating all v1 endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.v1.endpoints import health, predict, profiles

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(profiles.router, tags=["Profiles"])
api_router.include_router(predict.router, tags=["Inference"])
