"""Profile discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.schemas.profiles import ProfilesResponse

router = APIRouter()


@router.get("/profiles", response_model=ProfilesResponse, tags=["Profiles"])
def list_profiles(request: Request) -> ProfilesResponse:
    """Return all available classification profiles, active status, classes, and thresholds."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return ProfilesResponse(profiles=[])

    return ProfilesResponse(profiles=registry.get_profile_details())
