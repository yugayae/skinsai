"""Asynchronous HTTP client communicating with SkinAI FastAPI backend."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("skinai.bot.client")


class SkinAIClient:
    """Client for SkinAI inference API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout

    async def get_health(self) -> Dict[str, Any]:
        """Fetch service health probe."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def get_profiles(self) -> Dict[str, Any]:
        """Fetch active model profiles and thresholds."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/v1/profiles")
            resp.raise_for_status()
            return resp.json()

    async def predict_image(
        self,
        image_bytes: bytes,
        filename: str = "lesion.jpg",
        profile: str = "smartphone_clinical",
        user_id: Optional[int] = None,
        anamnesis_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send image bytes and optional Glasgow checklist to inference backend."""
        data: Dict[str, Any] = {"profile": profile}
        if anamnesis_dict:
            data["anamnesis"] = json.dumps(anamnesis_dict)

        headers = {}
        if user_id:
            headers["X-User-ID"] = str(user_id)

        files = {"file": (filename, image_bytes, "image/jpeg")}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/predictions",
                data=data,
                files=files,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
