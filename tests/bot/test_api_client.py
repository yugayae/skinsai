"""Unit tests for the bot SkinAIClient."""

import httpx
import pytest
from bot.services.api_client import SkinAIClient

_ORIGINAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.mark.asyncio
async def test_api_client_health(monkeypatch):
    """Verify SkinAIClient.get_health returns parsed payload."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "healthy", "version": "0.1.0"})

    transport = httpx.MockTransport(handler)

    def mock_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return _ORIGINAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_async_client)

    client = SkinAIClient(base_url="http://testserver")
    res = await client.get_health()
    assert res["status"] == "healthy"
    assert res["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_api_client_predict(monkeypatch):
    """Verify SkinAIClient.predict_image posts data and files properly."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert "/v1/predictions" in str(request.url)
        return httpx.Response(200, json={"status": "success", "predicted_class_id": "bcc"})

    transport = httpx.MockTransport(handler)

    def mock_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return _ORIGINAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_async_client)

    client = SkinAIClient(base_url="http://testserver")
    res = await client.predict_image(
        image_bytes=b"fake_bytes",
        filename="sample.jpg",
        profile="smartphone_clinical",
        anamnesis_dict={"change_in_size": True},
    )
    assert res["status"] == "success"
    assert res["predicted_class_id"] == "bcc"
