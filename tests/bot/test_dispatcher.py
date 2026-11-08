"""Tests for bot dispatcher and router configuration."""

from bot.main import build_dispatcher


def test_dispatcher_routers_registered():
    """Verify dispatcher initializes and registers both common and analysis routers."""
    dp = build_dispatcher()
    assert dp is not None
    router_names = [r.name for r in dp.sub_routers]
    assert "common" in router_names
    assert "analysis" in router_names
