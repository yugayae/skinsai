"""Tests for localization strings consistency and interpolation."""

import json
from pathlib import Path

from bot.locales.loader import load_locales, t

LOCALES_DIR = Path("bot/locales")


def test_locales_parity():
    """Verify that RU, EN, and KO locales have identical keys."""
    with open(LOCALES_DIR / "ru.json", "r", encoding="utf-8") as f:
        ru_keys = set(json.load(f).keys())
    with open(LOCALES_DIR / "en.json", "r", encoding="utf-8") as f:
        en_keys = set(json.load(f).keys())
    with open(LOCALES_DIR / "ko.json", "r", encoding="utf-8") as f:
        ko_keys = set(json.load(f).keys())

    assert ru_keys == en_keys, f"RU and EN key mismatch: {ru_keys ^ en_keys}"
    assert ru_keys == ko_keys, f"RU and KO key mismatch: {ru_keys ^ ko_keys}"


def test_t_function():
    """Verify t() loads and returns correct translations."""
    load_locales()
    welcome_ru = t("welcome_title", "ru")
    assert "Добро пожаловать" in welcome_ru

    welcome_en = t("welcome_title", "en")
    assert "Welcome" in welcome_en

    welcome_ko = t("welcome_title", "ko")
    assert "환영합니다" in welcome_ko
