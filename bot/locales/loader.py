"""Localization loader and text interpolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

LOCALES_DIR = Path(__file__).parent

_STRINGS: Dict[str, Dict[str, str]] = {}


def load_locales() -> None:
    """Load all JSON locale files from directory into memory."""
    for loc_file in LOCALES_DIR.glob("*.json"):
        lang = loc_file.stem
        with open(loc_file, "r", encoding="utf-8") as f:
            _STRINGS[lang] = json.load(f)


def t(key: str, lang: str = "ru", **kwargs: Any) -> str:
    """Translate key for target language with fallback to English."""
    if not _STRINGS:
        load_locales()

    lang_strings = _STRINGS.get(lang, _STRINGS.get("ru", {}))
    template = lang_strings.get(key)
    if template is None:
        template = _STRINGS.get("en", {}).get(key, key)

    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
