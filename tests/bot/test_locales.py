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


def test_triage_decision_multilingual_rationales():
    """Verify that AnamnesisEngine generates localized rationales in RU, EN, and KO."""
    from ml.anamnesis.questionnaire import AnamnesisEngine, GlasgowChecklistInput

    anamnesis = GlasgowChecklistInput(
        change_in_size=True,
        irregular_shape=True,
        irregular_color=True,
    )
    decision = AnamnesisEngine.determine_triage(
        image_probabilities={"mel": 0.35, "nev": 0.65},
        anamnesis=anamnesis,
    )
    assert decision.triage_zone == "red"
    assert len(decision.rationales_ru) >= 2
    assert len(decision.rationales_en) >= 2
    assert len(decision.rationales_ko) >= 2

    # Verify no Cyrillic leaks in English or Korean rationales
    import re
    cyrillic_pattern = re.compile(r"[\u0400-\u04FF]")
    for r_en in decision.rationales_en:
        assert not cyrillic_pattern.search(r_en), f"Cyrillic found in English rationale: {r_en}"

    # Verify Korean characters present in Korean rationales
    hangul_pattern = re.compile(r"[\uAC00-\uD7A3]")
    for r_ko in decision.rationales_ko:
        assert hangul_pattern.search(r_ko), f"Korean characters missing in Korean rationale: {r_ko}"


def test_format_prediction_multilingual_parity():
    """Verify that _format_prediction_response outputs properly localized text for EN, RU, KO."""
    from bot.handlers.analysis import _format_prediction_response

    mock_resp = {
        "status": "success",
        "predicted_class_id": "mel",
        "predicted_class_name_ru": "Меланома",
        "predicted_class_name_en": "Melanoma",
        "predicted_class_name_ko": "악성 흑색종",
        "confidence": 0.88,
        "class_probabilities": {"mel": 0.88, "nev": 0.12},
        "similar_biopsy_cases": [
            {
                "rank": 1,
                "subtype_id": "SSM",
                "malignancy_label": "Malignant",
                "cosine_similarity": 0.942,
                "image_id": "ISIC_001",
            }
        ],
        "triage": {
            "tier": "urgent_referral",
            "triage_zone": "red",
            "title_ru": "Рекомендована приоритетная очная консультация",
            "title_en": "Priority face-to-face consultation recommended",
            "title_ko": "우선 대면 진료 권고",
            "recommendation_ru": "Выявлены клинические или оптические признаки...",
            "recommendation_en": "Clinical or visual features detected...",
            "recommendation_ko": "피부과 전문의의 대면 진찰이 필요합니다...",
            "rationales_ru": ["Модельная оценка подозрения на меланому (88.0%)"],
            "rationales_en": ["Model assessment of melanoma suspicion (88.0%)"],
            "rationales_ko": ["악성 흑색종 의심 모델 평가 (88.0%)"],
            "urgency_days": 14,
        },
        "referral_recommended": True,
        "referral_reason": "High risk",
        "latency_ms": 20.0,
        "disclaimer_ru": "Дисклеймер RU",
        "disclaimer_en": "Disclaimer EN",
        "disclaimer_ko": "면책 조항 KO",
    }

    # Russian output
    res_ru = _format_prediction_response(mock_resp, "ru")
    assert "Меланома" in res_ru
    assert "сходство" in res_ru
    assert "Злокачественное" in res_ru

    # English output
    res_en = _format_prediction_response(mock_resp, "en")
    assert "Melanoma" in res_en
    assert "similarity" in res_en
    assert "Malignant" in res_en
    assert "Priority face-to-face consultation recommended" in res_en
    assert "Model assessment of melanoma suspicion" in res_en

    # Korean output
    res_ko = _format_prediction_response(mock_resp, "ko")
    assert "악성 흑색종" in res_ko
    assert "유사도" in res_ko
    assert "악성" in res_ko
    assert "우선 대면 진료 권고" in res_ko
    assert "악성 흑색종 의심 모델 평가" in res_ko
