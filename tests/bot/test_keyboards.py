"""Tests for Telegram inline keyboards."""

from bot.keyboards.inline import (
    get_disclaimer_keyboard,
    get_glasgow_prompt_keyboard,
    get_glasgow_question_keyboard,
    get_language_keyboard,
    get_profile_keyboard,
)


def test_disclaimer_keyboard():
    kb = get_disclaimer_keyboard("ru")
    assert len(kb.inline_keyboard) == 1
    assert kb.inline_keyboard[0][0].callback_data == "disclaimer:accept"


def test_glasgow_prompt_keyboard():
    kb = get_glasgow_prompt_keyboard("ru")
    assert len(kb.inline_keyboard[0]) == 2
    callbacks = [btn.callback_data for btn in kb.inline_keyboard[0]]
    assert "glasgow:start" in callbacks
    assert "glasgow:skip" in callbacks


def test_language_keyboard():
    kb = get_language_keyboard()
    assert len(kb.inline_keyboard[0]) == 3
    callbacks = [btn.callback_data for btn in kb.inline_keyboard[0]]
    assert "lang:ru" in callbacks
    assert "lang:en" in callbacks
    assert "lang:ko" in callbacks


def test_profile_keyboard():
    kb = get_profile_keyboard("ru")
    callbacks = [row[0].callback_data for row in kb.inline_keyboard]
    assert "profile:smartphone_clinical" in callbacks
    assert "profile:dermoscopy" in callbacks


def test_glasgow_question_keyboard():
    kb = get_glasgow_question_keyboard("ru", step_idx=2)
    assert len(kb.inline_keyboard) == 2
    # Row 1: Yes / No
    assert kb.inline_keyboard[0][0].callback_data == "g_ans:2:yes"
    assert kb.inline_keyboard[0][1].callback_data == "g_ans:2:no"
    # Row 2: Skip
    assert kb.inline_keyboard[1][0].callback_data == "g_ans:2:skip"


def test_analysis_result_keyboard():
    from bot.keyboards.inline import get_analysis_result_keyboard
    kb = get_analysis_result_keyboard("ru")
    assert len(kb.inline_keyboard) == 1
    assert kb.inline_keyboard[0][0].callback_data == "pdf:download"
