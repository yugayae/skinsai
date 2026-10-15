"""Unit tests for bot command handlers with mocked Telegram events."""

from unittest.mock import AsyncMock
import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage, StorageKey
from aiogram.types import CallbackQuery, Message, User

from bot.handlers.analysis import _format_prediction_response
from bot.handlers.common import (
    cb_accept_disclaimer,
    cb_set_language,
    cb_set_profile,
    cmd_help,
    cmd_language,
    cmd_profile,
    cmd_start,
    get_user_lang,
    get_user_profile,
)


@pytest.fixture
def mock_fsm_context() -> FSMContext:
    storage = MemoryStorage()
    key = StorageKey(bot_id=12345, chat_id=999, user_id=999)
    return FSMContext(storage=storage, key=key)


@pytest.fixture
def mock_user() -> User:
    return User(id=999, is_bot=False, first_name="TestUser", language_code="ru")


@pytest.mark.asyncio
async def test_cmd_start(mock_fsm_context: FSMContext, mock_user: User):
    """Verify /start clears state and sends disclaimer text with inline button."""
    message = AsyncMock(spec=Message)
    message.from_user = mock_user
    message.answer = AsyncMock()

    await cmd_start(message, mock_fsm_context)
    message.answer.assert_called_once()
    args, kwargs = message.answer.call_args
    assert "SkinAI" in args[0]
    assert "reply_markup" in kwargs


@pytest.mark.asyncio
async def test_cmd_help(mock_user: User):
    """Verify /help sends guidance on lighting and focus."""
    message = AsyncMock(spec=Message)
    message.from_user = mock_user
    message.answer = AsyncMock()

    await cmd_help(message)
    message.answer.assert_called_once()
    args, _ = message.answer.call_args
    assert "Освещение" in args[0] or "Lighting" in args[0]


@pytest.mark.asyncio
async def test_cmd_language(mock_user: User):
    """Verify /language presents keyboard."""
    message = AsyncMock(spec=Message)
    message.from_user = mock_user
    message.answer = AsyncMock()

    await cmd_language(message)
    message.answer.assert_called_once()


@pytest.mark.asyncio
async def test_cmd_profile(mock_user: User):
    """Verify /profile presents profile selector keyboard."""
    message = AsyncMock(spec=Message)
    message.from_user = mock_user
    message.answer = AsyncMock()

    await cmd_profile(message)
    message.answer.assert_called_once()


@pytest.mark.asyncio
async def test_callbacks_common(mock_user: User):
    """Verify acceptance, language switch, and profile switch callbacks."""
    # 1. Accept disclaimer
    cb_disc = AsyncMock(spec=CallbackQuery)
    cb_disc.from_user = mock_user
    cb_disc.message = AsyncMock()
    await cb_accept_disclaimer(cb_disc)
    cb_disc.answer.assert_called_once()

    # 2. Set Language
    cb_lang = AsyncMock(spec=CallbackQuery)
    cb_lang.from_user = mock_user
    cb_lang.data = "lang:en"
    cb_lang.message = AsyncMock()
    await cb_set_language(cb_lang)
    assert get_user_lang(mock_user.id) == "en"

    # 3. Set Profile
    cb_prof = AsyncMock(spec=CallbackQuery)
    cb_prof.from_user = mock_user
    cb_prof.data = "profile:dermoscopy"
    cb_prof.message = AsyncMock()
    await cb_set_profile(cb_prof)
    assert get_user_profile(mock_user.id) == "dermoscopy"


def test_format_prediction_response_cases():
    """Verify format string utility handles success, rejected, and referral states."""
    # Case 1: Rejected quality
    rej = {"status": "rejected_quality", "rejection_reason": "blur", "disclaimer_ru": "Дисклеймер"}
    msg_rej = _format_prediction_response(rej, "ru")
    assert "не прошел контроль качества" in msg_rej

    # Case 2: Abstained
    abst = {"status": "abstained", "disclaimer_ru": "Дисклеймер"}
    msg_abst = _format_prediction_response(abst, "ru")
    assert "воздержалась" in msg_abst

    # Case 3: Success with referral
    succ = {
        "status": "success",
        "predicted_class_name_ru": "Меланома",
        "confidence": 0.65,
        "class_probabilities": {"mel": 0.65, "nev": 0.35},
        "referral_recommended": True,
        "referral_reason": "Высокий риск",
        "latency_ms": 25.0,
        "disclaimer_ru": "Дисклеймер",
    }
    msg_succ = _format_prediction_response(succ, "ru")
    assert "Меланома" in msg_succ
    assert "Рекомендация специалиста" in msg_succ
