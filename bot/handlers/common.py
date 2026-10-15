"""Common command and navigation handlers for the Telegram bot."""

from __future__ import annotations

from typing import Dict
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import (
    get_disclaimer_keyboard,
    get_language_keyboard,
    get_profile_keyboard,
)
from bot.locales.loader import t

import os

router = Router(name="common")

# User preferences cache (user_id -> {"lang": "ru", "profile": "smartphone_clinical", "disclaimer_accepted": bool})
USER_PREFS: Dict[int, Dict[str, str]] = {}


def get_allowed_users() -> set[int]:
    """Parse ALLOWED_TELEGRAM_USERS from environment if set."""
    raw = os.getenv("ALLOWED_TELEGRAM_USERS", "").strip()
    if not raw:
        return set()
    res = set()
    for item in raw.split(","):
        item = item.strip()
        if item.isdigit():
            res.add(int(item))
    return res


def is_user_allowed(user_id: int) -> bool:
    """Check whether user is permitted under fail-closed research allowlist."""
    allowed = get_allowed_users()
    if not allowed:
        return True  # Open dev mode when no allowlist is configured
    return user_id in allowed


def get_user_lang(user_id: int) -> str:
    """Retrieve selected language for user, defaulting to Russian."""
    return USER_PREFS.get(user_id, {}).get("lang", "ru")


def get_user_profile(user_id: int) -> str:
    """Retrieve selected profile for user, defaulting to smartphone_clinical."""
    return USER_PREFS.get(user_id, {}).get("profile", "smartphone_clinical")


def is_disclaimer_accepted(user_id: int) -> bool:
    """Check if user accepted terms."""
    return bool(USER_PREFS.get(user_id, {}).get("disclaimer_accepted", False))


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    """Handle /start command: present welcome text, mandatory disclaimer and acceptance button."""
    await state.clear()
    user_id = message.from_user.id if message.from_user else 0

    if not is_user_allowed(user_id):
        await message.answer(
            f"🔒 **Режим закрытого пилотного тестирования**\n\n"
            f"Бот SkinAI в настоящее время доступен только для участников закрытого врачебного тестирования.\n\n"
            f"Ваш Telegram ID: `{user_id}`.\n"
            f"Для запроса доступа передайте ваш ID координатору проекта.",
            parse_mode="Markdown",
        )
        return

    lang = get_user_lang(user_id)

    text = f"{t('welcome_title', lang)}\n\n{t('disclaimer_text', lang)}"
    await message.answer(text, reply_markup=get_disclaimer_keyboard(lang), parse_mode="Markdown")


@router.callback_query(F.data == "disclaimer:accept")
async def cb_accept_disclaimer(callback: CallbackQuery) -> None:
    """Handle user accepting terms."""
    try:
        await callback.answer()
    except Exception:
        pass

    user_id = callback.from_user.id
    if user_id not in USER_PREFS:
        USER_PREFS[user_id] = {"lang": "ru", "profile": "smartphone_clinical"}
    USER_PREFS[user_id]["disclaimer_accepted"] = "true"
    lang = get_user_lang(user_id)

    if callback.message:
        try:
            await callback.message.edit_text(
                f"{t('disclaimer_accepted', lang)}\n\n{t('help_text', lang)}",
                parse_mode="Markdown",
            )
        except Exception:
            pass


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command: guidelines for lighting, focus, and framing."""
    user_id = message.from_user.id if message.from_user else 0
    lang = get_user_lang(user_id)
    await message.answer(t("help_text", lang), parse_mode="Markdown")


@router.message(Command("language"))
async def cmd_language(message: Message) -> None:
    """Handle /language command: switch between RU and EN."""
    user_id = message.from_user.id if message.from_user else 0
    lang = get_user_lang(user_id)
    await message.answer(t("language_choose", lang), reply_markup=get_language_keyboard())


@router.callback_query(F.data.startswith("lang:"))
async def cb_set_language(callback: CallbackQuery) -> None:
    """Handle language selection callback."""
    try:
        await callback.answer()
    except Exception:
        pass

    lang = callback.data.split(":")[1]
    user_id = callback.from_user.id

    if user_id not in USER_PREFS:
        USER_PREFS[user_id] = {"profile": "smartphone_clinical"}
    USER_PREFS[user_id]["lang"] = lang

    if callback.message:
        try:
            await callback.message.edit_text(t("language_changed", lang))
        except Exception:
            pass


@router.message(Command("profile"))
async def cmd_profile(message: Message) -> None:
    """Handle /profile command: choose between smartphone and dermoscopy."""
    user_id = message.from_user.id if message.from_user else 0
    lang = get_user_lang(user_id)
    await message.answer(t("profile_choose", lang), reply_markup=get_profile_keyboard(lang))


@router.callback_query(F.data.startswith("profile:"))
async def cb_set_profile(callback: CallbackQuery) -> None:
    """Handle profile change callback."""
    try:
        await callback.answer()
    except Exception:
        pass

    profile_id = callback.data.split(":")[1]
    user_id = callback.from_user.id

    if user_id not in USER_PREFS:
        USER_PREFS[user_id] = {"lang": "ru"}
    USER_PREFS[user_id]["profile"] = profile_id
    lang = get_user_lang(user_id)

    key = "profile_selected_smartphone" if profile_id == "smartphone_clinical" else "profile_selected_dermoscopy"
    if callback.message:
        try:
            await callback.message.edit_text(t(key, lang), parse_mode="Markdown")
        except Exception:
            pass
