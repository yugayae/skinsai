"""Inline keyboards for Telegram bot user interactions."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.locales.loader import t


def get_disclaimer_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    """Keyboard requiring explicit user acknowledgement of research-only disclaimer."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t("disclaimer_agree_btn", lang),
                    callback_data="disclaimer:accept",
                )
            ]
        ]
    )


def get_language_keyboard() -> InlineKeyboardMarkup:
    """Language selection keyboard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang:ru"),
                InlineKeyboardButton(text="🇬🇧 English", callback_data="lang:en"),
                InlineKeyboardButton(text="🇰🇷 한국어", callback_data="lang:ko"),
            ]
        ]
    )


def get_profile_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    """Image acquisition profile selector."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t("profile_smartphone_btn", lang),
                    callback_data="profile:smartphone_clinical",
                )
            ],
            [
                InlineKeyboardButton(
                    text=t("profile_dermoscopy_btn", lang),
                    callback_data="profile:dermoscopy",
                )
            ],
        ]
    )


def get_glasgow_prompt_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    """Prompt keyboard asking whether user wants to complete Glasgow 7-point checklist."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t("glasgow_yes_btn", lang),
                    callback_data="glasgow:start",
                ),
                InlineKeyboardButton(
                    text=t("glasgow_skip_btn", lang),
                    callback_data="glasgow:skip",
                ),
            ]
        ]
    )


def get_glasgow_question_keyboard(lang: str = "ru", step_idx: int = 0) -> InlineKeyboardMarkup:
    """Three-button keyboard for answering each checklist question."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"✅ {t('ans_yes', lang)}",
                    callback_data=f"g_ans:{step_idx}:yes",
                ),
                InlineKeyboardButton(
                    text=f"❌ {t('ans_no', lang)}",
                    callback_data=f"g_ans:{step_idx}:no",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=f"🤷 {t('ans_not_sure', lang)} / {t('glasgow_skip_btn', lang)}",
                    callback_data=f"g_ans:{step_idx}:skip",
                )
            ],
        ]
    )


def get_analysis_result_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    """Action keyboard displayed below prediction results, offering PDF memo download."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t("download_pdf_btn", lang),
                    callback_data="pdf:download",
                )
            ]
        ]
    )


def get_age_group_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    """De-Confounded demographic age group selector (FS-6)."""
    skip_lbl = t("btn_skip", lang)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="< 30", callback_data="g_age:25"),
                InlineKeyboardButton(text="30–59", callback_data="g_age:45"),
                InlineKeyboardButton(text="60+", callback_data="g_age:68"),
            ],
            [
                InlineKeyboardButton(text=f"🤷 {skip_lbl}", callback_data="g_age:skip"),
            ],
        ]
    )


def get_anatom_site_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    """De-Confounded anatomical lesion site selector (FS-6)."""
    lbl_head = t("site_head_neck", lang)
    lbl_trunk = t("site_trunk", lang)
    lbl_upper = t("site_upper_extremity", lang)
    lbl_lower = t("site_lower_extremity", lang)
    lbl_skip = t("btn_skip", lang)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=lbl_head, callback_data="g_site:head/neck"),
                InlineKeyboardButton(text=lbl_trunk, callback_data="g_site:trunk"),
            ],
            [
                InlineKeyboardButton(text=lbl_upper, callback_data="g_site:upper extremity"),
                InlineKeyboardButton(text=lbl_lower, callback_data="g_site:lower extremity"),
            ],
            [
                InlineKeyboardButton(text=f"🤷 {lbl_skip}", callback_data="g_site:skip"),
            ],
        ]
    )


