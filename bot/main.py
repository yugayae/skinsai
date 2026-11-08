"""Telegram Bot main application entrypoint."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

from bot.handlers.analysis import router as analysis_router
from bot.handlers.common import router as common_router
from bot.locales.loader import load_locales

# Load environment configuration from .env if present
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("skinai.bot")


def build_dispatcher() -> Dispatcher:
    """Construct and configure aiogram Dispatcher with registered routers and memory storage."""
    load_locales()
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(common_router)
    dp.include_router(analysis_router)
    return dp


async def main() -> None:
    """Launch bot in long-polling mode."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        logger.error(
            "TELEGRAM_BOT_TOKEN is not configured! "
            "Please provide a valid bot token in your .env file or environment."
        )
        sys.exit(1)

    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )
    dp = build_dispatcher()

    logger.info("Starting SkinAI Telegram Bot polling...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
