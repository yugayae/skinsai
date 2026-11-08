@echo off
chcp 65001 > nul
echo ========================================================
echo        Запуск сервисов SkinAI (Backend + Telegram Bot)
echo ========================================================
echo.

echo [1/2] Запуск бэкенда FastAPI (http://127.0.0.1:8000)...
start "SkinAI - FastAPI Backend" cmd /k "chcp 65001 > nul && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak > nul

echo [2/2] Запуск Telegram-бота (aiogram 3)...
start "SkinAI - Telegram Bot" cmd /k "chcp 65001 > nul && python -m bot.main"

echo.
echo ========================================================
echo Оба сервиса запущены в отдельных окнах:
echo - FastAPI Docs: http://127.0.0.1:8000/docs
echo - Telegram Bot: готов к сообщениям (отправьте /start)
echo ========================================================
pause

