"""Application configuration and settings management using Pydantic Settings."""

from __future__ import annotations

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core configuration for SkinAI FastAPI backend."""

    PROJECT_NAME: str = "SkinAI Inference API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/v1"
    DEBUG: bool = False

    # Server binding
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # Database & Audit Telemetry
    DATABASE_URL: str = "sqlite:///./skinai_dev.db"
    TELEMETRY_SALT: str = "skinai_default_telemetry_salt_secret"
    API_SECRET_KEY: str = "skinai_secret_key_development_only"

    # Artifacts & Storage paths
    ARTIFACTS_DIR: str = "artifacts"
    DEFAULT_PROFILE: str = "smartphone_clinical"

    # Input constraints
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    MIN_IMAGE_DIMENSION: int = 128
    MAX_IMAGE_DIMENSION: int = 6000

    # Non-diagnostic medical disclaimers
    STATIC_DISCLAIMER_RU: str = (
        "SkinAI является исследовательским инструментом предварительного анализа "
        "и не заменяет очный визит к врачу-дерматологу."
    )
    STATIC_DISCLAIMER_EN: str = (
        "SkinAI is an experimental research tool for preliminary image analysis "
        "and does not replace an in-person consultation with a certified dermatologist."
    )
    STATIC_DISCLAIMER_KO: str = (
        "SkinAI는 예비 이미지 분석을 위한 실험용 연구 도구이며 "
        "공인 피부과 전문의의 대면 진료를 대신할 수 없습니다."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
