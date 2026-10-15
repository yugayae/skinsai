"""FastAPI application entrypoint and server lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.endpoints.health import get_health
from backend.app.api.v1.router import api_router
from backend.app.core.config import settings
from backend.app.db.database import init_db
from backend.app.services.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("skinai.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan managing database startup and ONNX model registry."""
    logger.info("Initializing SkinAI database schema...")
    init_db()

    logger.info("Initializing SkinAI model registry...")
    app.state.registry = ModelRegistry()
    loaded = app.state.registry.list_loaded_profiles()
    logger.info(f"SkinAI initialized successfully. Active profiles: {loaded}")

    yield

    logger.info("Shutting down SkinAI inference application.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "SkinAI Research Inference Service: Dual-profile skin lesion image "
        "classification with evidence-based Glasgow anamnesis and calibrated safety abstention."
    ),
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level health check probe for load balancers / Docker / Kubernetes
app.add_api_route("/health", get_health, methods=["GET"], tags=["Health"])

# Mount versioned API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Sanitize unexpected exceptions to prevent internal path and secret leakage."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An internal error occurred during request processing.",
        },
    )
