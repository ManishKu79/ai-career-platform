# backend/main.py

from contextlib import asynccontextmanager
from datetime import datetime
import logging
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import db_manager

# Routers
from backend.routers import (
    upload,
    nlp,
    skills,
    jobs,
    scoring,
    candidates,
    admin,
    pipeline,
)

# Middleware
from backend.middleware.cors import configure_cors
from backend.middleware.logging_middleware import (
    RequestIDMiddleware,
    TimingMiddleware,
)
from backend.middleware.error_handler import register_exception_handlers

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

APP_START_TIME = time.time()

# --------------------------------------------------
# Lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting AI Career Intelligence Platform...")

    try:
        # Connect database
        db_manager.connect()

        # Initialize indexes if available
        if hasattr(db_manager, "initialize_indexes"):
            await db_manager.initialize_indexes()

        logger.info("Database connected successfully")

    except Exception as e:
        logger.exception(f"Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down application...")

    try:
        db_manager.disconnect()
        logger.info("Database disconnected")

    except Exception as e:
        logger.exception(f"Shutdown error: {e}")

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="AI Career Intelligence Platform",
    version="1.0.0",
    description="""
AI-powered ATS and Candidate Ranking Platform

Features:
- Resume Upload
- NLP Processing
- Skill Extraction
- ATS Scoring
- Candidate Ranking
- Admin Analytics
""",
    lifespan=lifespan,
)

# --------------------------------------------------
# Middleware Registration
# --------------------------------------------------

configure_cors(app)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)

register_exception_handlers(app)

# --------------------------------------------------
# Router Registration
# --------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(upload.router, prefix=API_PREFIX)
app.include_router(nlp.router, prefix=API_PREFIX)
app.include_router(skills.router, prefix=API_PREFIX)
app.include_router(jobs.router, prefix=API_PREFIX)
app.include_router(scoring.router, prefix=API_PREFIX)
app.include_router(candidates.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)
app.include_router(pipeline.router, prefix=API_PREFIX)

# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/", tags=["System"])
async def root():
    return {
        "service": "AI Career Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }

# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health", tags=["System"])
async def health():

    uptime = int(time.time() - APP_START_TIME)

    return {
        "status": "alive",
        "uptime_s": uptime,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-career-platform",
        "version": "1.0.0",
    }

# --------------------------------------------------
# Readiness Check
# --------------------------------------------------

@app.get("/ready", tags=["System"])
async def ready():

    try:

        if hasattr(db_manager, "ping"):
            is_ready = await db_manager.ping()

            if not is_ready:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "database": "unreachable",
                    }
                )

        return {
            "status": "ready",
            "database": "connected",
        }

    except Exception as e:

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
            }
        )

# --------------------------------------------------
# Metrics Endpoint
# --------------------------------------------------

@app.get("/metrics", tags=["System"])
async def metrics():

    uptime = int(time.time() - APP_START_TIME)

    return {
        "uptime_seconds": uptime,
        "environment": {
            "api_host": settings.API_HOST,
            "api_port": settings.API_PORT,
            "spacy_model": settings.SPACY_MODEL,
            "ats_threshold": settings.ATS_SCORE_THRESHOLD,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        }
    }

# --------------------------------------------------
# Development Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info",
    )