# backend/main.py

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from backend.config import settings
from backend.database import db_manager

from backend.routers import (
    upload,
    nlp,
    skills,
    jobs,
    scoring,
    candidates,
    admin
)

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Application Lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Startup:
        - Connect MongoDB
        - Create indexes

    Shutdown:
        - Close MongoDB connection
    """

    logger.info("Starting AI Career Intelligence Platform API...")

    try:
        # Connect database
        db_manager.connect()

        # Initialize indexes if implemented
        if hasattr(db_manager, "initialize_indexes"):
            await db_manager.initialize_indexes()

        logger.info(
            "Database connected and indexes initialized successfully."
        )

    except Exception as e:
        logger.exception(f"Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down API...")

    try:
        db_manager.disconnect()
        logger.info("Database connection closed.")

    except Exception as e:
        logger.exception(f"Shutdown error: {e}")

# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="AI Career Intelligence Platform",
    description="""
Production-grade ATS and Candidate Intelligence Platform.

Features:
- Resume Upload & Parsing
- NLP Processing
- Skill Extraction
- ATS Scoring
- Candidate Ranking
- Job Management
- Analytics Dashboard
- Admin Operations
""",
    version="1.0.0",
    lifespan=lifespan
)

# --------------------------------------------------
# Router Registration
# --------------------------------------------------

app.include_router(upload.router, prefix="/api/v1")
app.include_router(nlp.router, prefix="/api/v1")
app.include_router(skills.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(scoring.router, prefix="/api/v1")
app.include_router(candidates.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")

# --------------------------------------------------
# System Endpoints
# --------------------------------------------------

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "AI Career Intelligence Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Career Intelligence Platform",
        "version": "1.0.0"
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
        reload=settings.API_RELOAD
    )