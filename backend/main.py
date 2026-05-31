# backend/main.py

# FastAPI: the web framework class
from fastapi import FastAPI

# contextlib.asynccontextmanager: creates async context manager for lifespan
from contextlib import asynccontextmanager

# Our database manager
from backend.database import db_manager

# Our router modules
from backend.routers import upload

# Settings
from backend.config import settings

# Logging
import logging

# Configure root logger format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    
    Code BEFORE yield: runs at application startup
    Code AFTER yield:  runs at application shutdown
    
    This replaces the deprecated @app.on_event("startup") pattern.
    Guarantees DB is connected before first request and disconnected cleanly.
    """
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Starting AI Career Intelligence Platform API...")
    db_manager.connect()
    logger.info("Database connected. API ready.")

    # yield hands control to FastAPI to handle requests
    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down API...")
    db_manager.disconnect()
    logger.info("Shutdown complete.")


# Create FastAPI app instance
app = FastAPI(
    title="AI Career Intelligence Platform",
    description="""
    Production-grade ATS (Applicant Tracking System) API.
    
    ## Features
    * **Resume Parsing**: Upload PDF and DOCX resumes
    * **NLP Processing**: Extract skills, entities, and features
    * **ATS Scoring**: TF-IDF + Cosine Similarity scoring
    * **Candidate Ranking**: Objective candidate comparison
    
    ## Workflow
    1. Upload resumes via `/upload/resume`
    2. Create job descriptions via `/jobs`
    3. Score resumes vs jobs via `/scoring`
    4. View rankings via `/candidates`
    """,
    version="1.0.0",
    lifespan=lifespan  # Register our lifespan handler
)


# ── Register Routers ──────────────────────────────────────────────────
# Each router handles a domain of the application
# prefix="/api/v1" namespaces all routes for versioning
app.include_router(upload.router, prefix="/api/v1")


# ── Health Check Endpoint ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for Docker, load balancers, and monitoring.
    Returns 200 if API is running. Production systems check this every 30s.
    """
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "service": "AI Career Intelligence Platform"
    }


# ── Root Endpoint ─────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root():
    """API root — redirects users to documentation."""
    return {
        "message": "AI Career Intelligence Platform API",
        "docs": "/docs",
        "health": "/health"
    }


# ── Development Server ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",        # module:app_variable string
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD  # Hot reload in development
    )