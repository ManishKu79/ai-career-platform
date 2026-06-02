from fastapi import APIRouter, HTTPException, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase
from backend.database import get_database, db_manager
from backend.services.db_service import db_service
from backend.services.aggregation import aggregation_service
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin & Analytics"])


@router.get(
    "/health",
    summary="Comprehensive database health check"
)
async def database_health(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Runs complete database health check including:
    - MongoDB ping and latency
    - Write/read/delete test
    - Collection existence verification
    - Index verification
    Used by Docker HEALTHCHECK and monitoring systems.
    """
    return await db_service.health_check(db)


@router.get(
    "/stats",
    summary="Database and collection statistics"
)
async def database_stats(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns database-level and per-collection statistics."""
    import asyncio
    db_stats, coll_stats = await asyncio.gather(
        db_service.get_database_stats(db),
        db_service.get_collection_stats(db),
    )
    return {
        "database":     db_stats,
        "collections":  coll_stats,
    }


@router.get(
    "/pipeline/status",
    summary="Resume processing pipeline status"
)
async def pipeline_status(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns all resumes with their current pipeline stage."""
    resumes    = await db_service.get_resume_pipeline_status(db)
    funnel     = await aggregation_service.candidate_pipeline_summary(db)
    return {
        "pipeline_funnel": funnel,
        "resumes":         resumes,
    }


@router.get(
    "/analytics/score-distribution/{job_id}",
    summary="Score distribution for a job"
)
async def score_distribution(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns score histogram and tier distribution for a job."""
    return await aggregation_service.score_distribution_by_job(db, job_id)


@router.get(
    "/analytics/top-skills",
    summary="Most common skills across all resumes"
)
async def top_skills(
    limit: int = Query(default=20, le=50),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns the N most common skills across all uploaded resumes."""
    return await aggregation_service.top_skills_across_resumes(db, limit=limit)


@router.get(
    "/analytics/skill-gap/{job_id}",
    summary="Skill gap heatmap for a job"
)
async def skill_gap_heatmap(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns skill coverage and gap data for all candidates for a job."""
    return await aggregation_service.skill_gap_heatmap(db, job_id)


@router.get(
    "/analytics/funnel",
    summary="Hiring funnel conversion metrics"
)
async def hiring_funnel(
    job_id: Optional[str] = Query(default=None),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns hiring funnel metrics with conversion rates."""
    return await aggregation_service.hiring_funnel_metrics(db, job_id)


@router.post(
    "/maintenance/cleanup-scores",
    summary="Delete old score documents"
)
async def cleanup_scores(
    days_old: int = Query(default=90, ge=7, le=365),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Manually deletes score documents older than `days_old` days.
    TTL index handles this automatically, but this endpoint
    allows immediate manual cleanup.
    """
    deleted = await db_service.cleanup_old_scores(db, days_old)
    return {
        "deleted_count": deleted,
        "message": f"Deleted {deleted} score documents older than {days_old} days."
    }


@router.post(
    "/maintenance/rebuild-indexes",
    summary="Rebuild all database indexes"
)
async def rebuild_indexes(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Drops and rebuilds all indexes. Use after bulk data imports.
    WARNING: Index drops are irreversible. Use with caution.
    """
    try:
        await db_manager.initialize_indexes()
        verification = await db_service.verify_indexes(db)
        return {
            "message": "Indexes rebuilt successfully.",
            "verification": verification
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed: {str(e)}"
        )
