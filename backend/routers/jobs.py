# backend/routers/jobs.py

from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from backend.database import get_database
from backend.models.job_models import JobCreate, JobResponse
from backend.services.nlp_pipeline import nlp_pipeline
from backend.services.skill_extractor import skill_extractor
from backend.services.parser import resume_parser
import logging
import re

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["Job Descriptions"])


@router.post(
    "/",
    response_model=JobResponse,
    status_code=201,
    summary="Create a new job description"
)
async def create_job(
    job_data: JobCreate,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Creates a new job description and runs NLP + skill extraction pipeline.

    The job description undergoes the same processing as a resume:
    1. Text cleaning
    2. NLP feature extraction
    3. Skill extraction
    4. Storage in MongoDB jobs collection
    """

    # ── Clean job description text ────────────────────────────────────
    cleaned = resume_parser._clean_text(job_data.description)
    job_data.cleaned_description = cleaned

    # ── Run NLP pipeline ──────────────────────────────────────────────
    try:
        nlp_features = nlp_pipeline.process(cleaned)
        job_data.nlp_features = nlp_features.model_dump(mode="json")
    except Exception as e:
        logger.error(f"NLP failed for job {job_data.job_id}: {e}")

    # ── Run skill extraction ──────────────────────────────────────────
    try:
        skill_result = skill_extractor.extract(nlp_features, cleaned)
        job_data.skill_extraction_result = skill_result.model_dump(mode="json")
        job_data.required_skills = skill_result.all_skills
        job_data.status = "active"
    except Exception as e:
        logger.error(f"Skill extraction failed for job {job_data.job_id}: {e}")

    # ── Persist to MongoDB ────────────────────────────────────────────
    job_dict = job_data.model_dump(mode="json")

    # Check for duplicate titles from same company
    existing = await db["jobs"].find_one({
        "title": job_data.title,
        "company": job_data.company,
        "is_active": True
    })

    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Active job '{job_data.title}' at {job_data.company} "
                f"already exists. Use PUT to update."
            )
        )

    await db["jobs"].insert_one(job_dict)

    logger.info(
        f"Job created: {job_data.job_id} | "
        f"'{job_data.title}' | "
        f"skills={len(job_data.required_skills)}"
    )

    return JobResponse(
        job_id=job_data.job_id,
        title=job_data.title,
        company=job_data.company,
        status=job_data.status,
        required_skills=job_data.required_skills,
        created_at=job_data.created_at,
        is_active=job_data.is_active,
        message=(
            f"Job '{job_data.title}' created. "
            f"Found {len(job_data.required_skills)} required skills."
        )
    )


@router.get(
    "/",
    summary="List all active job descriptions"
)
async def list_jobs(
    db: AsyncIOMotorDatabase = Depends(get_database),
    active_only: bool = True
):
    """Returns all job descriptions, optionally filtered to active only."""
    query = {"is_active": True} if active_only else {}

    cursor = db["jobs"].find(
        query,
        {
            "_id": 0, "job_id": 1, "title": 1, "company": 1,
            "status": 1, "required_skills": 1, "created_at": 1,
            "employment_type": 1, "remote_policy": 1
        }
    ).sort("created_at", -1)

    jobs = await cursor.to_list(length=100)
    return {"total": len(jobs), "jobs": jobs}


@router.get(
    "/{job_id}",
    summary="Get a specific job description"
)
async def get_job(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns full job document including NLP features and skills."""
    job = await db["jobs"].find_one(
        {"job_id": job_id},
        {"_id": 0}
    )
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.delete(
    "/{job_id}",
    summary="Deactivate a job description"
)
async def deactivate_job(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Soft-deletes a job by setting is_active=False."""
    result = await db["jobs"].update_one(
        {"job_id": job_id},
        {"$set": {"is_active": False, "status": "closed"}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"message": f"Job {job_id} deactivated.", "job_id": job_id}