# backend/routers/scoring.py

from fastapi import APIRouter, HTTPException, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase
from backend.database import get_database
from backend.services.ats_scorer import ats_scorer
from backend.models.score_models import ATSScoreResult, BatchScoreRequest
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scoring", tags=["ATS Scoring"])


@router.post(
    "/score",
    response_model=ATSScoreResult,
    summary="Score one resume against one job"
)
async def score_resume(
    resume_file_id: str,
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Computes full ATS score for one resume vs one job description.

    Prerequisites:
    - Resume must have status 'skills_extracted'
    - Job must have status 'active'

    Returns complete score with component breakdown,
    skill gap analysis, and recruiter recommendation.
    """

    # ── Fetch documents ───────────────────────────────────────────────
    resume = await db["resumes"].find_one(
        {"file_id": resume_file_id},
        {"_id": 0}
    )
    if not resume:
        raise HTTPException(
            status_code=404,
            detail=f"Resume not found: {resume_file_id}"
        )

    job = await db["jobs"].find_one(
        {"job_id": job_id},
        {"_id": 0}
    )
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )

    # ── Validate pipeline completion ──────────────────────────────────
    if not resume.get("extracted_skills"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Resume {resume_file_id} not fully processed. "
                f"Run the full pipeline: upload → nlp → skills first."
            )
        )

    # ── Compute score ─────────────────────────────────────────────────
    try:
        result = ats_scorer.score(resume, job)
    except Exception as e:
        logger.error(f"Scoring failed: {resume_file_id} vs {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scoring computation failed: {str(e)}"
        )

    # ── Persist score to MongoDB ──────────────────────────────────────
    score_dict = result.model_dump(mode="json")
    await db["scores"].insert_one(score_dict)

    logger.info(
        f"Score saved: {result.score_id} | "
        f"{result.final_score_percent:.1f}% ({result.score_tier})"
    )

    return result


@router.post(
    "/batch",
    summary="Score all resumes against a job description"
)
async def batch_score(
    request: BatchScoreRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Scores multiple resumes against one job in one API call.

    If resume_file_ids is empty, scores ALL resumes in the database
    that have completed the skills extraction pipeline.

    Returns summary statistics and the top 20 scored candidates.
    """

    # ── Fetch job ─────────────────────────────────────────────────────
    job = await db["jobs"].find_one(
        {"job_id": request.job_id},
        {"_id": 0}
    )
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {request.job_id}"
        )

    # ── Fetch resumes ─────────────────────────────────────────────────
    if request.resume_file_ids:
        # Score specific resumes
        query = {
            "file_id": {"$in": request.resume_file_ids},
            "status": "skills_extracted"
        }
    else:
        # Score all processed resumes
        query = {"status": "skills_extracted"}

    cursor = db["resumes"].find(query, {"_id": 0})
    resumes = await cursor.to_list(length=500)

    if not resumes:
        return {
            "message": "No eligible resumes found for scoring",
            "job_id": request.job_id,
            "scored": 0
        }

    # ── Batch score ───────────────────────────────────────────────────
    results = ats_scorer.score_batch(
        resumes, job, threshold=request.score_threshold
    )

    # ── Persist all scores ────────────────────────────────────────────
    if results:
        score_dicts = [r.model_dump(mode="json") for r in results]
        await db["scores"].insert_many(score_dicts)

    # ── Compute summary statistics ────────────────────────────────────
    scores = [r.final_score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    above_threshold = sum(1 for s in scores if s >= settings_threshold)

    from backend.config import settings
    settings_threshold = settings.ATS_SCORE_THRESHOLD

    return {
        "job_id": request.job_id,
        "job_title": job.get("title", ""),
        "total_resumes": len(resumes),
        "scored": len(results),
        "average_score": round(avg_score * 100, 2),
        "above_threshold": sum(
            1 for r in results
            if r.final_score >= settings.ATS_SCORE_THRESHOLD
        ),
        "score_distribution": {
            "excellent": sum(1 for r in results if r.final_score >= 0.80),
            "good":      sum(1 for r in results if 0.65 <= r.final_score < 0.80),
            "fair":      sum(1 for r in results if 0.50 <= r.final_score < 0.65),
            "poor":      sum(1 for r in results if r.final_score < 0.50),
        },
        "top_candidates": [
            {
                "resume_file_id": r.resume_file_id,
                "final_score": r.final_score_percent,
                "score_tier":  r.score_tier,
                "passes":      r.passes_threshold,
            }
            for r in results[:20]
        ]
    }


@router.get(
    "/results/{resume_file_id}",
    summary="Get all scores for a resume"
)
async def get_resume_scores(
    resume_file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Returns all scoring results for a given resume across all jobs."""
    cursor = db["scores"].find(
        {"resume_file_id": resume_file_id},
        {"_id": 0}
    ).sort("scored_at", -1)

    scores = await cursor.to_list(length=50)
    if not scores:
        raise HTTPException(
            status_code=404,
            detail=f"No scores found for resume: {resume_file_id}"
        )
    return {"resume_file_id": resume_file_id, "scores": scores}


@router.get(
    "/leaderboard/{job_id}",
    summary="Get ranked candidates for a job"
)
async def get_leaderboard(
    job_id: str,
    limit: int = Query(default=20, le=100),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Returns ranked candidate list for a job, sorted by final_score desc.
    Used as the primary data source for Module 9's dashboard rankings page.
    """
    cursor = db["scores"].find(
        {
            "job_id": job_id,
            "final_score": {"$gte": min_score}
        },
        {"_id": 0}
    ).sort("final_score", -1).limit(limit)

    scores = await cursor.to_list(length=limit)

    if not scores:
        raise HTTPException(
            status_code=404,
            detail=f"No scores found for job: {job_id}"
        )

    return {
        "job_id": job_id,
        "total_candidates": len(scores),
        "leaderboard": scores
    }