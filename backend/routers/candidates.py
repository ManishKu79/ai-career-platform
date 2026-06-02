from fastapi import APIRouter, HTTPException, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase
from backend.database import get_database
from backend.services.ranker import candidate_ranker
from backend.models.ranking_models import (
    CandidateRanking,
    RankedCandidate,
    ShortlistRequest,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/candidates", tags=["Candidate Ranking"])


@router.post(
    "/rank/{job_id}",
    response_model=CandidateRanking,
    summary="Generate ranked candidate list for a job"
)
async def rank_candidates(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Runs the full ranking pipeline for all scored candidates
    for a given job. Fetches scores, enriches candidates,
    computes composite scores, sorts, assigns percentiles,
    and generates shortlists.

    Prerequisites:
    - Resumes must be processed through full pipeline
    - POST /scoring/batch must have been called for this job_id
    """

    try:
        ranking = await candidate_ranker.rank_candidates(job_id, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ranking failed for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ranking computation failed: {str(e)}"
        )

    if ranking.total_candidates == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No scored candidates found for job {job_id}. "
                f"Run POST /api/v1/scoring/batch first."
            )
        )

    # Persist ranking to MongoDB
    ranking_dict = ranking.model_dump(mode="json")
    await db["rankings"].insert_one(ranking_dict)

    logger.info(
        f"Ranking persisted for job {job_id}: "
        f"{ranking.total_candidates} candidates"
    )

    return ranking


@router.get(
    "/ranking/{job_id}",
    response_model=CandidateRanking,
    summary="Retrieve most recent ranking for a job"
)
async def get_ranking(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Returns the most recently generated ranking for a job.
    Use POST /rank/{job_id} to generate a fresh ranking.
    """

    # Find most recent ranking for this job
    ranking = await db["rankings"].find_one(
        {"job_id": job_id},
        {"_id": 0},
        sort=[("generated_at", -1)]  # Most recent first
    )

    if not ranking:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No ranking found for job {job_id}. "
                f"Run POST /api/v1/candidates/rank/{job_id} first."
            )
        )

    return ranking


@router.get(
    "/shortlist/{job_id}",
    summary="Get shortlisted candidates for a job"
)
async def get_shortlist(
    job_id: str,
    tier: str = Query(
        default="priority",
        description="Tier: priority | standard | reserve"
    ),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Returns candidates in the specified shortlist tier
    from the most recent ranking for this job.

    Tiers:
    - priority: ATS ≥ 80% — immediate interview
    - standard: ATS ≥ 65% — phone screen
    - reserve:  ATS ≥ 50% — hold for consideration
    """

    valid_tiers = {"priority", "standard", "reserve"}
    if tier not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier '{tier}'. Must be one of: {valid_tiers}"
        )

    ranking = await db["rankings"].find_one(
        {"job_id": job_id},
        {"_id": 0},
        sort=[("generated_at", -1)]
    )

    if not ranking:
        raise HTTPException(
            status_code=404,
            detail=f"No ranking found for job {job_id}."
        )

    # Get file_ids for the requested tier
    shortlist_key = f"{tier}_shortlist"
    shortlisted_ids = ranking.get(shortlist_key, [])

    # Fetch full candidate details for shortlisted IDs
    # Filter ranked_candidates list to only shortlisted candidates
    all_ranked = ranking.get("ranked_candidates", [])
    shortlisted = [
        c for c in all_ranked
        if c["file_id"] in shortlisted_ids
    ]

    return {
        "job_id": job_id,
        "tier": tier,
        "count": len(shortlisted),
        "candidates": shortlisted
    }


@router.get(
    "/detail/{file_id}",
    summary="Get ranking detail for a specific candidate"
)
async def get_candidate_detail(
    file_id: str,
    job_id: str = Query(..., description="Job ID this candidate was ranked for"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Returns the complete ranked candidate record for one candidate
    against a specific job, including all criteria breakdowns,
    skill gap, and recommendation.
    """

    ranking = await db["rankings"].find_one(
        {"job_id": job_id},
        {"_id": 0},
        sort=[("generated_at", -1)]
    )

    if not ranking:
        raise HTTPException(
            status_code=404,
            detail=f"No ranking found for job {job_id}."
        )

    # Search for candidate within ranked_candidates list
    all_ranked = ranking.get("ranked_candidates", [])
    candidate  = next(
        (c for c in all_ranked if c["file_id"] == file_id),
        None
    )

    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {file_id} not found in ranking for job {job_id}."
        )

    return candidate


@router.post(
    "/shortlist/custom",
    summary="Generate a custom shortlist with filters"
)
async def custom_shortlist(
    request: ShortlistRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Generates a custom shortlist with specific filters:
    - Minimum ATS score
    - Required skills (subset)
    - Number of candidates
    - Threshold-only flag

    This is used by recruiters who want to apply additional
    criteria beyond the standard tier thresholds.
    """

    ranking = await db["rankings"].find_one(
        {"job_id": request.job_id},
        {"ranked_candidates": 1, "_id": 0},
        sort=[("generated_at", -1)]
    )

    if not ranking:
        raise HTTPException(
            status_code=404,
            detail=f"No ranking found for job {request.job_id}."
        )

    all_ranked = ranking.get("ranked_candidates", [])

    # Apply filters
    filtered = []
    for candidate in all_ranked:
        # Filter by minimum ATS score
        if candidate.get("ats_score", 0) < request.min_ats_score:
            continue

        # Filter by threshold pass
        if request.threshold_only and not candidate.get("passes_threshold"):
            continue

        # Filter by required skills (candidate must have ALL specified skills)
        if request.required_skills:
            candidate_skills = set(candidate.get("matched_skills", []))
            required_set     = set(request.required_skills)
            if not required_set.issubset(candidate_skills):
                continue

        filtered.append(candidate)

        # Stop when we have enough
        if len(filtered) >= request.top_n:
            break

    return {
        "job_id": request.job_id,
        "filters_applied": {
            "min_ats_score":    request.min_ats_score,
            "required_skills":  request.required_skills,
            "threshold_only":   request.threshold_only,
            "top_n":            request.top_n,
        },
        "count": len(filtered),
        "candidates": filtered
    }


@router.get(
    "/statistics/{job_id}",
    summary="Get pool statistics for a job"
)
async def get_pool_statistics(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Returns score distribution statistics for the candidate pool.
    Used by the dashboard analytics page.
    """

    ranking = await db["rankings"].find_one(
        {"job_id": job_id},
        {
            "_id": 0,
            "score_statistics": 1,
            "tier_distribution": 1,
            "total_candidates": 1,
            "job_title": 1,
            "generated_at": 1
        },
        sort=[("generated_at", -1)]
    )

    if not ranking:
        raise HTTPException(
            status_code=404,
            detail=f"No ranking found for job {job_id}."
        )

    return ranking
