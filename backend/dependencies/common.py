# backend/dependencies/common.py

from fastapi import Query, Depends, HTTPException, status

# motor database type
from motor.motor_asyncio import AsyncIOMotorDatabase

# Our database getter
from backend.database import get_database

# pydantic: data model for common parameters
from pydantic import BaseModel

# typing: type annotations
from typing import Optional

# Our settings
from backend.config import settings


class PaginationParams(BaseModel):
    """
    Standardized pagination parameters used across all list endpoints.

    Using a Pydantic model allows:
    - Default values
    - Validation (skip ≥ 0, limit ≤ MAX)
    - Type coercion (string query params → int)
    - Reuse across all paginated endpoints

    Convention: skip + limit pagination (offset-based)
    skip=0,  limit=20 → first page
    skip=20, limit=20 → second page
    skip=40, limit=20 → third page
    """
    skip:  int = 0    # Number of records to skip
    limit: int = 20   # Maximum records to return


def get_pagination(
    skip:  int = Query(default=0,  ge=0,    description="Records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return")
) -> PaginationParams:
    """
    FastAPI dependency for paginated list endpoints.

    Query parameter constraints:
    - skip:  ge=0         → must be ≥ 0 (no negative skip)
    - limit: ge=1, le=100 → between 1 and 100

    FastAPI enforces these via Pydantic before the route handler runs.
    Clients sending skip=-1 or limit=999 get a 422 error automatically.

    Usage:
        @router.get("/items")
        async def list_items(pagination = Depends(get_pagination)):
            return db.find().skip(pagination.skip).limit(pagination.limit)

    Args:
        skip:  Query parameter from URL: ?skip=0
        limit: Query parameter from URL: ?limit=20

    Returns:
        PaginationParams object with validated values
    """
    return PaginationParams(skip=skip, limit=limit)


async def get_resume_or_404(
    file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> dict:
    """
    Dependency that fetches a resume by file_id or raises HTTP 404.

    Pattern: "get_X_or_404" dependencies eliminate boilerplate in routes.
    Without: Every route that needs a resume has 5 lines of fetch + check.
    With:    One line `resume = Depends(get_resume_or_404)`.

    The dependency is cached within a single request lifecycle —
    if two route handlers use it in the same request, MongoDB is
    only queried once.

    Args:
        file_id: Resume file identifier from path parameter
        db:      Async database from get_database dependency

    Returns:
        Resume document dict from MongoDB

    Raises:
        HTTPException 404: If no resume found with this file_id
    """
    resume = await db["resumes"].find_one(
        {"file_id": file_id},
        {"_id": 0}  # Exclude MongoDB internal _id
    )

    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume not found: {file_id}. "
                   f"Upload a resume first via POST /api/v1/upload/resume"
        )

    return resume


async def get_job_or_404(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> dict:
    """
    Dependency that fetches a job by job_id or raises HTTP 404.

    Args:
        job_id: Job identifier from path parameter
        db:     Async database from get_database dependency

    Returns:
        Job document dict from MongoDB

    Raises:
        HTTPException 404: If no job found with this job_id
    """
    job = await db["jobs"].find_one(
        {"job_id": job_id},
        {"_id": 0}
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}. "
                   f"Create a job first via POST /api/v1/jobs/"
        )

    return job


def validate_pipeline_stage(
    required_status: str,
    current_status: str,
    file_id: str
) -> None:
    """
    Validates a resume has completed a required pipeline stage.

    Pipeline order:
    uploaded → nlp_processed → skills_extracted → scored

    If a route requires 'skills_extracted' but the resume
    is only 'nlp_processed', raises 422 with actionable message.

    Args:
        required_status: Minimum status required ("nlp_processed", etc.)
        current_status:  Current status of the resume document
        file_id:         Resume file_id for error messages

    Raises:
        HTTPException 422: If current_status is below required_status
    """

    # Define pipeline stage order
    # Higher index = further in the pipeline
    stage_order = {
        "uploaded":         0,
        "nlp_processed":    1,
        "skills_extracted": 2,
        "scored":           3,
    }

    required_stage = stage_order.get(required_status, 0)
    current_stage  = stage_order.get(current_status, 0)

    # Map stage to the endpoint that advances it
    stage_endpoints = {
        "nlp_processed":    f"POST /api/v1/nlp/process/{file_id}",
        "skills_extracted": f"POST /api/v1/skills/extract/{file_id}",
        "scored":           f"POST /api/v1/scoring/score?resume_file_id={file_id}",
    }

    if current_stage < required_stage:
        next_step = stage_endpoints.get(required_status, "the processing pipeline")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Resume {file_id} has status '{current_status}' "
                f"but '{required_status}' is required. "
                f"Run: {next_step}"
            )
        )
