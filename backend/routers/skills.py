from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from backend.database import get_database
from backend.services.skill_extractor import skill_extractor
from backend.services.nlp_pipeline import nlp_pipeline
from backend.models.skill_models import SkillExtractionResult
from nlp.skill_taxonomy import (
    SKILL_TAXONOMY, CATEGORY_WEIGHTS, ALL_SKILLS
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["Skill Extraction"])


@router.post(
    "/extract/{file_id}",
    summary="Extract skills from a processed resume",
    description="""
    Runs skill extraction on a resume that has already completed NLP processing.

    Prerequisites:
    - Resume must have status 'nlp_processed'
    - Call POST /nlp/process/{file_id} first if status is 'uploaded'

    Returns full skill extraction result including categories,
    scores, frequencies, and extraction methods.
    """
)
async def extract_skills(
    file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Extract and store skills for a single resume."""

    # ── Fetch resume with NLP features ───────────────────────────────
    resume = await db["resumes"].find_one(
        {"file_id": file_id},
        {"nlp_features": 1, "cleaned_text": 1, "status": 1, "file_id": 1}
    )

    if not resume:
        raise HTTPException(status_code=404, detail=f"Resume not found: {file_id}")

    # Validate NLP features exist
    if not resume.get("nlp_features"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"NLP features missing for {file_id}. "
                f"Run POST /api/v1/nlp/process/{file_id} first."
            )
        )

    # ── Reconstruct NLPFeatures object ───────────────────────────────
    # MongoDB returns dict — we reconstruct the Pydantic model
    # for type safety in the extractor
    try:
        from backend.models.nlp_models import NLPFeatures
        nlp_features = NLPFeatures(**resume["nlp_features"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse stored NLP features: {str(e)}"
        )

    cleaned_text = resume.get("cleaned_text", "")

    # ── Run skill extraction ──────────────────────────────────────────
    try:
        result = skill_extractor.extract(nlp_features, cleaned_text)
    except Exception as e:
        logger.error(f"Skill extraction failed for {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Skill extraction failed: {str(e)}"
        )

    # ── Persist to MongoDB ────────────────────────────────────────────
    result_dict = result.model_dump(mode="json")

    await db["resumes"].update_one(
        {"file_id": file_id},
        {
            "$set": {
                "skill_extraction_result": result_dict,
                "extracted_skills": result.all_skills,
                "status": "skills_extracted"
            }
        }
    )

    logger.info(
        f"Skills extracted for {file_id}: "
        f"{result.total_skills_found} skills found"
    )

    return {
        "file_id": file_id,
        "status": "skills_extracted",
        "total_skills_found": result.total_skills_found,
        "top_skills": result.top_skills,
        "skills_by_category": result.skills_by_category,
        "skills_per_category": result.skills_per_category,
        "category_coverage": result.category_coverage,
        "processing_time_seconds": result.processing_time_seconds
    }


@router.get(
    "/{file_id}",
    response_model=SkillExtractionResult,
    summary="Retrieve extracted skills for a resume"
)
async def get_skills(
    file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Retrieve previously extracted skills for a resume."""

    resume = await db["resumes"].find_one(
        {"file_id": file_id},
        {"skill_extraction_result": 1, "_id": 0}
    )

    if not resume:
        raise HTTPException(status_code=404, detail=f"Resume not found: {file_id}")

    if not resume.get("skill_extraction_result"):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Skills not yet extracted for {file_id}. "
                f"Call POST /api/v1/skills/extract/{file_id} first."
            )
        )

    return resume["skill_extraction_result"]


@router.post(
    "/gap-analysis",
    summary="Compare resume skills against job requirements"
)
async def skill_gap_analysis(
    resume_file_id: str,
    job_file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Computes skill gap between a resume and job description.

    Returns:
    - matched: skills candidate has that job requires
    - missing: skills job requires that candidate lacks
    - extra:   skills candidate has beyond job requirements
    - match_rate: percentage of job skills covered
    """

    # Fetch resume skills
    resume = await db["resumes"].find_one(
        {"file_id": resume_file_id},
        {"extracted_skills": 1}
    )

    # Fetch job skills
    job = await db["jobs"].find_one(
        {"job_id": job_file_id},
        {"required_skills": 1}
    )

    if not resume:
        raise HTTPException(
            status_code=404,
            detail=f"Resume not found: {resume_file_id}"
        )
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_file_id}"
        )

    resume_skills = set(resume.get("extracted_skills", []))
    job_skills = set(job.get("required_skills", []))

    gap = skill_extractor.compute_skill_gap(resume_skills, job_skills)

    return {
        "resume_file_id": resume_file_id,
        "job_file_id": job_file_id,
        "gap_analysis": gap,
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(job_skills)
    }


@router.get(
    "/taxonomy/browse",
    summary="Browse the full skill taxonomy"
)
async def browse_taxonomy():
    """Returns the complete skill taxonomy organized by category."""
    return {
        "total_skills": len(ALL_SKILLS),
        "categories": {
            category: {
                "skills": skills,
                "count": len(skills),
                "weight": CATEGORY_WEIGHTS.get(category, 1.0)
            }
            for category, skills in SKILL_TAXONOMY.items()
        }
    }


@router.post(
    "/extract/batch",
    summary="Extract skills from all NLP-processed resumes"
)
async def batch_extract_skills(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Process all resumes with status 'nlp_processed'."""

    from backend.models.nlp_models import NLPFeatures

    cursor = db["resumes"].find(
        {"status": "nlp_processed"},
        {"file_id": 1, "nlp_features": 1, "cleaned_text": 1}
    )
    resumes = await cursor.to_list(length=200)

    if not resumes:
        return {"message": "No resumes pending skill extraction", "processed": 0}

    processed, errors = 0, []

    for resume in resumes:
        try:
            nlp_features = NLPFeatures(**resume["nlp_features"])
            result = skill_extractor.extract(
                nlp_features,
                resume.get("cleaned_text", "")
            )
            result_dict = result.model_dump(mode="json")

            await db["resumes"].update_one(
                {"file_id": resume["file_id"]},
                {
                    "$set": {
                        "skill_extraction_result": result_dict,
                        "extracted_skills": result.all_skills,
                        "status": "skills_extracted"
                    }
                }
            )
            processed += 1

        except Exception as e:
            errors.append({"file_id": resume["file_id"], "error": str(e)})
            logger.error(f"Batch skill extraction failed for {resume['file_id']}: {e}")

    return {"processed": processed, "errors": len(errors), "error_details": errors}
