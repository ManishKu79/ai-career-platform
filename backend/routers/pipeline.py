# backend/routers/pipeline.py

# fastapi: all required imports for a complex router
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    BackgroundTasks,
    status,
)

# motor database
from motor.motor_asyncio import AsyncIOMotorDatabase

# Our database dependency
from backend.database import get_database

# All processing services
from backend.services.parser import resume_parser
from backend.services.nlp_pipeline import nlp_pipeline
from backend.services.skill_extractor import skill_extractor
from backend.services.ats_scorer import ats_scorer
from backend.services.ranker import candidate_ranker

# NLP models for type reconstruction
from backend.models.nlp_models import NLPFeatures

# logging and time
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline Orchestration"])


@router.post(
    "/process",
    status_code=status.HTTP_201_CREATED,
    summary="Full pipeline: upload → parse → NLP → skills → score",
    description="""
    Single endpoint that runs the complete processing pipeline:

    **Stage 1 — Parse:** Extract text from PDF/DOCX file
    **Stage 2 — NLP:** Tokenization, POS, NER, lemmatization
    **Stage 3 — Skills:** Taxonomy matching + fuzzy extraction
    **Stage 4 — Score:** TF-IDF cosine similarity + multi-component ATS score
    **Stage 5 — Store:** Persist all results to MongoDB

    Returns complete processing result in one API call.
    Use `run_ranking=true` to also trigger candidate re-ranking.
    """
)
async def process_resume_full_pipeline(
    # File upload — required multipart field
    file: UploadFile = File(..., description="PDF or DOCX resume file"),

    # Job ID to score against — passed as form field alongside file
    # Form() is used instead of Body() because we're in multipart context
    job_id: str = Form(..., description="Job ID to score the resume against"),

    # Optional: trigger ranking after scoring
    run_ranking: bool = Form(default=False, description="Re-rank all candidates after scoring"),

    # Database dependency
    db: AsyncIOMotorDatabase = Depends(get_database),

    # Background tasks for async operations
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Executes the complete 5-stage resume processing pipeline.

    Each stage builds on the previous:
    parse → nlp → skills → score → (rank)

    Errors at any stage return immediately with details
    about which stage failed and why.
    """
    pipeline_start = time.time()
    stage_timings  = {}

    # ── Validate job exists before processing ────────────────────────
    job = await db["jobs"].find_one(
        {"job_id": job_id},
        {"_id": 0}
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Job '{job_id}' not found. "
                f"Create it first via POST /api/v1/jobs/"
            )
        )

    # ────────────────────────────────────────────────────────────────
    # STAGE 1: PARSE
    # ────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info(f"Pipeline Stage 1: Parsing '{file.filename}'")

    try:
        # Read file bytes
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # Parse resume → ResumeCreate Pydantic model
        parsed = resume_parser.parse(
            file_bytes=file_bytes,
            filename=file.filename
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Stage 1 (Parse) failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Parse stage error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage 1 (Parse) unexpected error: {str(e)}"
        )

    stage_timings["parse_ms"] = round((time.time() - stage_start) * 1000, 2)
    logger.info(
        f"Stage 1 complete: {parsed.metadata.word_count} words "
        f"in {stage_timings['parse_ms']}ms"
    )

    # ────────────────────────────────────────────────────────────────
    # STAGE 2: NLP PROCESSING
    # ────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info(f"Pipeline Stage 2: NLP processing file_id={parsed.file_id}")

    try:
        nlp_features = nlp_pipeline.process(parsed.cleaned_text)

    except Exception as e:
        logger.error(f"NLP stage error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage 2 (NLP) failed: {str(e)}"
        )

    stage_timings["nlp_ms"] = round((time.time() - stage_start) * 1000, 2)
    logger.info(
        f"Stage 2 complete: {nlp_features.total_tokens} tokens, "
        f"{len(nlp_features.entities)} entities "
        f"in {stage_timings['nlp_ms']}ms"
    )

    # ────────────────────────────────────────────────────────────────
    # STAGE 3: SKILL EXTRACTION
    # ────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info(f"Pipeline Stage 3: Skill extraction")

    try:
        skill_result = skill_extractor.extract(
            nlp_features,
            parsed.cleaned_text
        )

    except Exception as e:
        logger.error(f"Skill extraction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage 3 (Skills) failed: {str(e)}"
        )

    stage_timings["skills_ms"] = round((time.time() - stage_start) * 1000, 2)
    logger.info(
        f"Stage 3 complete: {skill_result.total_skills_found} skills "
        f"in {stage_timings['skills_ms']}ms"
    )

    # ────────────────────────────────────────────────────────────────
    # STAGE 4: PERSIST TO MONGODB
    # ────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info(f"Pipeline Stage 4: Persisting to MongoDB")

    try:
        # Build complete resume document
        resume_doc = parsed.model_dump(mode="json")
        resume_doc["nlp_features"]            = nlp_features.model_dump(mode="json")
        resume_doc["skill_extraction_result"] = skill_result.model_dump(mode="json")
        resume_doc["extracted_skills"]        = skill_result.all_skills
        resume_doc["status"]                  = "skills_extracted"
        resume_doc["upload_timestamp"]        = datetime.utcnow().isoformat()

        # Upsert by file_id — idempotent if same file re-uploaded
        await db["resumes"].update_one(
            {"file_id": parsed.file_id},
            {"$set": resume_doc},
            upsert=True
        )

    except Exception as e:
        logger.error(f"MongoDB persist error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage 4 (Persist) failed: {str(e)}"
        )

    stage_timings["persist_ms"] = round((time.time() - stage_start) * 1000, 2)

    # ────────────────────────────────────────────────────────────────
    # STAGE 5: ATS SCORING
    # ────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info(f"Pipeline Stage 5: ATS scoring")

    ats_result = None
    try:
        ats_result = ats_scorer.score(resume_doc, job)

        # Upsert score (handles re-scoring the same resume)
        from backend.services.db_service import db_service
        await db_service.upsert_score(db, ats_result.model_dump(mode="json"))

        # Update resume status to scored
        await db["resumes"].update_one(
            {"file_id": parsed.file_id},
            {"$set": {"status": "scored"}}
        )

    except Exception as e:
        logger.error(f"Scoring error: {e}", exc_info=True)
        # Scoring failure is non-fatal — return partial result
        logger.warning("Scoring failed — returning parse and NLP results only")

    stage_timings["score_ms"] = round((time.time() - stage_start) * 1000, 2)

    # ────────────────────────────────────────────────────────────────
    # OPTIONAL BACKGROUND: RE-RANK ALL CANDIDATES
    # ────────────────────────────────────────────────────────────────
    if run_ranking and ats_result:
        # Add ranking as a background task — runs AFTER response is sent
        # Client gets the score result immediately without waiting for ranking
        background_tasks.add_task(
            _background_rank_candidates,
            job_id=job_id,
            db=db
        )
        logger.info(f"Background ranking task queued for job {job_id}")

    # ── Assemble final response ───────────────────────────────────────
    total_ms = round((time.time() - pipeline_start) * 1000, 2)

    response = {
        "pipeline_status": "complete",
        "file_id":         parsed.file_id,
        "job_id":          job_id,
        "stage_timings":   stage_timings,
        "total_ms":        total_ms,

        # Stage 1 results
        "parse_result": {
            "candidate_name":  parsed.candidate_name,
            "candidate_email": parsed.candidate_email,
            "filename":        parsed.metadata.filename,
            "file_type":       parsed.metadata.file_type,
            "word_count":      parsed.metadata.word_count,
            "page_count":      parsed.metadata.page_count,
        },

        # Stage 2 results
        "nlp_summary": {
            "total_tokens":      nlp_features.total_tokens,
            "unique_tokens":     nlp_features.unique_tokens,
            "lexical_diversity": nlp_features.lexical_diversity,
            "entities_found":    len(nlp_features.entities),
            "organizations":     nlp_features.organizations[:5],
            "achievement_count": len(nlp_features.achievement_sentences),
            "top_words":         list(nlp_features.word_frequency.items())[:10],
        },

        # Stage 3 results
        "skill_summary": {
            "total_skills_found": skill_result.total_skills_found,
            "top_skills":         skill_result.top_skills[:10],
            "skills_by_category": skill_result.skills_per_category,
        },

        # Stage 5 results (None if scoring failed)
        "ats_score": {
            "final_score":         ats_result.final_score if ats_result else None,
            "final_score_percent": ats_result.final_score_percent if ats_result else None,
            "score_tier":          ats_result.score_tier if ats_result else None,
            "passes_threshold":    ats_result.passes_threshold if ats_result else None,
            "recommendation":      ats_result.recommendation if ats_result else None,
            "matched_keywords":    ats_result.matched_keywords[:10] if ats_result else [],
            "skill_gap": {
                "matched": ats_result.skill_gap.get("matched", []) if ats_result else [],
                "missing": ats_result.skill_gap.get("missing", []) if ats_result else [],
                "match_rate": ats_result.skill_gap.get("match_rate", 0) if ats_result else 0,
            },
        } if ats_result else None,

        "ranking_queued": run_ranking and ats_result is not None,
    }

    logger.info(
        f"Full pipeline complete: {parsed.file_id[:8]}... | "
        f"score={ats_result.final_score_percent:.1f}% | "
        f"total={total_ms}ms"
    )

    return response


async def _background_rank_candidates(job_id: str, db: AsyncIOMotorDatabase):
    """
    Background task: re-ranks all candidates for a job.
    Runs after the HTTP response is sent to the client.

    This prevents the ranking computation (which requires fetching
    all scored resumes) from blocking the pipeline endpoint response.

    Args:
        job_id: Job to re-rank candidates for
        db:     Database connection (passed from request context)
    """
    try:
        logger.info(f"Background task: re-ranking candidates for job {job_id}")
        ranking = await candidate_ranker.rank_candidates(job_id, db)

        # Persist updated ranking
        ranking_dict = ranking.model_dump(mode="json")
        await db["rankings"].update_one(
            {"job_id": job_id},
            {"$set": ranking_dict},
            upsert=True
        )

        logger.info(
            f"Background ranking complete: {ranking.total_candidates} "
            f"candidates ranked for job {job_id}"
        )
    except Exception as e:
        logger.error(f"Background ranking failed for job {job_id}: {e}")


@router.post(
    "/process/batch",
    summary="Batch process multiple resumes against a job",
    description="""
    Uploads and processes multiple resume files in one request.
    Returns a summary of all processing results.

    Maximum: 20 files per batch request.
    """
)
async def batch_process_resumes(
    files: list[UploadFile] = File(..., description="PDF or DOCX resume files"),
    job_id: str = Form(..., description="Job ID to score all resumes against"),
    db: AsyncIOMotorDatabase = Depends(get_database),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Processes multiple resumes through the full pipeline.
    Returns per-file results and aggregate statistics.
    """

    # Enforce batch size limit
    if len(files) > 20:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch limit is 20 files. Received {len(files)}."
        )

    # Validate job exists
    job = await db["jobs"].find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )

    results     = []
    errors      = []
    file_ids    = []
    batch_start = time.time()

    for file in files:
        try:
            file_bytes = await file.read()

            # Full pipeline for each file
            parsed       = resume_parser.parse(file_bytes, file.filename)
            nlp_features = nlp_pipeline.process(parsed.cleaned_text)
            skill_result = skill_extractor.extract(nlp_features, parsed.cleaned_text)

            # Build and persist document
            resume_doc = parsed.model_dump(mode="json")
            resume_doc.update({
                "nlp_features":            nlp_features.model_dump(mode="json"),
                "skill_extraction_result": skill_result.model_dump(mode="json"),
                "extracted_skills":        skill_result.all_skills,
                "status":                  "skills_extracted",
                "upload_timestamp":        datetime.utcnow().isoformat(),
            })

            await db["resumes"].update_one(
                {"file_id": parsed.file_id},
                {"$set": resume_doc},
                upsert=True
            )

            # Score
            ats_result = ats_scorer.score(resume_doc, job)
            from backend.services.db_service import db_service
            await db_service.upsert_score(db, ats_result.model_dump(mode="json"))
            await db["resumes"].update_one(
                {"file_id": parsed.file_id},
                {"$set": {"status": "scored"}}
            )

            file_ids.append(parsed.file_id)
            results.append({
                "file_id":         parsed.file_id,
                "filename":        file.filename,
                "candidate_name":  parsed.candidate_name,
                "ats_score":       ats_result.final_score_percent,
                "score_tier":      ats_result.score_tier,
                "skills_found":    skill_result.total_skills_found,
                "status":          "success",
            })

        except Exception as e:
            logger.error(f"Batch item failed ({file.filename}): {e}")
            errors.append({
                "filename": file.filename,
                "error":    str(e),
                "status":   "failed",
            })

    # Queue background ranking after all files processed
    if file_ids:
        background_tasks.add_task(
            _background_rank_candidates,
            job_id=job_id,
            db=db
        )

    total_ms = round((time.time() - batch_start) * 1000, 2)

    return {
        "batch_status":   "complete",
        "job_id":         job_id,
        "total_files":    len(files),
        "successful":     len(results),
        "failed":         len(errors),
        "total_ms":       total_ms,
        "results":        results,
        "errors":         errors,
        "ranking_queued": len(file_ids) > 0,
    }


@router.get(
    "/status/{file_id}",
    summary="Get processing status for a resume"
)
async def get_pipeline_status(
    file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Returns current pipeline stage and available results for a resume.
    Used for polling status during async processing.
    """
    resume = await db["resumes"].find_one(
        {"file_id": file_id},
        {
            "_id": 0,
            "file_id": 1,
            "status": 1,
            "candidate_name": 1,
            "upload_timestamp": 1,
            "metadata.filename": 1,
            "metadata.word_count": 1,
            "extracted_skills": 1,
        }
    )

    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume not found: {file_id}"
        )

    # Define which stages are complete based on status
    stage_order = ["uploaded", "nlp_processed", "skills_extracted", "scored"]
    current_idx = stage_order.index(resume.get("status", "uploaded"))

    stages_complete = {
        stage: (i <= current_idx)
        for i, stage in enumerate(stage_order)
    }

    return {
        "file_id":         file_id,
        "current_status":  resume.get("status"),
        "candidate_name":  resume.get("candidate_name"),
        "stages_complete": stages_complete,
        "skills_count":    len(resume.get("extracted_skills", [])),
        "top_skills":      resume.get("extracted_skills", [])[:5],
        "uploaded_at":     resume.get("upload_timestamp"),
    }