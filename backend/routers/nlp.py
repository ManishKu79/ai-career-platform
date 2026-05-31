# backend/routers/nlp.py

from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from backend.database import get_database
from backend.services.nlp_pipeline import nlp_pipeline
from backend.models.nlp_models import NLPFeatures
import logging

logger = logging.getLogger(__name__)

# Router with /nlp prefix and "NLP" tag for OpenAPI grouping
router = APIRouter(prefix="/nlp", tags=["NLP Processing"])


@router.post(
    "/process/{file_id}",
    summary="Run NLP pipeline on a parsed resume",
    description="""
    Runs the full NLP pipeline on a previously uploaded resume.

    Pipeline:
    1. Fetches cleaned_text from MongoDB
    2. Runs spaCy tokenization, POS, NER, lemmatization
    3. Extracts bigrams, trigrams, noun chunks
    4. Computes word frequency and lexical diversity
    5. Updates MongoDB with nlp_features
    6. Updates status to 'nlp_processed'
    """
)
async def process_resume_nlp(
    file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Process NLP for a single resume by file_id."""

    # ── Step 1: Fetch resume from MongoDB ────────────────────────────
    resume = await db["resumes"].find_one(
        {"file_id": file_id},
        {"cleaned_text": 1, "status": 1, "file_id": 1}
    )

    # Raise 404 if resume not found
    if not resume:
        raise HTTPException(
            status_code=404,
            detail=f"Resume not found: {file_id}"
        )

    # Get cleaned text from the resume document
    cleaned_text = resume.get("cleaned_text", "")

    if not cleaned_text:
        raise HTTPException(
            status_code=422,
            detail=f"Resume {file_id} has no cleaned text. Re-upload the file."
        )

    # ── Step 2: Run NLP Pipeline ──────────────────────────────────────
    try:
        features = nlp_pipeline.process(cleaned_text)
    except Exception as e:
        logger.error(f"NLP pipeline failed for {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"NLP processing failed: {str(e)}"
        )

    # ── Step 3: Serialize features ────────────────────────────────────
    # model_dump() converts Pydantic model to dict for MongoDB
    # mode="json" ensures datetime objects are serialized to strings
    features_dict = features.model_dump(mode="json")

    # ── Step 4: Update MongoDB ────────────────────────────────────────
    # $set: updates only specified fields, leaves others unchanged
    # This is an upsert-style update: replaces nlp_features if exists
    await db["resumes"].update_one(
        {"file_id": file_id},
        {
            "$set": {
                "nlp_features": features_dict,
                "status": "nlp_processed"
            }
        }
    )

    logger.info(
        f"NLP processed: {file_id} | "
        f"tokens={features.total_tokens} | "
        f"entities={len(features.entities)} | "
        f"time={features.processing_time_seconds}s"
    )

    # Return summary (not full features — too verbose for default response)
    return {
        "file_id": file_id,
        "status": "nlp_processed",
        "summary": {
            "total_tokens": features.total_tokens,
            "unique_tokens": features.unique_tokens,
            "lexical_diversity": features.lexical_diversity,
            "bigrams_extracted": len(features.bigrams),
            "trigrams_extracted": len(features.trigrams),
            "entities_found": len(features.entities),
            "organizations": features.organizations[:5],
            "achievement_sentences": len(features.achievement_sentences),
            "top_words": list(features.word_frequency.items())[:10],
            "processing_time_seconds": features.processing_time_seconds
        }
    }


@router.get(
    "/features/{file_id}",
    response_model=NLPFeatures,
    summary="Retrieve NLP features for a resume"
)
async def get_nlp_features(
    file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Retrieve stored NLP features for a previously processed resume."""

    resume = await db["resumes"].find_one(
        {"file_id": file_id},
        {"nlp_features": 1, "_id": 0}
    )

    if not resume:
        raise HTTPException(status_code=404, detail=f"Resume not found: {file_id}")

    if "nlp_features" not in resume or not resume["nlp_features"]:
        raise HTTPException(
            status_code=404,
            detail=f"NLP features not yet computed for {file_id}. "
                   f"Call POST /nlp/process/{file_id} first."
        )

    return resume["nlp_features"]


@router.post(
    "/process/batch",
    summary="Run NLP pipeline on all unprocessed resumes"
)
async def process_batch_nlp(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Process all resumes with status 'uploaded' (not yet NLP processed).
    Uses spaCy's efficient batch processing via nlp.pipe().
    """

    # Find all resumes that have been parsed but not NLP processed
    cursor = db["resumes"].find(
        {"status": "uploaded"},
        {"file_id": 1, "cleaned_text": 1}
    )
    resumes = await cursor.to_list(length=100)

    if not resumes:
        return {"message": "No resumes pending NLP processing", "processed": 0}

    logger.info(f"Batch NLP processing {len(resumes)} resumes")

    # Extract texts for batch processing
    texts = [r.get("cleaned_text", "") for r in resumes]
    file_ids = [r["file_id"] for r in resumes]

    # Process all texts in one efficient batch
    features_list = nlp_pipeline.process_batch(texts)

    # Update MongoDB for each processed resume
    processed_count = 0
    errors = []

    for file_id, features in zip(file_ids, features_list):
        try:
            features_dict = features.model_dump(mode="json")
            await db["resumes"].update_one(
                {"file_id": file_id},
                {"$set": {"nlp_features": features_dict, "status": "nlp_processed"}}
            )
            processed_count += 1
        except Exception as e:
            errors.append({"file_id": file_id, "error": str(e)})
            logger.error(f"Failed to update {file_id}: {e}")

    return {
        "processed": processed_count,
        "errors": len(errors),
        "error_details": errors
    }