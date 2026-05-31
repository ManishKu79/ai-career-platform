# backend/routers/upload.py

# FastAPI imports:
# APIRouter: creates a modular router to be included in main app
# UploadFile: FastAPI's file upload type with async read capability
# File: marks a parameter as a file upload field
# HTTPException: raises HTTP errors with status codes and messages
# Depends: dependency injection
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

# Motor database type for type hints
from motor.motor_asyncio import AsyncIOMotorDatabase

# Our database dependency
from backend.database import get_database

# Our parser service
from backend.services.parser import resume_parser

# Response model
from backend.models.resume import ResumeResponse

# logging for production observability
import logging

# datetime for timestamp in response
from datetime import datetime

# Configure router with prefix and tags
# prefix="/upload": all routes in this router start with /upload
# tags=["Upload"]: groups these routes together in OpenAPI docs
router = APIRouter(prefix="/upload", tags=["Upload"])

# Module logger
logger = logging.getLogger(__name__)


@router.post(
    "/resume",
    response_model=ResumeResponse,
    status_code=201,  # 201 Created is correct for resource creation
    summary="Upload and parse a resume",
    description="""
    Accepts a PDF or DOCX resume file.
    
    Processing pipeline:
    1. Validates file type and size
    2. Extracts text content
    3. Cleans and normalizes text  
    4. Extracts contact information
    5. Stores parsed data in MongoDB
    6. Returns file_id for subsequent operations
    """
)
async def upload_resume(
    # UploadFile: FastAPI's async file wrapper
    # File(...): "..." means this parameter is required (no default)
    file: UploadFile = File(..., description="PDF or DOCX resume file"),

    # Depends(get_database): FastAPI injects the database object
    # from our get_database() function before the route handler runs
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Upload and parse a single resume file.
    
    Returns the file_id which must be used for all subsequent
    scoring, ranking, and retrieval operations.
    """

    # ── Step 1: Read file bytes ──────────────────────────────────────
    # file.read() is async — doesn't block the event loop
    # Returns raw bytes of the entire file
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {str(e)}"
        )

    # ── Step 2: Validate file is not empty ───────────────────────────
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty. Please upload a valid resume."
        )

    logger.info(
        f"Received upload: '{file.filename}' "
        f"({len(file_bytes)} bytes, type: {file.content_type})"
    )

    # ── Step 3: Parse the resume ─────────────────────────────────────
    # resume_parser.parse() handles format detection, text extraction,
    # cleaning, and contact info extraction
    try:
        parsed_resume = resume_parser.parse(
            file_bytes=file_bytes,
            filename=file.filename
        )
    except ValueError as e:
        # ValueError means invalid file — return 422 Unprocessable Entity
        logger.warning(f"Parse failed for '{file.filename}': {e}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error — return 500 Internal Server Error
        logger.error(f"Unexpected parse error for '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error during resume parsing. Please try again."
        )

    # ── Step 4: Store in MongoDB ──────────────────────────────────────
    # parsed_resume.model_dump() converts Pydantic model to dict
    # MongoDB stores documents as BSON (Binary JSON) — dict is compatible
    try:
        resume_dict = parsed_resume.model_dump()

        # Convert datetime to string for MongoDB compatibility
        # MongoDB stores dates as BSON Date, but we store as ISO string
        # for simplicity and cross-platform compatibility
        resume_dict['upload_timestamp'] = (
            parsed_resume.upload_timestamp.isoformat()
        )

        # Insert document into the "resumes" collection
        # MongoDB creates the collection automatically if it doesn't exist
        result = await db["resumes"].insert_one(resume_dict)

        logger.info(
            f"Resume stored in MongoDB: file_id={parsed_resume.file_id}, "
            f"mongo_id={result.inserted_id}"
        )

    except Exception as e:
        logger.error(f"MongoDB insert failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to store resume data. Please try again."
        )

    # ── Step 5: Return success response ──────────────────────────────
    # ResumeResponse only includes safe, relevant fields for the client
    # Never return raw_text (too large) or internal IDs
    return ResumeResponse(
        file_id=parsed_resume.file_id,
        candidate_name=parsed_resume.candidate_name,
        upload_timestamp=parsed_resume.upload_timestamp,
        status=parsed_resume.status,
        word_count=parsed_resume.metadata.word_count,
        message=(
            f"Resume '{file.filename}' uploaded and parsed successfully. "
            f"Word count: {parsed_resume.metadata.word_count}. "
            f"Use file_id '{parsed_resume.file_id}' for scoring."
        )
    )


@router.get(
    "/resume/{file_id}",
    summary="Retrieve a parsed resume by ID",
    description="Returns the stored parsed resume data for a given file_id."
)
async def get_resume(
    file_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Retrieve a previously uploaded and parsed resume.
    
    Args:
        file_id: The UUID returned when the resume was uploaded
        
    Returns:
        Full resume document from MongoDB (excluding raw_text for size)
    """

    # Query MongoDB for document matching file_id
    # find_one() returns the document dict or None
    # {"_id": 0}: exclude MongoDB's internal _id from response
    # {"raw_text": 0}: exclude raw_text (too large for API response)
    resume = await db["resumes"].find_one(
        {"file_id": file_id},
        {"_id": 0, "raw_text": 0}
    )

    # If no document found, return 404
    if not resume:
        raise HTTPException(
            status_code=404,
            detail=f"No resume found with file_id: {file_id}"
        )

    return resume


@router.get(
    "/resumes",
    summary="List all uploaded resumes",
    description="Returns metadata for all resumes stored in the database."
)
async def list_resumes(
    db: AsyncIOMotorDatabase = Depends(get_database),
    limit: int = 50,   # Query parameter with default value
    skip: int = 0      # For pagination: skip N records
):
    """
    List all uploaded resumes with pagination support.
    
    Args:
        limit: Maximum number of resumes to return (default 50)
        skip: Number of resumes to skip for pagination (default 0)
    """

    # Build MongoDB cursor
    # Projection: only return lightweight metadata fields, not full text
    # sort: newest first (upload_timestamp descending = -1)
    cursor = db["resumes"].find(
        {},  # empty filter = match all documents
        {    # projection: include only these fields
            "_id": 0,
            "file_id": 1,
            "candidate_name": 1,
            "candidate_email": 1,
            "upload_timestamp": 1,
            "status": 1,
            "metadata.filename": 1,
            "metadata.word_count": 1,
            "metadata.file_type": 1
        }
    ).sort("upload_timestamp", -1).skip(skip).limit(limit)

    # Convert cursor to list — await is required for async iteration
    resumes = await cursor.to_list(length=limit)

    return {
        "total_returned": len(resumes),
        "skip": skip,
        "limit": limit,
        "resumes": resumes
    }