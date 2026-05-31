# backend/models/resume.py

# pydantic: data validation library — BaseModel is the base class for all schemas
from pydantic import BaseModel, Field

# datetime: for upload timestamps
from datetime import datetime

# typing: Optional means the field can be None, List for arrays
from typing import Optional, List, Dict, Any

# uuid: generate unique identifiers
import uuid


class ResumeMetadata(BaseModel):
    """
    Metadata extracted from the resume file itself.
    This is structural information about the document, not its content.
    """

    # Original filename as uploaded by the user
    filename: str

    # File type: "pdf" or "docx"
    file_type: str

    # File size in bytes — used for storage tracking
    file_size_bytes: int

    # Number of pages (PDF) or sections (DOCX)
    page_count: Optional[int] = None

    # Word count of extracted text — quick quality signal
    word_count: Optional[int] = None


class ResumeCreate(BaseModel):
    """
    Schema for creating a new resume record.
    Used when the parser sends data to MongoDB for insertion.
    
    Field() allows us to add metadata like descriptions and defaults.
    default_factory=... means the default is computed at runtime (not at class definition).
    """

    # Unique identifier — generated at parse time, not by MongoDB
    # uuid4() generates a random UUID like: "550e8400-e29b-41d4-a716-446655440000"
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Candidate's full name if extractable from resume
    candidate_name: Optional[str] = None

    # Candidate's email address
    candidate_email: Optional[str] = None

    # Candidate's phone number
    candidate_phone: Optional[str] = None

    # Raw text as extracted directly from PDF/DOCX before any cleaning
    # Preserved for debugging and re-processing
    raw_text: str

    # Cleaned, normalized text ready for NLP processing
    cleaned_text: str

    # Structural metadata about the file
    metadata: ResumeMetadata

    # ISO timestamp of when resume was uploaded
    # datetime.utcnow is called at object creation time
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Processing status: "uploaded" → "parsed" → "nlp_processed" → "scored"
    # Tracks where in the pipeline this resume currently is
    status: str = "uploaded"

    # Placeholder for skills extracted in Module 4
    extracted_skills: List[str] = []

    # Placeholder for NLP features extracted in Module 3
    nlp_features: Dict[str, Any] = {}


class ResumeResponse(BaseModel):
    """
    Schema for API responses when returning resume data to clients.
    Excludes raw_text (too large to send) and includes a success message.
    
    This is what the client sees — intentionally smaller than ResumeCreate.
    """

    # Unique file identifier the client stores for future API calls
    file_id: str

    # Candidate name for display
    candidate_name: Optional[str] = None

    # Upload confirmation timestamp
    upload_timestamp: datetime

    # Current pipeline status
    status: str

    # Word count for UI display
    word_count: Optional[int] = None

    # Human-readable message for the API consumer
    message: str = "Resume uploaded and parsed successfully"


class ResumeInDB(ResumeCreate):
    """
    Schema representing a resume as stored in MongoDB.
    Inherits all fields from ResumeCreate and adds the MongoDB-generated _id.
    
    MongoDB automatically adds _id to every document — this model
    captures that additional field.
    """

    # MongoDB's native document identifier
    # alias="_id" tells Pydantic this field maps to MongoDB's _id field
    id: Optional[str] = Field(None, alias="_id")

    class Config:
        # Allow population of fields using their alias (_id → id)
        populate_by_name = True