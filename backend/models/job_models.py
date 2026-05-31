# backend/models/job_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


class JobCreate(BaseModel):
    """
    Schema for creating a new job description.
    Submitted by recruiters via the dashboard or API.
    """

    # Unique identifier — generated at creation, not by MongoDB
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Job title — used for display and search
    title: str

    # Company posting the role
    company: Optional[str] = None

    # Full job description text — the PRIMARY input to TF-IDF scoring
    # Should include: responsibilities, requirements, nice-to-haves
    description: str

    # Explicitly listed required skills — extracted from description
    # Populated by the skill extraction pipeline (same as resumes)
    required_skills: List[str] = []

    # Optional: minimum years of experience required
    min_experience_years: Optional[int] = None

    # Optional: education requirements ("Bachelor's", "Master's", "PhD")
    education_requirement: Optional[str] = None

    # Employment type: "full_time", "part_time", "contract", "internship"
    employment_type: Optional[str] = "full_time"

    # Remote policy: "remote", "hybrid", "onsite"
    remote_policy: Optional[str] = None

    # Salary range (optional — used for candidate filtering in Module 6)
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None

    # Processing status — mirrors resume pipeline
    # "created" → "nlp_processed" → "skills_extracted" → "active"
    status: str = "created"

    # Cleaned text — populated after NLP processing
    cleaned_description: Optional[str] = None

    # NLP features extracted from job description text
    nlp_features: Dict[str, Any] = {}

    # Skill extraction result for this job posting
    skill_extraction_result: Dict[str, Any] = {}

    # ISO creation timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Whether the job is actively accepting applications
    is_active: bool = True


class JobResponse(BaseModel):
    """
    Lightweight API response schema for job listings.
    Excludes full description text for list endpoints.
    """
    job_id: str
    title: str
    company: Optional[str]
    status: str
    required_skills: List[str]
    created_at: datetime
    is_active: bool
    message: str = "Job description created successfully"


class JobInDB(JobCreate):
    """Job description as stored in MongoDB, with _id field."""
    id: Optional[str] = Field(None, alias="_id")

    class Config:
        populate_by_name = True