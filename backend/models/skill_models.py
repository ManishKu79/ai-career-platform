# backend/models/skill_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class ExtractionMethod(str, Enum):
    """
    How a skill was identified.
    Stored per-skill for debugging and confidence scoring.

    Enum inherits from str so values serialize to strings in JSON —
    critical for MongoDB storage and API responses.
    """
    # Direct token match: "python" found in token list
    EXACT_TOKEN = "exact_token"

    # Bigram/trigram match: "machine learning" found in phrase list
    PHRASE_MATCH = "phrase_match"

    # Alias resolution: "postgres" → "postgresql" via SKILL_ALIASES
    ALIAS_MATCH = "alias_match"

    # Fuzzy match: "Kubernets" → "kubernetes" via edit distance
    FUZZY_MATCH = "fuzzy_match"

    # spaCy NER ORG entity matched a known tool/technology
    NER_ENTITY = "ner_entity"

    # Found in spaCy noun chunk: "kubernetes cluster management"
    NOUN_CHUNK = "noun_chunk"


class ExtractedSkill(BaseModel):
    """
    Represents a single identified skill with full metadata.

    This granular representation enables:
    - Explainable scoring: "Python found 5 times via exact match"
    - Confidence filtering: only show skills above threshold
    - Audit trail: how was each skill found?
    - Category grouping: display skills by domain in UI
    """

    # Canonical skill name from taxonomy
    # Example: "postgresql" (not "postgres" or "Postgres")
    canonical_name: str

    # Surface form as it appeared in the resume text
    # Example: "postgres" (what was written before alias resolution)
    surface_form: str

    # Taxonomy category
    # Example: "databases", "programming_languages"
    category: str

    # How this skill was found
    extraction_method: ExtractionMethod

    # How many times this skill appeared in the resume
    # Higher frequency = stronger signal of expertise
    frequency: int = 1

    # Confidence score: 0.0 to 1.0
    # exact_token = 1.0, fuzzy_match = 0.7-0.9, ner_entity = 0.85
    confidence: float = 1.0

    # Category weight from CATEGORY_WEIGHTS
    category_weight: float = 1.0

    # Final weighted score: frequency × confidence × category_weight
    # Used for ranking skills by importance
    weighted_score: float = 0.0


class SkillGap(BaseModel):
    """
    Represents a skill required by a job description
    but absent from a candidate's resume.
    Generated during ATS scoring in Module 5.
    """

    # Canonical skill name that is missing
    skill_name: str

    # Which category the missing skill belongs to
    category: str

    # How important this skill is to the job (0.0 to 1.0)
    # Derived from job description skill frequency
    importance: float

    # Whether this is explicitly listed in the job posting
    # vs inferred from context
    is_explicit_requirement: bool = True


class SkillExtractionResult(BaseModel):
    """
    Complete output of the skill extraction engine for one resume.

    This object is stored as resumes.skill_extraction_result in MongoDB
    and is the primary input to the ATS Scoring Engine (Module 5).
    """

    # ── Flat skill lists ─────────────────────────────────────────────

    # All unique canonical skill names found (for quick lookups)
    all_skills: List[str] = []

    # Detailed skill objects with full metadata
    skill_details: List[ExtractedSkill] = []

    # ── Categorized skills ───────────────────────────────────────────

    # Skills grouped by taxonomy category
    # Example: {"programming_languages": ["python", "java"], ...}
    skills_by_category: Dict[str, List[str]] = {}

    # ── Scoring data ─────────────────────────────────────────────────

    # Canonical name → weighted score for ATS engine
    skill_scores: Dict[str, float] = {}

    # Canonical name → raw occurrence count
    skill_frequency: Dict[str, int] = {}

    # Canonical name → confidence score
    skill_confidence: Dict[str, float] = {}

    # ── Summary statistics ───────────────────────────────────────────

    # Total number of unique skills found
    total_skills_found: int = 0

    # Skills found per category (for radar chart in dashboard)
    skills_per_category: Dict[str, int] = {}

    # Percentage of each category covered vs taxonomy size
    # Example: {"programming_languages": 0.15} = 15% of all langs known
    category_coverage: Dict[str, float] = {}

    # Skills sorted by weighted_score descending (top skills for this resume)
    top_skills: List[str] = []

    # ── Processing metadata ──────────────────────────────────────────

    processed_at: datetime = Field(default_factory=datetime.utcnow)

    # How many extraction methods were used
    methods_used: List[str] = []

    # Processing time for performance monitoring
    processing_time_seconds: float = 0.0