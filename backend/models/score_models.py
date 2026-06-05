

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class ComponentScore(BaseModel):
    """
    One component of the overall ATS score with full breakdown.

    Storing components separately enables:
    - Explainability: "You scored 0.45 on TF-IDF but 0.90 on skills"
    - Debugging: identify which component is underperforming
    - Weight tuning: adjust weights without re-scoring
    """

    # Component name: "tfidf_similarity", "skill_match", etc.
    name: str

    # Raw score: 0.0 to 1.0
    raw_score: float

    # Weight applied to this component in final score
    weight: float

    # Contribution to final score: raw_score × weight
    weighted_contribution: float

    # Human-readable explanation of this score
    explanation: str


class ATSScoreResult(BaseModel):
    """
    Complete ATS scoring result for one resume against one job.
    Stored as one document in the MongoDB 'scores' collection.

    This is the primary output of Module 5 and primary input to
    Module 6 (Candidate Ranking).
    """

    # Unique score record identifier
    score_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Resume being scored
    resume_file_id: str

    # Job being scored against
    job_id: str

    # ── Score components ─────────────────────────────────────────────

    # Final weighted aggregate score: 0.0 to 1.0
    # This is the primary ranking signal
    final_score: float

    # Normalized to 0-100 for human-readable display
    final_score_percent: float

    # Individual component scores
    component_scores: List[ComponentScore] = []

    # ── Detailed breakdowns ──────────────────────────────────────────

    # Skill gap analysis result
    # {"matched": [...], "missing": [...], "extra": [...], "match_rate": 0.8}
    skill_gap: Dict = {}

    # Keywords from job description found in resume
    matched_keywords: List[str] = []

    # Keywords from job description NOT found in resume
    missing_keywords: List[str] = []

    # TF-IDF vector overlap (top contributing terms)
    top_matching_terms: List[str] = []

    # ── Interpretation ────────────────────────────────────────────────

    # Text tier: "Excellent", "Good", "Fair", "Poor"
    score_tier: str

    # Actionable text recommendation for recruiter
    recommendation: str

    # Whether this candidate passes the minimum threshold
    passes_threshold: bool

    # ── Metadata ──────────────────────────────────────────────────────

    scored_at: datetime = Field(default_factory=datetime.utcnow)

    # Scoring engine version for reproducibility
    scorer_version: str = "1.0.0"

    # Processing duration
    processing_time_seconds: float = 0.0


class BatchScoreRequest(BaseModel):
    """Request body for batch scoring multiple resumes against one job."""

    # Job to score against
    job_id: str

    # List of resume file_ids to score
    # Empty list = score ALL resumes in database
    resume_file_ids: List[str] = []

    # Minimum score threshold (0.0-1.0) — resumes below are excluded
    score_threshold: float = 0.0


class ScoringWeights(BaseModel):
    """
    Configurable scoring weights for the ATS engine.
    Must sum to 1.0. Used to customize scoring for different job types.

    Example: For data science roles, increase data_science weight.
    For DevOps roles, increase cloud_and_devops weight.
    """
    tfidf_similarity: float = 0.35
    skill_match:      float = 0.30
    keyword_match:    float = 0.20
    experience_match: float = 0.10
    education_match:  float = 0.05

    def validate_sum(self) -> bool:
        """Validates all weights sum to approximately 1.0."""
        total = (
            self.tfidf_similarity +
            self.skill_match +
            self.keyword_match +
            self.experience_match +
            self.education_match
        )
        # Allow small floating point variance
        return abs(total - 1.0) < 0.001
