# pydantic: data validation and schema definition
from pydantic import BaseModel, Field

# typing: type annotations
from typing import List, Dict, Optional, Any

# datetime: timestamps for ranking records
from datetime import datetime

# uuid: unique ranking record IDs
import uuid


class RankCriterion(BaseModel):
    """
    One criterion used in composite ranking score computation.

    Storing criteria individually enables full explainability:
    "This candidate ranked 3rd because skill_depth was high
    but achievement_score was below average."
    """

    # Criterion identifier: "ats_score", "skill_depth", etc.
    name: str

    # Raw criterion value before weighting (0.0 to 1.0)
    raw_value: float

    # Weight applied in composite calculation
    weight: float

    # Contribution = raw_value × weight
    contribution: float

    # Percentile of this criterion relative to all candidates in pool
    # "This candidate is in the 87th percentile for skill depth"
    percentile: Optional[float] = None


class RankedCandidate(BaseModel):
    """
    A single candidate's complete ranking record.

    This is the primary object exposed to the dashboard.
    Contains everything a recruiter needs to make a decision:
    rank position, scores, skill gaps, recommendation, and explanation.
    """

    # ── Identity ──────────────────────────────────────────────────────

    # Resume file identifier
    file_id: str

    # Candidate name from resume parsing
    candidate_name: Optional[str] = None

    # Candidate email
    candidate_email: Optional[str] = None

    # Job this candidate was ranked against
    job_id: str

    # ── Ranking position ──────────────────────────────────────────────

    # 1-based rank position (1 = best candidate)
    rank: int

    # Total candidates in the pool (for context: "rank 3 of 47")
    total_candidates: int

    # Percentile: 95.0 means better than 95% of applicants
    # Computed as (total - rank) / total × 100
    percentile: float

    # ── Scores ────────────────────────────────────────────────────────

    # Final ATS score from Module 5 (0.0 to 1.0)
    ats_score: float

    # Human-readable percentage
    ats_score_percent: float

    # Composite ranking score (weighted multi-criteria)
    composite_score: float

    # ATS score tier: "Excellent", "Good", "Fair", "Poor", "Very Poor"
    score_tier: str

    # ── Multi-criteria breakdown ──────────────────────────────────────

    # Individual ranking criteria with weights and percentiles
    rank_criteria: List[RankCriterion] = []

    # ── Skill analysis ────────────────────────────────────────────────

    # Skills candidate has that job requires
    matched_skills: List[str] = []

    # Skills job requires that candidate lacks
    missing_skills: List[str] = []

    # Skills candidate has beyond requirements (transferable assets)
    extra_skills: List[str] = []

    # Percentage of required skills covered (0.0 to 1.0)
    skill_match_rate: float = 0.0

    # Candidate's top skills by weighted score
    top_skills: List[str] = []

    # ── Quality signals ───────────────────────────────────────────────

    # Number of achievement sentences (quantified impact statements)
    # "Led team of 10 engineers, reducing deployment time by 40%"
    achievement_count: int = 0

    # Lexical diversity score (vocabulary breadth)
    lexical_diversity: float = 0.0

    # Total skills found in resume
    total_skills_found: int = 0

    # ── Recommendation ────────────────────────────────────────────────

    # Whether candidate passes minimum ATS threshold
    passes_threshold: bool = False

    # Recruiter action recommendation
    recommendation: str = ""

    # Shortlist tier: "priority", "standard", "reserve", "reject"
    shortlist_tier: str = "reserve"

    # ── Processing metadata ───────────────────────────────────────────

    # When this ranking was computed
    ranked_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateRanking(BaseModel):
    """
    Complete ranking result for all candidates against one job.
    Stored as one document in MongoDB 'rankings' collection.
    The primary data source for the Module 9 dashboard rankings page.
    """

    # Unique identifier for this ranking run
    ranking_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Job this ranking was computed for
    job_id: str

    # Job title for display
    job_title: str

    # All candidates in ranked order (rank 1 = best)
    ranked_candidates: List[RankedCandidate] = []

    # ── Pool statistics ───────────────────────────────────────────────

    # Total candidates in pool
    total_candidates: int = 0

    # Breakdown by tier
    tier_distribution: Dict[str, int] = {}

    # Score statistics across the pool
    score_statistics: Dict[str, float] = {}

    # ── Shortlists ────────────────────────────────────────────────────

    # Priority candidates (Excellent tier): interview immediately
    priority_shortlist: List[str] = []   # file_ids

    # Standard candidates (Good tier): phone screen
    standard_shortlist: List[str] = []  # file_ids

    # Reserve candidates (Fair tier): consider if priority exhausted
    reserve_shortlist: List[str] = []   # file_ids

    # ── Metadata ──────────────────────────────────────────────────────

    # When this ranking was generated
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Processing duration in seconds
    processing_time_seconds: float = 0.0


class ShortlistRequest(BaseModel):
    """
    Request body for generating a custom shortlist.
    Allows recruiters to specify how many candidates they want
    and what minimum criteria they require.
    """

    # Job to generate shortlist for
    job_id: str

    # Number of candidates to include
    top_n: int = 10

    # Minimum ATS score (0.0-1.0)
    min_ats_score: float = 0.5

    # Required skills (subset) candidate must have
    required_skills: List[str] = []

    # Whether to include candidates who pass threshold only
    threshold_only: bool = True
