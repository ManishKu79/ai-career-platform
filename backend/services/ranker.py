# backend/services/ranker.py

# numpy: numerical operations for percentile and statistical computations
import numpy as np

# time: performance measurement
import time

# logging: structured application logging
import logging

# typing: type annotations throughout
from typing import List, Dict, Tuple, Optional, Any

# datetime: timestamps for ranking records
from datetime import datetime

# collections.defaultdict: dict with automatic default factory
from collections import defaultdict

# Our Pydantic models
from backend.models.ranking_models import (
    RankedCandidate,
    CandidateRanking,
    RankCriterion,
)

# Motor database type for type hints
from motor.motor_asyncio import AsyncIOMotorDatabase

# Module logger
logger = logging.getLogger(__name__)


class CandidateRanker:
    """
    Multi-criteria candidate ranking engine.

    Ranking pipeline:
    ┌─────────────────────────────────────────────┐
    │ 1. Fetch ATS scores + resume data           │
    │    MongoDB: scores + resumes collections    │
    ├─────────────────────────────────────────────┤
    │ 2. Enrich each candidate                    │
    │    Compute derived quality metrics          │
    ├─────────────────────────────────────────────┤
    │ 3. Compute composite ranking score          │
    │    Weighted sum of 5 criteria               │
    ├─────────────────────────────────────────────┤
    │ 4. Stable sort with tie-breaking            │
    │    Primary → secondary → tertiary           │
    ├─────────────────────────────────────────────┤
    │ 5. Assign ranks and percentiles             │
    │    1-based rank, percentile within pool     │
    ├─────────────────────────────────────────────┤
    │ 6. Generate shortlists                      │
    │    priority / standard / reserve / reject   │
    ├─────────────────────────────────────────────┤
    │ 7. Compute pool statistics                  │
    │    mean, median, std, percentile bands      │
    └─────────────────────────────────────────────┘

    Composite ranking weights:
    These weights are calibrated for general software engineering roles.
    Override per-job via custom weights parameter.
    """

    # Composite ranking score weights
    # Must sum to 1.0
    RANKING_WEIGHTS = {
        "ats_score":            0.60,  # Primary: TF-IDF + component scores
        "skill_depth_score":    0.15,  # Weighted skill importance sum
        "achievement_score":    0.10,  # Quantified achievement sentences
        "experience_score":     0.10,  # Experience component from ATS
        "lexical_quality_score":0.05,  # Resume communication quality
    }

    # Shortlist tier thresholds
    SHORTLIST_TIERS = {
        "priority": 0.80,   # Excellent: interview immediately
        "standard": 0.65,   # Good: phone screen
        "reserve":  0.50,   # Fair: consider if priority pool insufficient
        # Below 0.50 → "reject": does not pass minimum threshold
    }

    # Maximum candidates to include in each shortlist tier
    SHORTLIST_LIMITS = {
        "priority": 5,
        "standard": 10,
        "reserve":  15,
    }

    async def rank_candidates(
        self,
        job_id: str,
        db: AsyncIOMotorDatabase,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> CandidateRanking:
        """
        Main entry point. Fetches data and runs full ranking pipeline.

        This method is async because it fetches data from MongoDB
        using Motor's async driver. All downstream computation is
        synchronous (pure Python/numpy), so no further async is needed.

        Args:
            job_id:          Job to rank candidates for
            db:              AsyncIOMotorDatabase injected from FastAPI
            custom_weights:  Optional dict overriding RANKING_WEIGHTS

        Returns:
            CandidateRanking: Complete ranking with all candidates
        """
        start_time = time.time()

        # Use custom weights if provided, else defaults
        weights = custom_weights or self.RANKING_WEIGHTS

        # Validate custom weights if provided
        if custom_weights:
            total = sum(custom_weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Custom ranking weights must sum to 1.0. "
                    f"Current sum: {total:.3f}"
                )

        # ── Stage 1: Fetch data from MongoDB ──────────────────────────
        logger.info(f"Ranking candidates for job: {job_id}")

        scores, resumes, job = await self._fetch_data(job_id, db)

        if not scores:
            logger.warning(f"No scores found for job: {job_id}")
            return CandidateRanking(
                job_id=job_id,
                job_title=job.get("title", "Unknown") if job else "Unknown",
                total_candidates=0
            )

        logger.info(
            f"Fetched: {len(scores)} scores, "
            f"{len(resumes)} resumes for job {job_id}"
        )

        # Build a lookup dict: file_id → resume document
        # O(1) lookup during enrichment loop
        resume_lookup: Dict[str, Dict] = {
            r["file_id"]: r for r in resumes
        }

        # ── Stage 2: Enrich each candidate ────────────────────────────
        enriched_candidates = []

        for score_doc in scores:
            file_id = score_doc.get("resume_file_id", "")
            resume  = resume_lookup.get(file_id, {})

            enriched = self._enrich_candidate(score_doc, resume, job or {})
            enriched_candidates.append(enriched)

        # ── Stage 3: Compute composite scores ─────────────────────────
        # Pass all candidates together so we can compute per-criterion
        # percentiles across the pool before final composite score
        scored_candidates = self._compute_composite_scores(
            enriched_candidates, weights
        )

        # ── Stage 4: Stable sort with tie-breaking ─────────────────────
        ranked_list = self._stable_sort(scored_candidates)

        # ── Stage 5: Assign ranks and percentiles ─────────────────────
        total = len(ranked_list)
        ranked_candidates = []

        for i, candidate_data in enumerate(ranked_list):
            rank      = i + 1  # 1-based rank
            # Percentile: fraction of candidates ranked AT OR BELOW this one
            # Candidate ranked 1st of 100 = 99th percentile
            # (1 - rank/total) × 100 gives intuitive "better than X%" value
            percentile = round((1 - rank / total) * 100, 1) if total > 1 else 100.0

            # Determine shortlist tier
            ats_score   = candidate_data["ats_score"]
            shortlist_tier = self._assign_shortlist_tier(ats_score)

            # Build recommendation string
            recommendation = self._build_recommendation(
                rank, total, ats_score, shortlist_tier,
                candidate_data.get("achievement_count", 0),
                candidate_data.get("missing_skills", [])
            )

            ranked_candidate = RankedCandidate(
                file_id=candidate_data["file_id"],
                candidate_name=candidate_data.get("candidate_name"),
                candidate_email=candidate_data.get("candidate_email"),
                job_id=job_id,
                rank=rank,
                total_candidates=total,
                percentile=percentile,
                ats_score=round(candidate_data["ats_score"], 4),
                ats_score_percent=round(candidate_data["ats_score"] * 100, 2),
                composite_score=round(candidate_data["composite_score"], 4),
                score_tier=candidate_data["score_tier"],
                rank_criteria=candidate_data["rank_criteria"],
                matched_skills=candidate_data.get("matched_skills", []),
                missing_skills=candidate_data.get("missing_skills", []),
                extra_skills=candidate_data.get("extra_skills", []),
                skill_match_rate=candidate_data.get("skill_match_rate", 0.0),
                top_skills=candidate_data.get("top_skills", []),
                achievement_count=candidate_data.get("achievement_count", 0),
                lexical_diversity=candidate_data.get("lexical_diversity", 0.0),
                total_skills_found=candidate_data.get("total_skills_found", 0),
                passes_threshold=candidate_data.get("passes_threshold", False),
                recommendation=recommendation,
                shortlist_tier=shortlist_tier,
            )
            ranked_candidates.append(ranked_candidate)

        # ── Stage 6: Generate shortlists ──────────────────────────────
        shortlists = self._generate_shortlists(ranked_candidates)

        # ── Stage 7: Compute pool statistics ──────────────────────────
        pool_stats    = self._compute_pool_statistics(ranked_candidates)
        tier_dist     = self._compute_tier_distribution(ranked_candidates)

        # ── Assemble final CandidateRanking ───────────────────────────
        processing_time = time.time() - start_time

        ranking = CandidateRanking(
            job_id=job_id,
            job_title=job.get("title", "Unknown") if job else "Unknown",
            ranked_candidates=ranked_candidates,
            total_candidates=total,
            tier_distribution=tier_dist,
            score_statistics=pool_stats,
            priority_shortlist=shortlists["priority"],
            standard_shortlist=shortlists["standard"],
            reserve_shortlist=shortlists["reserve"],
            processing_time_seconds=round(processing_time, 3)
        )

        logger.info(
            f"Ranking complete: {total} candidates | "
            f"priority={len(shortlists['priority'])}, "
            f"standard={len(shortlists['standard'])}, "
            f"time={processing_time:.3f}s"
        )

        return ranking

    # ─────────────────────────────────────────────────────────────────
    # STAGE 1: DATA FETCHING
    # ─────────────────────────────────────────────────────────────────

    async def _fetch_data(
        self,
        job_id: str,
        db: AsyncIOMotorDatabase
    ) -> Tuple[List[Dict], List[Dict], Optional[Dict]]:
        """
        Fetches all data required for ranking from MongoDB.

        Three queries:
        1. scores collection:  All ATS score documents for this job
        2. resumes collection: Resume data for all scored candidates
        3. jobs collection:    Job document for context and title

        Uses projection to limit data transfer — only fetches fields
        needed for ranking, not full documents with large text fields.

        Args:
            job_id: Job identifier to fetch scores for
            db:     Async MongoDB database object

        Returns:
            Tuple of (scores_list, resumes_list, job_dict)
        """

        # ── Query 1: Fetch ATS scores for this job ────────────────────
        # Sort by final_score descending as a pre-sort hint
        # (our ranker re-sorts, but pre-sorting reduces stable sort work)
        scores_cursor = db["scores"].find(
            {"job_id": job_id},
            {
                # Include all scoring fields
                "_id": 0,
                "score_id": 1,
                "resume_file_id": 1,
                "job_id": 1,
                "final_score": 1,
                "final_score_percent": 1,
                "component_scores": 1,
                "skill_gap": 1,
                "score_tier": 1,
                "passes_threshold": 1,
                "matched_keywords": 1,
                "missing_keywords": 1,
                "scored_at": 1,
            }
        ).sort("final_score", -1)

        scores = await scores_cursor.to_list(length=500)

        if not scores:
            return [], [], None

        # Collect all resume IDs from scores
        resume_ids = [s["resume_file_id"] for s in scores]

        # ── Query 2: Fetch resume data for all scored candidates ───────
        # $in operator: fetch all matching in one query (not N queries)
        resumes_cursor = db["resumes"].find(
            {"file_id": {"$in": resume_ids}},
            {
                "_id": 0,
                "file_id": 1,
                "candidate_name": 1,
                "candidate_email": 1,
                "extracted_skills": 1,
                "skill_extraction_result": 1,
                "nlp_features": 1,
                "upload_timestamp": 1,
                "status": 1,
            }
        )

        resumes = await resumes_cursor.to_list(length=500)

        # ── Query 3: Fetch job document ────────────────────────────────
        job = await db["jobs"].find_one(
            {"job_id": job_id},
            {"_id": 0, "title": 1, "required_skills": 1, "company": 1}
        )

        return scores, resumes, job

    # ─────────────────────────────────────────────────────────────────
    # STAGE 2: CANDIDATE ENRICHMENT
    # ─────────────────────────────────────────────────────────────────

    def _enrich_candidate(
        self,
        score_doc: Dict,
        resume_doc: Dict,
        job_doc: Dict
    ) -> Dict:
        """
        Computes additional quality metrics for one candidate beyond
        the raw ATS score. These metrics are the "secondary criteria"
        used in multi-criteria ranking.

        Enrichment adds 5 derived metrics:

        1. skill_depth_score:
           Weighted sum of matched skill scores from extraction result.
           A candidate who has Python used 8 times outranks one with
           Python mentioned once, even at same ATS score.

        2. achievement_score:
           Normalized count of achievement sentences (action verb +
           quantifier pairs). From Module 3's sentence analysis.
           "Led team of 15 engineers, improving throughput by 35%"

        3. experience_score:
           Extracted from ATS component scores — the experience
           component raw score from Module 5.

        4. lexical_quality_score:
           Lexical diversity from NLP features. Higher diversity
           signals a candidate who communicates with varied vocabulary —
           a proxy for communication quality.

        5. skill_match_rate:
           Fraction of job's required skills covered by candidate.
           Directly from skill gap analysis.

        Args:
            score_doc:  MongoDB ATS score document
            resume_doc: MongoDB resume document
            job_doc:    MongoDB job document

        Returns:
            Enriched candidate dict with all metrics for composite scoring
        """

        file_id    = score_doc.get("resume_file_id", "")
        ats_score  = score_doc.get("final_score", 0.0)
        score_tier = score_doc.get("score_tier", "Very Poor")
        skill_gap  = score_doc.get("skill_gap", {})
        passes     = score_doc.get("passes_threshold", False)

        # ── Metric 1: Skill depth score ───────────────────────────────
        # Sums weighted scores of MATCHED skills from extraction result
        # High weighted_score = skill appears frequently in high-weight category
        skill_depth_score = self._compute_skill_depth(
            resume_doc.get("skill_extraction_result", {}),
            set(skill_gap.get("matched", []))
        )

        # ── Metric 2: Achievement score ───────────────────────────────
        # Normalized achievement sentence count
        # Raw count normalized to [0,1] using log scale:
        # 0 achievements → 0.0
        # 1 achievement  → 0.33
        # 3 achievements → 0.58
        # 10 achievements → 0.92
        nlp_features = resume_doc.get("nlp_features", {})
        achievement_sentences = nlp_features.get("achievement_sentences", [])
        achievement_count = len(achievement_sentences)

        # Log normalization prevents outliers from dominating
        # 1 + log(x+1) ensures log(0) = 0 (defined)
        # Divide by 1 + log(11) to normalize to roughly [0,1]
        import math
        achievement_score = (
            math.log(achievement_count + 1) /
            math.log(12)  # log(12) ≈ 2.485 normalizes to ~1.0 at 11 achievements
        )
        achievement_score = float(np.clip(achievement_score, 0.0, 1.0))

        # ── Metric 3: Experience score ────────────────────────────────
        # Extract from ATS component scores (computed in Module 5)
        # component_scores is a list of ComponentScore objects (as dicts)
        experience_score = 0.7  # Default neutral
        component_scores = score_doc.get("component_scores", [])
        for comp in component_scores:
            if comp.get("name") == "experience_match":
                experience_score = comp.get("raw_score", 0.7)
                break

        # ── Metric 4: Lexical quality score ───────────────────────────
        # Lexical diversity from NLP pipeline
        # Range: 0.0 (all words repeated) to 1.0 (all words unique)
        # Professional resumes typically score 0.6 - 0.8
        lexical_diversity = nlp_features.get("lexical_diversity", 0.5)
        lexical_quality_score = float(lexical_diversity)

        # ── Skill gap data ────────────────────────────────────────────
        matched_skills  = skill_gap.get("matched", [])
        missing_skills  = skill_gap.get("missing", [])
        extra_skills    = skill_gap.get("extra", [])
        skill_match_rate = skill_gap.get("match_rate", 0.0)

        # ── Top skills from extraction result ─────────────────────────
        skill_result = resume_doc.get("skill_extraction_result", {})
        top_skills = skill_result.get("top_skills", [])[:10]

        # ── Total skills found ────────────────────────────────────────
        total_skills = skill_result.get("total_skills_found", 0)

        return {
            # Identity
            "file_id":         file_id,
            "candidate_name":  resume_doc.get("candidate_name"),
            "candidate_email": resume_doc.get("candidate_email"),
            "upload_timestamp": resume_doc.get("upload_timestamp"),

            # Primary ATS score
            "ats_score":    ats_score,
            "score_tier":   score_tier,
            "passes_threshold": passes,

            # Derived quality metrics (for composite ranking)
            "skill_depth_score":     skill_depth_score,
            "achievement_score":     achievement_score,
            "experience_score":      experience_score,
            "lexical_quality_score": lexical_quality_score,

            # Raw counts
            "achievement_count":   achievement_count,
            "lexical_diversity":   lexical_diversity,
            "total_skills_found":  total_skills,

            # Skill analysis
            "matched_skills":   matched_skills,
            "missing_skills":   missing_skills,
            "extra_skills":     extra_skills,
            "skill_match_rate": skill_match_rate,
            "top_skills":       top_skills,
        }

    def _compute_skill_depth(
        self,
        skill_extraction_result: Dict,
        matched_skills: set
    ) -> float:
        """
        Computes skill depth score for matched skills only.

        Skill depth = sum of weighted_scores of matched skills,
        normalized by the theoretical maximum.

        Logic: A candidate who has Python (weighted_score=4.5) and
        Kubernetes (3.2) outranks one with Python (2.0) and Docker (1.5)
        even if both have the same number of matched skills.

        Args:
            skill_extraction_result: Full skill extraction dict from MongoDB
            matched_skills:          Set of skill names that matched job req

        Returns:
            Normalized skill depth score [0.0, 1.0]
        """

        if not skill_extraction_result or not matched_skills:
            return 0.0

        # skill_scores: Dict[canonical_name, weighted_score]
        skill_scores = skill_extraction_result.get("skill_scores", {})

        if not skill_scores:
            return 0.5  # Neutral if no detailed scores available

        # Sum weighted scores for MATCHED skills only
        # (skills that actually cover job requirements)
        matched_score_sum = sum(
            skill_scores.get(skill, 0.0)
            for skill in matched_skills
        )

        # Theoretical maximum: max possible score × number of matched skills
        # Using the maximum observed score as the ceiling
        if skill_scores:
            max_single_score = max(skill_scores.values())
            theoretical_max  = max_single_score * max(len(matched_skills), 1)
        else:
            theoretical_max = 1.0

        # Normalize
        depth_score = (
            matched_score_sum / theoretical_max
            if theoretical_max > 0 else 0.0
        )

        return float(np.clip(depth_score, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────
    # STAGE 3: COMPOSITE SCORE COMPUTATION
    # ─────────────────────────────────────────────────────────────────

    def _compute_composite_scores(
        self,
        candidates: List[Dict],
        weights: Dict[str, float]
    ) -> List[Dict]:
        """
        Computes composite ranking scores for all candidates.

        Process:
        1. Extract each criterion's values across all candidates
        2. Compute per-criterion percentiles (requires all values)
        3. Compute weighted composite score for each candidate
        4. Build RankCriterion objects with percentile data

        Why compute percentiles first?
        Percentile of a single value requires knowing the full
        distribution. We must see all candidates before assigning
        "this candidate is in the 87th percentile for skill depth."

        Args:
            candidates: List of enriched candidate dicts
            weights:    Dict of criterion_name → weight

        Returns:
            List of candidate dicts with composite_score and rank_criteria added
        """

        if not candidates:
            return []

        # ── Extract criterion arrays across all candidates ─────────────
        # Each criterion becomes a numpy array for percentile computation

        criteria_values: Dict[str, List[float]] = {
            "ats_score":            [c["ats_score"] for c in candidates],
            "skill_depth_score":    [c["skill_depth_score"] for c in candidates],
            "achievement_score":    [c["achievement_score"] for c in candidates],
            "experience_score":     [c["experience_score"] for c in candidates],
            "lexical_quality_score":[c["lexical_quality_score"] for c in candidates],
        }

        # ── Compute per-criterion percentile arrays ────────────────────
        # np.percentile_rank equivalent: for each value, what fraction
        # of all values is it greater than or equal to?
        # We use scipy-style: percentile = (rank - 1) / (n - 1) × 100
        # But we implement without scipy using numpy searchsorted

        criteria_percentiles: Dict[str, np.ndarray] = {}
        for criterion, values in criteria_values.items():
            arr = np.array(values, dtype=float)
            # For each value, compute its percentile in the distribution
            # np.argsort argsort gives rank order → convert to percentile
            n = len(arr)
            if n == 1:
                # Single candidate = 100th percentile by default
                criteria_percentiles[criterion] = np.array([100.0])
            else:
                # Rank each value (0-based)
                ranks = np.argsort(np.argsort(arr))  # double argsort = rank
                # Convert rank to percentile: rank / (n-1) × 100
                percentile_arr = (ranks / (n - 1)) * 100
                criteria_percentiles[criterion] = percentile_arr

        # ── Compute composite score for each candidate ─────────────────
        result_candidates = []

        for i, candidate in enumerate(candidates):

            # Weighted sum of all criteria
            composite = (
                candidate["ats_score"]            * weights.get("ats_score", 0.60) +
                candidate["skill_depth_score"]    * weights.get("skill_depth_score", 0.15) +
                candidate["achievement_score"]    * weights.get("achievement_score", 0.10) +
                candidate["experience_score"]     * weights.get("experience_score", 0.10) +
                candidate["lexical_quality_score"]* weights.get("lexical_quality_score", 0.05)
            )

            # Clamp composite to [0, 1]
            composite = float(np.clip(composite, 0.0, 1.0))

            # Build RankCriterion objects for explainability
            rank_criteria = []
            for criterion_name, weight in weights.items():
                raw_val = candidate.get(criterion_name, 0.0)
                contribution = raw_val * weight

                # Get this candidate's percentile for this criterion
                percentile_arr = criteria_percentiles.get(criterion_name)
                percentile = float(percentile_arr[i]) if percentile_arr is not None else 0.0

                rank_criteria.append(RankCriterion(
                    name=criterion_name,
                    raw_value=round(raw_val, 4),
                    weight=weight,
                    contribution=round(contribution, 4),
                    percentile=round(percentile, 1)
                ))

            # Sort rank criteria by contribution descending
            # (show most impactful criterion first in UI)
            rank_criteria.sort(key=lambda c: c.contribution, reverse=True)

            # Add composite score and criteria to candidate dict
            enriched = {**candidate}
            enriched["composite_score"] = composite
            enriched["rank_criteria"]   = rank_criteria

            result_candidates.append(enriched)

        return result_candidates

    # ─────────────────────────────────────────────────────────────────
    # STAGE 4: STABLE SORT WITH TIE-BREAKING
    # ─────────────────────────────────────────────────────────────────

    def _stable_sort(self, candidates: List[Dict]) -> List[Dict]:
        """
        Sorts candidates with multi-level tie-breaking using Python's
        stable Timsort algorithm.

        Primary sort:   composite_score DESC (highest ranked first)
        Secondary sort: skill_match_rate DESC (more required skills covered)
        Tertiary sort:  achievement_count DESC (more quantified achievements)
        Quaternary sort: total_skills_found DESC (broader qualification)
        Final sort:     upload_timestamp ASC (earlier applicants first — fairness)

        Python's sorted() with a tuple key applies all criteria simultaneously.
        Tuples are compared lexicographically: first element first, then
        second element only if first elements are equal, etc.

        Negating scores achieves descending sort within ascending tuple comparison:
            (-composite, -skill_match, -achievement, timestamp)
            sorted() ascending → highest composite first

        Timsort properties:
        - Stable: equal elements preserve insertion order
        - O(n log n) worst case, O(n) for nearly-sorted data
        - In-place variant used by list.sort()
        - Python's default sort algorithm since Python 2.3

        Args:
            candidates: List of enriched+composite-scored candidate dicts

        Returns:
            Sorted list (descending by composite score with tie-breaking)
        """

        def sort_key(candidate: Dict) -> Tuple:
            """
            Returns a tuple used for multi-level sort comparison.
            Negated values achieve descending sort (higher = ranked first).
            timestamp is NOT negated — earlier upload = better tie-break.
            """
            # Handle missing/None upload_timestamp gracefully
            # datetime.min is used as fallback so None-timestamp candidates
            # sort last within a tie group
            timestamp = candidate.get("upload_timestamp")
            if timestamp is None:
                timestamp = datetime.min
            elif isinstance(timestamp, str):
                # MongoDB may return ISO string — parse it
                try:
                    from datetime import datetime as dt
                    timestamp = dt.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    timestamp = datetime.min

            return (
                -candidate.get("composite_score", 0.0),   # Primary:   highest first
                -candidate.get("skill_match_rate", 0.0),  # Secondary: highest first
                -candidate.get("achievement_count", 0),   # Tertiary:  highest first
                -candidate.get("total_skills_found", 0),  # Quaternary: highest first
                timestamp                                  # Final: earliest first
            )

        # sorted() creates a new list — does not modify in-place
        # Timsort guarantees stability: equal elements keep original order
        return sorted(candidates, key=sort_key)

    # ─────────────────────────────────────────────────────────────────
    # STAGE 5: SHORTLIST GENERATION
    # ─────────────────────────────────────────────────────────────────

    def _assign_shortlist_tier(self, ats_score: float) -> str:
        """
        Assigns a shortlist tier based on ATS score thresholds.

        Tiers:
        - priority (≥0.80): Excellent match, interview immediately
        - standard (≥0.65): Good match, phone screen
        - reserve  (≥0.50): Fair match, consider if priority pool insufficient
        - reject   (<0.50): Insufficient match

        Args:
            ats_score: Final ATS score (0.0 to 1.0)

        Returns:
            Tier string: "priority", "standard", "reserve", or "reject"
        """
        if ats_score >= self.SHORTLIST_TIERS["priority"]:
            return "priority"
        elif ats_score >= self.SHORTLIST_TIERS["standard"]:
            return "standard"
        elif ats_score >= self.SHORTLIST_TIERS["reserve"]:
            return "reserve"
        else:
            return "reject"

    def _generate_shortlists(
        self,
        ranked_candidates: List[RankedCandidate]
    ) -> Dict[str, List[str]]:
        """
        Generates prioritized shortlists of file_ids for each tier.

        Candidates are already in ranked order (rank 1 = best).
        We partition by shortlist_tier and take top N per tier.

        Args:
            ranked_candidates: List of RankedCandidate in rank order

        Returns:
            Dict with keys: "priority", "standard", "reserve"
            Each maps to a list of file_ids
        """
        shortlists: Dict[str, List[str]] = {
            "priority": [],
            "standard": [],
            "reserve":  [],
        }

        counts = defaultdict(int)

        for candidate in ranked_candidates:
            tier  = candidate.shortlist_tier
            limit = self.SHORTLIST_LIMITS.get(tier, 0)

            # Only add non-rejected candidates within tier limits
            if tier != "reject" and counts[tier] < limit:
                shortlists[tier].append(candidate.file_id)
                counts[tier] += 1

        return shortlists

    def _build_recommendation(
        self,
        rank: int,
        total: int,
        ats_score: float,
        shortlist_tier: str,
        achievement_count: int,
        missing_skills: List[str]
    ) -> str:
        """
        Builds a human-readable, actionable recruiter recommendation.

        Combines rank position, score tier, achievement signal,
        and skill gap to produce a concise decision-support statement.

        Args:
            rank:              1-based rank position
            total:             Total candidates in pool
            ats_score:         Final ATS score
            shortlist_tier:    Assigned tier
            achievement_count: Number of quantified achievement sentences
            missing_skills:    Skills required but not found in resume

        Returns:
            Recommendation string for recruiter display
        """

        # Position context
        position_str = f"Rank {rank} of {total}"
        score_str    = f"{ats_score * 100:.1f}%"

        # Tier-based action
        tier_actions = {
            "priority": "Recommend for immediate technical interview.",
            "standard": "Recommend for 30-minute phone screen.",
            "reserve":  "Hold in reserve — consider if priority pool is insufficient.",
            "reject":   "Does not meet minimum requirements. Not recommended to advance."
        }
        action = tier_actions.get(shortlist_tier, "Manual review recommended.")

        # Achievement signal
        if achievement_count >= 3:
            achievement_note = " Strong achievement language with quantified impact."
        elif achievement_count >= 1:
            achievement_note = " Some quantified achievements present."
        else:
            achievement_note = " Limited quantified achievements in resume."

        # Skill gap note
        if missing_skills:
            top_missing = ", ".join(missing_skills[:3])
            skill_note = f" Missing required skills: {top_missing}."
        else:
            skill_note = " Covers all required skills."

        return (
            f"{position_str} ({score_str}). "
            f"{action}"
            f"{achievement_note}"
            f"{skill_note}"
        )

    # ─────────────────────────────────────────────────────────────────
    # STAGE 7: POOL STATISTICS
    # ─────────────────────────────────────────────────────────────────

    def _compute_pool_statistics(
        self,
        ranked_candidates: List[RankedCandidate]
    ) -> Dict[str, float]:
        """
        Computes descriptive statistics across the entire candidate pool.

        Statistics computed:
        - mean_score:   Average ATS score in the pool
        - median_score: Median ATS score (50th percentile)
        - std_score:    Standard deviation of ATS scores
        - min_score:    Lowest ATS score
        - max_score:    Highest ATS score
        - p25_score:    25th percentile score
        - p75_score:    75th percentile score
        - mean_skills:  Average number of skills found per resume

        These statistics power the dashboard's distribution chart
        and help recruiters calibrate the quality of the applicant pool.

        Args:
            ranked_candidates: List of RankedCandidate in rank order

        Returns:
            Dict mapping statistic_name → float_value
        """

        if not ranked_candidates:
            return {}

        # Extract score arrays using numpy for vectorized statistics
        scores     = np.array([c.ats_score for c in ranked_candidates])
        skill_cnts = np.array([c.total_skills_found for c in ranked_candidates])

        return {
            "mean_score":   round(float(np.mean(scores)), 4),
            "median_score": round(float(np.median(scores)), 4),
            "std_score":    round(float(np.std(scores)), 4),
            "min_score":    round(float(np.min(scores)), 4),
            "max_score":    round(float(np.max(scores)), 4),
            "p25_score":    round(float(np.percentile(scores, 25)), 4),
            "p75_score":    round(float(np.percentile(scores, 75)), 4),
            "mean_skills":  round(float(np.mean(skill_cnts)), 2),
            "total_above_threshold": int(np.sum(scores >= 0.50)),
        }

    def _compute_tier_distribution(
        self,
        ranked_candidates: List[RankedCandidate]
    ) -> Dict[str, int]:
        """
        Counts candidates per shortlist tier.

        Used to populate the tier distribution chart in the dashboard:
        "5 Excellent, 12 Good, 18 Fair, 65 Poor"

        Args:
            ranked_candidates: Full ranked candidate list

        Returns:
            Dict mapping tier_label → count
        """
        distribution: Dict[str, int] = defaultdict(int)

        for candidate in ranked_candidates:
            distribution[candidate.shortlist_tier] += 1

        # Ensure all tiers present even if count = 0
        for tier in ["priority", "standard", "reserve", "reject"]:
            if tier not in distribution:
                distribution[tier] = 0

        return dict(distribution)


# Single shared instance
# All async methods use injected db — no shared state
candidate_ranker = CandidateRanker()