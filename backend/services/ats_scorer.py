# backend/services/ats_scorer.py

# sklearn.feature_extraction.text.TfidfVectorizer:
# Converts text documents into TF-IDF feature matrices.
# Handles tokenization, vocabulary building, TF-IDF weighting internally.
from sklearn.feature_extraction.text import TfidfVectorizer

# sklearn.metrics.pairwise.cosine_similarity:
# Computes cosine similarity between rows of sparse/dense matrices.
# Returns a 2D array: result[i][j] = similarity between doc_i and doc_j
from sklearn.metrics.pairwise import cosine_similarity

# numpy: numerical computing library
# Used for vector operations and matrix manipulation
import numpy as np

# re: regex for experience year extraction from text
import re

# time: performance measurement
import time

# logging: structured logging
import logging

# typing: type annotations
from typing import List, Dict, Tuple, Set, Optional, Any

# collections.Counter: fast frequency counting for keyword analysis
from collections import Counter

# Our Pydantic models
from backend.models.score_models import (
    ATSScoreResult,
    ComponentScore,
    ScoringWeights,
)

# Skill extractor for gap analysis
from backend.services.skill_extractor import skill_extractor

# Category weights from taxonomy
from nlp.skill_taxonomy import CATEGORY_WEIGHTS, SKILL_TO_CATEGORY

# Settings
from backend.config import settings

logger = logging.getLogger(__name__)


class ATSScorer:
    """
    Multi-component ATS scoring engine.

    Architecture:
    ┌──────────────────────────────────────┐
    │  Input: resume_doc + job_doc         │
    │  Both have: text, skills, nlp_feats  │
    └──────────────────┬───────────────────┘
                       │
            ┌──────────▼──────────┐
            │  5 Component Scores │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Weighted Aggregation│
            │  + Interpretation   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  ATSScoreResult     │
            └─────────────────────┘

    TF-IDF Note:
    We fit TfidfVectorizer on BOTH documents together (2-document corpus).
    This gives IDF values specific to the resume/job pair being scored.
    Alternative: fit on the full corpus — better IDF but requires
    re-fitting when new documents are added.
    """

    # Default scoring weights — sum must equal 1.0
    DEFAULT_WEIGHTS = ScoringWeights(
        tfidf_similarity = 0.35,
        skill_match      = 0.30,
        keyword_match    = 0.20,
        experience_match = 0.10,
        education_match  = 0.05,
    )

    # Score tier thresholds and labels
    SCORE_TIERS = [
        (0.80, "Excellent",  True,  "Strong match. Recommend for immediate interview."),
        (0.65, "Good",       True,  "Good match. Recommend for phone screen."),
        (0.50, "Fair",       False, "Partial match. Review manually before deciding."),
        (0.35, "Poor",       False, "Weak match. Likely missing key requirements."),
        (0.00, "Very Poor",  False, "Insufficient match. Does not meet minimum requirements."),
    ]

    # Education level keywords ordered by seniority
    # Higher index = higher qualification level
    EDUCATION_LEVELS = [
        "high school", "associate", "bachelor", "bs", "ba", "b.s", "b.a",
        "master", "ms", "ma", "m.s", "m.a", "mba",
        "phd", "ph.d", "doctorate", "doctoral"
    ]

    # Experience extraction patterns
    # Handles: "5 years", "five years", "5+ years", "5-7 years"
    EXPERIENCE_PATTERNS = [
        r'(\d+)\+?\s*(?:to|-)\s*(\d+)\s*years?',  # "5-7 years" or "5 to 7 years"
        r'(\d+)\+\s*years?',                        # "5+ years"
        r'(\d+)\s*years?\s*(?:of\s*)?experience',  # "5 years of experience"
        r'over\s*(\d+)\s*years?',                   # "over 5 years"
        r'more\s*than\s*(\d+)\s*years?',            # "more than 5 years"
    ]

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize scorer with default or custom weights.

        Args:
            weights: Optional custom ScoringWeights object.
                     If None, uses DEFAULT_WEIGHTS.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Validate weights sum to 1.0
        if not self.weights.validate_sum():
            raise ValueError(
                "Scoring weights must sum to 1.0. "
                f"Current sum: {sum(vars(self.weights).values())}"
            )

        logger.info(
            f"ATSScorer initialized. Weights: "
            f"tfidf={self.weights.tfidf_similarity}, "
            f"skills={self.weights.skill_match}, "
            f"keywords={self.weights.keyword_match}"
        )

    def score(
        self,
        resume_doc: Dict[str, Any],
        job_doc: Dict[str, Any]
    ) -> ATSScoreResult:
        """
        Main scoring entry point. Computes full ATS score for one
        resume against one job description.

        Args:
            resume_doc: MongoDB resume document dict.
                        Must contain: file_id, cleaned_text,
                        extracted_skills, nlp_features
            job_doc:    MongoDB job document dict.
                        Must contain: job_id, description,
                        required_skills, nlp_features

        Returns:
            ATSScoreResult: Complete scoring result with all components
        """
        start_time = time.time()

        # ── Extract core fields from documents ───────────────────────
        resume_id    = resume_doc.get("file_id", "")
        job_id       = job_doc.get("job_id", "")
        resume_text  = resume_doc.get("cleaned_text", "")
        job_text     = job_doc.get("description", "")

        # Fallback: use description directly if cleaned not available
        if not job_text:
            job_text = job_doc.get("cleaned_description", "")

        # Skills as sets for intersection/difference operations
        resume_skills = set(resume_doc.get("extracted_skills", []))
        job_skills    = set(job_doc.get("required_skills", []))

        # NLP feature dicts from MongoDB
        resume_nlp = resume_doc.get("nlp_features", {})
        job_nlp    = job_doc.get("nlp_features", {})

        logger.info(
            f"Scoring: resume={resume_id[:8]}... vs job={job_id[:8]}... | "
            f"resume_skills={len(resume_skills)}, job_skills={len(job_skills)}"
        )

        # ── Compute 5 component scores ────────────────────────────────

        # Component 1: TF-IDF cosine similarity (text-level match)
        tfidf_score, top_terms = self._compute_tfidf_similarity(
            resume_text, job_text
        )

        # Component 2: Skill match score (taxonomy-weighted intersection)
        skill_score, skill_gap = self._compute_skill_match_score(
            resume_skills, job_skills,
            resume_doc.get("skill_extraction_result", {})
        )

        # Component 3: Keyword match score (direct frequency overlap)
        keyword_score, matched_kw, missing_kw = self._compute_keyword_match(
            resume_nlp.get("tokens", []),
            job_nlp.get("tokens", [])
        )

        # Component 4: Experience match score
        experience_score = self._compute_experience_score(
            resume_text,
            job_doc.get("min_experience_years")
        )

        # Component 5: Education match score
        education_score = self._compute_education_score(
            resume_text,
            job_doc.get("education_requirement", "")
        )

        # ── Build ComponentScore objects with explanations ────────────
        components = self._build_component_scores(
            tfidf_score, skill_score, keyword_score,
            experience_score, education_score
        )

        # ── Weighted aggregation ──────────────────────────────────────
        final_score = (
            tfidf_score    * self.weights.tfidf_similarity +
            skill_score    * self.weights.skill_match +
            keyword_score  * self.weights.keyword_match +
            experience_score * self.weights.experience_match +
            education_score  * self.weights.education_match
        )

        # Clamp to [0.0, 1.0] — floating point arithmetic can drift slightly
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # ── Interpretation ────────────────────────────────────────────
        score_tier, recommendation, passes = self._interpret_score(final_score)

        processing_time = time.time() - start_time

        # ── Assemble result ───────────────────────────────────────────
        result = ATSScoreResult(
            resume_file_id=resume_id,
            job_id=job_id,
            final_score=round(final_score, 4),
            final_score_percent=round(final_score * 100, 2),
            component_scores=components,
            skill_gap=skill_gap,
            matched_keywords=matched_kw[:20],   # Top 20 matched keywords
            missing_keywords=missing_kw[:20],   # Top 20 missing keywords
            top_matching_terms=top_terms[:15],  # Top 15 TF-IDF terms
            score_tier=score_tier,
            recommendation=recommendation,
            passes_threshold=passes,
            processing_time_seconds=round(processing_time, 3)
        )

        logger.info(
            f"Score computed: {resume_id[:8]}... vs {job_id[:8]}... | "
            f"final={final_score:.4f} ({score_tier}) | "
            f"time={processing_time:.3f}s"
        )

        return result

    # ─────────────────────────────────────────────────────────────────
    # COMPONENT 1: TF-IDF COSINE SIMILARITY
    # ─────────────────────────────────────────────────────────────────

    def _compute_tfidf_similarity(
        self,
        resume_text: str,
        job_text: str
    ) -> Tuple[float, List[str]]:
        """
        Computes TF-IDF cosine similarity between resume and job description.

        Algorithm:
        1. Fit TfidfVectorizer on [resume_text, job_text] as a 2-doc corpus
        2. Transform both texts into TF-IDF sparse vectors
        3. Compute cosine similarity between the two vectors
        4. Extract top contributing terms for explainability

        TfidfVectorizer parameters:
        - ngram_range=(1,3): includes unigrams, bigrams, trigrams
          captures "machine learning" as a single feature
        - max_features=10000: vocabulary capped at 10K most frequent terms
          prevents memory explosion on large document sets
        - sublinear_tf=True: applies 1+log(tf) instead of raw tf
          reduces dominance of frequently repeated terms
        - min_df=1: include terms appearing in ≥1 document
          (with 2 docs, min_df=1 includes all terms)
        - analyzer='word': tokenize on word boundaries
        - stop_words='english': sklearn's built-in stop word list
          (our custom stop words already applied in NLP pipeline)

        Args:
            resume_text: Cleaned resume text string
            job_text:    Job description text string

        Returns:
            Tuple of (similarity_score, top_matching_terms)
        """

        # Handle empty text edge cases
        if not resume_text or not resume_text.strip():
            return 0.0, []
        if not job_text or not job_text.strip():
            return 0.0, []

        try:
            # ── Initialize TF-IDF Vectorizer ──────────────────────────
            vectorizer = TfidfVectorizer(
                # ngram_range=(1,3): capture unigrams, bigrams, trigrams
                # "machine learning engineer" becomes a single feature
                ngram_range=(1, 3),

                # max_features: vocabulary ceiling
                # prevents memory issues and speeds up matrix operations
                max_features=10_000,

                # sublinear_tf: applies 1 + log(tf) transformation
                # A term appearing 100× is not 100× more important than
                # one appearing 10×. Sublinear scaling reduces this bias.
                sublinear_tf=True,

                # min_df: minimum document frequency
                # 1 = include terms appearing in at least 1 document
                min_df=1,

                # analyzer: how to tokenize
                # 'word' splits on whitespace and punctuation
                analyzer='word',

                # stop_words: sklearn's built-in English stop words
                # Applied on top of our NLP pipeline stop word removal
                stop_words='english',

                # lowercase: normalize all text to lowercase
                # (our NLP pipeline already does this, but belt-and-suspenders)
                lowercase=True,

                # token_pattern: regex for valid tokens
                # Default matches word characters and hyphens
                # Allows: "c++", "node.js", "ci/cd" to be single tokens
                token_pattern=r"(?u)\b\w[\w\.\+\#\/\-]*\w\b|\b\w\b",
            )

            # ── Fit and transform ─────────────────────────────────────
            # Corpus = [resume, job_description]
            # vectorizer.fit_transform():
            #   1. Builds vocabulary from all tokens in both documents
            #   2. Computes IDF for each term based on document frequency
            #   3. Returns TF-IDF matrix: shape (2, vocab_size)
            #   Returns scipy sparse matrix for memory efficiency
            #   (most entries are 0 for typical documents)

            tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

            # tfidf_matrix[0]: resume's TF-IDF vector (1 × vocab_size)
            # tfidf_matrix[1]: job's TF-IDF vector    (1 × vocab_size)

            # ── Compute cosine similarity ─────────────────────────────
            # cosine_similarity expects 2D arrays
            # tfidf_matrix[0] is a sparse matrix row — already 2D when sliced
            # Result: 1×1 matrix [[similarity_value]]
            similarity_matrix = cosine_similarity(
                tfidf_matrix[0],   # resume vector
                tfidf_matrix[1]    # job description vector
            )

            # Extract scalar from the 1×1 result matrix
            # [0][0] = first row, first column = our single similarity score
            raw_score = float(similarity_matrix[0][0])

            # ── Extract top contributing terms ────────────────────────
            top_terms = self._extract_top_tfidf_terms(
                vectorizer, tfidf_matrix
            )

            return raw_score, top_terms

        except Exception as e:
            logger.error(f"TF-IDF computation failed: {e}")
            return 0.0, []

    def _extract_top_tfidf_terms(
        self,
        vectorizer: TfidfVectorizer,
        tfidf_matrix
    ) -> List[str]:
        """
        Identifies terms with high TF-IDF scores in BOTH documents.
        These are the most discriminating terms — they appear importantly
        in the resume AND the job description.

        Strategy:
        1. Get feature names (vocabulary) from vectorizer
        2. Get TF-IDF scores for resume (row 0) and job (row 1)
        3. Multiply element-wise: high score in BOTH = high overlap score
        4. Sort by overlap score and return top terms

        Args:
            vectorizer: Fitted TfidfVectorizer with vocabulary
            tfidf_matrix: (2, vocab_size) sparse TF-IDF matrix

        Returns:
            List of top matching term strings
        """
        try:
            # Get vocabulary (list of feature strings)
            # feature_names_out() returns numpy array of strings
            # Example: ["python", "machine learning", "docker", ...]
            feature_names = vectorizer.get_feature_names_out()

            # Convert sparse rows to dense 1D arrays
            # toarray() converts scipy sparse to numpy dense
            # [0] extracts the 1D array from shape (1, vocab_size)
            resume_scores = np.asarray(tfidf_matrix[0].todense()).flatten()
            job_scores    = np.asarray(tfidf_matrix[1].todense()).flatten()

            # Element-wise product: high only when both are high
            # This identifies terms important to both documents
            # (not just common in resume or just common in job)
            overlap_scores = resume_scores * job_scores

            # Get indices of top 15 terms by overlap score
            # np.argsort returns indices that would sort the array ascending
            # [::-1] reverses to descending
            # [:15] takes top 15
            top_indices = np.argsort(overlap_scores)[::-1][:15]

            # Filter to only include indices with non-zero overlap
            # (terms present in both documents)
            top_terms = [
                str(feature_names[i])
                for i in top_indices
                if overlap_scores[i] > 0
            ]

            return top_terms

        except Exception as e:
            logger.error(f"Top terms extraction failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────
    # COMPONENT 2: SKILL MATCH SCORE
    # ─────────────────────────────────────────────────────────────────

    def _compute_skill_match_score(
        self,
        resume_skills: Set[str],
        job_skills: Set[str],
        skill_extraction_result: Dict
    ) -> Tuple[float, Dict]:
        """
        Computes taxonomy-weighted skill overlap between resume and job.

        Unlike TF-IDF which treats all words equally, this component
        uses category weights and individual skill scores to give more
        credit for matching high-value skills.

        Algorithm:
        1. Compute raw intersection: matched = resume_skills ∩ job_skills
        2. For each matched skill: add category_weight to matched_weight
        3. For each job skill: add category_weight to total_weight
        4. Skill match score = matched_weight / total_weight

        Example:
            Job requires: ["python", "kubernetes", "docker"]
            Weights:       [1.5,      1.2,          1.2    ]
            Total weight = 3.9

            Resume has: ["python", "docker"]
            Matched weight = 1.5 + 1.2 = 2.7

            Score = 2.7 / 3.9 = 0.692

        Args:
            resume_skills:          Set of canonical skill names from resume
            job_skills:             Set of canonical skill names from job
            skill_extraction_result: Full extraction result for frequency data

        Returns:
            Tuple of (skill_match_score, skill_gap_dict)
        """

        # Handle edge case: no skills in job description
        if not job_skills:
            return 0.5, {"matched": [], "missing": [], "extra": [], "match_rate": 0.0}

        # Compute gap
        gap = skill_extractor.compute_skill_gap(resume_skills, job_skills)

        # If no skills extracted from resume at all
        if not resume_skills:
            return 0.0, gap

        # Get skill frequency data for weighting matched skills
        # Skills mentioned more frequently in resume score higher
        skill_freq = {}
        if skill_extraction_result:
            skill_freq = skill_extraction_result.get("skill_frequency", {})

        # ── Compute weighted scores ───────────────────────────────────

        # Total weighted importance of all job requirements
        total_weight = 0.0
        for skill in job_skills:
            # Look up this skill's category weight
            category = SKILL_TO_CATEGORY.get(skill, "engineering_practices")
            cat_weight = CATEGORY_WEIGHTS.get(category, 1.0)
            total_weight += cat_weight

        # Weighted sum of matched (covered) skills
        matched_weight = 0.0
        for skill in gap["matched"]:
            category = SKILL_TO_CATEGORY.get(skill, "engineering_practices")
            cat_weight = CATEGORY_WEIGHTS.get(category, 1.0)

            # Frequency bonus: skills appearing more often score higher
            # Cap frequency multiplier at 1.3 to prevent over-boosting
            freq = skill_freq.get(skill, 1)
            freq_multiplier = min(1.0 + (freq - 1) * 0.05, 1.3)

            matched_weight += cat_weight * freq_multiplier

        # Normalize to [0, 1]
        skill_match_score = (
            matched_weight / total_weight if total_weight > 0 else 0.0
        )

        # Clamp to [0, 1]
        skill_match_score = float(np.clip(skill_match_score, 0.0, 1.0))

        return skill_match_score, gap

    # ─────────────────────────────────────────────────────────────────
    # COMPONENT 3: KEYWORD MATCH SCORE
    # ─────────────────────────────────────────────────────────────────

    def _compute_keyword_match(
        self,
        resume_tokens: List[str],
        job_tokens: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Computes direct keyword overlap between resume and job tokens.

        This component captures important job keywords that aren't in
        our skill taxonomy — company values, domain terms, methodologies.

        Algorithm:
        1. Get unique vocabulary of job description tokens
        2. Count how many appear in resume tokens (as sets)
        3. Score = |resume_vocab ∩ job_vocab| / |job_vocab|
        4. Weight by term frequency in job — important terms count more

        Frequency-weighted version:
        - job_token_counts["agile"] = 5 → agile is important to this job
        - If resume also has "agile" → high contribution
        - If resume missing "agile" → significant gap

        Args:
            resume_tokens: Filtered tokens from resume NLP features
            job_tokens:    Filtered tokens from job NLP features

        Returns:
            Tuple of (keyword_score, matched_keywords, missing_keywords)
        """

        if not job_tokens:
            return 0.5, [], []

        if not resume_tokens:
            return 0.0, [], list(set(job_tokens))[:20]

        # Count token frequencies in each document
        resume_counter = Counter(resume_tokens)
        job_counter    = Counter(job_tokens)

        # Unique vocabularies as sets
        resume_vocab = set(resume_counter.keys())
        job_vocab    = set(job_counter.keys())

        # Matched and missing keywords
        matched  = list(resume_vocab & job_vocab)
        missing  = list(job_vocab - resume_vocab)

        # Sort matched by job frequency (most important matched terms first)
        matched.sort(key=lambda t: job_counter[t], reverse=True)

        # Sort missing by job frequency (most important gaps first)
        missing.sort(key=lambda t: job_counter[t], reverse=True)

        # ── Frequency-weighted score ──────────────────────────────────

        # Total weighted importance of all job keywords
        total_freq = sum(job_counter.values())

        if total_freq == 0:
            return 0.0, matched, missing

        # Sum frequencies of job terms that appear in resume
        matched_freq = sum(job_counter[term] for term in matched)

        # Score = fraction of job term frequency covered by resume
        # This gives more credit for matching high-frequency job terms
        keyword_score = matched_freq / total_freq

        # Clamp to [0, 1]
        keyword_score = float(np.clip(keyword_score, 0.0, 1.0))

        return keyword_score, matched, missing

    # ─────────────────────────────────────────────────────────────────
    # COMPONENT 4: EXPERIENCE MATCH SCORE
    # ─────────────────────────────────────────────────────────────────

    def _compute_experience_score(
        self,
        resume_text: str,
        min_years_required: Optional[int]
    ) -> float:
        """
        Estimates years of experience from resume and compares to job requirement.

        Extraction strategy:
        1. Regex search for experience patterns in resume text
        2. Take the maximum extracted value (most recent/senior experience)
        3. Compare to job's minimum_experience_years

        Scoring logic:
        - If no requirement: return 0.7 (neutral — can't score)
        - If no experience found in resume: return 0.3 (uncertain)
        - If resume_years >= required: score scales up to 1.0
        - If resume_years < required: penalized proportionally

        Scoring formula when underqualified:
            score = (resume_years / required_years) × 0.6
            Max 0.6 even if close — missing requirements always penalized

        Args:
            resume_text:        Full cleaned resume text
            min_years_required: Minimum experience from job description

        Returns:
            Experience match score 0.0 to 1.0
        """

        # If job has no experience requirement, return neutral score
        if not min_years_required or min_years_required <= 0:
            return 0.7  # Neutral — no penalty, no bonus

        # Extract years of experience from resume text
        resume_years = self._extract_years_experience(resume_text)

        # If we couldn't determine experience from resume
        if resume_years is None:
            return 0.4  # Slightly penalized — uncertain

        # Candidate meets or exceeds requirement
        if resume_years >= min_years_required:
            # Bonus for exceeding requirement (capped at 1.0)
            # 1 extra year → 0.05 bonus, 3 extra years → 0.15 bonus
            bonus = min((resume_years - min_years_required) * 0.05, 0.15)
            return min(1.0, 0.85 + bonus)

        # Candidate is underqualified
        # Proportional penalty: 4 years for 5 required = 0.6 × 0.8 = 0.48
        ratio = resume_years / min_years_required
        score = ratio * 0.6  # Max 0.6 when just under requirement

        return float(np.clip(score, 0.0, 0.6))

    def _extract_years_experience(self, text: str) -> Optional[int]:
        """
        Extracts years of experience from resume text using regex patterns.

        Handles patterns:
        - "5 years of experience"
        - "5+ years"
        - "5-7 years"
        - "over 5 years"
        - "more than 5 years"

        Returns the maximum extracted value (senior-most claim).

        Args:
            text: Resume text to search

        Returns:
            Integer years, or None if not extractable
        """
        text_lower = text.lower()
        found_years = []

        for pattern in self.EXPERIENCE_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Some patterns return tuples (range patterns like "5-7")
                if isinstance(match, tuple):
                    # For range "5-7 years", take the upper bound (7)
                    for val in match:
                        if val:
                            try:
                                found_years.append(int(val))
                            except ValueError:
                                pass
                else:
                    try:
                        found_years.append(int(match))
                    except ValueError:
                        pass

        if not found_years:
            return None

        # Return maximum found value — represents most senior claim
        # Filter out unrealistic values (> 40 years is likely a year, not experience)
        valid_years = [y for y in found_years if 0 < y <= 40]
        return max(valid_years) if valid_years else None

    # ─────────────────────────────────────────────────────────────────
    # COMPONENT 5: EDUCATION MATCH SCORE
    # ─────────────────────────────────────────────────────────────────

    def _compute_education_score(
        self,
        resume_text: str,
        education_requirement: Optional[str]
    ) -> float:
        """
        Scores education level match between resume and job requirement.

        Strategy:
        1. Find highest education level mentioned in resume
           (using EDUCATION_LEVELS list — higher index = higher level)
        2. Find required education level from job description
        3. Score based on whether resume meets or exceeds requirement

        EDUCATION_LEVELS is ordered by seniority:
            ["high school"(0), ..., "bachelor"(5), ..., "phd"(14)]

        Scoring:
        - Meets requirement:   1.0
        - Exceeds requirement: 1.0 (bonus does not exceed cap)
        - One level below:     0.7
        - Two levels below:    0.4
        - Three+ levels below: 0.2

        Args:
            resume_text:           Cleaned resume text
            education_requirement: Required education string from job

        Returns:
            Education match score 0.0 to 1.0
        """

        # No education requirement = neutral score
        if not education_requirement:
            return 0.7

        text_lower = resume_text.lower()
        req_lower  = education_requirement.lower()

        # ── Find resume education level ───────────────────────────────
        resume_edu_index = -1
        for i, level in enumerate(self.EDUCATION_LEVELS):
            if level in text_lower:
                resume_edu_index = i  # Track highest found (last iteration wins)

        # ── Find required education level ─────────────────────────────
        required_edu_index = -1
        for i, level in enumerate(self.EDUCATION_LEVELS):
            if level in req_lower:
                required_edu_index = i
                break  # Take first/lowest requirement mentioned

        # Can't determine requirement from text
        if required_edu_index == -1:
            return 0.7  # Neutral

        # Can't determine resume education
        if resume_edu_index == -1:
            return 0.4  # Uncertain — slight penalty

        # Compute gap: positive means exceeds, negative means below
        gap = resume_edu_index - required_edu_index

        if gap >= 0:
            return 1.0   # Meets or exceeds
        elif gap == -1:
            return 0.7   # One level below (e.g., associate vs bachelor)
        elif gap == -2:
            return 0.4   # Two levels below
        else:
            return 0.2   # Significantly underqualified

    # ─────────────────────────────────────────────────────────────────
    # SCORE ASSEMBLY AND INTERPRETATION
    # ─────────────────────────────────────────────────────────────────

    def _build_component_scores(
        self,
        tfidf_score: float,
        skill_score: float,
        keyword_score: float,
        experience_score: float,
        education_score: float
    ) -> List[ComponentScore]:
        """
        Builds ComponentScore objects with weighted contributions and
        human-readable explanations for each scoring dimension.

        These explanations are surfaced in the dashboard so recruiters
        understand WHY a candidate scored a particular number.

        Args:
            Five component score floats, all in range [0.0, 1.0]

        Returns:
            List of ComponentScore Pydantic objects
        """
        # Data for each component: (name, score, weight, explanation_template)
        components_data = [
            (
                "tfidf_similarity",
                tfidf_score,
                self.weights.tfidf_similarity,
                self._explain_tfidf(tfidf_score)
            ),
            (
                "skill_match",
                skill_score,
                self.weights.skill_match,
                self._explain_skill(skill_score)
            ),
            (
                "keyword_match",
                keyword_score,
                self.weights.keyword_match,
                self._explain_keyword(keyword_score)
            ),
            (
                "experience_match",
                experience_score,
                self.weights.experience_match,
                self._explain_experience(experience_score)
            ),
            (
                "education_match",
                education_score,
                self.weights.education_match,
                self._explain_education(education_score)
            ),
        ]

        components = []
        for name, raw_score, weight, explanation in components_data:
            components.append(ComponentScore(
                name=name,
                raw_score=round(raw_score, 4),
                weight=weight,
                weighted_contribution=round(raw_score * weight, 4),
                explanation=explanation
            ))

        return components

    def _explain_tfidf(self, score: float) -> str:
        """Human-readable TF-IDF score explanation."""
        if score >= 0.7:
            return f"Strong text similarity ({score:.0%}). Resume language closely mirrors job description."
        elif score >= 0.5:
            return f"Moderate text similarity ({score:.0%}). Significant vocabulary overlap with job posting."
        elif score >= 0.3:
            return f"Limited text similarity ({score:.0%}). Some common terms but substantial language gap."
        else:
            return f"Weak text similarity ({score:.0%}). Resume vocabulary largely different from job description."

    def _explain_skill(self, score: float) -> str:
        """Human-readable skill match score explanation."""
        if score >= 0.8:
            return f"Excellent skill coverage ({score:.0%}). Candidate has nearly all required skills."
        elif score >= 0.6:
            return f"Good skill coverage ({score:.0%}). Candidate has most required skills."
        elif score >= 0.4:
            return f"Partial skill coverage ({score:.0%}). Key skills are missing."
        else:
            return f"Poor skill coverage ({score:.0%}). Candidate lacks most required technical skills."

    def _explain_keyword(self, score: float) -> str:
        """Human-readable keyword match score explanation."""
        if score >= 0.7:
            return f"High keyword match ({score:.0%}). Resume uses job-relevant terminology throughout."
        elif score >= 0.5:
            return f"Moderate keyword match ({score:.0%}). Resume uses some job-specific language."
        else:
            return f"Low keyword match ({score:.0%}). Resume missing important job-specific keywords."

    def _explain_experience(self, score: float) -> str:
        """Human-readable experience score explanation."""
        if score >= 0.85:
            return "Experience requirement met or exceeded."
        elif score >= 0.7:
            return "Experience level not specified or requirement unclear."
        elif score >= 0.5:
            return "Experience level slightly below requirement."
        else:
            return "Experience level significantly below job requirement."

    def _explain_education(self, score: float) -> str:
        """Human-readable education score explanation."""
        if score >= 0.9:
            return "Education requirement met or exceeded."
        elif score >= 0.6:
            return "Education level not verified or requirement unclear."
        elif score >= 0.3:
            return "Education level slightly below requirement."
        else:
            return "Education level below stated requirement."

    def _interpret_score(
        self,
        final_score: float
    ) -> Tuple[str, str, bool]:
        """
        Converts a numeric score into a tier, recommendation, and pass/fail.

        Iterates through SCORE_TIERS thresholds in descending order.
        First threshold that final_score exceeds defines the tier.

        Args:
            final_score: Weighted aggregate score 0.0 to 1.0

        Returns:
            Tuple of (tier_label, recommendation_text, passes_threshold)
        """
        for threshold, tier, passes, recommendation in self.SCORE_TIERS:
            if final_score >= threshold:
                return tier, recommendation, passes

        # Fallback (should never reach here due to 0.00 threshold)
        return "Very Poor", "Does not meet minimum requirements.", False

    # ─────────────────────────────────────────────────────────────────
    # BATCH SCORING
    # ─────────────────────────────────────────────────────────────────

    def score_batch(
        self,
        resume_docs: List[Dict[str, Any]],
        job_doc: Dict[str, Any],
        threshold: float = 0.0
    ) -> List[ATSScoreResult]:
        """
        Scores multiple resumes against a single job description.

        Used by the batch scoring endpoint and candidate ranking module.
        Processes resumes sequentially — for true parallelism, wrap in
        asyncio.gather() or a ProcessPoolExecutor in production.

        Args:
            resume_docs: List of MongoDB resume document dicts
            job_doc:     Single job description document dict
            threshold:   Minimum score filter (0.0 = return all)

        Returns:
            List of ATSScoreResult sorted by final_score descending
        """
        results = []
        errors = []

        for i, resume in enumerate(resume_docs):
            try:
                result = self.score(resume, job_doc)

                # Apply threshold filter
                if result.final_score >= threshold:
                    results.append(result)

            except Exception as e:
                resume_id = resume.get("file_id", f"index_{i}")
                logger.error(f"Batch score failed for {resume_id}: {e}")
                errors.append(resume_id)

        if errors:
            logger.warning(f"Batch scoring: {len(errors)} failures: {errors}")

        # Sort by final_score descending (best candidates first)
        results.sort(key=lambda r: r.final_score, reverse=True)

        logger.info(
            f"Batch scoring complete: {len(results)} scored, "
            f"{len(errors)} errors, threshold={threshold}"
        )

        return results


# Single shared instance
# ATSScorer is stateless after initialization
ats_scorer = ATSScorer()