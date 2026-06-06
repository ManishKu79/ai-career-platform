

import pytest
from backend.services.ats_scorer import ATSScorer
from backend.models.score_models import ScoringWeights

scorer = ATSScorer()

# ── Sample documents for testing ─────────────────────────────────────

RESUME_DOC = {
    "file_id": "test-resume-001",
    "cleaned_text": """
        Senior Python developer with 7 years experience.
        Built FastAPI microservices deployed on AWS using Docker and Kubernetes.
        Implemented machine learning pipelines with scikit-learn and pandas.
        Managed PostgreSQL and MongoDB databases.
        Led agile team using GitHub for version control and CI/CD with Jenkins.
        Bachelor of Science in Computer Science from MIT.
    """,
    "extracted_skills": [
        "python", "fastapi", "aws", "docker", "kubernetes",
        "machine learning", "scikit-learn", "pandas",
        "postgresql", "mongodb", "git", "jenkins", "agile"
    ],
    "skill_extraction_result": {
        "skill_frequency": {
            "python": 3, "fastapi": 2, "aws": 2,
            "docker": 2, "kubernetes": 1
        }
    },
    "nlp_features": {
        "tokens": [
            "python", "developer", "fastapi", "microservices",
            "aws", "docker", "kubernetes", "machine", "learning",
            "postgresql", "mongodb", "agile", "jenkins"
        ]
    }
}

JOB_DOC = {
    "job_id": "test-job-001",
    "title": "Senior Python Engineer",
    "description": """
        We are seeking a Senior Python Engineer with 5+ years experience.
        Must have experience with FastAPI or Django for REST API development.
        Required: Docker, Kubernetes, AWS cloud infrastructure.
        Experience with PostgreSQL or MongoDB databases required.
        Machine learning knowledge is a strong plus.
        Bachelor's degree in Computer Science or related field required.
        Agile development environment using GitHub and CI/CD practices.
    """,
    "required_skills": [
        "python", "fastapi", "docker", "kubernetes", "aws",
        "postgresql", "mongodb", "machine learning", "agile", "git"
    ],
    "min_experience_years": 5,
    "education_requirement": "bachelor",
    "nlp_features": {
        "tokens": [
            "python", "engineer", "fastapi", "django", "docker",
            "kubernetes", "aws", "postgresql", "mongodb",
            "machine", "learning", "agile", "github", "bachelor"
        ]
    }
}

UNRELATED_RESUME = {
    "file_id": "test-resume-002",
    "cleaned_text": "Marketing manager with 10 years in brand management and advertising.",
    "extracted_skills": ["marketing", "branding", "advertising"],
    "skill_extraction_result": {"skill_frequency": {}},
    "nlp_features": {"tokens": ["marketing", "manager", "brand", "advertising"]}
}


class TestTFIDFSimilarity:

    def test_similar_documents_score_higher(self):
        """Matching resume should score higher than unrelated one."""
        match_score, _ = scorer._compute_tfidf_similarity(
            RESUME_DOC["cleaned_text"],
            JOB_DOC["description"]
        )
        nomatch_score, _ = scorer._compute_tfidf_similarity(
            UNRELATED_RESUME["cleaned_text"],
            JOB_DOC["description"]
        )
        assert match_score > nomatch_score

    def test_identical_documents_score_one(self):
        """Same text compared to itself should return 1.0."""
        score, _ = scorer._compute_tfidf_similarity(
            "python developer fastapi docker",
            "python developer fastapi docker"
        )
        assert abs(score - 1.0) < 0.001

    def test_empty_resume_returns_zero(self):
        score, _ = scorer._compute_tfidf_similarity("", JOB_DOC["description"])
        assert score == 0.0

    def test_top_terms_returned(self):
        _, terms = scorer._compute_tfidf_similarity(
            RESUME_DOC["cleaned_text"],
            JOB_DOC["description"]
        )
        assert len(terms) > 0
        assert all(isinstance(t, str) for t in terms)


class TestSkillMatchScore:

    def test_perfect_skill_match(self):
        skills = {"python", "docker", "kubernetes"}
        score, _ = scorer._compute_skill_match_score(skills, skills, {})
        assert score >= 0.95

    def test_zero_skill_match(self):
        resume = {"marketing", "advertising"}
        job    = {"python", "docker", "aws"}
        score, _ = scorer._compute_skill_match_score(resume, job, {})
        assert score == 0.0

    def test_partial_skill_match_proportional(self):
        resume = {"python", "docker"}
        job    = {"python", "docker", "kubernetes", "aws"}
        score, _ = scorer._compute_skill_match_score(resume, job, {})
        # Should be between 0.3 and 0.7 for 50% coverage
        assert 0.3 < score < 0.7

    def test_skill_gap_structure(self):
        resume = {"python", "docker"}
        job    = {"python", "docker", "kubernetes"}
        _, gap = scorer._compute_skill_match_score(resume, job, {})
        assert "matched" in gap
        assert "missing" in gap
        assert "extra" in gap
        assert "kubernetes" in gap["missing"]


class TestExperienceScoring:

    def test_meets_requirement(self):
        text = "7 years of experience in software development"
        score = scorer._compute_experience_score(text, 5)
        assert score >= 0.85

    def test_below_requirement(self):
        text = "2 years of experience"
        score = scorer._compute_experience_score(text, 5)
        assert score < 0.5

    def test_no_requirement_neutral(self):
        text = "experienced developer"
        score = scorer._compute_experience_score(text, None)
        assert score == 0.7

    def test_extracts_range_years(self):
        text = "5-7 years experience building distributed systems"
        years = scorer._extract_years_experience(text)
        assert years == 7  # Upper bound of range

    def test_extracts_plus_years(self):
        years = scorer._extract_years_experience("5+ years of Python")
        assert years == 5


class TestEducationScoring:

    def test_meets_bachelor_requirement(self):
        resume = "Bachelor of Science in Computer Science from Stanford"
        score = scorer._compute_education_score(resume, "bachelor")
        assert score == 1.0

    def test_exceeds_requirement_with_masters(self):
        resume = "Master of Science in Computer Science"
        score = scorer._compute_education_score(resume, "bachelor")
        assert score == 1.0

    def test_phd_exceeds_all(self):
        resume = "PhD in Machine Learning from MIT"
        score = scorer._compute_education_score(resume, "master")
        assert score == 1.0

    def test_no_requirement_neutral(self):
        score = scorer._compute_education_score("some resume text", "")
        assert score == 0.7


class TestFinalScoring:

    def test_full_score_returns_result(self):
        result = scorer.score(RESUME_DOC, JOB_DOC)
        assert result is not None
        assert 0.0 <= result.final_score <= 1.0

    def test_matching_resume_beats_unrelated(self):
        good_result  = scorer.score(RESUME_DOC, JOB_DOC)
        bad_result   = scorer.score(UNRELATED_RESUME, JOB_DOC)
        assert good_result.final_score > bad_result.final_score

    def test_score_components_sum_correctly(self):
        result = scorer.score(RESUME_DOC, JOB_DOC)
        summed = sum(c.weighted_contribution for c in result.component_scores)
        assert abs(summed - result.final_score) < 0.01

    def test_score_tier_present(self):
        result = scorer.score(RESUME_DOC, JOB_DOC)
        valid_tiers = {"Excellent", "Good", "Fair", "Poor", "Very Poor"}
        assert result.score_tier in valid_tiers

    def test_passes_threshold_for_good_match(self):
        result = scorer.score(RESUME_DOC, JOB_DOC)
        assert result.passes_threshold is True

    def test_fails_threshold_for_unrelated(self):
        result = scorer.score(UNRELATED_RESUME, JOB_DOC)
        assert result.passes_threshold is False

    def test_skill_gap_in_result(self):
        result = scorer.score(RESUME_DOC, JOB_DOC)
        assert "matched" in result.skill_gap
        assert "missing" in result.skill_gap


class TestScoringWeights:

    def test_default_weights_sum_to_one(self):
        weights = ScoringWeights()
        assert weights.validate_sum() is True

    def test_invalid_weights_detected(self):
        weights = ScoringWeights(
            tfidf_similarity=0.5,
            skill_match=0.5,
            keyword_match=0.5,
            experience_match=0.1,
            education_match=0.1
        )
        assert weights.validate_sum() is False

    def test_custom_weights_accepted(self):
        custom = ScoringWeights(
            tfidf_similarity=0.40,
            skill_match=0.35,
            keyword_match=0.15,
            experience_match=0.05,
            education_match=0.05
        )
        scorer_custom = ATSScorer(weights=custom)
        assert scorer_custom is not None
