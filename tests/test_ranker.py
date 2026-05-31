# tests/test_ranker.py

import pytest
import numpy as np
from backend.services.ranker import CandidateRanker
from backend.models.ranking_models import RankedCandidate

ranker = CandidateRanker()


# ── Sample data fixtures ──────────────────────────────────────────────

def make_candidate(
    file_id: str,
    ats_score: float,
    skill_match_rate: float = 0.7,
    achievement_count: int = 2,
    total_skills: int = 8,
    timestamp: str = "2024-01-01T00:00:00"
) -> dict:
    """Factory for test candidate enriched dicts."""
    return {
        "file_id": file_id,
        "candidate_name": f"Candidate {file_id}",
        "candidate_email": f"{file_id}@test.com",
        "upload_timestamp": timestamp,
        "ats_score": ats_score,
        "score_tier": "Good" if ats_score >= 0.65 else "Fair",
        "passes_threshold": ats_score >= 0.50,
        "skill_depth_score":     skill_match_rate,
        "achievement_score":     min(achievement_count / 10, 1.0),
        "experience_score":      0.85,
        "lexical_quality_score": 0.72,
        "achievement_count":     achievement_count,
        "lexical_diversity":     0.72,
        "total_skills_found":    total_skills,
        "matched_skills":        ["python", "docker"],
        "missing_skills":        ["kubernetes"],
        "extra_skills":          ["rust"],
        "skill_match_rate":      skill_match_rate,
        "top_skills":            ["python", "docker", "aws"],
    }


SAMPLE_CANDIDATES = [
    make_candidate("c001", 0.85, skill_match_rate=0.90, achievement_count=5),
    make_candidate("c002", 0.72, skill_match_rate=0.70, achievement_count=3),
    make_candidate("c003", 0.72, skill_match_rate=0.80, achievement_count=4),
    make_candidate("c004", 0.55, skill_match_rate=0.50, achievement_count=1),
    make_candidate("c005", 0.40, skill_match_rate=0.30, achievement_count=0),
]


class TestStableSort:

    def test_higher_ats_score_ranked_first(self):
        """Candidate with higher composite score should rank first."""
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        sorted_list = ranker._stable_sort(scored)
        assert sorted_list[0]["file_id"] == "c001"

    def test_lowest_score_ranked_last(self):
        """Candidate with lowest score should rank last."""
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        sorted_list = ranker._stable_sort(scored)
        assert sorted_list[-1]["file_id"] == "c005"

    def test_tie_broken_by_skill_match(self):
        """
        c002 and c003 have identical ATS scores (0.72).
        c003 has higher skill_match_rate (0.80 vs 0.70).
        c003 should rank above c002.
        """
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        sorted_list = ranker._stable_sort(scored)

        c002_rank = next(i for i, c in enumerate(sorted_list) if c["file_id"] == "c002")
        c003_rank = next(i for i, c in enumerate(sorted_list) if c["file_id"] == "c003")

        # c003 should have lower index (better rank) than c002
        assert c003_rank < c002_rank

    def test_returns_all_candidates(self):
        """Sort should not drop any candidates."""
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        sorted_list = ranker._stable_sort(scored)
        assert len(sorted_list) == len(SAMPLE_CANDIDATES)


class TestCompositeScores:

    def test_composite_within_range(self):
        """All composite scores should be between 0.0 and 1.0."""
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        for c in scored:
            assert 0.0 <= c["composite_score"] <= 1.0

    def test_rank_criteria_count(self):
        """Each candidate should have criteria for all 5 dimensions."""
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        for c in scored:
            assert len(c["rank_criteria"]) == 5

    def test_criteria_contributions_sum_to_composite(self):
        """Sum of criterion contributions ≈ composite_score."""
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        for c in scored:
            contribution_sum = sum(cr.contribution for cr in c["rank_criteria"])
            assert abs(contribution_sum - c["composite_score"]) < 0.01

    def test_percentiles_range(self):
        """All percentiles should be between 0 and 100."""
        scored = ranker._compute_composite_scores(
            SAMPLE_CANDIDATES, ranker.RANKING_WEIGHTS
        )
        for c in scored:
            for criterion in c["rank_criteria"]:
                assert 0 <= criterion.percentile <= 100


class TestShortlistTiers:

    def test_priority_tier_assignment(self):
        assert ranker._assign_shortlist_tier(0.85) == "priority"
        assert ranker._assign_shortlist_tier(0.80) == "priority"

    def test_standard_tier_assignment(self):
        assert ranker._assign_shortlist_tier(0.70) == "standard"
        assert ranker._assign_shortlist_tier(0.65) == "standard"

    def test_reserve_tier_assignment(self):
        assert ranker._assign_shortlist_tier(0.55) == "reserve"
        assert ranker._assign_shortlist_tier(0.50) == "reserve"

    def test_reject_tier_assignment(self):
        assert ranker._assign_shortlist_tier(0.49) == "reject"
        assert ranker._assign_shortlist_tier(0.00) == "reject"

    def test_shortlist_generation_structure(self):
        """Generated shortlists should have correct keys."""
        candidates_mock = []
        for c in SAMPLE_CANDIDATES:
            scored = ranker._compute_composite_scores(
                [c], ranker.RANKING_WEIGHTS
            )[0]
            tier = ranker._assign_shortlist_tier(c["ats_score"])
            ranked = RankedCandidate(
                file_id=c["file_id"],
                job_id="job-001",
                rank=1, total_candidates=5, percentile=80.0,
                ats_score=c["ats_score"],
                ats_score_percent=c["ats_score"] * 100,
                composite_score=scored["composite_score"],
                score_tier=c["score_tier"],
                passes_threshold=c["passes_threshold"],
                shortlist_tier=tier,
            )
            candidates_mock.append(ranked)

        shortlists = ranker._generate_shortlists(candidates_mock)
        assert "priority" in shortlists
        assert "standard" in shortlists
        assert "reserve" in shortlists


class TestPoolStatistics:

    def test_statistics_structure(self):
        """Pool statistics should contain all expected keys."""
        candidates = [
            RankedCandidate(
                file_id=f"c{i}", job_id="j1",
                rank=i+1, total_candidates=5,
                percentile=float(80 - i * 10),
                ats_score=score,
                ats_score_percent=score * 100,
                composite_score=score,
                score_tier="Good",
                passes_threshold=score >= 0.5,
                shortlist_tier="standard",
                total_skills_found=8
            )
            for i, score in enumerate([0.85, 0.75, 0.65, 0.55, 0.45])
        ]

        stats = ranker._compute_pool_statistics(candidates)
        required_keys = [
            "mean_score", "median_score", "std_score",
            "min_score", "max_score", "p25_score", "p75_score"
        ]
        for key in required_keys:
            assert key in stats

    def test_mean_score_correct(self):
        """Mean score should be average of all ATS scores."""
        scores = [0.80, 0.70, 0.60]
        candidates = [
            RankedCandidate(
                file_id=f"c{i}", job_id="j1",
                rank=i+1, total_candidates=3,
                percentile=float(80 - i * 20),
                ats_score=s,
                ats_score_percent=s * 100,
                composite_score=s,
                score_tier="Good",
                passes_threshold=True,
                shortlist_tier="standard",
                total_skills_found=5
            )
            for i, s in enumerate(scores)
        ]
        stats = ranker._compute_pool_statistics(candidates)
        expected_mean = sum(scores) / len(scores)
        assert abs(stats["mean_score"] - expected_mean) < 0.001


class TestSkillDepth:

    def test_no_skills_returns_zero(self):
        score = ranker._compute_skill_depth({}, set())
        assert score == 0.0

    def test_matched_skills_score_higher(self):
        skill_result = {
            "skill_scores": {
                "python": 4.5,
                "docker": 3.2,
                "rust":   2.0
            }
        }
        matched_all   = {"python", "docker", "rust"}
        matched_partial = {"python"}

        score_all     = ranker._compute_skill_depth(skill_result, matched_all)
        score_partial = ranker._compute_skill_depth(skill_result, matched_partial)

        assert score_all > score_partial

    def test_depth_score_within_range(self):
        skill_result = {
            "skill_scores": {"python": 5.0, "docker": 3.0}
        }
        score = ranker._compute_skill_depth(skill_result, {"python", "docker"})
        assert 0.0 <= score <= 1.0