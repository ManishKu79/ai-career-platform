

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.db_service import DatabaseService
from backend.services.aggregation import AggregationService

db_service     = DatabaseService()
agg_service    = AggregationService()


class TestSafePercentage:
    """Tests for the _pct helper method."""

    def test_normal_percentage(self):
        assert agg_service._pct(50, 100) == 50.0

    def test_zero_denominator(self):
        assert agg_service._pct(10, 0) == 0.0

    def test_full_percentage(self):
        assert agg_service._pct(100, 100) == 100.0

    def test_rounding(self):
        result = agg_service._pct(1, 3)
        assert result == 33.3


class TestHistogramBuilder:
    """Tests for the _build_histogram helper."""

    def test_empty_scores(self):
        result = agg_service._build_histogram([])
        assert result == []

    def test_correct_bin_count(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = agg_service._build_histogram(scores, bins=5)
        assert len(result) == 5

    def test_all_scores_counted(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = agg_service._build_histogram(scores, bins=5)
        total_count = sum(bin_data["count"] for bin_data in result)
        assert total_count == len(scores)

    def test_bin_labels_format(self):
        scores = [0.5]
        result = agg_service._build_histogram(scores, bins=10)
        for bin_data in result:
            assert "%" in bin_data["label"]
            assert "bin_start" in bin_data
            assert "bin_end" in bin_data

    def test_scores_at_boundaries(self):
        # Edge case: score exactly at 1.0 should be in last bin
        scores = [0.0, 1.0]
        result = agg_service._build_histogram(scores, bins=10)
        total = sum(b["count"] for b in result)
        assert total == 2


class TestDatabaseServiceConstants:
    """Tests for DatabaseService configuration constants."""

    def test_all_collections_defined(self):
        required = {"resumes", "jobs", "scores", "rankings"}
        defined  = set(db_service.COLLECTIONS.values())
        assert required == defined

    def test_expected_indexes_defined(self):
        for collection in db_service.COLLECTIONS.values():
            assert collection in db_service.EXPECTED_INDEXES

    def test_resumes_has_unique_index(self):
        resume_indexes = db_service.EXPECTED_INDEXES["resumes"]
        assert "idx_resumes_file_id_unique" in resume_indexes

    def test_scores_has_compound_index(self):
        score_indexes = db_service.EXPECTED_INDEXES["scores"]
        assert "idx_scores_job_score_compound" in score_indexes


class TestScoringWeightValidation:
    """Tests for scoring-related logic in db_service."""

    def test_tier_thresholds_ordered(self):
        """Score tier thresholds should be in descending order."""
        from backend.services.ranker import CandidateRanker
        ranker = CandidateRanker()
        tiers = ranker.SHORTLIST_TIERS
        values = list(tiers.values())
        assert values == sorted(values, reverse=True)

    def test_ranking_weights_sum_to_one(self):
        """Ranking weights must sum to 1.0."""
        from backend.services.ranker import CandidateRanker
        ranker = CandidateRanker()
        total = sum(ranker.RANKING_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001
