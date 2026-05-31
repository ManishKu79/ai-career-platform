# tests/test_nlp.py

import pytest
from backend.services.nlp_pipeline import NLPPipeline

# Use a fresh pipeline instance for tests
pipeline = NLPPipeline()

# Standard test resume text used across multiple tests
SAMPLE_RESUME_TEXT = """
John Smith
john.smith@email.com | (555) 123-4567 | San Francisco, CA

SUMMARY
Senior Software Engineer with 7 years of experience building scalable
distributed systems. Led a team of 10 engineers to deliver a machine
learning platform that reduced model training time by 40%.

EXPERIENCE
Senior Software Engineer - Google, 2020-2024
Architected and deployed microservices using Python and FastAPI.
Managed cloud infrastructure on AWS including EC2, S3, and Lambda.
Built CI/CD pipelines using Docker and Kubernetes.
Increased API response time by 60% through database query optimization.

Software Engineer - Startup Inc, 2018-2020
Developed REST APIs using Django and PostgreSQL.
Implemented machine learning models using scikit-learn and pandas.

SKILLS
Python, FastAPI, Django, PostgreSQL, MongoDB, AWS, Docker,
Kubernetes, Machine Learning, scikit-learn, pandas, numpy
"""


class TestTokenization:
    """Tests for token filtering."""

    def test_returns_nonempty_tokens(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert len(features.tokens) > 0

    def test_stop_words_removed(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        # Common stop words should not appear in tokens
        common_stops = {"the", "a", "an", "is", "was", "with", "and"}
        token_set = set(features.tokens)
        overlap = common_stops & token_set
        assert len(overlap) == 0, f"Stop words found in tokens: {overlap}"

    def test_technical_terms_preserved(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        token_set = set(features.tokens)
        # Technical terms should survive filtering
        for term in ["python", "fastapi", "docker", "kubernetes"]:
            assert term in token_set, f"'{term}' was incorrectly filtered"


class TestNGrams:
    """Tests for bigram and trigram extraction."""

    def test_bigrams_extracted(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert len(features.bigrams) > 0

    def test_bigrams_are_two_words(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        for bigram in features.bigrams[:20]:
            parts = bigram.split()
            assert len(parts) == 2, f"Bigram has wrong token count: '{bigram}'"

    def test_trigrams_are_three_words(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        for trigram in features.trigrams[:20]:
            parts = trigram.split()
            assert len(parts) == 3, f"Trigram has wrong token count: '{trigram}'"


class TestNER:
    """Tests for named entity recognition."""

    def test_entities_extracted(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert len(features.entities) > 0

    def test_organizations_detected(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        # "Google" should be detected as an organization
        org_texts = [org.lower() for org in features.organizations]
        assert any("google" in org for org in org_texts), \
            f"Google not found in organizations: {features.organizations}"

    def test_dates_extracted(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        # Resume contains "2020-2024" and "2018-2020"
        assert len(features.dates) > 0


class TestStatisticalFeatures:
    """Tests for word frequency and lexical diversity."""

    def test_word_frequency_populated(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert len(features.word_frequency) > 0

    def test_lexical_diversity_range(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        # Must be between 0 and 1 (it's a ratio)
        assert 0.0 <= features.lexical_diversity <= 1.0

    def test_total_tokens_positive(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert features.total_tokens > 0

    def test_unique_tokens_lte_total(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert features.unique_tokens <= features.total_tokens


class TestSentenceAnalysis:
    """Tests for sentence-level feature extraction."""

    def test_sentences_extracted(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert len(features.sentences) > 0

    def test_achievement_sentences_detected(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        # Resume has quantified achievements: "reduced by 40%", "increased by 60%"
        assert len(features.achievement_sentences) > 0, \
            "No achievement sentences detected in resume with quantified results"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_string_returns_empty_features(self):
        features = pipeline.process("")
        assert features.total_tokens == 0

    def test_short_text_does_not_crash(self):
        features = pipeline.process("Python developer")
        assert features is not None

    def test_processing_time_recorded(self):
        features = pipeline.process(SAMPLE_RESUME_TEXT)
        assert features.processing_time_seconds > 0