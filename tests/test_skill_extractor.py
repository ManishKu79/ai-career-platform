# tests/test_skill_extractor.py

import pytest
from backend.services.skill_extractor import SkillExtractor
from backend.models.nlp_models import NLPFeatures

extractor = SkillExtractor()

# Synthetic NLPFeatures for controlled unit testing
def make_features(tokens=None, bigrams=None, trigrams=None,
                  organizations=None, noun_chunks=None):
    return NLPFeatures(
        tokens=tokens or [],
        bigrams=bigrams or [],
        trigrams=trigrams or [],
        organizations=organizations or [],
        noun_chunks=noun_chunks or [],
        lemmas=[], pos_tags=[], nouns=[], verbs=[],
        adjectives=[], entities=[], dates=[], quantities=[],
        word_frequency={}, total_tokens=0,
        unique_tokens=0, lexical_diversity=0.0
    )


class TestExactTokenMatch:

    def test_detects_python(self):
        features = make_features(tokens=["python", "developer"])
        result = extractor.extract(features, "python developer")
        assert "python" in result.all_skills

    def test_detects_docker(self):
        features = make_features(tokens=["docker", "kubernetes"])
        result = extractor.extract(features, "docker kubernetes")
        assert "docker" in result.all_skills

    def test_no_false_positives(self):
        # Random words should not match any skills
        features = make_features(tokens=["hello", "world", "foo", "bar"])
        result = extractor.extract(features, "hello world foo bar")
        assert result.total_skills_found == 0


class TestPhraseMatch:

    def test_detects_machine_learning(self):
        features = make_features(bigrams=["machine learning"])
        result = extractor.extract(features, "machine learning")
        assert "machine learning" in result.all_skills

    def test_detects_natural_language_processing(self):
        features = make_features(trigrams=["natural language processing"])
        result = extractor.extract(features, "natural language processing")
        assert "natural language processing" in result.all_skills

    def test_detects_spring_boot(self):
        features = make_features(bigrams=["spring boot"])
        result = extractor.extract(features, "spring boot")
        assert "spring boot" in result.all_skills


class TestAliasMatch:

    def test_postgres_maps_to_postgresql(self):
        features = make_features(tokens=["postgres"])
        result = extractor.extract(features, "postgres")
        assert "postgresql" in result.all_skills
        assert "postgres" not in result.all_skills

    def test_k8s_maps_to_kubernetes(self):
        features = make_features(tokens=["k8s"])
        result = extractor.extract(features, "k8s")
        assert "kubernetes" in result.all_skills

    def test_ml_maps_to_machine_learning(self):
        features = make_features(tokens=["ml"])
        result = extractor.extract(features, "ml")
        assert "machine learning" in result.all_skills


class TestFuzzyMatch:

    def test_matches_kubernetes_typo(self):
        result = extractor._find_closest_skill("kubernets")
        match, distance = result
        assert match == "kubernetes"
        assert distance <= 2

    def test_levenshtein_identical_strings(self):
        assert extractor._levenshtein_distance("python", "python") == 0

    def test_levenshtein_single_substitution(self):
        # "pythin" vs "python": 1 substitution (i→o)
        assert extractor._levenshtein_distance("pythin", "python") == 1

    def test_levenshtein_insertion(self):
        # "pytho" → "python": insert 'n' = 1 edit
        assert extractor._levenshtein_distance("pytho", "python") == 1


class TestSkillGapAnalysis:

    def test_matched_skills(self):
        resume = {"python", "docker", "fastapi"}
        job    = {"python", "docker", "kubernetes"}
        gap = extractor.compute_skill_gap(resume, job)
        assert set(gap["matched"]) == {"python", "docker"}

    def test_missing_skills(self):
        resume = {"python", "docker"}
        job    = {"python", "docker", "kubernetes", "aws"}
        gap = extractor.compute_skill_gap(resume, job)
        assert set(gap["missing"]) == {"kubernetes", "aws"}

    def test_extra_skills(self):
        resume = {"python", "docker", "rust"}
        job    = {"python", "docker"}
        gap = extractor.compute_skill_gap(resume, job)
        assert "rust" in gap["extra"]

    def test_match_rate_perfect(self):
        skills = {"python", "docker"}
        gap = extractor.compute_skill_gap(skills, skills)
        assert gap["match_rate"] == 1.0

    def test_match_rate_zero(self):
        resume = {"rust", "erlang"}
        job    = {"python", "java"}
        gap = extractor.compute_skill_gap(resume, job)
        assert gap["match_rate"] == 0.0


class TestSkillCategorization:

    def test_python_in_programming_languages(self):
        features = make_features(tokens=["python"])
        result = extractor.extract(features, "python")
        assert "python" in result.skills_by_category.get(
            "programming_languages", []
        )

    def test_docker_in_cloud_devops(self):
        features = make_features(tokens=["docker"])
        result = extractor.extract(features, "docker")
        assert "docker" in result.skills_by_category.get(
            "cloud_and_devops", []
        )

    def test_weighted_score_positive(self):
        features = make_features(tokens=["python", "python", "python"])
        result = extractor.extract(features, "python python python")
        python_score = result.skill_scores.get("python", 0)
        assert python_score > 0