# backend/services/skill_extractor.py

# time: performance measurement
import time

# re: regex for text preprocessing during extraction
import re

# logging: structured application logging
import logging

# typing: type annotations for all function signatures
from typing import List, Dict, Set, Tuple, Optional

# collections.Counter: fast frequency counting
# collections.defaultdict: dict with automatic default values
from collections import Counter, defaultdict

# Our taxonomy — the knowledge base
from nlp.skill_taxonomy import (
    SKILL_TAXONOMY,
    SKILL_ALIASES,
    CATEGORY_WEIGHTS,
    ALL_SKILLS,
    SKILL_TO_CATEGORY,
    MULTI_WORD_SKILLS,
    SINGLE_WORD_SKILLS,
    COMPLETE_SKILL_LOOKUP,
)

# Our Pydantic models
from backend.models.skill_models import (
    SkillExtractionResult,
    ExtractedSkill,
    ExtractionMethod,
)

# NLP features produced by Module 3
from backend.models.nlp_models import NLPFeatures

logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Multi-strategy skill extraction engine.

    Extraction pipeline (5 strategies, run in order):
    ┌─────────────────────────────────────────────┐
    │ Strategy 1: Exact Token Match               │
    │   tokens → intersect with SINGLE_WORD_SKILLS│
    │   confidence: 1.0                           │
    ├─────────────────────────────────────────────┤
    │ Strategy 2: Phrase Match                    │
    │   bigrams + trigrams → intersect MULTI_WORD │
    │   confidence: 1.0                           │
    ├─────────────────────────────────────────────┤
    │ Strategy 3: Alias Resolution                │
    │   tokens → check SKILL_ALIASES dict         │
    │   confidence: 0.95                          │
    ├─────────────────────────────────────────────┤
    │ Strategy 4: NER Entity Matching             │
    │   org entities → match known tech terms     │
    │   confidence: 0.85                          │
    ├─────────────────────────────────────────────┤
    │ Strategy 5: Fuzzy Matching                  │
    │   unmatched tokens → Levenshtein distance   │
    │   confidence: 0.70 - 0.90                   │
    └─────────────────────────────────────────────┘
         │
         ▼
    Deduplication + Frequency Count
         │
         ▼
    Score Computation
         │
         ▼
    SkillExtractionResult
    """

    # Confidence scores per extraction method
    # These are calibrated heuristics — in production, calibrated via
    # precision/recall against a labeled dataset
    CONFIDENCE_SCORES = {
        ExtractionMethod.EXACT_TOKEN:  1.00,
        ExtractionMethod.PHRASE_MATCH: 1.00,
        ExtractionMethod.ALIAS_MATCH:  0.95,
        ExtractionMethod.NER_ENTITY:   0.85,
        ExtractionMethod.NOUN_CHUNK:   0.80,
        ExtractionMethod.FUZZY_MATCH:  0.75,
    }

    # Minimum confidence threshold — skills below this are discarded
    MIN_CONFIDENCE = 0.70

    # Maximum Levenshtein distance for fuzzy matching
    # Distance 2 catches most typos without too many false positives
    MAX_EDIT_DISTANCE = 2

    # Minimum skill name length to attempt fuzzy matching
    # Fuzzy matching short strings produces too many false positives
    MIN_FUZZY_LENGTH = 5

    def extract(
        self,
        nlp_features: NLPFeatures,
        cleaned_text: str
    ) -> SkillExtractionResult:
        """
        Main entry point. Runs all extraction strategies and assembles result.

        Args:
            nlp_features: NLPFeatures object from Module 3 pipeline
            cleaned_text:  Original cleaned resume text for regex fallback

        Returns:
            SkillExtractionResult: Fully populated skill extraction object
        """
        start_time = time.time()

        # ── Step 1: Run all extraction strategies ────────────────────
        # Each strategy returns: List[Tuple[surface_form, canonical, method]]
        # surface_form: what was written in the resume
        # canonical: standardized taxonomy name
        # method: which strategy found it

        raw_matches: List[Tuple[str, str, ExtractionMethod]] = []

        # Strategy 1: Exact single-word token matching
        exact_matches = self._exact_token_match(nlp_features.tokens)
        raw_matches.extend(exact_matches)

        # Strategy 2: Multi-word phrase matching (bigrams + trigrams)
        phrase_matches = self._phrase_match(
            nlp_features.bigrams,
            nlp_features.trigrams
        )
        raw_matches.extend(phrase_matches)

        # Strategy 3: Alias resolution for abbreviations and variants
        alias_matches = self._alias_match(
            nlp_features.tokens,
            nlp_features.bigrams
        )
        raw_matches.extend(alias_matches)

        # Strategy 4: NER entity matching (ORG entities = tech companies/tools)
        ner_matches = self._ner_entity_match(nlp_features.organizations)
        raw_matches.extend(ner_matches)

        # Strategy 5: Noun chunk extraction for compound skills
        chunk_matches = self._noun_chunk_match(nlp_features.noun_chunks)
        raw_matches.extend(chunk_matches)

        # Strategy 6: Fuzzy matching for typos and near-misses
        # Only run on tokens not already matched — avoid double-processing
        already_matched_surfaces = {m[0] for m in raw_matches}
        unmatched_tokens = [
            t for t in nlp_features.tokens
            if t not in already_matched_surfaces
        ]
        fuzzy_matches = self._fuzzy_match(unmatched_tokens)
        raw_matches.extend(fuzzy_matches)

        logger.info(
            f"Raw matches: exact={len(exact_matches)}, "
            f"phrase={len(phrase_matches)}, alias={len(alias_matches)}, "
            f"ner={len(ner_matches)}, chunk={len(chunk_matches)}, "
            f"fuzzy={len(fuzzy_matches)}"
        )

        # ── Step 2: Aggregate and deduplicate ────────────────────────
        # Multiple strategies may find the same canonical skill
        # Aggregate by canonical name, tracking best method and frequency
        aggregated = self._aggregate_matches(raw_matches, cleaned_text)

        # ── Step 3: Build ExtractedSkill objects ─────────────────────
        skill_details = self._build_skill_details(aggregated)

        # ── Step 4: Assemble result object ────────────────────────────
        result = self._assemble_result(skill_details, start_time)

        logger.info(
            f"Skill extraction complete: {result.total_skills_found} skills "
            f"in {result.processing_time_seconds}s | "
            f"categories: {list(result.skills_per_category.keys())}"
        )

        return result

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY 1: EXACT TOKEN MATCH
    # ─────────────────────────────────────────────────────────────────

    def _exact_token_match(
        self,
        tokens: List[str]
    ) -> List[Tuple[str, str, ExtractionMethod]]:
        """
        Checks each token against SINGLE_WORD_SKILLS using set intersection.

        Set intersection is O(min(len(tokens), len(skills))) — extremely fast.
        For 500 tokens and 200 single-word skills, this runs in microseconds.

        Example:
            tokens = ["python", "developer", "fastapi", "experience"]
            SINGLE_WORD_SKILLS = {"python", "fastapi", "docker", ...}
            matches = [("python", "python"), ("fastapi", "fastapi")]

        Args:
            tokens: Filtered tokens from NLPFeatures

        Returns:
            List of (surface_form, canonical_name, ExtractionMethod) tuples
        """
        matches = []

        # Convert tokens to set for O(1) per-lookup after O(n) conversion
        token_set = set(tokens)

        # Intersection: tokens that exist in our single-word skills set
        # Both sides already lowercase from NLP pipeline
        found_skills = token_set & SINGLE_WORD_SKILLS

        for skill in found_skills:
            # Verify skill is in taxonomy (sanity check)
            if skill in SKILL_TO_CATEGORY:
                matches.append((
                    skill,                      # surface form = canonical for exact match
                    skill,                      # canonical name
                    ExtractionMethod.EXACT_TOKEN
                ))

        return matches

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY 2: PHRASE MATCH
    # ─────────────────────────────────────────────────────────────────

    def _phrase_match(
        self,
        bigrams: List[str],
        trigrams: List[str]
    ) -> List[Tuple[str, str, ExtractionMethod]]:
        """
        Checks bigrams and trigrams against MULTI_WORD_SKILLS.

        This is the critical strategy for compound skills:
            "machine learning", "deep learning", "natural language processing"
            "ruby on rails", "test driven development", "spring boot"

        Without phrase matching, "machine" and "learning" would match
        nothing individually but together match "machine learning".

        Args:
            bigrams:  List of 2-word phrases from NLP pipeline
            trigrams: List of 3-word phrases from NLP pipeline

        Returns:
            List of (surface_form, canonical_name, ExtractionMethod) tuples
        """
        matches = []

        # Combine bigrams and trigrams into one phrase list
        all_phrases = bigrams + trigrams

        # Convert to set for efficient lookup
        phrase_set = set(all_phrases)

        # Check intersection with multi-word taxonomy skills
        found_phrases = phrase_set & MULTI_WORD_SKILLS

        for phrase in found_phrases:
            if phrase in SKILL_TO_CATEGORY:
                matches.append((
                    phrase,
                    phrase,
                    ExtractionMethod.PHRASE_MATCH
                ))

        return matches

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY 3: ALIAS RESOLUTION
    # ─────────────────────────────────────────────────────────────────

    def _alias_match(
        self,
        tokens: List[str],
        bigrams: List[str]
    ) -> List[Tuple[str, str, ExtractionMethod]]:
        """
        Resolves abbreviations and variants to canonical skill names.

        The SKILL_ALIASES dict maps surface forms to canonical names:
            "postgres" → "postgresql"
            "k8s"      → "kubernetes"
            "ml"       → "machine learning"
            "sklearn"  → "scikit-learn"

        We check both single tokens and bigrams against the alias dict.

        Args:
            tokens:  Single-word tokens from NLP features
            bigrams: Two-word phrases from NLP features

        Returns:
            List of (surface_form, canonical_name, ExtractionMethod) tuples
        """
        matches = []

        # Track already-resolved canonicals to avoid duplicates
        # (both "ml" and "machine learning" shouldn't both resolve to
        # "machine learning" — the phrase match already handles the latter)
        resolved_canonicals: Set[str] = set()

        # Check single tokens against aliases
        for token in tokens:
            if token in SKILL_ALIASES:
                canonical = SKILL_ALIASES[token]
                # Only add if this canonical not yet found by other strategies
                if canonical not in resolved_canonicals:
                    matches.append((
                        token,       # surface: "postgres"
                        canonical,   # canonical: "postgresql"
                        ExtractionMethod.ALIAS_MATCH
                    ))
                    resolved_canonicals.add(canonical)

        # Check bigrams against aliases (catches "google cloud" → "gcp")
        for bigram in bigrams:
            if bigram in SKILL_ALIASES:
                canonical = SKILL_ALIASES[bigram]
                if canonical not in resolved_canonicals:
                    matches.append((
                        bigram,
                        canonical,
                        ExtractionMethod.ALIAS_MATCH
                    ))
                    resolved_canonicals.add(canonical)

        return matches

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY 4: NER ENTITY MATCHING
    # ─────────────────────────────────────────────────────────────────

    def _ner_entity_match(
        self,
        organizations: List[str]
    ) -> List[Tuple[str, str, ExtractionMethod]]:
        """
        Matches NER-identified ORG entities against the skill taxonomy.

        spaCy's NER labels many technology names as ORG entities:
        "Google", "AWS", "MongoDB" → ORG
        "Python", "Docker" → sometimes ORG, sometimes PRODUCT

        By checking the extracted ORG entities against our taxonomy,
        we catch skills that appear in context like:
            "worked at Google" → "google" in organizations → no skill
            "deployed on AWS"  → "aws" in organizations → skill matched
            "used MongoDB"     → "mongodb" in organizations → skill matched

        Args:
            organizations: List of ORG entity strings from NER

        Returns:
            List of (surface_form, canonical_name, ExtractionMethod) tuples
        """
        matches = []

        for org in organizations:
            # Normalize: lowercase and strip whitespace
            org_lower = org.lower().strip()

            # Direct taxonomy match
            if org_lower in ALL_SKILLS:
                matches.append((
                    org,         # preserve original casing as surface form
                    org_lower,   # lowercase as canonical
                    ExtractionMethod.NER_ENTITY
                ))

            # Alias resolution for org entities
            elif org_lower in SKILL_ALIASES:
                canonical = SKILL_ALIASES[org_lower]
                matches.append((
                    org,
                    canonical,
                    ExtractionMethod.NER_ENTITY
                ))

        return matches

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY 5: NOUN CHUNK MATCHING
    # ─────────────────────────────────────────────────────────────────

    def _noun_chunk_match(
        self,
        noun_chunks: List[str]
    ) -> List[Tuple[str, str, ExtractionMethod]]:
        """
        Finds skills within spaCy's syntactic noun chunks.

        Noun chunks are syntactically richer than raw n-grams.
        They capture: "advanced machine learning techniques"
        Substrings: "machine learning" ← this is our skill

        Strategy:
        1. Check full chunk against taxonomy (direct)
        2. Check if any MULTI_WORD_SKILL is a substring of the chunk
        3. Check if any single-word skill appears in the chunk

        Args:
            noun_chunks: Syntactic noun phrases from NLP pipeline

        Returns:
            List of (surface_form, canonical_name, ExtractionMethod) tuples
        """
        matches = []
        seen: Set[str] = set()

        for chunk in noun_chunks:
            chunk_lower = chunk.lower().strip()

            # Check 1: Full chunk is a known skill
            if chunk_lower in ALL_SKILLS and chunk_lower not in seen:
                matches.append((chunk, chunk_lower, ExtractionMethod.NOUN_CHUNK))
                seen.add(chunk_lower)
                continue

            # Check 2: A multi-word skill is contained within the chunk
            # Example: chunk="advanced machine learning pipeline"
            #          skill="machine learning" → substring match
            for multi_skill in MULTI_WORD_SKILLS:
                if multi_skill in chunk_lower and multi_skill not in seen:
                    matches.append((chunk, multi_skill, ExtractionMethod.NOUN_CHUNK))
                    seen.add(multi_skill)

            # Check 3: Individual taxonomy skills appear as words in chunk
            chunk_words = set(chunk_lower.split())
            for word in chunk_words:
                if word in SINGLE_WORD_SKILLS and word not in seen:
                    matches.append((chunk, word, ExtractionMethod.NOUN_CHUNK))
                    seen.add(word)

        return matches

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY 6: FUZZY MATCHING
    # ─────────────────────────────────────────────────────────────────

    def _fuzzy_match(
        self,
        unmatched_tokens: List[str]
    ) -> List[Tuple[str, str, ExtractionMethod]]:
        """
        Finds approximate skill matches for tokens not caught by other strategies.
        Uses Levenshtein edit distance to handle typos and minor variants.

        Levenshtein distance algorithm:
        - Build a matrix where dp[i][j] = min edits to transform s1[:i] to s2[:j]
        - Operations: insert (+1), delete (+1), substitute (+1 if chars differ)
        - Result: dp[len(s1)][len(s2)]

        Example:
            "Kubernets" vs "kubernetes"
            Edits: insert 'e' after 'n' → distance = 1 → MATCH

        We only fuzzy-match tokens that are:
        - At least MIN_FUZZY_LENGTH characters (avoids false positives on short words)
        - Not already matched by other strategies

        Performance note: O(m×n) per comparison where m,n = string lengths.
        For 100 unmatched tokens × 400 skills = 40,000 comparisons.
        At ~1μs each = ~40ms. Acceptable for per-request processing.

        Args:
            unmatched_tokens: Tokens not matched by strategies 1-5

        Returns:
            List of (surface_form, canonical_name, ExtractionMethod) tuples
        """
        matches = []
        seen_canonicals: Set[str] = set()

        for token in unmatched_tokens:
            # Skip short tokens — too many false positives
            if len(token) < self.MIN_FUZZY_LENGTH:
                continue

            # Skip tokens that look like numbers or codes
            if not any(c.isalpha() for c in token):
                continue

            # Find the closest skill in taxonomy by edit distance
            best_match, best_distance = self._find_closest_skill(token)

            # Only accept matches within our distance threshold
            if best_match and best_distance <= self.MAX_EDIT_DISTANCE:
                # Avoid duplicate canonical entries
                if best_match not in seen_canonicals:
                    matches.append((
                        token,
                        best_match,
                        ExtractionMethod.FUZZY_MATCH
                    ))
                    seen_canonicals.add(best_match)

        return matches

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Computes Levenshtein edit distance between two strings.

        Dynamic programming implementation:
        - Create matrix of size (len(s1)+1) × (len(s2)+1)
        - dp[i][j] = minimum edits to transform s1[:i] into s2[:j]
        - Base cases: dp[i][0] = i (delete i chars), dp[0][j] = j (insert j chars)
        - Transition: if chars equal → dp[i-1][j-1], else 1 + min of 3 operations

        Time complexity:  O(m×n) where m,n = string lengths
        Space complexity: O(m×n) — can be optimized to O(min(m,n)) with 2 rows

        Args:
            s1: First string
            s2: Second string

        Returns:
            Integer edit distance (0 = identical, higher = more different)
        """
        m, n = len(s1), len(s2)

        # Create the DP matrix: (m+1) rows × (n+1) columns
        # dp[i][j] will hold the edit distance for s1[:i] vs s2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base case: transforming empty string requires i insertions
        for i in range(m + 1):
            dp[i][0] = i

        # Base case: transforming to empty string requires j deletions
        for j in range(n + 1):
            dp[0][j] = j

        # Fill the DP table row by row
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    # Characters match: no edit needed
                    # Cost = same as diagonal (s1[:i-1] vs s2[:j-1])
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Characters differ: take minimum of 3 operations
                    # dp[i-1][j]   + 1 = delete from s1 (move up)
                    # dp[i][j-1]   + 1 = insert into s1 (move left)
                    # dp[i-1][j-1] + 1 = substitute character (diagonal)
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],     # deletion
                        dp[i][j - 1],     # insertion
                        dp[i - 1][j - 1]  # substitution
                    )

        # Bottom-right cell = edit distance for complete strings
        return dp[m][n]

    def _find_closest_skill(
        self,
        token: str
    ) -> Tuple[Optional[str], int]:
        """
        Finds the closest skill in the taxonomy to a given token.

        Optimization: Only compare against skills of similar length.
        A word of length 5 cannot have edit distance ≤ 2 from a word of length 10.
        Specifically: |len(s1) - len(s2)| > MAX_EDIT_DISTANCE → skip comparison.

        This reduces comparisons by ~60% for typical token distributions.

        Args:
            token: The unmatched token to find a match for

        Returns:
            Tuple of (best_matching_skill, minimum_edit_distance)
            Returns (None, 999) if no close match found
        """
        best_match = None
        best_distance = 999  # Initialize to impossibly high value

        token_len = len(token)

        # Only compare against single-word skills (multi-word handled by phrases)
        for skill in SINGLE_WORD_SKILLS:
            # Length pruning: skip impossible matches early
            # If length difference alone exceeds threshold, skip
            if abs(token_len - len(skill)) > self.MAX_EDIT_DISTANCE:
                continue

            # Compute actual edit distance
            distance = self._levenshtein_distance(token, skill)

            # Track best (lowest distance) match
            if distance < best_distance:
                best_distance = distance
                best_match = skill

        return best_match, best_distance

    # ─────────────────────────────────────────────────────────────────
    # AGGREGATION AND SCORING
    # ─────────────────────────────────────────────────────────────────

    def _aggregate_matches(
        self,
        raw_matches: List[Tuple[str, str, ExtractionMethod]],
        text: str
    ) -> Dict[str, Dict]:
        """
        Deduplicates matches, counts frequencies, and resolves conflicts.

        When the same canonical skill is found by multiple strategies:
        - Keep the highest-confidence extraction method
        - Sum the frequency counts from all sources
        - Store all surface forms seen

        Also computes frequency directly from the cleaned text for
        verification and more accurate counting.

        Args:
            raw_matches: All matches from all strategies
            text: Original cleaned text for frequency counting

        Returns:
            Dict mapping canonical_name → {method, frequency, surfaces, ...}
        """
        # Aggregate by canonical skill name
        # defaultdict means missing keys get the default factory value
        aggregated: Dict[str, Dict] = defaultdict(lambda: {
            "canonical": "",
            "method": None,
            "frequency": 0,
            "surfaces": set(),
            "confidence": 0.0,
        })

        for surface, canonical, method in raw_matches:
            entry = aggregated[canonical]
            entry["canonical"] = canonical
            entry["surfaces"].add(surface)

            # Keep method with highest confidence
            # (exact token > phrase > alias > ner > chunk > fuzzy)
            method_confidence = self.CONFIDENCE_SCORES[method]
            if method_confidence > entry["confidence"]:
                entry["method"] = method
                entry["confidence"] = method_confidence

            # Increment raw frequency counter
            entry["frequency"] += 1

        # Refine frequency: count actual occurrences in full text
        # This is more accurate than counting matches across strategies
        text_lower = text.lower()
        for canonical, entry in aggregated.items():
            # Count occurrences in text using word-boundary regex
            # \b ensures we match whole words only ("java" not "javascript")
            pattern = r'\b' + re.escape(canonical) + r'\b'
            text_count = len(re.findall(pattern, text_lower))

            # Use text count if higher than aggregated strategy count
            # (strategies may have deduplicated occurrences)
            if text_count > entry["frequency"]:
                entry["frequency"] = text_count

            # Minimum frequency of 1 for any matched skill
            if entry["frequency"] == 0:
                entry["frequency"] = 1

        return dict(aggregated)

    def _build_skill_details(
        self,
        aggregated: Dict[str, Dict]
    ) -> List[ExtractedSkill]:
        """
        Converts aggregated match data into ExtractedSkill Pydantic objects.

        Computes the weighted_score for each skill:
            weighted_score = frequency × confidence × category_weight

        This score is used in Module 5's ATS Scoring to boost skills
        that appear frequently and belong to high-weight categories.

        Args:
            aggregated: Dict from _aggregate_matches

        Returns:
            List of ExtractedSkill objects, sorted by weighted_score desc
        """
        skill_details = []

        for canonical, entry in aggregated.items():
            # Skip if below confidence threshold
            if entry["confidence"] < self.MIN_CONFIDENCE:
                continue

            # Look up category from taxonomy
            category = SKILL_TO_CATEGORY.get(canonical, "unknown")
            if category == "unknown":
                continue

            # Get category weight multiplier
            category_weight = CATEGORY_WEIGHTS.get(category, 1.0)

            # Compute weighted score
            # frequency: raw occurrence count (1-10 typical range)
            # confidence: extraction reliability (0.70-1.00)
            # category_weight: taxonomy importance (1.0-1.5)
            weighted_score = (
                entry["frequency"] *
                entry["confidence"] *
                category_weight
            )

            # Best surface form: prefer the canonical form if it appeared,
            # otherwise use the most common surface form seen
            surfaces_list = list(entry["surfaces"])
            best_surface = (
                canonical if canonical in surfaces_list
                else surfaces_list[0]
            )

            skill = ExtractedSkill(
                canonical_name=canonical,
                surface_form=best_surface,
                category=category,
                extraction_method=entry["method"],
                frequency=entry["frequency"],
                confidence=round(entry["confidence"], 3),
                category_weight=category_weight,
                weighted_score=round(weighted_score, 4)
            )
            skill_details.append(skill)

        # Sort by weighted_score descending (most important skills first)
        skill_details.sort(key=lambda s: s.weighted_score, reverse=True)

        return skill_details

    def _assemble_result(
        self,
        skill_details: List[ExtractedSkill],
        start_time: float
    ) -> SkillExtractionResult:
        """
        Assembles all computed data into the final SkillExtractionResult.

        Computes derived structures:
        - skills_by_category: Dict[category → List[skill_names]]
        - skill_scores:       Dict[skill → weighted_score]
        - skill_frequency:    Dict[skill → count]
        - skills_per_category: Dict[category → count]
        - category_coverage:  Dict[category → fraction_of_taxonomy]
        - top_skills:         Top 15 by weighted_score

        Args:
            skill_details: List of ExtractedSkill objects
            start_time: Processing start time for duration calculation

        Returns:
            Complete SkillExtractionResult object
        """
        # ── Build derived structures ──────────────────────────────────

        # Group by category
        skills_by_category: Dict[str, List[str]] = defaultdict(list)
        skill_scores: Dict[str, float] = {}
        skill_frequency: Dict[str, int] = {}
        skill_confidence: Dict[str, float] = {}

        for skill in skill_details:
            skills_by_category[skill.category].append(skill.canonical_name)
            skill_scores[skill.canonical_name] = skill.weighted_score
            skill_frequency[skill.canonical_name] = skill.frequency
            skill_confidence[skill.canonical_name] = skill.confidence

        # Count skills per category
        skills_per_category = {
            cat: len(skills)
            for cat, skills in skills_by_category.items()
        }

        # Compute category coverage
        # Coverage = skills found in category / total skills in taxonomy category
        category_coverage = {}
        for cat, found_skills in skills_by_category.items():
            taxonomy_size = len(SKILL_TAXONOMY.get(cat, []))
            if taxonomy_size > 0:
                coverage = len(found_skills) / taxonomy_size
                category_coverage[cat] = round(coverage, 3)

        # All unique canonical skill names
        all_skills = [s.canonical_name for s in skill_details]

        # Top 15 skills by weighted score
        top_skills = all_skills[:15]

        # Methods used in this extraction
        methods_used = list({s.extraction_method.value for s in skill_details})

        return SkillExtractionResult(
            all_skills=all_skills,
            skill_details=skill_details,
            skills_by_category=dict(skills_by_category),
            skill_scores=skill_scores,
            skill_frequency=skill_frequency,
            skill_confidence=skill_confidence,
            total_skills_found=len(skill_details),
            skills_per_category=skills_per_category,
            category_coverage=category_coverage,
            top_skills=top_skills,
            processed_at=__import__('datetime').datetime.utcnow(),
            methods_used=methods_used,
            processing_time_seconds=round(time.time() - start_time, 3)
        )

    def compute_skill_gap(
        self,
        resume_skills: Set[str],
        job_skills: Set[str]
    ) -> Dict[str, List[str]]:
        """
        Computes the skill gap between a resume and job description.

        Gap analysis:
        - matched:    skills in both resume AND job (strengths)
        - missing:    skills in job but NOT in resume (gaps to address)
        - extra:      skills in resume but NOT in job (transferable assets)

        Used by Module 5 (ATS Scoring) and Module 9 (Dashboard) to
        display actionable skill gap visualizations to recruiters.

        Args:
            resume_skills: Set of canonical skill names from resume
            job_skills:    Set of canonical skill names from job description

        Returns:
            Dict with keys: "matched", "missing", "extra"
        """
        matched = list(resume_skills & job_skills)
        missing = list(job_skills - resume_skills)
        extra   = list(resume_skills - job_skills)

        return {
            "matched": sorted(matched),
            "missing": sorted(missing),
            "extra":   sorted(extra),
            "match_rate": round(
                len(matched) / len(job_skills) if job_skills else 0.0,
                3
            )
        }


# Single shared instance — stateless, thread-safe
skill_extractor = SkillExtractor()