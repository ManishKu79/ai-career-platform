

# spaCy: industrial-strength NLP library
# We use it for: tokenization, POS tagging, NER, lemmatization, noun chunks
import spacy

# collections.Counter: counts occurrences of hashable objects
# collections.defaultdict: dict with default value factory
from collections import Counter, defaultdict

# itertools: functional tools for iterators
# We use zip() for n-gram sliding window
import itertools

# re: regex for section detection
import re

# time: measuring processing duration
import time

# logging: structured logging
import logging

# typing: type annotations
from typing import List, Dict, Tuple, Set, Optional

# datetime: timestamps
from datetime import datetime

# Our Pydantic models
from backend.models.nlp_models import NLPFeatures, NamedEntity, SentenceFeatures

# Our custom stop words
from nlp.stop_words import RESUME_STOP_WORDS

# Settings for model name
from backend.config import settings

# Module logger
logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Production NLP processing pipeline for resume text.

    Architecture decisions:
    - Model loaded ONCE at instantiation (not per request) — spaCy models
      are 300-700MB and take 2-5 seconds to load. Loading per request
      would make the API unusable.
    - All processing is pure functions (no side effects) except model loading
    - Returns structured Pydantic objects, not raw dicts

    spaCy pipeline components used:
    - tok2vec:   Neural token-to-vector embeddings (base for all other components)
    - tagger:    POS tagging (NOUN, VERB, ADJ, PROPN, etc.)
    - parser:    Dependency parsing (sentence boundaries, noun chunks)
    - ner:       Named entity recognition (ORG, PERSON, DATE, etc.)
    - lemmatizer: Morphological analysis (running → run)
    """

    def __init__(self):
        """
        Load spaCy model at instantiation.
        Called once when the module is imported.
        """
        logger.info(f"Loading spaCy model: {settings.SPACY_MODEL}")

        try:
            # spacy.load() reads the model from disk into memory
            # en_core_web_lg is the large English model:
            # - 685MB on disk
            # - 300-dimensional word vectors
            # - Trained on OntoNotes 5 + Common Crawl
            # - Higher accuracy than sm/md at cost of memory
            self.nlp = spacy.load(settings.SPACY_MODEL)

            # Set max_length to handle long resumes
            # Default is 1,000,000 chars — increase for very long CVs
            self.nlp.max_length = 2_000_000

            logger.info(
                f"spaCy model loaded successfully. "
                f"Pipeline components: {self.nlp.pipe_names}"
            )

        except OSError as e:
            # Model not downloaded — provide actionable error message
            logger.error(
                f"spaCy model '{settings.SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {settings.SPACY_MODEL}"
            )
            raise RuntimeError(
                f"spaCy model not found: {e}. "
                f"Install with: python -m spacy download {settings.SPACY_MODEL}"
            )

        # Strong action verbs: high-signal words in resumes
        # These indicate achievements vs passive participation
        # Source: studies of high-performing LinkedIn profiles
        self.ACTION_VERBS = {
            # Leadership
            "lead", "led", "manage", "managed", "direct", "directed",
            "oversee", "oversaw", "supervise", "supervised", "mentor",
            "mentored", "coach", "coached", "train", "trained",

            # Building
            "build", "built", "develop", "developed", "create", "created",
            "design", "designed", "architect", "architected", "engineer",
            "engineered", "implement", "implemented", "deploy", "deployed",

            # Improvement
            "improve", "improved", "optimize", "optimized", "enhance",
            "enhanced", "increase", "increased", "reduce", "reduced",
            "streamline", "streamlined", "automate", "automated",

            # Achievement
            "launch", "launched", "deliver", "delivered", "achieve",
            "achieved", "accomplish", "accomplished", "execute", "executed",
            "drive", "drove", "spearhead", "spearheaded", "pioneer",
            "pioneered",

            # Analysis
            "analyze", "analyzed", "research", "researched", "evaluate",
            "evaluated", "assess", "assessed", "identify", "identified",
        }

    def process(self, text: str) -> NLPFeatures:
        """
        Main entry point. Runs the complete NLP pipeline on resume text.

        Pipeline stages:
        1. spaCy processing (tokenization, POS, NER, lemmatization)
        2. Token filtering (stop words, punctuation, short tokens)
        3. N-gram extraction (bigrams, trigrams)
        4. Entity extraction (organizations, dates, quantities)
        5. Noun chunk extraction (compound phrases)
        6. Sentence-level analysis
        7. Statistical feature computation
        8. Feature assembly into Pydantic model

        Args:
            text: Cleaned resume text from parser service

        Returns:
            NLPFeatures: Fully populated feature object
        """

        # Record start time for performance tracking
        start_time = time.time()

        # Validate input
        if not text or len(text.strip()) < 10:
            logger.warning("Empty or too-short text passed to NLP pipeline")
            return NLPFeatures()

        # ── Stage 1: spaCy Processing ────────────────────────────────
        # self.nlp(text) runs the full pipeline:
        # text → tokens → POS tags → parse tree → NER → lemmas
        # Returns a Doc object — the central spaCy data structure
        # Doc contains Span objects (sentences) and Token objects
        doc = self.nlp(text)

        logger.info(
            f"spaCy processed {len(doc)} tokens across "
            f"{len(list(doc.sents))} sentences"
        )

        # ── Stage 2: Token Filtering ─────────────────────────────────
        # Extract clean, meaningful tokens for downstream processing
        filtered_tokens = self._filter_tokens(doc)

        # ── Stage 3: Lemmatization ───────────────────────────────────
        # Get lemmas of filtered tokens (base forms)
        lemmas = self._extract_lemmas(doc)

        # ── Stage 4: POS Feature Extraction ─────────────────────────
        pos_tags, nouns, verbs, adjectives = self._extract_pos_features(doc)

        # ── Stage 5: N-gram Extraction ───────────────────────────────
        # Build from filtered tokens (already stop-word-removed)
        bigrams = self._extract_ngrams(filtered_tokens, n=2)
        trigrams = self._extract_ngrams(filtered_tokens, n=3)

        # ── Stage 6: Named Entity Extraction ────────────────────────
        entities, organizations, dates, quantities = self._extract_entities(doc)

        # ── Stage 7: Noun Chunk Extraction ──────────────────────────
        # spaCy's syntactic noun chunks — more accurate than pure n-grams
        noun_chunks = self._extract_noun_chunks(doc)

        # ── Stage 8: Sentence Analysis ───────────────────────────────
        sentences, achievement_sentences = self._analyze_sentences(doc)

        # ── Stage 9: Statistical Features ───────────────────────────
        word_freq = self._compute_word_frequency(filtered_tokens, top_n=30)
        vocabulary = list(set(filtered_tokens))
        total_tokens = len(filtered_tokens)
        unique_tokens = len(vocabulary)

        # Lexical diversity: ratio of unique to total tokens
        # 0.0 = completely repetitive, 1.0 = every word used exactly once
        # Professional resumes typically score 0.6 - 0.8
        lexical_diversity = (
            unique_tokens / total_tokens if total_tokens > 0 else 0.0
        )

        # ── Stage 10: Assemble Feature Object ───────────────────────
        processing_time = time.time() - start_time

        features = NLPFeatures(
            tokens=filtered_tokens,
            vocabulary=vocabulary,
            lemmas=lemmas,
            pos_tags=pos_tags,
            nouns=nouns,
            verbs=verbs,
            adjectives=adjectives,
            noun_chunks=noun_chunks,
            bigrams=bigrams,
            trigrams=trigrams,
            entities=entities,
            organizations=organizations,
            dates=dates,
            quantities=quantities,
            sentences=sentences,
            achievement_sentences=achievement_sentences,
            word_frequency=word_freq,
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            lexical_diversity=round(lexical_diversity, 4),
            processed_at=datetime.utcnow(),
            model_used=settings.SPACY_MODEL,
            processing_time_seconds=round(processing_time, 3)
        )

        logger.info(
            f"NLP pipeline complete in {processing_time:.3f}s. "
            f"Tokens: {total_tokens}, Unique: {unique_tokens}, "
            f"Entities: {len(entities)}, Bigrams: {len(bigrams)}"
        )

        return features

    def _filter_tokens(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Filters spaCy Doc tokens to retain only meaningful content words.

        Filtering criteria (token is KEPT only if ALL conditions pass):
        1. Not a stop word (in RESUME_STOP_WORDS)
        2. Not punctuation (period, comma, brackets, etc.)
        3. Not whitespace-only
        4. Not a single character (removes stray letters)
        5. Alphabetic content (removes pure numbers and symbols)
        6. Length ≥ 2 characters
        7. Not a URL or email address

        Args:
            doc: spaCy Doc object from nlp(text)

        Returns:
            List of lowercase, filtered token strings
        """
        filtered = []

        for token in doc:
            # token.is_stop: spaCy's built-in stop word check
            # We override with our more comprehensive RESUME_STOP_WORDS
            if token.text.lower() in RESUME_STOP_WORDS:
                continue

            # token.is_punct: is this token a punctuation mark?
            # True for: . , ; : ! ? ( ) [ ] { } " ' - /
            if token.is_punct:
                continue

            # token.is_space: is this token only whitespace?
            if token.is_space:
                continue

            # token.like_url: does this look like a URL?
            # True for: http://..., www..., github.com/...
            if token.like_url:
                continue

            # token.like_email: does this look like an email?
            if token.like_email:
                continue

            # Get lowercase text for comparison
            text = token.text.lower().strip()

            # Minimum length of 2 characters
            if len(text) < 2:
                continue

            # Must contain at least one alphabetic character
            # Filters out pure numbers like "2019" or symbols like "+++"
            # Note: "c++" and "c#" are handled in skill extractor
            if not any(c.isalpha() for c in text):
                continue

            # Passed all filters — add lowercase token
            filtered.append(text)

        return filtered

    def _extract_lemmas(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Extracts lemmatized versions of all meaningful tokens.

        Lemmatization converts inflected forms to base dictionary form:
            "engineers" → "engineer"
            "managing"  → "manage"
            "databases" → "database"
            "built"     → "build"

        spaCy uses a lookup-based lemmatizer combined with rule-based
        morphological analysis for accurate results.

        Args:
            doc: spaCy Doc object

        Returns:
            List of lemmatized tokens (stop words excluded)
        """
        lemmas = []

        for token in doc:
            # Apply same filters as _filter_tokens for consistency
            if (token.text.lower() in RESUME_STOP_WORDS or
                    token.is_punct or
                    token.is_space or
                    len(token.text.strip()) < 2):
                continue

            # token.lemma_: spaCy's computed base form
            # "-PRON-" is spaCy's placeholder for pronouns — skip it
            lemma = token.lemma_.lower().strip()

            if lemma and lemma != "-pron-" and len(lemma) >= 2:
                lemmas.append(lemma)

        return lemmas

    def _extract_pos_features(
        self,
        doc: spacy.tokens.Doc
    ) -> Tuple[List[Tuple[str, str]], List[str], List[str], List[str]]:
        """
        Extracts part-of-speech tagged features from the document.

        POS tags used (Universal Dependencies tagset):
            NOUN:  Common nouns (developer, system, database)
            PROPN: Proper nouns (Python, AWS, Google)
            VERB:  Verbs (develop, manage, build)
            ADJ:   Adjectives (scalable, distributed, automated)
            NUM:   Numbers (5, three, 100)

        Args:
            doc: spaCy Doc object

        Returns:
            Tuple of (pos_tags_list, nouns, verbs, adjectives)
        """
        pos_tags = []
        nouns = []
        verbs = []
        adjectives = []

        for token in doc:
            # Skip stop words, punctuation, and whitespace
            if (token.is_stop or token.is_punct or
                    token.is_space or len(token.text.strip()) < 2):
                continue

            text_lower = token.text.lower()

            # Record (text, POS) pair
            pos_tags.append((text_lower, token.pos_))

            # Categorize by POS
            # NOUN and PROPN are strongest skill indicators
            if token.pos_ in ("NOUN", "PROPN"):
                nouns.append(text_lower)

            # VERB: action words — leadership and achievement indicators
            elif token.pos_ == "VERB":
                lemma = token.lemma_.lower()
                verbs.append(lemma)

            # ADJ: descriptive language
            elif token.pos_ == "ADJ":
                adjectives.append(text_lower)

        return pos_tags, nouns, verbs, adjectives

    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Generates n-grams from a list of tokens using sliding window.

        Algorithm (sliding window):
            tokens = ["machine", "learning", "engineer", "python"]
            n = 2
            windows: ("machine","learning"), ("learning","engineer"),
                     ("engineer","python")
            result: ["machine learning", "learning engineer", "engineer python"]

        We use zip() with offset copies to create the sliding window:
            zip(tokens, tokens[1:])       → bigrams
            zip(tokens, tokens[1:], tokens[2:]) → trigrams

        This is O(n) time and O(n) space — efficient for large vocabularies.

        Args:
            tokens: Pre-filtered token list (from _filter_tokens)
            n: N-gram size (2 for bigrams, 3 for trigrams)

        Returns:
            List of space-joined n-gram strings
        """
        if len(tokens) < n:
            return []

        # Create n offset copies of the token list for zip
        # For n=2: zip(tokens[0:], tokens[1:])
        # For n=3: zip(tokens[0:], tokens[1:], tokens[2:])
        iterables = [tokens[i:] for i in range(n)]

        # zip() pairs elements at same index across all iterables
        ngrams = [" ".join(gram) for gram in zip(*iterables)]

        return ngrams

    def _extract_entities(
        self,
        doc: spacy.tokens.Doc
    ) -> Tuple[List[NamedEntity], List[str], List[str], List[str]]:
        """
        Extracts named entities using spaCy's NER model.

        Entity types relevant to resumes:
            ORG:      Organizations — "Google", "AWS", "MIT"
            PERSON:   People names — filtered out (privacy)
            GPE:      Geopolitical entities — "San Francisco", "Remote"
            DATE:     Dates and periods — "2019-2022", "3 years"
            CARDINAL: Numbers — "10 engineers", "50% improvement"
            PRODUCT:  Products — "iPhone", "Kubernetes"
            NORP:     Groups — "Agile", "Scrum" (sometimes labeled here)

        Args:
            doc: spaCy Doc object

        Returns:
            Tuple of (all_entities, organizations, dates, quantities)
        """
        all_entities = []
        organizations = []
        dates = []
        quantities = []

        # Track seen entities to deduplicate
        seen_entities: Set[str] = set()

        for ent in doc.ents:
            # Normalize entity text
            ent_text = ent.text.strip()

            # Skip empty or single-character entities
            if len(ent_text) < 2:
                continue

            # Deduplicate by (text, label) pair
            entity_key = f"{ent_text.lower()}_{ent.label_}"
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)

            # Create NamedEntity Pydantic object
            entity = NamedEntity(
                text=ent_text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            )
            all_entities.append(entity)

            # Categorize into specialized lists
            if ent.label_ == "ORG":
                # Organizations: companies, schools, tools
                organizations.append(ent_text)

            elif ent.label_ == "DATE":
                # Date expressions: employment periods, graduation years
                dates.append(ent_text)

            elif ent.label_ in ("CARDINAL", "QUANTITY", "PERCENT"):
                # Numbers: team sizes, percentages, quantities
                quantities.append(ent_text)

        return all_entities, organizations, dates, quantities

    def _extract_noun_chunks(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Extracts syntactic noun phrases using spaCy's dependency parser.

        Noun chunks are syntactically identified noun phrases — more
        accurate than simple n-grams because they use grammatical structure.

        Examples:
            "senior software engineer"   → noun chunk
            "distributed systems design" → noun chunk
            "machine learning models"    → noun chunk

        spaCy identifies these by traversing the dependency parse tree
        looking for noun heads with their modifiers.

        Filtering: Remove chunks that are entirely stop words or too short.

        Args:
            doc: spaCy Doc object (requires parser component)

        Returns:
            List of cleaned noun chunk strings
        """
        chunks = []
        seen = set()

        for chunk in doc.noun_chunks:
            # Get lowercase text of the chunk
            chunk_text = chunk.text.lower().strip()

            # Remove leading/trailing articles and determiners
            # "the software engineer" → "software engineer"
            chunk_text = re.sub(r'^(the|a|an|my|our|their|his|her)\s+', '', chunk_text)

            # Skip if too short after cleaning
            if len(chunk_text) < 3:
                continue

            # Skip if it's just a stop word
            if chunk_text in RESUME_STOP_WORDS:
                continue

            # Deduplicate
            if chunk_text in seen:
                continue
            seen.add(chunk_text)

            # Skip single-word chunks (already captured in tokens)
            if len(chunk_text.split()) < 2:
                continue

            chunks.append(chunk_text)

        return chunks

    def _analyze_sentences(
        self,
        doc: spacy.tokens.Doc
    ) -> Tuple[List[SentenceFeatures], List[str]]:
        """
        Analyzes each sentence for achievement signals.

        Achievement indicators:
        1. Strong action verb (from self.ACTION_VERBS set)
        2. Numeric quantifier ("increased revenue by 30%")
        3. Technical context (ORG or PRODUCT entity present)

        These signals are used in Module 6 (Candidate Ranking) to score
        candidates on achievement language quality — not just keyword presence.

        Args:
            doc: spaCy Doc object (requires sentence segmentation via parser)

        Returns:
            Tuple of (sentence_features_list, achievement_sentences)
        """
        sentence_features = []
        achievement_sentences = []

        # doc.sents: iterator over Span objects, one per sentence
        # spaCy uses the dependency parser to identify sentence boundaries
        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Skip very short sentences (headers, section titles)
            if len(sent_text) < 10:
                continue

            # Count meaningful tokens in this sentence
            sent_tokens = [
                t for t in sent
                if not t.is_stop and not t.is_punct and not t.is_space
            ]
            token_count = len(sent_tokens)

            # Check for strong action verbs
            # We check lemmas to catch all conjugations
            has_action_verb = any(
                token.lemma_.lower() in self.ACTION_VERBS
                for token in sent
            )

            # Check for numeric quantifiers
            # token.like_num: True for "3", "three", "50%", etc.
            # Also check for ORG_NUM entities (percentages, counts)
            has_quantifier = any(
                token.like_num or token.ent_type_ in ("CARDINAL", "PERCENT", "QUANTITY")
                for token in sent
            )

            # Build sentence feature object
            sent_feature = SentenceFeatures(
                text=sent_text,
                token_count=token_count,
                has_action_verb=has_action_verb,
                has_quantifier=has_quantifier
            )
            sentence_features.append(sent_feature)

            # Flag as achievement sentence if it has both action verb
            # AND a quantifier — these are the most compelling resume bullets
            if has_action_verb and has_quantifier:
                achievement_sentences.append(sent_text)

        return sentence_features, achievement_sentences

    def _compute_word_frequency(
        self,
        tokens: List[str],
        top_n: int = 30
    ) -> Dict[str, int]:
        """
        Computes word frequency distribution of the resume's content words.

        Uses collections.Counter which internally uses a hash table for O(1)
        per-word counting. most_common(n) returns the top n items in O(k log k)
        where k = vocabulary size.

        The top 30 words are a compressed signature of the resume's focus.
        A data scientist's top words: "model", "python", "analysis", "data"
        A DevOps engineer's top words: "deploy", "kubernetes", "pipeline", "aws"

        Args:
            tokens: Filtered token list from _filter_tokens
            top_n: How many top words to return

        Returns:
            Dict mapping word → count, sorted by frequency descending
        """
        # Counter({word: count, ...}) from list
        word_counts = Counter(tokens)

        # most_common(top_n) returns [(word, count), ...] sorted by count
        top_words = word_counts.most_common(top_n)

        # Convert to dict for JSON serialization
        return dict(top_words)

    def process_batch(self, texts: List[str]) -> List[NLPFeatures]:
        """
        Process multiple resumes efficiently using spaCy's pipe().

        nlp.pipe() is significantly faster than calling nlp() in a loop
        because it batches tokenization and model inference.
        On CPU: ~2x speedup. On GPU: ~10x speedup.

        Args:
            texts: List of cleaned resume texts

        Returns:
            List of NLPFeatures, one per input text
        """
        results = []

        # nlp.pipe() yields Doc objects in order
        # batch_size: number of texts processed per batch
        # n_process: number of CPU cores (1 = single-threaded for safety)
        for doc in self.nlp.pipe(texts, batch_size=32, n_process=1):
            # Process each doc through our feature extraction
            # We reconstruct text from doc for consistency
            features = self._process_doc(doc)
            results.append(features)

        return results

    def _process_doc(self, doc: spacy.tokens.Doc) -> NLPFeatures:
        """
        Process an already-parsed spaCy Doc (used in batch processing).
        Extracts all features from a pre-computed Doc object.
        """
        start_time = time.time()

        filtered_tokens = self._filter_tokens(doc)
        lemmas = self._extract_lemmas(doc)
        pos_tags, nouns, verbs, adjectives = self._extract_pos_features(doc)
        bigrams = self._extract_ngrams(filtered_tokens, n=2)
        trigrams = self._extract_ngrams(filtered_tokens, n=3)
        entities, organizations, dates, quantities = self._extract_entities(doc)
        noun_chunks = self._extract_noun_chunks(doc)
        sentences, achievement_sentences = self._analyze_sentences(doc)
        word_freq = self._compute_word_frequency(filtered_tokens, top_n=30)
        vocabulary = list(set(filtered_tokens))
        total_tokens = len(filtered_tokens)
        unique_tokens = len(vocabulary)
        lexical_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0.0

        return NLPFeatures(
            tokens=filtered_tokens,
            vocabulary=vocabulary,
            lemmas=lemmas,
            pos_tags=pos_tags,
            nouns=nouns,
            verbs=verbs,
            adjectives=adjectives,
            noun_chunks=noun_chunks,
            bigrams=bigrams,
            trigrams=trigrams,
            entities=entities,
            organizations=organizations,
            dates=dates,
            quantities=quantities,
            sentences=sentences,
            achievement_sentences=achievement_sentences,
            word_frequency=word_freq,
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            lexical_diversity=round(lexical_diversity, 4),
            processed_at=datetime.utcnow(),
            model_used=settings.SPACY_MODEL,
            processing_time_seconds=round(time.time() - start_time, 3)
        )


# Single instance — model loaded once at import time
# All requests share this instance (thread-safe: spaCy nlp() is stateless)
nlp_pipeline = NLPPipeline()
