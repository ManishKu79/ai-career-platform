from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class NamedEntity(BaseModel):
    """
    Represents a single named entity extracted by spaCy's NER model.

    Example:
        text="Google", label="ORG", start=45, end=51
    """

    # The surface text of the entity as it appears in the resume
    text: str

    # spaCy entity label: ORG, PERSON, GPE, DATE, CARDINAL, etc.
    label: str

    # Character offset where entity starts in the original text
    # Useful for highlighting in UI
    start: int

    # Character offset where entity ends
    end: int


class SentenceFeatures(BaseModel):
    """
    Features extracted at the sentence level.
    Sentence-level analysis captures context that word-level misses.
    Example: 'Led a team of 10 engineers' signals leadership at sentence level.
    """

    # The sentence text
    text: str

    # Number of tokens in sentence
    token_count: int

    # Does this sentence contain a strong action verb?
    # Signals achievement-oriented language
    has_action_verb: bool

    # Does this sentence contain a number/quantity?
    # "managed 15 engineers", "reduced latency by 40%"
    has_quantifier: bool


class NLPFeatures(BaseModel):
    """
    Complete NLP feature set extracted from one resume.

    This is the central output of Module 3 and the primary input
    to Module 4 (Skill Extraction) and Module 5 (ATS Scoring).

    Stored in MongoDB under resumes.nlp_features
    """

    # ── Token-level features ─────────────────────────────────────────

    # All tokens after stop word removal and lemmatization
    # Example: ["python", "develop", "api", "design", "database"]
    tokens: List[str] = []

    # Unique vocabulary of the resume
    # len(vocabulary) measures breadth of technical language
    vocabulary: List[str] = []

    # Lemmatized tokens: past tense → base form
    # "managed" → "manage", "developed" → "develop"
    lemmas: List[str] = []

    # ── Grammatical features ─────────────────────────────────────────

    # (token, POS_tag) pairs for grammatical analysis
    # Example: [("Python", "PROPN"), ("developer", "NOUN")]
    pos_tags: List[Tuple[str, str]] = []

    # Words tagged as NOUN or PROPN — strongest signal for skills
    nouns: List[str] = []

    # Words tagged as VERB — signal for action-oriented language
    verbs: List[str] = []

    # Words tagged as ADJ — signal for descriptive language
    adjectives: List[str] = []

    # ── Phrase-level features ────────────────────────────────────────

    # spaCy noun chunks: "machine learning", "cloud infrastructure"
    # These are syntactically identified compound noun phrases
    noun_chunks: List[str] = []

    # Sequential 2-word combinations from cleaned tokens
    # Critical for compound skill detection
    bigrams: List[str] = []

    # Sequential 3-word combinations from cleaned tokens
    trigrams: List[str] = []

    # ── Named Entity features ────────────────────────────────────────

    # All named entities from spaCy NER
    entities: List[NamedEntity] = []

    # Only ORG entities: companies, universities, tools (often labeled ORG)
    organizations: List[str] = []

    # Only DATE entities: employment periods, graduation years
    dates: List[str] = []

    # Only CARDINAL/QUANTITY: years of experience, team sizes
    quantities: List[str] = []

    # ── Sentence-level features ──────────────────────────────────────

    # Individual sentence analysis
    sentences: List[SentenceFeatures] = []

    # Sentences containing strong action verbs
    achievement_sentences: List[str] = []

    # ── Statistical features ─────────────────────────────────────────

    # Top N most frequent content words by occurrence
    # Dict maps word → count: {"python": 8, "api": 5}
    word_frequency: Dict[str, int] = {}

    # Total token count after stop word removal
    total_tokens: int = 0

    # Unique token count (vocabulary size)
    unique_tokens: int = 0

    # Lexical diversity: unique_tokens / total_tokens
    # Higher = more varied vocabulary
    lexical_diversity: float = 0.0

    # ── Processing metadata ──────────────────────────────────────────

    # ISO timestamp when NLP was run
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    # spaCy model version used (for reproducibility)
    model_used: str = ""

    # Processing time in seconds
    processing_time_seconds: float = 0.0
