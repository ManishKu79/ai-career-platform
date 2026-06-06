

# We extend spaCy's built-in English stop words with resume-specific terms
# spacy.lang.en.stop_words contains ~300+ standard English stop words
import spacy.lang.en.stop_words as spacy_stops

# Resume-specific stop words — words that appear frequently in resumes
# but carry no discriminating signal for skills or qualifications matching.
#
# Categories:
# 1. Resume boilerplate verbs    → "responsible", "duties", "worked"
# 2. Weak action words           → "helped", "assisted", "supported"
# 3. Filler phrases              → "including", "various", "multiple"
# 4. Temporal noise              → "currently", "previously", "ongoing"
# 5. Generic descriptors         → "strong", "excellent", "proven"

RESUME_SPECIFIC_STOP_WORDS = {
    # ── Resume boilerplate ──────────────────────────────────────────
    "responsible",
    "responsibilities",
    "duties",
    "duty",
    "role",
    "roles",
    "position",
    "positions",
    "job",
    "jobs",
    "work",
    "worked",
    "working",
    "workplace",

    # ── Weak verbs that add no signal ───────────────────────────────
    "helped",
    "assisted",
    "supported",
    "contributed",
    "participated",
    "involved",
    "handled",
    "performed",
    "carried",

    # ── Filler and connector words ──────────────────────────────────
    "including",
    "included",
    "includes",
    "include",
    "various",
    "multiple",
    "several",
    "many",
    "also",
    "well",
    "etc",
    "e.g",
    "i.e",
    "within",
    "across",
    "throughout",
    "overall",
    "general",
    "general",
    "basis",

    # ── Temporal and status words ────────────────────────────────────
    "currently",
    "previously",
    "formerly",
    "ongoing",
    "present",
    "current",
    "recent",
    "previous",
    "prior",
    "past",

    # ── Generic quality descriptors (noise for similarity) ──────────
    "strong",
    "excellent",
    "exceptional",
    "outstanding",
    "proven",
    "demonstrated",
    "extensive",
    "solid",
    "proficient",
    "skilled",
    "experienced",
    "knowledgeable",
    "ability",
    "skills",
    "skill",
    "expertise",
    "experience",

    # ── Education section filler ─────────────────────────────────────
    "university",
    "college",
    "school",
    "institute",
    "degree",
    "graduated",
    "graduation",
    "major",
    "minor",
    "coursework",
    "gpa",
    "honors",
    "cum",
    "laude",

    # ── Contact/header boilerplate ───────────────────────────────────
    "email",
    "phone",
    "address",
    "linkedin",
    "github",
    "portfolio",
    "website",
    "reference",
    "references",
    "available",
    "request",
}

# Combine spaCy's stop words with our resume-specific ones
# STOP_WORDS from spaCy is a frozenset — convert to set for union operation
ALL_STOP_WORDS = set(spacy_stops.STOP_WORDS) | RESUME_SPECIFIC_STOP_WORDS

# Export the combined set for use in NLP pipeline
RESUME_STOP_WORDS = ALL_STOP_WORDS
