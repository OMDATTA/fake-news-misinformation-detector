"""Text normalisation, tokenisation and handcrafted linguistic features.

Everything in this module is mirrored 1:1 in ``web/assets/js/model.js`` so the
browser produces the same feature vector as the Python training pipeline.
``tests/test_parity.py`` checks that the two implementations agree.

Keep the two files in sync: if you change a regex or a lexicon here, change it
in the JavaScript as well and re-run the parity test.
"""
from __future__ import annotations

import math
import re
import unicodedata

# ---------------------------------------------------------------------------
# 1. Source-artifact scrubbing
# ---------------------------------------------------------------------------
# The ISOT corpus has strong "shortcut" signals: reliable articles come from
# Reuters and carry datelines / the word "Reuters"; many misleading articles
# carry "Featured image via ..." credits, embedded tweets, or site names.
# A model that learns those markers is detecting *publishers*, not *language*.
# We remove them before any feature extraction. See scripts/02_source_topic_analysis.py
# for how much accuracy these artifacts alone would give.

_SCRUB_PATTERNS_I = [  # case-insensitive
    r"https?://\S+",
    r"www\.\S+",
    r"pic\.twitter\.com/\S+",
    r"@\w+",
    r"\(reuters\)",
    r"\breuters\b",
    r"21st century wire",
    r"featured image[^\n.]*",
    r"photo by[^\n.]*",
    r"image via[^\n.]*",
    r"getty images",
    r"\bvia\s+(?:twitter|youtube|facebook)\b",
    r"[\[(](?:video|videos|image|images|tweet|tweets|watch|photo|photos)[\])]",
    r"\bbit\.ly/?\S*",
    r"\b\d{1,4}\s?(?:edt|est|gmt|bst|cet)\b",
    r"\bfactbox\b",
    r"\bread more\b:?",
    r"\bvia\b",
]
_SCRUB_RE_I = [re.compile(p, re.IGNORECASE) for p in _SCRUB_PATTERNS_I]
# Upper-case wire dateline at the start, e.g. "WASHINGTON -" or "BEIJING/SEOUL -".
_DATELINE_RE = re.compile(r"^\s*[A-Z][A-Z .,/'-]{2,40}\s+-\s+")
_WS_RE = re.compile(r"\s+")


def ascii_fold(text: str) -> str:
    """Unify quotes/apostrophes and drop non-ASCII characters."""
    text = (
        text.replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
    )
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def scrub(text: str) -> str:
    """Remove publisher/source artifacts so the model focuses on language."""
    text = ascii_fold(text or "")
    text = _DATELINE_RE.sub(" ", text)
    for rx in _SCRUB_RE_I:
        text = rx.sub(" ", text)
    # apostrophes are inconsistently stripped in the corpus ("Trump s"); unify.
    text = text.replace("'", " ")
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# 2. Tokenisation for TF-IDF (must match JS exactly)
# ---------------------------------------------------------------------------
_DIGIT_RE = re.compile(r"[0-9]")
TOKEN_PATTERN = r"[a-z0-9_]{2,}"
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def for_tfidf(clean_text: str) -> str:
    """Lower-case and map every digit to 0 (so 2016 and 2017 share a token)."""
    return _DIGIT_RE.sub("0", clean_text.lower())


def tokenize(clean_text: str) -> list[str]:
    return _TOKEN_RE.findall(for_tfidf(clean_text))


# ---------------------------------------------------------------------------
# 3. Linguistic / stylometric features
# ---------------------------------------------------------------------------
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours", "ya"}
SENSATIONAL = {
    "shocking", "shocked", "outrageous", "unbelievable", "insane", "disgusting",
    "destroy", "destroys", "destroyed", "slam", "slams", "slammed", "bombshell",
    "horrifying", "epic", "hilarious", "pathetic", "idiot", "idiots", "stupid",
    "moron", "lies", "liar", "lying", "disgrace", "disgraceful", "breaking",
    "exposed", "exposes", "busted", "meltdown", "furious", "rant", "brutal",
    "brutally", "humiliating", "humiliated", "wow", "absolutely", "totally",
    "literally", "completely", "massive", "huge", "hysterical", "crazy",
}
HEDGES = {
    "reportedly", "allegedly", "may", "might", "could", "possibly", "perhaps",
    "appears", "appeared", "suggests", "suggested", "likely", "unlikely",
    "apparently", "estimated", "approximately",
}
ATTRIBUTION = {
    "said", "says", "told", "according", "stated", "spokesman", "spokeswoman",
    "spokesperson", "statement", "announced", "confirmed", "official",
    "officials", "added",
}

LING_FEATURES = [
    ("log_words", "Article length (log words)"),
    ("avg_word_len", "Average word length"),
    ("type_token_ratio", "Vocabulary diversity"),
    ("avg_sentence_len", "Average sentence length"),
    ("exclaim_rate", "Exclamation marks / 100 words"),
    ("question_rate", "Question marks / 100 words"),
    ("caps_ratio", "ALL-CAPS word share"),
    ("quote_rate", "Quotation marks / 100 words"),
    ("first_person_rate", "First-person pronouns / 100 words"),
    ("second_person_rate", "Second-person pronouns / 100 words"),
    ("sensational_rate", "Sensational words / 100 words"),
    ("hedge_rate", "Hedging words / 100 words"),
    ("attribution_rate", "Attribution verbs / 100 words"),
    ("digit_ratio", "Numeric token share"),
    ("flesch", "Flesch reading ease"),
]
LING_NAMES = [k for k, _ in LING_FEATURES]

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_RE = re.compile(r"[.!?]+")
_VOWEL_RE = re.compile(r"[aeiouy]+")
_HAS_DIGIT = re.compile(r"[0-9]")
_HAS_ALPHA = re.compile(r"[A-Z]")


def _syllables(word: str) -> int:
    return max(1, len(_VOWEL_RE.findall(word)))


def linguistic_features(clean_text: str) -> list[float]:
    words = _WORD_RE.findall(clean_text)
    n = len(words)
    nz = max(n, 1)
    lower = [w.lower() for w in words]
    sentences = [s for s in _SENT_RE.split(clean_text) if s.strip()]
    n_sent = max(len(sentences), 1)
    head = lower[:300]
    ttr = len(set(head)) / max(len(head), 1)
    caps = sum(1 for w in words if len(w) >= 2 and w.isupper() and _HAS_ALPHA.search(w))
    syl = sum(_syllables(w) for w in lower)
    per100 = 100.0 / nz

    def rate(lex: set[str]) -> float:
        return sum(1 for w in lower if w in lex) * per100

    flesch = 206.835 - 1.015 * (n / n_sent) - 84.6 * (syl / nz)
    flesch = min(max(flesch, -50.0), 120.0)
    return [
        math.log1p(n),
        sum(len(w) for w in words) / nz,
        ttr,
        min(n / n_sent, 120.0),
        clean_text.count("!") * per100,
        clean_text.count("?") * per100,
        caps / nz,
        clean_text.count('"') * per100,
        rate(FIRST_PERSON),
        rate(SECOND_PERSON),
        rate(SENSATIONAL),
        rate(HEDGES),
        rate(ATTRIBUTION),
        sum(1 for w in words if _HAS_DIGIT.search(w)) / nz,
        flesch,
    ]
