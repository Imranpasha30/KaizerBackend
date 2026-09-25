# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/emphasis.py.
# Changes from upstream: none (verbatim copy; origin header added).
"""Caption emphasis flags — PURE, deterministic, no LLM.

Marks ``isEmphasis`` on caption words so the Remotion caption component can
give them a stronger word-pop treatment. The heuristic MIRRORS the component
side so server + component agree on which words pop:

  1. NUMBERS — a word containing a digit, or a spelled-out cardinal in the
     small English/Telugu lexicon, is always emphasised.
  2. LONGEST CONTENT WORD PER WINDOW — within each ~1.5s window, the single
     longest "content" word (>= 4 letters, not a stopword) is emphasised.

Deterministic tie-break: earliest word wins a length tie. Cheap (single pass),
no network, no model.

Input/Output: a sequence of caption dicts ``{w|word, start_s, end_s, ...}``.
Returns NEW dicts (immutable) with an added ``isEmphasis: bool`` key; original
dicts are not mutated.
"""

from __future__ import annotations

import re
from typing import Sequence

# Window size for the "longest content word" pass (seconds).
_WINDOW_S = 1.5
# Minimum letters for a word to count as an emphasis-eligible "content" word.
_MIN_CONTENT_LEN = 4

_DIGIT_RE = re.compile(r"\d")

# Small stopword set (English) — kept tiny + deterministic, not linguistic.
_STOPWORDS = frozenset(
    {
        "the", "and", "that", "this", "with", "have", "from", "they", "them",
        "then", "than", "were", "what", "when", "your", "yours", "there",
        "their", "would", "could", "should", "about", "which", "into", "just",
        "like", "been", "some", "very", "also", "only", "more", "most", "over",
    }
)

# Spelled-out cardinals treated as numbers (EN + a Telugu starter set).
_NUMBER_WORDS = frozenset(
    {
        # English
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "twenty", "thirty",
        "forty", "fifty", "hundred", "thousand", "million", "billion",
        # Telugu (transliterated + script)
        "okati", "rendu", "moodu", "nalugu", "aidu",
        "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "వంద", "వెయ్యి",
    }
)


def _word_text(cap: dict) -> str:
    return str(cap.get("w", cap.get("word", ""))).strip()


def _is_number(text: str) -> bool:
    if _DIGIT_RE.search(text):
        return True
    return text.lower().strip(".,!?;:%") in _NUMBER_WORDS


def _letters_len(text: str) -> int:
    """Length counting alphabetic characters only (ignores punctuation)."""
    return sum(1 for ch in text if ch.isalpha())


def _is_content_word(text: str) -> bool:
    stripped = text.lower().strip(".,!?;:'\"")
    return _letters_len(stripped) >= _MIN_CONTENT_LEN and stripped not in _STOPWORDS


def mark_emphasis(
    captions: Sequence[dict],
    *,
    window_s: float = _WINDOW_S,
) -> list[dict]:
    """Return new caption dicts with an ``isEmphasis`` bool added.

    A word is emphasised if it is a number, OR it is the longest content word
    in its ~``window_s`` window. Pure + deterministic.
    """
    out: list[dict] = [dict(c) for c in captions]
    # First pass: numbers always emphasised.
    for c in out:
        c["isEmphasis"] = _is_number(_word_text(c))

    if not out:
        return out

    # Second pass: longest content word per window.
    base = float(out[0].get("start_s", 0.0))
    # index -> window bucket
    buckets: dict[int, list[int]] = {}
    for i, c in enumerate(out):
        start = float(c.get("start_s", 0.0))
        b = int((start - base) // window_s)
        buckets.setdefault(b, []).append(i)

    for idxs in buckets.values():
        best_i = -1
        best_len = 0
        for i in idxs:
            text = _word_text(out[i])
            if not _is_content_word(text):
                continue
            ln = _letters_len(text)
            if ln > best_len:  # strict > => earliest wins ties
                best_len = ln
                best_i = i
        if best_i >= 0:
            out[best_i]["isEmphasis"] = True

    return out


__all__ = ["mark_emphasis"]
