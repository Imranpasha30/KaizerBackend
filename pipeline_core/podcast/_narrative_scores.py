# Ported from kaizer-platform@d5fd482 server/pipeline_core/narrative.py +
# server/pipeline_core/clip_boundaries.py. Changes from upstream: this file EXTRACTS only the
# pure scoring helpers promo.py reuses (_hook_score/_completion_score/_composite_score +
# _sentences_in_window from narrative.py; detect_completion + its lexicons/regexes from
# clip_boundaries.py) — the full narrative/clip_boundaries modules were not ported. Function
# bodies are verbatim except the cross-module lazy imports, which now resolve locally.
"""Pure narrative scoring helpers for the podcast promo planner.

The upstream promo planner reuses ``pipeline_core.narrative``'s pure hook /
completion / composite scorers. This tree does not carry the full narrative
engine (an ASR->Gemini clip orchestrator), so the handful of PURE functions the
podcast promo planner needs are extracted here verbatim. No I/O, no network —
everything is total over its inputs.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("pipeline_core.podcast._narrative_scores")

# ── From narrative.py ──────────────────────────────────────────────────────

# Hook window: the first N seconds of a clip are what must grab the viewer.
_HOOK_WINDOW_S: float = 3.0


def _sentences_in_window(sentences: list, start: float, end: float) -> list:
    """Return sentences whose midpoint falls within [start, end]."""
    result = []
    for s in sentences:
        s_start = getattr(s, "start", 0.0)
        s_end = getattr(s, "end", 0.0)
        mid = (s_start + s_end) / 2.0
        if start <= mid <= end:
            result.append(s)
    return result


def _hook_score(sentences: list, clip_start: float, clip_end: float) -> float:
    """Score the hook strength of the first _HOOK_WINDOW_S of the clip.

    Heuristics:
      - Has content in the first window (not silent).
      - First sentence starts with a question or exclamation (+0.2).
      - First sentence is short and punchy (≤ 10 words, +0.2).
      - First word is a strong trigger word (verb/imperative/number, +0.2).
      - First sentence has no unresolved pronoun reference (+0.1).

    Score range: [0, 1].
    """
    hook_end = clip_start + _HOOK_WINDOW_S
    hook_sentences = _sentences_in_window(sentences, clip_start, min(hook_end, clip_end))

    if not hook_sentences:
        return 0.1  # clip has content but no mapped transcript in hook window

    score = 0.3  # base score for having any hook content
    first_sent_text = (getattr(hook_sentences[0], "text", "") or "").strip()

    # Question or exclamation
    if re.search(r"[?!]", first_sent_text):
        score += 0.2

    # Short punchy sentence
    word_count = len(first_sent_text.split())
    if 0 < word_count <= 10:
        score += 0.2

    # Starts with a number, verb-like word, or common news opener
    first_word = re.split(r"\s+", first_sent_text)[0].lower() if first_sent_text else ""
    _strong_openers = {
        "breaking", "watch", "listen", "revealed", "exclusive", "alert",
        "warning", "new", "just", "now", "today", "urgent",
    }
    if first_word in _strong_openers or re.match(r"^\d", first_word):
        score += 0.2

    # No unresolved pronoun
    if not _DANGLING_PRONOUN_RE.match(first_sent_text):
        score += 0.1

    return min(1.0, score)


def _completion_score(sentences: list, clip_start: float, clip_end: float) -> float:
    """Score how cleanly the clip ends on a narrative beat.

    Uses :func:`detect_completion` on the last sentence in the clip window.

    Returns a float in [0, 1]:
      - 1.0 → last sentence is complete (≥2 heuristics pass)
      - 0.5 → only 1 heuristic passes
      - 0.2 → no sentence found in window
    """
    in_window = _sentences_in_window(sentences, clip_start, clip_end)
    if not in_window:
        return 0.2

    last_sent = in_window[-1]
    is_complete, reasons = detect_completion(last_sent)

    n_reasons = len(reasons)
    if is_complete:
        return 1.0
    elif n_reasons == 1:
        return 0.5
    else:
        return 0.2


def _composite_score(
    importance: float,
    hook: float,
    completion: float,
    mode: str,
) -> float:
    """Compute the composite score using mode-specific weights."""
    if mode == "trailer":
        # Trailers live or die on the hook
        return 0.5 * hook + 0.25 * importance + 0.25 * completion
    else:
        # Default blend for standalone, series, promo, highlight, full_narrative
        return 0.4 * importance + 0.3 * hook + 0.3 * completion


# ── From clip_boundaries.py ────────────────────────────────────────────────

# Discourse markers that indicate an incomplete thought.
_DANGLING_MARKERS: frozenset[str] = frozenset({
    "but", "and", "however", "although", "though", "yet", "so", "because",
    "since", "while", "whereas", "unless", "until", "if", "when", "that",
    "which", "who", "whom", "whose",
    # Hindi / Telugu common connectors transliterated
    "aur", "lekin", "magar", "kintu", "parantu",
})

# Shallow regex for subject+verb detection (English and transliterated Indic).
# Accepts any run of non-space chars as a "word"; looks for a pattern where at
# least two words appear, which broadly implies a subject + predicate.
_SUBJ_VERB_RE = re.compile(r"\S+\s+\S+", re.UNICODE)

# Unresolved-pronoun heuristic: sentence begins with a pronoun that suggests
# the context is elsewhere.
_DANGLING_PRONOUN_RE = re.compile(
    r"^(he|she|they|it|this|that|these|those|his|her|their|its)\b",
    re.IGNORECASE,
)


def detect_completion(sentence) -> tuple[bool, list[str]]:
    """Determine whether a sentence represents a completed thought.

    Applies four shallow heuristics:

    1. **Terminal punctuation** — text ends with ``.``, ``?``, ``!``, or ``।``.
    2. **No dangling discourse marker** — last word is not in the set of
       connectors/subordinators that imply more content follows.
    3. **Subject + verb present** — at least two whitespace-separated tokens
       (broad approximation; language-agnostic).
    4. **No unresolved pronoun** — the sentence does not begin with a pronoun
       that references an implicit antecedent (``he``, ``she``, ``they``, …).

    If at least 2 of the 4 heuristics pass, the sentence is considered complete.

    Parameters
    ----------
    sentence : object
        A sentence object with a ``text`` attribute.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_complete, reasons)`` where *reasons* lists the names of the
        heuristics that passed.
    """
    text: str = (getattr(sentence, "text", None) or "").strip()
    reasons: list[str] = []

    if not text:
        return False, []

    # Heuristic 1: terminal punctuation
    if re.search(r"[.?!।]\s*$", text):
        reasons.append("terminal_punctuation")

    # Heuristic 2: no dangling discourse marker at the end
    last_word = re.split(r"\s+", text.rstrip(".?!। ").strip())[-1].lower()
    if last_word not in _DANGLING_MARKERS:
        reasons.append("no_dangling_marker")

    # Heuristic 3: has at least subject + verb (two+ tokens)
    if _SUBJ_VERB_RE.search(text):
        reasons.append("has_subject_verb")

    # Heuristic 4: no unresolved leading pronoun
    first_word_match = re.match(r"\s*(\S+)", text)
    if first_word_match:
        first_word = first_word_match.group(1).rstrip(".?!।")
        if not _DANGLING_PRONOUN_RE.match(first_word):
            reasons.append("no_unresolved_pronoun")

    is_complete = len(reasons) >= 2

    logger.debug(
        "detect_completion: %r → complete=%s reasons=%s",
        text[:60], is_complete, reasons,
    )
    return is_complete, reasons


__all__ = [
    "_HOOK_WINDOW_S",
    "_sentences_in_window",
    "_hook_score",
    "_completion_score",
    "_composite_score",
    "detect_completion",
]
