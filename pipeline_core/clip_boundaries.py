"""
kaizer.pipeline.clip_boundaries
================================
Snap-to-beat boundary snapping for the Kaizer News video pipeline.

Takes a proposed (start, end) clip window and snaps each edge to the nearest
signal — shot boundary, sentence start/end, or audio RMS valley — so that clip
edges feel intentional rather than arbitrary.

Usage
-----
    from pipeline_core.clip_boundaries import snap_boundaries, detect_completion, SnapResult
    from pipeline_core.asr import Sentence

    result = snap_boundaries(
        proposed_start=12.4,
        proposed_end=57.8,
        shots=[5.0, 14.0, 30.0, 58.5],
        sentences=transcription.sentences,
        valleys=[10.2, 28.9, 57.1],
    )
    print(result.start, result.end, result.start_sources, result.end_sources)

    is_done, reasons = detect_completion(sentence)

SnapResult fields
-----------------
  start         : float       — Snapped start time (seconds).
  end           : float       — Snapped end time (seconds).
  start_sources : list[str]   — Signals that voted for the snapped start.
  end_sources   : list[str]   — Signals that voted for the snapped end.
  adjusted_s    : float       — Total seconds of adjustment from the proposal.
  warnings      : list[str]   — Non-fatal issues.

detect_completion return value
------------------------------
  (is_complete: bool, reasons: list[str])
    is_complete is True when ≥2 of the 4 heuristics pass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.clip_boundaries")


# ── Public dataclass ───────────────────────────────────────────────────────────

@dataclass
class SnapResult:
    """Result of a :func:`snap_boundaries` call.

    Attributes
    ----------
    start : float
        Snapped clip start in seconds.
    end : float
        Snapped clip end in seconds.
    start_sources : list[str]
        Which signals contributed to the snapped start
        (``"sentence"``, ``"shot"``, or ``"valley"``).
    end_sources : list[str]
        Which signals contributed to the snapped end.
    adjusted_s : float
        Total absolute adjustment in seconds compared to the proposal.
    warnings : list[str]
        Non-fatal issues encountered during snapping.
    """

    start: float
    end: float
    start_sources: list[str] = field(default_factory=list)
    end_sources: list[str] = field(default_factory=list)
    adjusted_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ── Discourse markers that indicate an incomplete thought ─────────────────────

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


# ── Public functions ───────────────────────────────────────────────────────────

def snap_boundaries(
    proposed_start: float,
    proposed_end: float,
    *,
    shots: list[float],
    sentences: Optional[list] = None,          # list[asr.Sentence]
    valleys: Optional[list[float]] = None,
    search_window_s: float = 2.0,
    min_clip_duration_s: float = 5.0,
    max_clip_duration_s: float = 179.0,
) -> SnapResult:
    """Snap proposed clip edges to the nearest natural boundary signal.

    For each edge the preference ordering is:

    **Start edge:** sentence start ≻ shot boundary ≻ RMS valley end
    **End edge:**   sentence end (with terminal punctuation) ≻ RMS valley start ≻ shot boundary

    If no signal falls within *search_window_s* of a proposed edge, that edge
    is kept as-is and a warning is added.

    Parameters
    ----------
    proposed_start, proposed_end : float
        The initially proposed clip boundaries in seconds.
    shots : list[float]
        Shot boundary timestamps (unsorted is fine; sorted internally).
    sentences : list[Sentence] | None
        Sentence objects from :mod:`pipeline_core.asr`.
    valleys : list[float] | None
        Audio RMS valley timestamps (quietest moments in the source).
    search_window_s : float
        Maximum distance in seconds for a candidate signal to be considered.
    min_clip_duration_s, max_clip_duration_s : float
        Hard duration clamps applied after snapping.

    Returns
    -------
    SnapResult
    """
    warnings: list[str] = []
    shots_sorted = sorted(shots or [])
    valleys_sorted = sorted(valleys or [])

    # ── Collect start-edge candidates ─────────────────────────────────────────
    # (timestamp, priority, source_label)
    # Lower priority number = preferred signal for the start edge.
    start_candidates: list[tuple[float, int, str]] = []

    if sentences:
        for sent in sentences:
            if abs(sent.start - proposed_start) <= search_window_s:
                start_candidates.append((sent.start, 0, "sentence"))

    for t in shots_sorted:
        if abs(t - proposed_start) <= search_window_s:
            start_candidates.append((t, 1, "shot"))

    for t in valleys_sorted:
        if abs(t - proposed_start) <= search_window_s:
            start_candidates.append((t, 2, "valley"))

    # ── Collect end-edge candidates ────────────────────────────────────────────
    # For end we prefer sentence ends that have terminal punctuation.
    end_candidates: list[tuple[float, int, str]] = []

    if sentences:
        for sent in sentences:
            if abs(sent.end - proposed_end) <= search_window_s:
                has_punct = bool(re.search(r"[.?!।]$", sent.text.strip()))
                prio = 0 if has_punct else 3  # complete sentence vs plain
                end_candidates.append((sent.end, prio, "sentence"))

    for t in valleys_sorted:
        if abs(t - proposed_end) <= search_window_s:
            end_candidates.append((t, 1, "valley"))

    for t in shots_sorted:
        if abs(t - proposed_end) <= search_window_s:
            end_candidates.append((t, 2, "shot"))

    # ── Select best candidate for each edge ───────────────────────────────────

    def _pick_best(
        candidates: list[tuple[float, int, str]],
        proposed: float,
    ) -> tuple[float, list[str]]:
        """Return (chosen_time, source_labels) for the best candidate."""
        if not candidates:
            return proposed, []
        # Sort by priority first, then by proximity to proposed edge
        candidates.sort(key=lambda c: (c[1], abs(c[0] - proposed)))
        chosen_t, _, _ = candidates[0]

        # Gather all sources with the same priority at the chosen timestamp
        # (within 0.05 s tolerance — multiple signals at essentially the same point)
        best_prio = candidates[0][1]
        sources: list[str] = []
        for t, prio, src in candidates:
            if prio == best_prio and abs(t - chosen_t) < 0.05:
                if src not in sources:
                    sources.append(src)
        return chosen_t, sources

    snapped_start, start_sources = _pick_best(start_candidates, proposed_start)
    snapped_end, end_sources = _pick_best(end_candidates, proposed_end)

    if not start_sources:
        warnings.append(
            f"No signal found within ±{search_window_s}s of proposed start "
            f"{proposed_start:.2f}s; keeping original."
        )
    if not end_sources:
        warnings.append(
            f"No signal found within ±{search_window_s}s of proposed end "
            f"{proposed_end:.2f}s; keeping original."
        )

    # ── Duration clamping ──────────────────────────────────────────────────────
    duration = snapped_end - snapped_start
    if duration < min_clip_duration_s:
        # Expand end outward
        snapped_end = snapped_start + min_clip_duration_s
        warnings.append(
            f"Clip duration {duration:.2f}s < minimum {min_clip_duration_s:.0f}s; "
            f"end expanded to {snapped_end:.2f}s."
        )

    duration = snapped_end - snapped_start
    if duration > max_clip_duration_s:
        # Shrink end inward from start
        snapped_end = snapped_start + max_clip_duration_s
        warnings.append(
            f"Clip duration {duration:.2f}s > maximum {max_clip_duration_s:.0f}s; "
            f"end clamped to {snapped_end:.2f}s."
        )

    adjusted_s = abs(snapped_start - proposed_start) + abs(snapped_end - proposed_end)

    logger.debug(
        "snap_boundaries: [%.2f→%.2f] → [%.2f→%.2f]  adj=%.2fs  "
        "start_src=%s end_src=%s",
        proposed_start, proposed_end, snapped_start, snapped_end, adjusted_s,
        start_sources, end_sources,
    )

    return SnapResult(
        start=snapped_start,
        end=snapped_end,
        start_sources=start_sources,
        end_sources=end_sources,
        adjusted_s=adjusted_s,
        warnings=warnings,
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
    sentence : asr.Sentence
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
