# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/cutlist.py.
# Changes from upstream: none (verbatim copy; origin header added).
"""Podcast cut-list builder — PURE functions (no I/O, no ffmpeg).

Consumes a word-level transcript (the ``Word`` shape from
``pipeline_v2.models``: ``.w`` text, ``.s`` start-sec, ``.e`` end-sec,
``.speaker`` int|None, ``.confidence`` float|None) and produces
``keep_ranges`` — a list of ``(start_s, end_s)`` tuples ready to feed
straight into ``pipeline_v2.render.edl_builder.build_extraction_edl``.

Three cut mechanisms (rules 10-12 of POD_RENDER_RULES.md):

  (a) SILENCE cuts  — a gap between consecutive words longer than
      ``silence_cut_threshold_ms`` is removed. Each surviving kept span
      is padded by ``pad_ms`` on each side (clamped to neighbours) so a
      cut at a word boundary does not clip inhaled breaths.
  (b) FILLER removal — words matching the English filler lexicon
      (``um``/``uh``/``erm`` always; ``like``/``you know`` only when the
      ASR confidence is low OR the word repeats — context-guarded) plus a
      two-tier Telugu lexicon: a hard-drop set (``ante``/``mari``/
      ``kadha``/``enti``/``sare`` + romanization variants) and a
      context-guarded soft set (``emo``/``ade``/``ika``/``edo``/
      ``nijanga``, guarded the same way as the EN soft fillers). Still
      flagged ``[inferred]`` — see the lexicon's own module comment for
      the confidence/provenance breakdown per entry.
  (c) STUTTER / repeat — a consecutive duplicate word (case-insensitive)
      whose start is within ``stutter_window_s`` of the previous one has
      its EARLIER occurrence(s) dropped, keeping the last clean take.

The three passes run over the word list, marking words as dropped; the
surviving words are then coalesced into keep_ranges (silence-splitting is
applied on the surviving stream so filler/stutter gaps don't accidentally
merge across a genuine silence).

Determinism: every function is pure and total over its inputs. Tests build
``Word`` lists by hand — no audio, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence


# ── Word structural type ────────────────────────────────────────────────
# We accept anything with the pipeline_v2.models.Word attribute shape via a
# Protocol so callers may pass either the real pydantic Word or a lightweight
# test double, without importing pipeline_v2 here (keeps this module import-
# light and free of the pydantic dependency for pure-function tests).


class WordT(Protocol):
    w: str
    s: float
    e: float
    speaker: int | None
    confidence: float | None


# ── Lexicons ────────────────────────────────────────────────────────────

# Always-drop English fillers (rule 11, [confirmed]).
_EN_FILLERS_HARD: frozenset[str] = frozenset({
    "um", "uh", "erm", "uhh", "umm", "hmm", "mm", "mmm", "ah", "er",
})

# Context-guarded English fillers: only removed when ASR confidence is low
# OR the token repeats immediately. "like" and discourse "you know".
_EN_FILLERS_SOFT: frozenset[str] = frozenset({
    "like", "basically", "literally", "actually", "yknow",
})
# Multi-word soft filler phrases handled as an n-gram guard.
_EN_FILLER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("you", "know"),
    ("i", "mean"),
    ("sort", "of"),
    ("kind", "of"),
)

# Telugu filler lexicon — [inferred], tune later. Extended 2026-07-09
# (HANDOFF Podcast Phase 3) from the original 4-word starter set to a
# broader, deliberately two-tier list of documented Telugu discourse
# fillers / hesitation markers (the Telugu equivalents of "um"/"like"/
# "you know"). Romanised + native (Telugu-script) forms are both included
# so matching works regardless of which script the ASR/transcript uses
# for a given word (mirrors the pattern in content_policy/lexicons.py).
#
# HONESTY / PROVENANCE (read before extending further): this list is
# built from general linguistic knowledge of common spoken-register
# Telugu discourse particles, NOT verified against a native-speaker
# review pass or a corpus study. It is split into two confidence/risk
# tiers rather than one flat list:
#
#   _TE_FILLERS_HARD  — unconditionally dropped when remove_te_fillers is
#                        on. Judged LOW risk of colliding with a
#                        meaningful standalone word in ordinary spoken
#                        Telugu.
#   _TE_FILLERS_SOFT  — context-guarded (same low-ASR-confidence-OR-
#                        immediate-repeat guard already used for the EN
#                        soft fillers "like"/"actually"/etc. below).
#                        Each entry ALSO has a legitimate, meaningful
#                        standalone sense (a demonstrative, a real
#                        adverb, a real indefinite pronoun) in normal
#                        Telugu speech, so a bare token match risks
#                        stripping real content — guarding trades some
#                        missed fillers for fewer false positives.
#
# Only a handful of new entries were added, deliberately: each one is a
# word this pass has at least moderate confidence is a genuinely
# documented Telugu discourse/hesitation marker. Words considered and
# excluded outright (too low-confidence to include even guarded) are
# listed below the two sets, so a future session doesn't have to
# re-derive why they're missing.
#
# ROMANIZATION: STT transcript scripts vary (native Telugu vs. Latin
# transliteration), and even within Latin transliteration, spelling
# varies (aspirated consonants are often dropped/simplified in casual
# romanization, e.g. "kadha" -> "kada"). A few well-known spelling
# variants of existing entries are included below for this reason; this
# is NOT an exhaustive fuzzy-transliteration matcher — it is still exact
# token matching (see ``_norm``/membership test), just against a wider
# set of known-common spellings.
_TE_FILLERS_HARD: frozenset[str] = frozenset({
    # Original starter set (unchanged) — [inferred].
    "ante", "అంటే",        # "I mean / that is" — very common filler.
    "mari", "మరి",          # "well, then" — discourse transition.
    "kadha", "కదా",         # tag particle "isn't it / right".
    "enti", "ఏంటి",         # "what" used as a hedge/filler.
    # Known common romanization variants of the above (aspiration
    # dropped / y-glide spelling) — same lexical items, not new words.
    "kada",                 # "kadha" without the aspirated "dh".
    "anthe",                # "ante" with the aspirated "th" spelling.
    "yenti",                # "enti" with an initial y-glide spelling.
    # New this pass — [inferred, moderate-to-high confidence]:
    "sare", "సరే",          # "okay / well / fine" — very common spoken
                             # acknowledgement/filler particle, close
                             # analog to English discourse "okay"/"well".
})

# Context-guarded Telugu fillers — see the tier explanation above.
_TE_FILLERS_SOFT: frozenset[str] = frozenset({
    "emo", "ఏమో",           # "who knows / dunno" hesitation marker.
                             # Linguistic confidence in this as a filler
                             # is HIGH, but it is guarded (not hard-
                             # dropped) for a SEPARATE reason: the
                             # romanized form "emo" collides with the
                             # unrelated English word "emo" (music
                             # subculture / adjective). Because the
                             # caller (build_keep_ranges) checks the EN
                             # and TE lexicons unconditionally regardless
                             # of the job's declared language, an
                             # ungated hard match here would risk cutting
                             # a real word out of English transcripts.
    "ade", "అదే",           # "that (thing)" — word-search filler, akin
                             # to English "that thing, you know" — also
                             # a real demonstrative ("that same"), hence
                             # guarded.
    "ika", "ఇక",            # "now / so then" discourse transition —
                             # also a real adverb ("further / anymore"),
                             # hence guarded.
    "edo", "ఏదో",           # "something / kind of" hedge, a close
                             # analog to the existing English soft
                             # filler "kind of" — also a real indefinite
                             # pronoun, hence guarded.
    "nijanga", "నిజంగా",    # "actually / really" — close analog to the
                             # existing English soft filler "actually" —
                             # also a sincere, meaningful adverb, hence
                             # guarded.
})

# Considered and DELIBERATELY EXCLUDED — confidence too low without a
# native-speaker review pass to include even in the soft-guarded set:
#   - chudu / chudandi ("look / see" as a discourse marker, cf. English
#     "you see") — overwhelmingly used as a literal imperative verb
#     ("look at this") in ordinary transcripts; a bare token match felt
#     too likely to strip real instructions/content.
#   - aa / haa (non-lexical hesitation vocalizations, cf. "uh") — "aa"
#     is also the common demonstrative "that"; without audio-level
#     disfluency signals (pitch, elongation) this pipeline doesn't have
#     access to, a bare-token match would strip real demonstratives too
#     often.

# Confidence below this makes a soft filler eligible for removal.
_SOFT_FILLER_CONF_CEIL: float = 0.55


def _norm(token: str) -> str:
    """Lowercase + strip surrounding punctuation for lexicon matching."""
    return re.sub(r"^[^\wఀ-౿]+|[^\wఀ-౿]+$", "", token.lower())


# ── Config ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CutlistConfig:
    """All cut-list knobs. Defaults mirror POD_RENDER_RULES.md."""

    # Rule 10: silence.
    silence_cut_threshold_ms: float = 450.0
    pad_ms: float = 120.0
    # Rule 12: stutter window.
    stutter_window_s: float = 1.2
    # Rule 11 toggles.
    remove_en_fillers: bool = True
    remove_te_fillers: bool = True
    remove_soft_fillers: bool = True
    soft_filler_conf_ceil: float = _SOFT_FILLER_CONF_CEIL
    # Minimum kept-span duration (seconds) — spans shorter than this after
    # cutting are dropped (avoids single-frame flickers).
    min_keep_span_s: float = 0.20


# ── Drop-marking passes ─────────────────────────────────────────────────


@dataclass
class _Marked:
    word: WordT
    idx: int
    dropped: bool = False
    reason: str = ""


def _mark_fillers(marked: list[_Marked], cfg: CutlistConfig) -> None:
    """Mark filler words (pass b)."""
    n = len(marked)
    for i, m in enumerate(marked):
        if m.dropped:
            continue
        tok = _norm(m.word.w)
        if not tok:
            continue
        # Hard EN fillers.
        if cfg.remove_en_fillers and tok in _EN_FILLERS_HARD:
            m.dropped, m.reason = True, "filler_en_hard"
            continue
        # Telugu lexicon (hard tier — low collision risk).
        if cfg.remove_te_fillers and tok in _TE_FILLERS_HARD:
            m.dropped, m.reason = True, "filler_te"
            continue
        # Soft EN fillers — context guarded.
        if cfg.remove_soft_fillers and tok in _EN_FILLERS_SOFT:
            conf = m.word.confidence
            low_conf = conf is not None and conf < cfg.soft_filler_conf_ceil
            prev_tok = _norm(marked[i - 1].word.w) if i > 0 else ""
            next_tok = _norm(marked[i + 1].word.w) if i + 1 < n else ""
            repeated = tok == prev_tok or tok == next_tok
            if low_conf or repeated:
                m.dropped, m.reason = True, "filler_en_soft"
                continue
        # Soft Telugu lexicon — same context guard as soft EN fillers,
        # gated by the same remove_te_fillers toggle (a single toggle
        # turns off ALL Telugu filler removal, hard + soft).
        if cfg.remove_te_fillers and tok in _TE_FILLERS_SOFT:
            conf = m.word.confidence
            low_conf = conf is not None and conf < cfg.soft_filler_conf_ceil
            prev_tok = _norm(marked[i - 1].word.w) if i > 0 else ""
            next_tok = _norm(marked[i + 1].word.w) if i + 1 < n else ""
            repeated = tok == prev_tok or tok == next_tok
            if low_conf or repeated:
                m.dropped, m.reason = True, "filler_te_soft"
                continue
    # Soft filler PHRASES (n-grams) — only when confidence low on any token
    # of the phrase, or unguarded discourse "you know" mid-sentence.
    if cfg.remove_soft_fillers:
        for phrase in _EN_FILLER_PHRASES:
            plen = len(phrase)
            for i in range(0, n - plen + 1):
                window = marked[i:i + plen]
                if any(w.dropped for w in window):
                    continue
                toks = tuple(_norm(w.word.w) for w in window)
                if toks != phrase:
                    continue
                confs = [w.word.confidence for w in window
                         if w.word.confidence is not None]
                low_conf = bool(confs) and min(confs) < cfg.soft_filler_conf_ceil
                if low_conf:
                    for w in window:
                        w.dropped, w.reason = True, "filler_en_phrase"


def _mark_stutters(marked: list[_Marked], cfg: CutlistConfig) -> None:
    """Mark stutter/repeat words (pass c).

    A run of consecutive (ignoring already-dropped) identical tokens whose
    span is within ``stutter_window_s`` keeps only the LAST occurrence —
    the clean take — and drops the earlier ones.
    """
    live = [m for m in marked if not m.dropped]
    i = 0
    while i < len(live):
        j = i + 1
        tok_i = _norm(live[i].word.w)
        if not tok_i:
            i += 1
            continue
        # Extend the run while the next live word is the same token AND
        # starts within the stutter window of the run's first word.
        run_end = i
        while (
            j < len(live)
            and _norm(live[j].word.w) == tok_i
            and (live[j].word.s - live[i].word.s) <= cfg.stutter_window_s
        ):
            run_end = j
            j += 1
        if run_end > i:
            # Drop all but the last in the run.
            for k in range(i, run_end):
                live[k].dropped, live[k].reason = True, "stutter"
        i = j if j > i + 1 else i + 1


# ── keep_range coalescing (pass a — silence-aware) ──────────────────────


def _coalesce_keep_ranges(
    marked: list[_Marked],
    cfg: CutlistConfig,
) -> list[tuple[float, float]]:
    """Turn surviving words into padded keep_ranges, splitting on silence.

    Walk the surviving (non-dropped) words in time order. Start a span at
    the first survivor; extend it while the inter-word gap stays under the
    silence threshold. When a gap exceeds the threshold (rule 10) OR a
    dropped word created a break, close the current span and open a new one.
    Each closed span is padded by ``pad_ms`` on each side, clamped so it
    never overlaps the neighbouring survivor's boundary.
    """
    survivors = [m for m in marked if not m.dropped]
    if not survivors:
        return []

    thresh_s = cfg.silence_cut_threshold_ms / 1000.0
    pad_s = cfg.pad_ms / 1000.0

    spans: list[tuple[float, float]] = []
    cur_start = survivors[0].word.s
    cur_end = survivors[0].word.e

    for prev, nxt in zip(survivors, survivors[1:]):
        gap = nxt.word.s - prev.word.e
        # A break happens either on a true silence gap OR when the two
        # survivors were not adjacent in the original stream (a dropped
        # word/filler sat between them) AND that removed span itself
        # exceeded the silence threshold. We approximate "removed span"
        # by the same gap measure — a large gap means cut here.
        if gap > thresh_s:
            spans.append((cur_start, cur_end))
            cur_start = nxt.word.s
            cur_end = nxt.word.e
        else:
            cur_end = nxt.word.e
    spans.append((cur_start, cur_end))

    # Apply padding, clamped to neighbours and to >= 0.
    padded: list[tuple[float, float]] = []
    for k, (s, e) in enumerate(spans):
        lo = s - pad_s
        hi = e + pad_s
        if k > 0:
            prev_hi = spans[k - 1][1]
            # Don't let padding cross the midpoint of the removed gap.
            mid = (prev_hi + s) / 2.0
            lo = max(lo, mid)
        if k + 1 < len(spans):
            next_lo = spans[k + 1][0]
            mid = (e + next_lo) / 2.0
            hi = min(hi, mid)
        lo = max(0.0, lo)
        if hi - lo >= cfg.min_keep_span_s:
            padded.append((lo, hi))
    return padded


# ── Public API ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CutlistResult:
    """Result of ``build_keep_ranges``.

    ``keep_ranges`` feeds ``build_extraction_edl``. ``dropped`` records
    every removed word with its reason for transparency in the results
    JSON. ``removed_seconds`` is the total source time cut.
    """

    keep_ranges: tuple[tuple[float, float], ...]
    dropped: tuple[dict, ...]
    source_duration_s: float
    kept_seconds: float
    removed_seconds: float


def build_keep_ranges(
    words: Sequence[WordT],
    cfg: CutlistConfig | None = None,
) -> CutlistResult:
    """Build padded keep_ranges from a word-level transcript.

    Parameters
    ----------
    words
        Word-level transcript in playback order (``pipeline_v2.models.Word``
        shape). Must be sorted by start time; unsorted input is sorted
        defensively.
    cfg
        Cut-list knobs; defaults from ``CutlistConfig`` / POD_RENDER_RULES.

    Returns
    -------
    CutlistResult
    """
    cfg = cfg or CutlistConfig()
    ordered = sorted(words, key=lambda w: (float(w.s), float(w.e)))
    if not ordered:
        return CutlistResult((), (), 0.0, 0.0, 0.0)

    source_duration = float(ordered[-1].e) - float(ordered[0].s)
    marked = [_Marked(word=w, idx=i) for i, w in enumerate(ordered)]

    _mark_fillers(marked, cfg)
    _mark_stutters(marked, cfg)

    keep_ranges = _coalesce_keep_ranges(marked, cfg)

    kept = sum(e - s for s, e in keep_ranges)
    dropped = tuple(
        {
            "idx": m.idx,
            "w": m.word.w,
            "s": float(m.word.s),
            "e": float(m.word.e),
            "reason": m.reason,
        }
        for m in marked
        if m.dropped
    )
    removed = max(0.0, source_duration - kept)

    return CutlistResult(
        keep_ranges=tuple(keep_ranges),
        dropped=dropped,
        source_duration_s=source_duration,
        kept_seconds=kept,
        removed_seconds=removed,
    )


__all__ = [
    "WordT",
    "CutlistConfig",
    "CutlistResult",
    "build_keep_ranges",
]
