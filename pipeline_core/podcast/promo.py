# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/promo.py.
# Changes from upstream: scorer import rewired from pipeline_core.narrative (not present in
# this tree) to the locally-extracted pipeline_core.podcast._narrative_scores; logic unchanged.
"""Podcast promo / trailer planner — PURE (no I/O, no ffmpeg).

Given the word-level transcript + the main edit's ``keep_ranges``, score
candidate highlight segments for HOOK STRENGTH and select 3-6 of them,
totalling the promo-duration target, ordered strong-hook-first for a
narrative tease (POD_RENDER_RULES.md §B).

REUSE
-----
Hook scoring **reuses the existing pure heuristics** upstream housed in
``pipeline_core.narrative`` (extracted locally into
``pipeline_core.podcast._narrative_scores`` — this tree does not carry the
full narrative engine):

  * ``_hook_score``       — question/exclamation opener, short-punchy first
    sentence, strong-opener lexicon, no dangling pronoun.
  * ``_completion_score`` — clean narrative end via ``clip_boundaries.detect_completion``.
  * ``_composite_score``  — with ``mode="trailer"`` (0.5*hook + 0.25*importance
    + 0.25*completion) because promos live or die on the hook.

Those helpers expect "sentence" objects with ``.text``/``.start``/``.end``.
We synthesise sentence objects from the word transcript (grouping words at
sentence-final punctuation and long pauses), so we feed the SAME battle-tested
scorer rather than re-implementing it. On top of narrative's ``importance``
input we add promo-specific signals (numbers/claims, emotional words, speaker-
turn density) as the ``importance`` term.

NOTE on the other hook-scoring found in the repo: the shorts selector in
``pipeline_v2.stages.stage_3a_shorts`` scores "hook" 1-10 but does so via a
**Gemini LLM prompt** (network call) — unusable in pure/offline tests. The
``narrative`` heuristics are the reusable pure path, so we reuse those.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from pipeline_core.podcast.cutlist import WordT, _norm

# Reused pure scorers (extracted from upstream pipeline_core.narrative).
from pipeline_core.podcast._narrative_scores import (
    _hook_score,
    _completion_score,
    _composite_score,
)


# ── Config ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromoConfig:
    promo_duration_sec: float = 75.0      # target; range enforced below
    promo_min_sec: float = 60.0
    promo_max_sec: float = 120.0
    segment_min_s: float = 10.0
    segment_max_s: float = 25.0
    min_segments: int = 3
    max_segments: int = 6
    end_card_sec: float = 4.0             # reserved CTA tail (3-5s)
    # Pause (seconds) that ends a pseudo-sentence when building segments.
    sentence_pause_s: float = 0.6


# Emotional-word lexicon for the importance term.
_EMOTION_WORDS: frozenset[str] = frozenset({
    "love", "hate", "crazy", "insane", "shocking", "incredible", "terrifying",
    "amazing", "unbelievable", "heartbreaking", "furious", "thrilled", "scared",
    "devastated", "beautiful", "horrible", "wonderful", "outrageous",
})
_CLAIM_WORDS: frozenset[str] = frozenset({
    "percent", "million", "billion", "thousand", "first", "never", "always",
    "biggest", "largest", "record", "history", "ever", "most",
})


# ── Pseudo-sentence + segment types ─────────────────────────────────────


@dataclass
class _Sent:
    """Minimal sentence object matching narrative's duck-typed shape."""

    text: str
    start: float
    end: float
    words: list[WordT]


@dataclass(frozen=True)
class PromoSegment:
    """One selected highlight, in SOURCE coordinates."""

    start_s: float
    end_s: float
    score: float
    hook: float
    text_preview: str


@dataclass(frozen=True)
class PromoPlan:
    """Promo EDL: source keep_ranges (tease-ordered) + captions + telemetry."""

    keep_ranges: tuple[tuple[float, float], ...]
    segments: tuple[PromoSegment, ...]
    captions: tuple[dict, ...]
    target_duration_s: float
    total_duration_s: float
    end_card_sec: float


# ── Sentence + candidate construction ───────────────────────────────────


def _words_in_keep(
    words: Sequence[WordT], keep_ranges: Sequence[tuple[float, float]]
) -> list[WordT]:
    """Words whose start falls inside a kept range (playback order)."""
    if not keep_ranges:
        return sorted(words, key=lambda w: float(w.s))
    out: list[WordT] = []
    for w in sorted(words, key=lambda w: float(w.s)):
        for s, e in keep_ranges:
            if s <= float(w.s) <= e:
                out.append(w)
                break
    return out


def _build_sentences(words: list[WordT], cfg: PromoConfig) -> list[_Sent]:
    """Group words into pseudo-sentences on final punctuation / long pause."""
    sents: list[_Sent] = []
    cur: list[WordT] = []
    for i, w in enumerate(words):
        cur.append(w)
        raw = w.w.strip()
        end_punct = raw.endswith((".", "?", "!", "।"))  # incl. Devanagari danda
        gap_next = (
            i + 1 < len(words) and (float(words[i + 1].s) - float(w.e)) >= cfg.sentence_pause_s
        )
        if end_punct or gap_next or i == len(words) - 1:
            text = " ".join(x.w for x in cur).strip()
            sents.append(_Sent(text=text, start=float(cur[0].s), end=float(cur[-1].e), words=cur))
            cur = []
    return sents


def _importance(sent_window: list[_Sent]) -> float:
    """Promo importance term in [0,1] from numbers/claims/emotion/turn-density."""
    if not sent_window:
        return 0.0
    all_words = [w for s in sent_window for w in s.words]
    if not all_words:
        return 0.0
    text_tokens = [_norm(w.w) for w in all_words]
    n = len(text_tokens)

    has_number = any(re.search(r"\d", w.w) for w in all_words)
    claim_hits = sum(1 for t in text_tokens if t in _CLAIM_WORDS)
    emo_hits = sum(1 for t in text_tokens if t in _EMOTION_WORDS)

    # Speaker-turn density: changes per second.
    speakers = [w.speaker for w in all_words]
    turns = sum(1 for a, b in zip(speakers, speakers[1:]) if a != b)
    span = max(1e-6, all_words[-1].e - all_words[0].s)
    turn_density = turns / span  # turns/sec

    score = 0.0
    score += 0.25 if has_number else 0.0
    score += min(0.25, 0.12 * claim_hits)
    score += min(0.30, 0.12 * emo_hits)
    score += min(0.20, turn_density * 0.5)
    return min(1.0, score)


def _candidate_segments(sents: list[_Sent], cfg: PromoConfig) -> list[tuple[float, float, list[_Sent]]]:
    """Sliding windows of consecutive sentences within the segment length band."""
    cands: list[tuple[float, float, list[_Sent]]] = []
    n = len(sents)
    for i in range(n):
        window: list[_Sent] = []
        for j in range(i, n):
            window = sents[i:j + 1]
            dur = window[-1].end - window[0].start
            if dur < cfg.segment_min_s:
                continue
            if dur > cfg.segment_max_s:
                break
            cands.append((window[0].start, window[-1].end, list(window)))
    # If NO window reached segment_min (short source), fall back to whole-run
    # windows so a short podcast still yields a proportional promo.
    if not cands and sents:
        cands.append((sents[0].start, sents[-1].end, list(sents)))
    return cands


def _non_overlapping(
    ranked: list[PromoSegment], budget_s: float, cfg: PromoConfig
) -> list[PromoSegment]:
    """Greedily take highest-scoring non-overlapping segments within budget."""
    chosen: list[PromoSegment] = []
    used: list[tuple[float, float]] = []
    total = 0.0
    for seg in ranked:
        if len(chosen) >= cfg.max_segments:
            break
        if any(not (seg.end_s <= u0 or seg.start_s >= u1) for u0, u1 in used):
            continue  # overlaps an already-chosen segment
        seg_dur = seg.end_s - seg.start_s
        if total + seg_dur > budget_s and len(chosen) >= cfg.min_segments:
            continue
        chosen.append(seg)
        used.append((seg.start_s, seg.end_s))
        total += seg_dur
    return chosen


# ── Public API ──────────────────────────────────────────────────────────


def build_promo_plan(
    words: Sequence[WordT],
    keep_ranges: Sequence[tuple[float, float]],
    cfg: PromoConfig | None = None,
) -> PromoPlan:
    """Score highlights and select a tease-ordered promo EDL.

    Selection:
      1. Build pseudo-sentences from the kept words.
      2. Form candidate segments (10-25s bands).
      3. Score each with narrative's trailer-mode composite
         (0.5*hook + 0.25*importance + 0.25*completion).
      4. Greedily pick top non-overlapping segments up to the duration budget
         and segment-count band.
      5. Emit them **hook-descending** (strongest first) — no-spoiler tease.

    Short-source rule: if the kept runtime is below ``promo_min_sec``, the
    target collapses to ``min(promo_min_sec, 0.5 * kept_runtime)`` and the
    segment floor drops to 1.
    """
    cfg = cfg or PromoConfig()
    kept_words = _words_in_keep(words, keep_ranges)
    if not kept_words:
        return PromoPlan((), (), (), cfg.promo_duration_sec, 0.0, cfg.end_card_sec)

    kept_runtime = sum(
        (e - s) for s, e in keep_ranges
    ) if keep_ranges else (float(kept_words[-1].e) - float(kept_words[0].s))

    # Short-source scaling.
    min_segments = cfg.min_segments
    if kept_runtime < cfg.promo_min_sec:
        target = min(cfg.promo_min_sec, 0.5 * kept_runtime)
        min_segments = 1
        eff_cfg = PromoConfig(
            promo_duration_sec=target,
            promo_min_sec=cfg.promo_min_sec,
            promo_max_sec=cfg.promo_max_sec,
            segment_min_s=min(cfg.segment_min_s, max(3.0, kept_runtime / 3.0)),
            segment_max_s=cfg.segment_max_s,
            min_segments=1,
            max_segments=cfg.max_segments,
            end_card_sec=cfg.end_card_sec,
            sentence_pause_s=cfg.sentence_pause_s,
        )
    else:
        target = min(cfg.promo_max_sec, max(cfg.promo_min_sec, cfg.promo_duration_sec))
        eff_cfg = cfg

    sents = _build_sentences(kept_words, eff_cfg)
    cands = _candidate_segments(sents, eff_cfg)

    scored: list[PromoSegment] = []
    for start, end, window in cands:
        hook = _hook_score(window, start, end)
        completion = _completion_score(window, start, end)
        importance = _importance(window)
        comp = _composite_score(importance, hook, completion, mode="trailer")
        preview = window[0].text[:80] if window else ""
        scored.append(PromoSegment(
            start_s=start, end_s=end, score=comp, hook=hook, text_preview=preview,
        ))

    # Budget = target minus the reserved end-card slot.
    budget = max(eff_cfg.segment_min_s, target - eff_cfg.end_card_sec)
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    chosen = _non_overlapping(ranked, budget, eff_cfg)

    # Tease order: strongest HOOK first (front-load the grab). Do NOT resolve
    # last — teasers lead; the payoff is withheld.
    tease_ordered = sorted(chosen, key=lambda s: s.hook, reverse=True)

    promo_keep = tuple((s.start_s, s.end_s) for s in tease_ordered)
    total_dur = sum(e - s for s, e in promo_keep)

    # Word-pop captions for the promo, in promo-edited coordinates.
    captions = _promo_captions(kept_words, promo_keep)

    return PromoPlan(
        keep_ranges=promo_keep,
        segments=tuple(tease_ordered),
        captions=captions,
        target_duration_s=target,
        total_duration_s=total_dur,
        end_card_sec=eff_cfg.end_card_sec,
    )


def _promo_captions(
    words: Sequence[WordT], promo_keep: Sequence[tuple[float, float]]
) -> tuple[dict, ...]:
    """Per-word captions mapped into the PROMO edited timeline."""
    caps: list[dict] = []
    acc = 0.0
    for s, e in promo_keep:
        seg_words = [w for w in words if s <= float(w.s) <= e]
        for w in seg_words:
            et_start = acc + (float(w.s) - s)
            et_end = acc + (float(w.e) - s)
            caps.append({
                "w": w.w,
                "start_s": round(et_start, 3),
                "end_s": round(et_end, 3),
            })
        acc += e - s
    return tuple(caps)


__all__ = [
    "PromoConfig",
    "PromoSegment",
    "PromoPlan",
    "build_promo_plan",
]
