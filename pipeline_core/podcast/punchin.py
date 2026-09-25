# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/punchin.py.
# Changes from upstream: none (verbatim copy; origin header added).
"""Podcast punch-in planner — PURE (no I/O, no ffmpeg).

Given the ``keep_ranges`` from the cut-list and the word-level transcript,
emit a punch-in plan: a list of

    {start_s, end_s, zoom, mode, trigger}

where ``zoom`` is 1.08-1.15 (rule 6), ``mode`` is ``"snap"`` (150-400ms
emphasis punch) or ``"slow_push"`` (1500-3000ms emotional push, rule 7), and
``trigger`` names why the punch fired (rule 8: emphasis_word | question |
exclamation | monotony_breaker).

The plan is expressed in **edited-timeline coordinates** (seconds into the
final cut, after keep_ranges are concatenated) so the renderer can map each
punch onto the correct trim segment. Density is capped (rule: max 1 punch per
``min_gap_s`` seconds, default 10) so the edit doesn't feel jittery.

Triggers (Phase 1, transcript-only — no LLM emphasis pass yet):

  * QUESTION / EXCLAMATION — a kept word ending in ``?`` or ``!`` fires a
    snap punch on that word.
  * EMPHASIS WORD — an all-caps token (len>1) or a word from a small
    emphasis lexicon fires a snap punch.
  * MONOTONY BREAKER (rule 3 spirit) — when the SAME speaker holds the floor
    for longer than ``monotony_s`` (default 25s) of continuous edited time
    without any other punch, insert one slow_push variety punch.

Everything is deterministic and total; unit tests build Word lists by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from pipeline_core.podcast.cutlist import WordT, _norm


# ── Config ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PunchInConfig:
    zoom_snap: float = 1.12          # rule 6: emphasis punch
    zoom_slow: float = 1.08          # gentler for emotional/monotony push
    snap_ms: float = 250.0           # rule 7 snap 150-400
    slow_push_ms: float = 2000.0     # rule 7 slow-push 1500-3000
    min_gap_s: float = 10.0          # density cap: <=1 punch / 10s
    monotony_s: float = 25.0         # rule 3: >25s same speaker -> variety
    hold_s: float = 1.4              # how long a snap punch stays zoomed
    # bounds enforced on any zoom value
    zoom_min: float = 1.08
    zoom_max: float = 1.15


# Small emphasis lexicon (English). Kept tiny + honest — a real emphasis
# detector is an LLM pass (Phase 2+).
_EMPHASIS_WORDS: frozenset[str] = frozenset({
    "never", "always", "huge", "massive", "insane", "crazy", "unbelievable",
    "exactly", "absolutely", "everything", "nothing", "impossible", "shocking",
})


@dataclass(frozen=True)
class PunchIn:
    start_s: float          # edited-timeline start
    end_s: float            # edited-timeline end
    zoom: float
    mode: str               # "snap" | "slow_push"
    trigger: str


# ── Timeline mapping ────────────────────────────────────────────────────


def _edited_time_of(src_t: float, keep_ranges: Sequence[tuple[float, float]]) -> float | None:
    """Map a SOURCE timestamp to its EDITED-timeline position.

    Returns None if ``src_t`` falls inside a removed gap (not in any kept
    range).
    """
    acc = 0.0
    for s, e in keep_ranges:
        if s <= src_t <= e:
            return acc + (src_t - s)
        acc += e - s
    return None


def _edited_duration(keep_ranges: Sequence[tuple[float, float]]) -> float:
    return sum(e - s for s, e in keep_ranges)


def _is_emphasis(word: WordT) -> bool:
    raw = word.w.strip()
    tok = _norm(raw)
    if not tok:
        return False
    if tok in _EMPHASIS_WORDS:
        return True
    # ALL-CAPS token (>1 char, has letters) = shouted emphasis.
    letters = re.sub(r"[^A-Za-z]", "", raw)
    if len(letters) > 1 and letters.isupper():
        return True
    return False


# ── Public API ──────────────────────────────────────────────────────────


def build_punch_in_plan(
    keep_ranges: Sequence[tuple[float, float]],
    words: Sequence[WordT],
    cfg: PunchInConfig | None = None,
) -> list[PunchIn]:
    """Emit a density-capped punch-in plan in edited-timeline coordinates.

    Parameters
    ----------
    keep_ranges
        The cut-list output — source ``(start_s, end_s)`` ranges, in order.
    words
        The word-level transcript (source timestamps).
    cfg
        Punch-in knobs.
    """
    cfg = cfg or PunchInConfig()
    if not keep_ranges or not words:
        return []

    ordered = sorted(words, key=lambda w: float(w.s))

    def _clamp_zoom(z: float) -> float:
        return max(cfg.zoom_min, min(cfg.zoom_max, z))

    candidates: list[PunchIn] = []

    # --- Trigger 1+2: per-word question/exclamation/emphasis (snap) ---
    for w in ordered:
        et = _edited_time_of(float(w.s), keep_ranges)
        if et is None:
            continue  # word landed in a removed gap
        raw = w.w.strip()
        trigger: str | None = None
        if raw.endswith("?"):
            trigger = "question"
        elif raw.endswith("!"):
            trigger = "exclamation"
        elif _is_emphasis(w):
            trigger = "emphasis_word"
        if trigger is None:
            continue
        candidates.append(PunchIn(
            start_s=et,
            end_s=min(et + cfg.hold_s, _edited_duration(keep_ranges)),
            zoom=_clamp_zoom(cfg.zoom_snap),
            mode="snap",
            trigger=trigger,
        ))

    # --- Trigger 3: monotony breaker (slow_push) ---
    # Walk the words in edited time; track continuous same-speaker runs and
    # drop a variety push when a run exceeds monotony_s.
    run_speaker: int | None = None
    run_start_et: float | None = None
    last_et = 0.0
    for w in ordered:
        et = _edited_time_of(float(w.s), keep_ranges)
        if et is None:
            continue
        last_et = et
        spk = w.speaker
        if spk != run_speaker:
            run_speaker = spk
            run_start_et = et
            continue
        if run_start_et is not None and (et - run_start_et) >= cfg.monotony_s:
            candidates.append(PunchIn(
                start_s=et,
                end_s=min(et + cfg.slow_push_ms / 1000.0 + cfg.hold_s,
                          _edited_duration(keep_ranges)),
                zoom=_clamp_zoom(cfg.zoom_slow),
                mode="slow_push",
                trigger="monotony_breaker",
            ))
            # Reset the run anchor so we don't fire every subsequent word.
            run_start_et = et

    # --- Density cap: sort by start, greedily keep >= min_gap_s apart ---
    candidates.sort(key=lambda p: p.start_s)
    plan: list[PunchIn] = []
    last_kept_start = -1e9
    for c in candidates:
        if c.start_s - last_kept_start >= cfg.min_gap_s:
            plan.append(c)
            last_kept_start = c.start_s
    return plan


__all__ = ["PunchInConfig", "PunchIn", "build_punch_in_plan"]
