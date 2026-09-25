# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/editorial.py.
# Changes from upstream: none (verbatim copy; origin header added). Uses anthropic
# (ANTHROPIC_API_KEY) / google.genai (GEMINI_API_KEY|GOOGLE_API_KEY) directly with env keys — kept as-is.
"""Podcast Director — LLM editorial pass (Phase A of the podcast upgrade).

This is the podcast-mode counterpart of pipeline_v2 Stage 2 (Continuity
Editor). It reads the FULL word-level transcript (with speaker labels when
the STT provider supplied them) and makes SEMANTIC editing decisions the
mechanical cutlist cannot:

  * drop spans with a labeled reason — rambling, aside/off-topic, retake
    (implicit phrase restarts), hesitation runs, crew talk, self-correction,
    warm-up chatter;
  * emphasis words chosen by MEANING (claims, numbers-in-context, turning
    points) instead of the longest-word-per-window heuristic;
  * promo picks — the story's actual hooks/highlights, scored 1-10.

Provider plumbing mirrors pipeline_v2/stages/stage_2_providers.py (Claude
``messages.parse`` with a cached system prompt / Gemini ``response_schema``)
but runs SYNCHRONOUSLY because the podcast pipeline executes on a daemon
thread (routers/podcast.py), and is self-contained so podcast mode does not
import Stage-2's bulletin-specific prompt/models.

Honest availability rule (same spirit as STT): if no LLM key is configured
this module reports unavailable and the caller falls back to the mechanical
cutlist — it must NEVER fake an editorial result.

Env knobs
---------
  KAIZER_PODCAST_EDITORIAL           "auto" (default) | "off"
  KAIZER_PODCAST_EDITORIAL_PROVIDER  "claude" | "gemini"  (default: by key)
  KAIZER_PODCAST_EDITORIAL_MODEL     model override for the chosen provider
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from pipeline_core.podcast.cutlist import WordT
from pipeline_core.podcast.promo import PromoConfig, PromoPlan, PromoSegment

logger = logging.getLogger("pipeline_core.podcast.editorial")

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
MAX_OUTPUT_TOKENS = 16384
# One LLM call handles this many words; longer transcripts are chunked and
# the results merged (indices are global, so merging is trivial).
CHUNK_WORDS = 8000

DropCategory = Literal[
    "warm_up", "retake", "hesitation", "crew_talk",
    "aside", "self_correction", "rambling", "dead_air",
]


# ── Structured output contract ──────────────────────────────────────────


class EditorialDrop(BaseModel):
    """A contiguous word-index span to remove, with the editorial reason."""

    start_word: int = Field(description="first word index of the span (inclusive)")
    end_word: int = Field(description="last word index of the span (inclusive)")
    category: DropCategory
    reason: str = Field(description="one short sentence: why this span is cut")


class EditorialPromoPick(BaseModel):
    """A self-contained highlight window worth teasing in the promo."""

    start_word: int
    end_word: int
    score: int = Field(ge=1, le=10, description="10 = strongest hook of the episode")
    hook: str = Field(description="one short sentence: why this moment grabs")


class PodcastEditorial(BaseModel):
    """The director's full decision sheet for one transcript (or chunk)."""

    drops: list[EditorialDrop] = Field(default_factory=list)
    emphasis_words: list[int] = Field(
        default_factory=list,
        description="word indices to word-pop (semantically important words)",
    )
    promo_picks: list[EditorialPromoPick] = Field(default_factory=list)
    audit: str = Field(description="one-sentence summary of what was cut and why")


@dataclass
class EditorialResult:
    """Validated + index-clamped editorial decisions in caller-friendly form."""

    engine: str                                   # e.g. "claude:claude-sonnet-4-6"
    drops: list[dict] = field(default_factory=list)      # {start_s,end_s,category,reason,text}
    drop_word_indices: set[int] = field(default_factory=set)
    emphasis_starts: set[float] = field(default_factory=set)  # original word start_s
    promo_picks: list[EditorialPromoPick] = field(default_factory=list)
    audit: str = ""


# ── Availability / provider resolution ──────────────────────────────────


def editorial_provider() -> Optional[str]:
    """Resolve the editorial provider, or None when the pass must be skipped.

    Explicit provider override must actually have its key; otherwise resolve
    by key presence (Claude preferred — Sonnet is the tier this task needs).
    """
    if os.environ.get("KAIZER_PODCAST_EDITORIAL", "auto").strip().lower() == "off":
        return None
    override = os.environ.get("KAIZER_PODCAST_EDITORIAL_PROVIDER", "").strip().lower()
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    has_gemini = bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if override == "claude":
        return "claude" if has_claude else None
    if override == "gemini":
        return "gemini" if has_gemini else None
    if has_claude:
        return "claude"
    if has_gemini:
        return "gemini"
    return None


# ── Prompt ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the PODCAST DIRECTOR — the editorial brain of an \
automated podcast editor. You receive a word-level transcript (one word per \
line: `index|speaker|start|end|word`) of a raw single-camera podcast \
recording, possibly in Telugu, Hindi, English, or code-mixed. Timestamps are \
seconds. Speaker is an integer label from diarization, or `-` when unknown.

Your job is THREE decisions, returned as structured output:

1. drops — contiguous word-index spans to REMOVE, each with a category:
   * warm_up          — chatter/setup before the real conversation starts
   * retake           — a false start: the speaker restarts the same phrase
                        (matching opening + brief halt + the retry). Keep the
                        LAST complete take, drop the earlier attempt(s).
   * hesitation       — runs of fillers ("um", "uh", "ante", "mari" …) that
                        a single-word filter would miss (e.g. "I- I mean- so")
   * crew_talk        — off-mic/producer interruptions
   * aside            — off-topic digression that breaks the episode's thread
   * self_correction  — a factual slip immediately corrected; drop the slip
   * rambling         — the speaker circles without adding anything: repeated
                        points, unfocused filler sentences, dead weight
   * dead_air         — spans that carry no words worth keeping

2. emphasis_words — word indices that deserve a visual word-pop because they
   carry the MEANING: key claims, surprising numbers IN CONTEXT, names at
   reveal moments, emotional peaks, turning points. Aim for roughly one every
   5–10 seconds of kept content; never mark stopwords.

3. promo_picks — 3 to 6 self-contained windows of roughly 10–25 seconds
   (use the timestamps) that would make someone want to hear the episode:
   strong claims, questions that create curiosity, emotional or surprising
   moments. Score 1–10 (10 = the single strongest hook). A pick must make
   sense with zero surrounding context and must not sit inside a dropped span.

RULES
- Be CONSERVATIVE with drops: when unsure, KEEP. Never cut the payoff of a
  story, and never leave a sentence grammatically broken — extend the span to
  the natural boundary instead.
- Spans are inclusive word-index ranges into the EXACT list you were given.
  Never invent indices outside the range you received.
- Silence between words is handled downstream — do NOT emit drops for pure
  silence gaps unless the span also matches a category above.
- audit: ONE sentence summarizing what you removed and why (counts per
  category are welcome).
"""


def _payload_for(words: Sequence[WordT], lo: int, hi: int, language: Optional[str]) -> str:
    lines = [
        f"language_hint: {language or 'unknown'}",
        f"word_range: {lo}..{hi - 1} (inclusive)",
        "transcript:",
    ]
    for i in range(lo, hi):
        w = words[i]
        spk = getattr(w, "speaker", None)
        lines.append(
            f"{i}|{'-' if spk is None else spk}|{float(w.s):.2f}|{float(w.e):.2f}|{w.w}"
        )
    return "\n".join(lines)


# ── Provider calls (sync — the podcast pipeline runs on a worker thread) ──


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    return re.sub(r"\s*```$", "", t)


def _call_claude(payload: str, model: str) -> PodcastEditorial:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
    # Cached system prompt + parse(): mirrors stage_2_providers.ClaudeStage2Provider.
    response = client.messages.parse(
        output_format=PodcastEditorial,
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=[{
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": payload}],
        temperature=0.0,
        thinking={"type": "disabled"},
    )
    # The parsed object lives on a content block, not the response (see
    # stage_2_providers.py — top-level response has no parsed_output).
    for block in response.content:
        candidate = getattr(block, "parsed_output", None)
        if isinstance(candidate, PodcastEditorial):
            return candidate
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            return PodcastEditorial.model_validate(json.loads(_strip_code_fence(text)))
    raise RuntimeError("Claude response had no parsed_output and no text block")


def _call_gemini(payload: str, model: str) -> PodcastEditorial:
    from google import genai
    from google.genai import types as gtypes

    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")).strip()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[_SYSTEM_PROMPT + "\n\n" + payload],
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PodcastEditorial,
            temperature=0.2,
        ),
    )
    # NEVER trust response.parsed — it silently swallows validation errors
    # (same rule as stage_2_providers). Validate response.text ourselves.
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini response.text is empty")
    return PodcastEditorial.model_validate(json.loads(_strip_code_fence(text)))


# ── Merge / normalize ─────────────────────────────────────────────────────


def _merge_chunk(
    total: PodcastEditorial, chunk: PodcastEditorial, lo: int, hi: int
) -> None:
    """Fold one chunk's decisions into the running total, clamping indices to
    the chunk's own window so a confabulated index can't damage other chunks."""
    for d in chunk.drops:
        s, e = max(d.start_word, lo), min(d.end_word, hi - 1)
        if s <= e:
            total.drops.append(EditorialDrop(
                start_word=s, end_word=e, category=d.category, reason=d.reason))
    total.emphasis_words.extend(i for i in chunk.emphasis_words if lo <= i < hi)
    for p in chunk.promo_picks:
        s, e = max(p.start_word, lo), min(p.end_word, hi - 1)
        if s <= e:
            total.promo_picks.append(EditorialPromoPick(
                start_word=s, end_word=e, score=p.score, hook=p.hook))
    total.audit = (total.audit + " " + chunk.audit).strip()


def run_editorial(
    words: Sequence[WordT],
    *,
    language: Optional[str] = None,
    provider: Optional[str] = None,
) -> EditorialResult:
    """Run the director pass over the full transcript. Raises on failure —
    the caller decides whether to fall back to the mechanical path."""
    prov = provider or editorial_provider()
    if prov is None:
        raise RuntimeError(
            "no editorial LLM available (set ANTHROPIC_API_KEY or GEMINI_API_KEY, "
            "or KAIZER_PODCAST_EDITORIAL=off to silence this)"
        )
    model = os.environ.get("KAIZER_PODCAST_EDITORIAL_MODEL", "").strip() or (
        DEFAULT_CLAUDE_MODEL if prov == "claude" else DEFAULT_GEMINI_MODEL
    )
    call = _call_claude if prov == "claude" else _call_gemini

    n = len(words)
    merged = PodcastEditorial(audit="")
    for lo in range(0, n, CHUNK_WORDS):
        hi = min(lo + CHUNK_WORDS, n)
        chunk = call(_payload_for(words, lo, hi, language), model)
        _merge_chunk(merged, chunk, lo, hi)
        logger.info("editorial chunk %d..%d: %d drops, %d emphasis, %d picks",
                    lo, hi - 1, len(chunk.drops), len(chunk.emphasis_words),
                    len(chunk.promo_picks))

    # Normalize into caller-friendly, time-resolved form.
    result = EditorialResult(engine=f"{prov}:{model}", audit=merged.audit)
    for d in merged.drops:
        span = words[d.start_word:d.end_word + 1]
        if not span:
            continue
        result.drop_word_indices.update(range(d.start_word, d.end_word + 1))
        result.drops.append({
            "start_s": round(float(span[0].s), 3),
            "end_s": round(float(span[-1].e), 3),
            "category": d.category,
            "reason": d.reason,
            "text": " ".join(w.w for w in span)[:160],
        })
    result.emphasis_starts = {
        round(float(words[i].s), 3)
        for i in merged.emphasis_words
        if 0 <= i < n and i not in result.drop_word_indices
    }
    # Promo picks that survived the drop pass, strongest first.
    result.promo_picks = sorted(
        (p for p in merged.promo_picks
         if not all(i in result.drop_word_indices
                    for i in range(p.start_word, p.end_word + 1))),
        key=lambda p: p.score, reverse=True,
    )
    return result


# ── Appliers (used by routers/podcast.py) ─────────────────────────────────


def filter_dropped_words(
    words: Sequence[WordT], result: EditorialResult
) -> list[WordT]:
    """Words that survive the editorial drops — feed THESE to the mechanical
    cutlist so silence/filler handling still runs on the kept content."""
    return [w for i, w in enumerate(words) if i not in result.drop_word_indices]


def captions_for_ranges(
    words: Sequence[WordT],
    ranges: Sequence[tuple[float, float]],
    emphasis_starts: set[float] | None = None,
) -> list[dict]:
    """Per-word captions mapped into the edited timeline of ``ranges``,
    tagging ``isEmphasis`` from the editorial marks while the ORIGINAL word
    start time is still known (it isn't recoverable afterwards)."""
    emphasis_starts = emphasis_starts or set()
    caps: list[dict] = []
    acc = 0.0
    for s, e in ranges:
        for w in words:
            ws = float(w.s)
            if s <= ws <= e:
                caps.append({
                    "w": w.w,
                    "start_s": round(acc + (ws - s), 3),
                    "end_s": round(acc + (float(w.e) - s), 3),
                    "isEmphasis": round(ws, 3) in emphasis_starts,
                })
        acc += e - s
    return caps


def promo_plan_from_picks(
    words: Sequence[WordT],
    result: EditorialResult,
    cfg: PromoConfig | None = None,
) -> PromoPlan:
    """Build a PromoPlan from the director's picks (already strongest-first,
    matching promo.py's tease order). Falls back caller-side to the heuristic
    build_promo_plan when there are no picks."""
    cfg = cfg or PromoConfig()
    budget = max(cfg.segment_min_s, cfg.promo_max_sec - cfg.end_card_sec)
    keep: list[tuple[float, float]] = []
    segments: list[PromoSegment] = []
    total = 0.0
    for p in result.promo_picks[:cfg.max_segments]:
        span = words[p.start_word:p.end_word + 1]
        if not span:
            continue
        s, e = float(span[0].s), float(span[-1].e)
        if total + (e - s) > budget and segments:
            continue
        keep.append((s, e))
        total += e - s
        segments.append(PromoSegment(
            start_s=s, end_s=e, score=p.score / 10.0, hook=p.score / 10.0,
            text_preview=p.hook[:80],
        ))
    captions = captions_for_ranges(words, keep, result.emphasis_starts)
    return PromoPlan(
        keep_ranges=tuple(keep),
        segments=tuple(segments),
        captions=tuple(captions),
        target_duration_s=cfg.promo_duration_sec,
        total_duration_s=total,
        end_card_sec=cfg.end_card_sec,
    )


__all__ = [
    "PodcastEditorial", "EditorialDrop", "EditorialPromoPick", "EditorialResult",
    "editorial_provider", "run_editorial", "filter_dropped_words",
    "captions_for_ranges", "promo_plan_from_picks",
]
