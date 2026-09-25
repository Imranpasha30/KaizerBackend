"""Claude prompts used in V4.

Two prompts:
  1. KEEP/CUT plan — given Deepgram word-level transcript, decide
     which segments to keep. Output is consumed by trim_engine to
     drive the atomic ffmpeg trim+concat.
  2. Per-story canvas plan — given the kept segments, produce per-story
     metadata (title, summary, image timing). Output is consumed by
     orchestrator to materialise the canvas.json.

Style: terse, schema-strict, no preamble. Claude returns ONE JSON
object per call. The orchestrator parses and validates.
"""
from __future__ import annotations


KEEP_CUT_SYSTEM = """You are a video editor for a multilingual news channel. Given a word-level transcript with precise timestamps, you decide which spans of speech to KEEP and which to CUT.

REMOVE:
- Dead silence / non-speech (gaps > 800ms between words count as cut candidates)
- Filler words (um, uh, ఆ, మంటే, हम्म, यानी)
- Pre-roll / post-roll throat-clearing
- Repetition (the same fact stated twice)
- Off-topic asides, trail-offs, mistakes, retakes
- Anything before the first substantive sentence

KEEP:
- Substantive sentences with clear subject + predicate
- Strong quotes, reveals, named entities
- Natural transitions between distinct news items (these mark story boundaries)

GROUP into STORIES:
A "story" is one news item. The same news item may span multiple kept spans (the anchor reads the headline, cuts away to footage, then continues). Group your kept spans into stories so the downstream renderer can build a per-story panel.

OUTPUT: a single JSON object. No prose. No markdown fences. Start with { end with }.

Schema:
{
  "stories": [
    {
      "story_index": 0,
      "title_native": "<headline in input script>",
      "title_english": "<short English summary>",
      "summary": "<1-sentence English summary>",
      "kept_spans": [
        {"start_sec": 12.34, "end_sec": 28.91, "reason": "intro"},
        {"start_sec": 31.04, "end_sec": 48.20, "reason": "detail"}
      ]
    }
  ],
  "removed_sec_total": 18.7,
  "summary_note": "<1-line note on what was cut overall>"
}

CONSTRAINTS:
- Times must come from the word transcript provided. Pick the start of the first kept word and the end of the last kept word in each span.
- kept_spans within a story must be sorted by start_sec ascending and non-overlapping.
- stories must be sorted by their first kept_span's start_sec.
- Target total kept duration between TARGET_MIN_SEC and TARGET_MAX_SEC (provided in the user prompt). If the source already fits, keep everything substantive. If it's much longer, be aggressive about cutting.
"""


def build_keep_cut_user_prompt(
    *,
    words: list[dict],
    target_min_sec: float,
    target_max_sec: float,
    language: str,
    duration_sec: float,
) -> str:
    """Build the user-side prompt for the KEEP/CUT call.

    ``words`` is the Deepgram word array, one entry per word with
    {w, s, e} (word text, start sec, end sec). We compact to compact
    JSON to fit Claude's context efficiently.
    """
    import json as _json

    # Compact representation: one word per line, "[index] start-end word".
    # This is dense enough for Claude to navigate while staying readable
    # for retry-feedback inspection.
    lines = []
    for i, w in enumerate(words):
        s = float(w.get("s") or w.get("start") or 0.0)
        e = float(w.get("e") or w.get("end") or 0.0)
        tok = (w.get("w") or w.get("word") or "").strip()
        if not tok:
            continue
        lines.append(f"[{i}] [{s:.2f}-{e:.2f}] {tok}")

    body = "\n".join(lines)

    return (
        f"Transcript language: {language}\n"
        f"Source duration: {duration_sec:.1f} seconds.\n"
        f"TARGET_MIN_SEC: {target_min_sec:.0f}\n"
        f"TARGET_MAX_SEC: {target_max_sec:.0f}\n\n"
        f"Word transcript ({len(lines)} words):\n"
        f"\"\"\"\n{body}\n\"\"\"\n"
    )


# ────────────────────────────────────────────────────────────────────
# Image-timing prompt (run per story after we know its duration)
# ────────────────────────────────────────────────────────────────────

IMAGE_TIMING_SYSTEM = """You are a news producer deciding which carousel images appear when on a news bulletin.

Given a story's text and a pool of relevant images, output a timing plan: which image shows at which seconds-from-start of the story, and for how long.

RULES:
- Story duration is fixed (provided). Your timings must cover [0, duration] with no gap.
- Each image is shown for at least 2.5 seconds and at most 6.0 seconds.
- An image must NOT repeat consecutively. Cycle through the pool.
- Prefer SHORT durations on stats / numbers (quick reveal feels punchy) and LONGER durations on people / location shots (viewer needs time to identify).
- If you have fewer images than fit the duration, cycle them.

OUTPUT: one JSON object, no preamble.

Schema:
{
  "images": [
    {"pool_index": 0, "t_start": 0.0, "t_end": 4.0},
    {"pool_index": 1, "t_start": 4.0, "t_end": 8.5}
  ]
}

pool_index is 0-based into the pool list provided in the user prompt.
"""


def build_image_timing_user_prompt(
    *,
    story_title: str,
    story_summary: str,
    duration_sec: float,
    image_pool: list[dict],
) -> str:
    """``image_pool`` is a list of {label, kind} describing each image.
    We give Claude semantic labels (not file paths) so it picks based
    on meaning."""
    lines = []
    for i, p in enumerate(image_pool):
        label = p.get("label") or p.get("filename") or f"image_{i}"
        kind = p.get("kind") or "photo"
        lines.append(f"[{i}] {kind}: {label}")
    pool_str = "\n".join(lines) or "(empty pool)"

    return (
        f"Story title: {story_title}\n"
        f"Story summary: {story_summary}\n"
        f"Story duration: {duration_sec:.1f} seconds\n\n"
        f"Available images ({len(image_pool)}):\n"
        f"{pool_str}\n"
    )


# ────────────────────────────────────────────────────────────────────
# Image-timing prompt V2 — transcript-grounded (the Phase-1 engine).
# V1 above mandated gapless pool-cycling; V2 is the opposite contract:
# an image appears ONLY while its subject is actually being spoken,
# and gaps are EXPECTED (the renderer cuts back to the main video).
# V1 is kept for reference; nothing calls it.
# ────────────────────────────────────────────────────────────────────

IMAGE_TIMING_SYSTEM_V2 = """You are a broadcast news producer deciding WHEN each image appears during a news story, so that every image shows EXACTLY while its subject is being spoken.

You get:
- the story's spoken words, each with [index] and start-end seconds (story-relative),
- an image manifest: [id] kind: label — the label says WHO/WHAT the image shows.

Match images to speech BY LABEL MEANING, never by guessing pixels. Words may be Telugu, Hindi, English or transliterations; match labels semantically across scripts (e.g. label "మోదీ (Modi)" matches spoken "Modi", "Modiji", "మోదీ").

RULES:
1. An image appears ONLY when its subject is being spoken about. If no image matches a stretch of speech, leave that stretch EMPTY — the broadcast cuts back to the anchor video. Gaps are normal and expected. NEVER stretch an image over speech about something else.
2. One image at a time. Windows must not overlap.
3. Each window: at least {min_dwell:.1f}s, at most {max_dwell:.1f}s (viewers need time to read a picture; longer gets stale).
4. Same subject spoken again later → reuse the SAME image id (never a different picture of the same subject).
5. Respect RESERVED intervals verbatim — never assign any image overlapping them.
6. For each window report: confidence (0-1: how sure the label truly matches those words), importance (0-1: how central this moment is to the story — a key reveal scores high), and anchor_word_indexes (the [index] numbers of the exact words that triggered the match).
7. When unsure whether an image matches, either lower the confidence or skip the window — a missing image is invisible; a WRONG image is a broadcast error.

OUTPUT: one JSON object, no preamble, no code fences.

Schema:
{{
  "images": [
    {{"pool_index": 0, "t_start": 4.2, "t_end": 9.0, "confidence": 0.86,
      "importance": 0.9, "anchor_word_indexes": [12, 13], "reason": "label X spoken here"}}
  ]
}}

pool_index is the [id] from the image manifest. Times are story-relative seconds.
An empty "images" list is a valid answer when nothing matches.
"""


def build_image_timing_user_prompt_v2(
    *,
    story_title: str,
    story_title_english: str = "",
    story_summary: str = "",
    duration_sec: float,
    words: list[dict],
    image_pool: list[dict],
    pinned_windows: list[tuple[float, float]] = (),
    language: str = "",
    max_words: int = 800,
) -> str:
    """User prompt for IMAGE_TIMING_SYSTEM_V2.

    ``words``  — story-relative [{"w","s","e"}, …] (TrimmedStory.words).
    ``image_pool`` — [{label, kind}, …]; index in this list == pool_index.
    ``pinned_windows`` — operator-pinned [t_start, t_end] intervals the
    model must treat as reserved."""
    pool_lines = []
    for i, p in enumerate(image_pool):
        label = p.get("label") or p.get("filename") or f"image_{i}"
        kind = p.get("kind") or "photo"
        pool_lines.append(f"[{i}] {kind}: {label}")
    pool_str = "\n".join(pool_lines) or "(empty pool)"

    word_lines = []
    for i, w in enumerate(words[:max_words]):
        try:
            word_lines.append(f"[{i}] {w.get('w', '')} {float(w.get('s', 0.0)):.2f}-{float(w.get('e', 0.0)):.2f}")
        except (TypeError, ValueError):
            continue
    words_str = "\n".join(word_lines) or "(no word timestamps)"
    truncated = " (truncated)" if len(words) > max_words else ""

    parts = [
        f"Story title: {story_title}",
    ]
    if story_title_english and story_title_english != story_title:
        parts.append(f"Story title (English): {story_title_english}")
    if story_summary:
        parts.append(f"Story summary: {story_summary}")
    if language:
        parts.append(f"Spoken language: {language}")
    parts.append(f"Story duration: {duration_sec:.1f} seconds")
    if pinned_windows:
        reserved = ", ".join(f"[{a:.1f}, {b:.1f}]" for a, b in pinned_windows)
        parts.append(f"RESERVED intervals (already taken — do not overlap): {reserved}")
    parts.append(f"\nImage manifest ({len(image_pool)}):\n{pool_str}")
    parts.append(f"\nSpoken words ([index] word start-end){truncated}:\n{words_str}")
    return "\n".join(parts) + "\n"
