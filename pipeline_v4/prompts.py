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
