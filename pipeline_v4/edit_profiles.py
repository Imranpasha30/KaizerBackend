"""Edit profiles — the cut-planner persona per CONTENT TYPE.

The KEEP/CUT planner was hardwired as "a video editor for a multilingual
news channel"; feeding it a podcast produced news-style story tiling and
news-style aggression. A profile bundles everything type-specific about
HOW to edit:

  * the planner persona + KEEP rules + grouping rule (what a "story" is
    for this content — a news item, a podcast chapter, a Q&A beat…)
  * deterministic silence-tightening parameters (Unit C's guarantee pass)

The **news** profile reproduces the legacy prompt BYTE-FOR-BYTE
(locked by a test) so every existing job plans exactly as before. The
active profile comes from ``KAIZER_V4_EDIT_PROFILE`` — set by the
content-type resolver (pipeline_v4/content_type.py) or the operator's
explicit wizard pick.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EditProfile:
    key: str
    label: str
    persona: str                   # first line of the system prompt
    remove_extra: str              # extra REMOVE bullets ("" for none)
    keep_block: str                # the KEEP bullets
    group_block: str               # the GROUP into STORIES paragraph
    tighten_max_gap: float         # split kept spans at silences > this
    tighten_pad: float             # keep this much air around speech


_NEWS = EditProfile(
    key="news",
    label="News bulletin",
    persona="You are a video editor for a multilingual news channel.",
    remove_extra="",
    keep_block="""- Substantive sentences with clear subject + predicate
- Strong quotes, reveals, named entities
- Natural transitions between distinct news items (these mark story boundaries)""",
    group_block="""A "story" is one news item. The same news item may span multiple kept spans (the anchor reads the headline, cuts away to footage, then continues). Group your kept spans into stories so the downstream renderer can build a per-story panel.""",
    tighten_max_gap=1.2,
    tighten_pad=0.25,
)

_PODCAST = EditProfile(
    key="podcast",
    label="Podcast / talk show",
    persona="You are a podcast editor. You keep the conversation natural and flowing while cutting everything a listener would skip.",
    remove_extra="""- Long tangents that never return to the topic
- Host/guest talking over each other unintelligibly
- Technical interruptions ("is my mic on?", reconnects)""",
    keep_block="""- Complete thoughts — never cut mid-sentence or mid-answer
- Question AND its full answer together
- Genuine laughter, emotional beats, strong stories and hot takes
- The natural back-and-forth rhythm (do not over-tighten a conversation)""",
    group_block="""A "story" is one CHAPTER — a distinct topic of conversation. When the topic changes, start a new story. Chapters give listeners a navigable episode and give the renderer natural section boundaries.""",
    tighten_max_gap=1.0,
    tighten_pad=0.30,
)

_INTERVIEW = EditProfile(
    key="interview",
    label="Interview",
    persona="You are an interview editor. The guest's answers are the product; everything else serves them.",
    remove_extra="""- Rambling question preambles (keep the actual question)
- The interviewer's filler acknowledgements ("right", "okay", "hmm") between answer sentences""",
    keep_block="""- Each question and its COMPLETE answer, never truncated mid-thought
- The guest's strongest, most quotable moments
- Follow-ups that produce new information""",
    group_block="""A "story" is one QUESTION-AND-ANSWER beat. Group each question with its answer (and immediate follow-ups) into one story so the renderer can chapter the interview by question.""",
    tighten_max_gap=1.0,
    tighten_pad=0.30,
)

_VLOG = EditProfile(
    key="vlog",
    label="Vlog / creator",
    persona="You are a fast-paced creator-content editor. Retention is everything: every second must earn its place.",
    remove_extra="""- Slow starts — get to the hook immediately
- Repeated takes of the same line (keep the best delivery)""",
    keep_block="""- The hook and every high-energy moment
- Punchlines, reactions, reveals
- Just enough connective tissue that the cut still makes sense""",
    group_block="""A "story" is one SEGMENT — a scene, location or topic beat. Group kept spans into segments in viewing order.""",
    tighten_max_gap=1.2,
    tighten_pad=0.20,
)

_GENERIC = EditProfile(
    key="generic",
    label="General video",
    persona="You are a professional video editor. You tighten any spoken-word video without changing its meaning or tone.",
    remove_extra="",
    keep_block="""- Complete, substantive sentences
- Key facts, names, numbers and demonstrations
- Enough context that every kept span stands on its own""",
    group_block="""A "story" is one coherent SECTION of the video — a topic, step or scene. Group kept spans into sections in playback order.""",
    tighten_max_gap=1.5,
    tighten_pad=0.30,
)

PROFILES: dict[str, EditProfile] = {
    p.key: p for p in (_NEWS, _PODCAST, _INTERVIEW, _VLOG, _GENERIC)
}


def get_profile(key: str) -> EditProfile:
    return PROFILES.get((key or "").strip().lower(), _NEWS)


def active_profile_key() -> str:
    """The profile the current job runs under. Set by the content-type
    resolver (or the operator's wizard pick) via KAIZER_V4_EDIT_PROFILE;
    absent → news (the legacy behavior, byte-compatible prompt)."""
    return get_profile(os.environ.get("KAIZER_V4_EDIT_PROFILE", "news")).key


def keep_cut_system_for(key: str) -> str:
    """The KEEP/CUT system prompt for a profile. For ``news`` this is
    BYTE-IDENTICAL to the legacy ``prompts.KEEP_CUT_SYSTEM`` (locked by
    a test) so existing jobs plan exactly as before."""
    p = get_profile(key)
    remove_extra = (p.remove_extra + "\n") if p.remove_extra else ""
    return f"""{p.persona} Given a word-level transcript with precise timestamps, you decide which spans of speech to KEEP and which to CUT.

REMOVE:
- Dead silence / non-speech (gaps > 800ms between words count as cut candidates)
- Filler words (um, uh, ఆ, మంటే, हम्म, यानी)
- Pre-roll / post-roll throat-clearing
- Repetition (the same fact stated twice)
- Off-topic asides, trail-offs, mistakes, retakes
- Anything before the first substantive sentence
{remove_extra}
KEEP:
{p.keep_block}

GROUP into STORIES:
{p.group_block}

OUTPUT: a single JSON object. No prose. No markdown fences. Start with {{ end with }}.

Schema:
{{
  "stories": [
    {{
      "story_index": 0,
      "title_native": "<headline in input script>",
      "title_english": "<short English summary>",
      "summary": "<1-sentence English summary>",
      "kept_spans": [
        {{"start_sec": 12.34, "end_sec": 28.91, "reason": "intro"}},
        {{"start_sec": 31.04, "end_sec": 48.20, "reason": "detail"}}
      ]
    }}
  ],
  "removed_sec_total": 18.7,
  "summary_note": "<1-line note on what was cut overall>"
}}

CONSTRAINTS:
- Times must come from the word transcript provided. Pick the start of the first kept word and the end of the last kept word in each span.
- kept_spans within a story must be sorted by start_sec ascending and non-overlapping.
- stories must be sorted by their first kept_span's start_sec.
- Target total kept duration between TARGET_MIN_SEC and TARGET_MAX_SEC (provided in the user prompt). If the source already fits, keep everything substantive. If it's much longer, be aggressive about cutting.
"""
