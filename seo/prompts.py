"""SEO prompt builders — CHANNEL-AGNOSTIC output.

New architecture (Content + Brand Overlay):
  - Gemini produces GENERIC SEO — no channel name in title, no mandatory
    hashtags, no footer.  Just the news content optimized for discoverability.
  - Branding (destination's name, hashtags, footer) is injected LATER at
    publish time by `seo.composer.compose(...)`.

Optional `style_source` teaches writing VOICE only — never identity.  Sanitizer
runs post-generation to strip any brand leaks Gemini may have slipped in.

Context layers fed to Gemini:
  1. Clip facts               — what the video is actually about
  2. Google News items        — factually grounded details
  3. Google Trends keywords   — terms people are searching RIGHT NOW
  4. YouTube top-5 titles     — winning title patterns for this topic this week
  5. Style corpus             — hook/title rhythm to emulate (from style_source)
  6. User retry feedback      — verifier failures from previous attempt(s)
"""
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

import models


# ── System prompt (content-only) ─────────────────────────────────────────────

def build_system_prompt(
    *,
    language: str = "te",
    style_source: Optional[models.Channel] = None,
    target_score: int = 95,
) -> str:
    """Channel-agnostic system prompt.  Tells Gemini to produce news-topic SEO
    with ZERO channel branding — that's added mechanically at publish time.
    """
    voice_block = ""
    if style_source:
        tf = (style_source.title_formula or "").strip()
        ds = (style_source.desc_style or "hook_first").strip()
        voice_block = f"""
# Writing voice reference (LEARN FROM — do NOT mention the reference channel)
You are studying a top-performing news channel's writing rhythm.
- Title formula pattern: {tf or "(none — use your best judgement for native-script news)"}
- Description style: {ds}
Do NOT include the reference channel's name, handle, hashtags, or any
branding in the output.  Use only its RHYTHM and WORDING STYLE.
"""

    return f"""\
You are an elite YouTube SEO strategist specializing in {language}-language news.
Your job: write viral, click-worthy, factually-honest SEO metadata that will
score ≥{target_score}/100 on our independent verifier.

# Critical rule — CHANNEL-AGNOSTIC OUTPUT
The output MUST be completely generic — it will be reused across many
different YouTube channels.  Therefore:
- Title MUST NOT end with " | ChannelName" or contain any channel's name.
- Description MUST NOT mention any channel by name, URL, or @handle.
- Description MUST NOT include subscribe lines, Follow-us blocks, or emoji
  footer blocks — those are added per-destination by downstream systems.
- Hashtags MUST NOT include channel-branded tags like #RTVTelugu or
  #TV9News — only TOPIC hashtags like #NPSRetirees or #RevanthReddy.
- Keywords MUST NOT contain channel names — only news-topic terms.

Any channel name, handle, or URL that leaks in will cost score points AND
will be mechanically stripped before publish.
{voice_block}
# Output contract (strict — response_schema is enforced)
- `title`: 50-95 characters.  Bilingual (English hook + native-script) OR
  fully native-script.  NO "| Channel" suffix anywhere.  Include a power
  word (Shocking, Breaking, Exclusive, Revealed, Viral, or the native-script
  equivalent: బిగ్, షాకింగ్, బ్రేకింగ్, వైరల్, ...).  Put the key person
  / place in the first 6 words.
- `description`: 700-1800 characters.  Plain text, no markdown.  Line 1 is
  the HOOK sentence, verbatim.  Then 3 context paragraphs with blank-line
  breaks.  Cite facts from "Live Google News context" if provided.  Do NOT
  include hashtag-only lines or subscribe lines.
- `keywords`: exactly 28-30 unique SEO tags.  Plain lowercase strings.  No
  '#' prefix.  Total combined length ≤500 chars (YouTube hard cap).  Must
  include 2-3 trending keywords from "Google Trends" if provided.  Mix
  English + native-script.
- `hashtags`: 10-12 unique hashtags with '#' prefix, strict CamelCase.  No
  spaces, no punctuation inside the tag.  Topic-only — NO channel brands.
- `hook`: one strong opening sentence reused on thumbnails + social copy.
- `thumbnail_text`: 2-5 shouting words, no punctuation.
- `metadata.sentiment`: one of shock | breaking | political | emotional | analytical | positive
- `metadata.category`: one of politics | cinema | sports | crime | national | state | viral | other
- `metadata.viral_score`: 0-100 (server recomputes — don't game)

# Editorial rules
1. News only, never entertainment fiction — use facts from the clip context
   and the Google News items.
2. Power words sparingly, not every sentence.
3. Conversational newsroom phrasing.  Native-script should feel spoken, not
   literary.
4. Prefer concrete numbers to vague claims.
5. Each of title / hook / thumbnail_text should use DIFFERENT phrasing.
6. Every keyword and hashtag must be earnable — no generic stuffing like
   "news news news".
"""


# ── User prompt (per-clip, with all context layers) ──────────────────────────

def build_user_prompt(
    *,
    clip: models.Clip,
    language: str = "te",
    news_items: Optional[List[Dict[str, Any]]] = None,
    trends: Optional[Dict[str, Any]] = None,
    yt_top: Optional[List[Dict[str, Any]]] = None,
    corpus: Optional[Dict[str, Any]] = None,
    style_source: Optional[models.Channel] = None,
    retry_feedback: Optional[List[str]] = None,
) -> str:
    """Per-clip user prompt with all grounded research layers + retry context."""
    try:
        meta = json.loads(clip.meta or "{}")
    except (ValueError, TypeError):
        meta = {}

    summary_en = (meta.get("summary") or "").strip()
    summary_native = (meta.get("summary_telugu") or meta.get("summary_native") or "").strip()
    headline = (clip.text or meta.get("text") or "").strip()
    key_people = meta.get("key_people") or meta.get("speakers") or []
    key_topics = meta.get("key_topics") or []
    key_locations = meta.get("key_locations") or []
    sentiment = (clip.sentiment or meta.get("mood") or "").strip()
    duration_str = f"{clip.duration:.1f}s" if clip.duration else "unknown"

    # ── News block ──
    news_block = ""
    if news_items:
        news_block = "\n# Live Google News context (ground wording in THESE facts)\n"
        for i, item in enumerate(news_items[:8], 1):
            src = item.get("source") or "Google News"
            news_block += f"{i}. {item['title']} — {src}\n"

    # ── Trends block ──
    trends_block = ""
    if trends and (trends.get("related_queries") or trends.get("rising_queries") or trends.get("trending_now")):
        trends_block = "\n# Google Trends — incorporate 2-3 of these into keywords AND at least 1 into title/description\n"
        if trends.get("related_queries"):
            trends_block += f"Related to topic: {', '.join(trends['related_queries'][:8])}\n"
        if trends.get("rising_queries"):
            trends_block += f"Rising queries: {', '.join(trends['rising_queries'][:6])}\n"
        if trends.get("trending_now"):
            trends_block += f"Trending now (regional): {', '.join(trends['trending_now'][:6])}\n"

    # ── YouTube top-5 block ──
    yt_block = ""
    if yt_top:
        yt_block = "\n# YouTube top-performing titles for this topic (last 7 days) — emulate the hook shape, do NOT copy verbatim\n"
        for i, v in enumerate(yt_top[:5], 1):
            views = v.get("views", 0)
            yt_block += f'{i}. [{views:,} views] "{v.get("title", "")}"\n'

    # ── Style corpus block (from style_source) ──
    corpus_block = ""
    if corpus and corpus.get("top_titles"):
        label = style_source.name if style_source else "reference style"
        corpus_block = (
            f"\n# Writing-voice corpus from {label} — emulate RHYTHM, do NOT mention them by name\n"
        )
        for t in (corpus["top_titles"] or [])[:8]:
            corpus_block += f"- {t}\n"
        if corpus.get("hook_patterns"):
            corpus_block += "Common hooks: " + " | ".join(corpus["hook_patterns"][:6]) + "\n"

    # ── Retry feedback (from verifier) ──
    retry_block = ""
    if retry_feedback:
        retry_block = (
            "\n# ⚠ PREVIOUS ATTEMPT SCORED BELOW TARGET — YOU MUST FIX EACH ISSUE BELOW\n"
            "# This is your chance to recover.  Every line below is a direct order.\n"
            "# Do NOT keep any element that violates these rules.  Regenerate fully.\n\n"
        )
        for i, fail in enumerate(retry_feedback[:14], 1):
            retry_block += f"  {i}. {fail}\n"
        retry_block += (
            "\n# How to respond to this feedback\n"
            "- Treat every item as a HARD constraint, not a suggestion.\n"
            "- Do NOT paraphrase the old title/description/keywords — rewrite them.\n"
            "- If a specific trending keyword or phrase was named, copy it VERBATIM\n"
            "  into the title or description (exact spelling, including case if native).\n"
            "- If a channel-suffix leak was flagged, the title must NOT end with '|'\n"
            "  followed by anything at all.  Zero exceptions.\n"
            "- Hit 95/100 this time — a verifier will regrade immediately.\n"
        )

    return f"""\
Generate YouTube SEO for this news clip.  Follow the response_schema exactly
and produce CHANNEL-AGNOSTIC content (no channel names, no footer).

# Clip facts
- Current headline on-screen: {headline or '(none)'}
- English summary: {summary_en or '(none)'}
- Native-script summary: {summary_native or '(none)'}
- Key people: {', '.join(key_people) if key_people else '(none)'}
- Key topics: {', '.join(key_topics) if key_topics else '(none)'}
- Key locations: {', '.join(key_locations) if key_locations else '(none)'}
- Sentiment / mood: {sentiment or '(unspecified)'}
- Clip duration: {duration_str}
- Target language: {language}
{news_block}{trends_block}{yt_block}{corpus_block}{retry_block}
Write the JSON now.
"""
