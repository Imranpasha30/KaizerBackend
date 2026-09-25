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

    The language is resolved through `languages.get(...)` so the prompt
    carries the FULL language name + native script directive instead of
    just the ISO code. Without this Gemini tends to fall back to English
    for `te`/`hi`/etc., which breaks the contract that the bulletin's
    SEO must match the language the operator selected in the wizard.
    """
    # Resolve to the rich language config — gives us name_english,
    # name_native, script. Falls back to Telugu silently for unknown
    # codes (same behaviour as the rest of the pipeline).
    try:
        import languages as _langs  # local import — avoid circular at module load
        cfg = _langs.get(language)
        lang_full = cfg.name_english          # "Telugu", "Hindi", ...
        lang_native = cfg.name_native          # "తెలుగు", "हिन्दी", ...
        script_name = cfg.script               # "Telugu", "Devanagari", ...
        lang_code = cfg.code or language
    except Exception:
        # Defensive fallback so a missing languages module never blocks
        # SEO generation entirely.
        lang_full = language
        lang_native = ""
        script_name = ""
        lang_code = language

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

    # Hard language directive — placed at the very top of the system
    # prompt so Gemini reads it before any other rule. The script name
    # ("Telugu", "Devanagari", ...) is the unambiguous instruction; the
    # native name ("తెలుగు", "हिन्दी", ...) is a concrete example
    # Gemini can pattern-match against when writing back.
    language_directive = (
        f"# ⚠ LANGUAGE CONTRACT (MOST IMPORTANT RULE)\n"
        f"The TITLE must be BILINGUAL: ENGLISH + {lang_full}\n"
        f"({script_name} script) mixed in ONE title. All OTHER SEO output\n"
        f"(description, keywords, hashtags, hook, thumbnail_text) MUST be\n"
        f"written in {lang_full} ({lang_native}) using the {script_name}\n"
        f"script natively. ISO code: {lang_code}.\n"
        f"\n"
        f"- The TITLE mixes BOTH scripts (operator mandate 2026-08): a\n"
        f"  searchable ENGLISH part (key person/place/topic in standard\n"
        f"  English spelling, e.g. \"Revanth Reddy\", \"Hyderabad\") PLUS\n"
        f"  the {script_name}-script core of the headline — this is the\n"
        f"  proven high-CTR pattern in {lang_full} news feeds, and the\n"
        f"  English terms make the video searchable in both languages.\n"
        f"  Never a romanised transliteration of whole {lang_full}\n"
        f"  sentences.\n"
        f"- The DESCRIPTION must be in {lang_full} ({script_name} script)\n"
        f"  with at most one English line for the opening hook. No\n"
        f"  romanised transliteration of {lang_full} words — write them\n"
        f"  in the proper script.\n"
        f"- KEYWORDS: 60% in {lang_full} ({script_name}), 40% English.\n"
        f"- HASHTAGS: at least 4 of the 10-12 hashtags must be in\n"
        f"  {script_name} script (e.g. native-script topic names).\n"
        f"- HOOK and THUMBNAIL_TEXT may be English IF that maximises CTR,\n"
        f"  but prefer the native script when the topic is local news.\n"
        f"\n"
        f"If the output language doesn't match the operator's selection,\n"
        f"our verifier rejects the SEO and the operator has to regenerate\n"
        f"manually — that's a failure mode we never want.\n\n"
    )

    return f"""\
{language_directive}You are an elite YouTube SEO strategist specializing in {lang_full} ({lang_native}) news.
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
- `title`: 50-95 characters, BILINGUAL (English + {lang_full} script
  mixed in one headline — searchable English person/place/topic terms
  plus the native-script core; never a romanised transliteration).
  NO "| Channel" suffix anywhere.  Include a power word in EITHER
  language (Shocking, Breaking, Exclusive, సంచలనం, షాకింగ్, ...) and
  VARY the hook form across videos — question, number, consequence,
  quote — never the same opener twice in a row.  Put the key person /
  place in the first 6 words.
- `description`: 700-1800 characters.  Plain text, no markdown.  Structure:
    1. Line 1: the HOOK sentence, verbatim.
    2. Three context paragraphs separated by blank lines.  Cite facts from
       "Live Google News context" if provided.
    3. A blank line, then a HASHTAG BLOCK as the FINAL line(s) — every
       hashtag from `hashtags` listed inline separated by spaces (e.g.
       `#TeluguNews #BreakingNews #HeeraGold ...`). This block is what
       YouTube surfaces above the title for viewers; without it the
       channel loses the top-of-watch-page hashtag chip.
  Do NOT include subscribe lines or "follow us" blocks anywhere — those
  land per-destination at publish time.
- `keywords`: exactly 28-30 unique SEO TAGS (the hidden YouTube tags
  field — NOT the hashtags). Plain lowercase strings.  No '#' prefix.
  Total combined length ≤500 chars (YouTube hard cap).  Must include
  2-3 trending keywords from "Google Trends" if provided.  Mix English
  + native-script. These TAGS and the HASHTAGS below are two separate
  surfaces with different roles — they MUST be substantially different
  sets (allow at most 2 overlapping topics, the rest unique). The
  hashtags drive viewer-facing topic chips; the keywords drive the
  recommendation algorithm.
- `hashtags`: 10-12 unique hashtags with '#' prefix, strict CamelCase.  No
  spaces, no punctuation inside the tag.  Topic-only — NO channel brands.
  These appear in the description block (see above) AND above the title
  on the YouTube watch page. Pick the 10-12 most clickable topic chips,
  not the 10-12 most searched terms — those go in `keywords`.
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
    learned: Optional[Dict[str, Any]] = None,
    explore_hook: Optional[str] = None,
    script_policy: str = "bilingual",
    competitor: Optional[Dict[str, Any]] = None,
    avoid_titles: Optional[List[str]] = None,
    angle_hint: Optional[str] = None,
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

    # ── Learned-policy block (learning/seo_learning.py — REAL measured
    #    performance of this channel's published videos; the wire that
    #    makes generation improve over time instead of staying static) ──
    learned_block = ""
    if learned:
        learned_block = (
            "\n# 📈 CHANNEL LEARNING — measured from this channel's REAL "
            "YouTube performance. FOLLOW IT.\n"
        )
        hooks = learned.get("best_hooks") or []
        if hooks:
            _hnames = {"question": "a QUESTION headline",
                       "number": "a NUMBER-led headline",
                       "quote": "a QUOTE-led headline",
                       "power": "a power-word hook",
                       "plain": "a plain factual headline"}
            learned_block += ("- Hook forms that WIN here: "
                              + ", ".join(_hnames.get(h, h) for h in hooks)
                              + " — use one of these forms.\n")
        if learned.get("best_script"):
            _snames = {"mixed": "English + native script MIXED",
                       "english": "mostly English",
                       "native": "mostly native script"}
            learned_block += (f"- Title script that earns the most here: "
                              f"{_snames.get(learned['best_script'], learned['best_script'])}.\n")
        if learned.get("best_len_band"):
            _bands = {"short": "under 50 chars", "sweet": "50-80 chars",
                      "long": "80-95 chars"}
            learned_block += (f"- Title length sweet spot here: "
                              f"{_bands.get(learned['best_len_band'], learned['best_len_band'])}.\n")
        kws = learned.get("top_keywords") or []
        if kws:
            learned_block += ("- Keywords that historically ride this "
                              "channel's winners (weave 2-4 IF relevant to "
                              "THIS video, never force them): "
                              + ", ".join(kws[:8]) + "\n")
        tops = learned.get("top_topics") or []
        if tops:
            learned_block += ("- Topic angles that WIN on this channel "
                              "(when THIS video touches one, lead with it): "
                              + ", ".join(tops[:5]) + "\n")
        if learned.get("based_on"):
            learned_block += (f"(learned from {learned['based_on']} published "
                              f"videos' measured views/hour and CTR)\n")
    # EXPLORATION directive (measured A/B): overrides the exploit hint so
    # the channel's policy keeps getting tested against alternatives.
    if explore_hook:
        _hnames2 = {"question": "a QUESTION headline",
                    "number": "a NUMBER-led headline",
                    "quote": "a QUOTE-led headline",
                    "power": "a power-word hook",
                    "plain": "a plain factual headline"}
        learned_block += (
            f"\n# 🧪 EXPLORATION (measured test — this round only)\n"
            f"Write the title as {_hnames2.get(explore_hook, explore_hook)} "
            f"this time, NOT the channel's usual form. This is a deliberate "
            f"A/B probe; its performance will be measured.\n")

    # ── Competitor intelligence block (opt-in; topic-matched only) ──
    competitor_block = ""
    if competitor and competitor.get("rivals"):
        competitor_block = (
            "\n# ⚔ MARKET INTELLIGENCE — tracked competitors' WINNERS on "
            "THIS topic (public data)\n"
            "Their best-performing videos on this story:\n")
        for rv in competitor["rivals"][:5]:
            competitor_block += (f"- [{rv['vph']} views/hr] "
                                 f"\"{rv['title']}\"\n")
        competitor_block += (
            "RULES: target the SAME search queries but DIFFERENTIATE — a "
            "fresher angle, never a near-copy of their titles.\n")
        if competitor.get("cover_terms"):
            competitor_block += (
                "- Query terms their winners carry — cover the relevant "
                "ones: " + ", ".join(competitor["cover_terms"][:8]) + "\n")
        if competitor.get("harvest_tags"):
            competitor_block += (
                "- Their proven tags — include the ones that fit THIS "
                "video in `keywords`: "
                + ", ".join(competitor["harvest_tags"][:12]) + "\n")

    # ── Per-channel distinctness (angle + sibling titles to avoid) ──
    distinct_block = ""
    if angle_hint:
        distinct_block += (
            f"\n# 🎯 ANGLE FOR THIS CHANNEL'S VERSION\n"
            f"Write this channel's title from a DISTINCT angle: {angle_hint}. "
            f"It must read differently from the sibling versions below.\n")
    if avoid_titles:
        distinct_block += (
            "\n# 🚫 SIBLING TITLES ALREADY USED — do NOT reuse their opening "
            "words, structure or phrasing:\n")
        for t in list(avoid_titles)[:8]:
            _t = str(t).strip()
            if _t:
                distinct_block += f'- "{_t}"\n'

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
- Target language: {language} (for description / tags / hashtags)
- Title language: {"mostly ENGLISH (this channel's measured winner)" if script_policy == "english" else ("NATIVE script (this channel's measured winner)" if script_policy == "native" else "BILINGUAL — English searchable terms + native-script core mixed in one headline")}
{news_block}{trends_block}{yt_block}{corpus_block}{learned_block}{competitor_block}{distinct_block}{retry_block}
Write the JSON now.
"""
