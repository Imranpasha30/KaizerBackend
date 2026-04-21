"""SEO prompt builders — system + user prompts for Gemini.

Ported from the Chrome extension's `buildSystemPrompt` + `buildUserPrompt` in
background.js. The system prompt encodes channel rules, title formula,
mandatory hashtags, and the output contract. The user prompt pipes the clip's
metadata + live news context into a single generate_content call.
"""
from __future__ import annotations

import json
from typing import Optional

import models


def build_system_prompt(
    destination: models.Channel,
    style_source: Optional[models.Channel] = None,
) -> str:
    """System prompt.

    - `destination` — whose YouTube channel the SEO is being generated for.
      All OUTPUT branding comes from this: title suffix, mandatory hashtags,
      fixed tags, footer, handle.
    - `style_source` — optional reference channel (e.g. a competitor) whose
      writing style (title formula + desc style) Gemini should emulate.  If
      None, the destination's own style is used.

    Destination never equals style_source in intent: destination is WHO we
    publish as, style_source is HOW we write.  The prompt makes this split
    explicit so Gemini doesn't bleed RTV-style branding into a Kaizer video.
    """
    name = destination.name
    handle = destination.handle or "(no handle)"
    footer = (destination.footer or "").strip()
    lang = destination.language or "te"

    mandatory = ", ".join(destination.mandatory_hashtags or []) or "(none)"
    fixed = ", ".join(destination.fixed_tags or []) or "(none)"

    has_footer = "yes — will be appended automatically; do NOT include subscribe/hashtag lines in description" if footer else "none"

    # Style comes from style_source if provided, else destination's own.
    style = style_source or destination
    title_formula = (style.title_formula or f"Viral Hook ({lang}) | {name}").strip()
    desc_style = style.desc_style or "hook_first"

    # Distinct-source block — only appears when learning from a different
    # channel, so Gemini understands "match rhythm/voice from X, publish as Y."
    style_ref_block = ""
    if style_source and style_source.id != destination.id:
        src_name = style_source.name
        src_handle = style_source.handle or "(no handle)"
        style_ref_block = f"""
# Writing style reference (LEARN FROM — do NOT copy branding)
You are learning the *writing rhythm* of **{src_name}** ({src_handle}): their
title-formula pattern, their description cadence, and their hook style.  Do
NOT put {src_name}'s name, handle, hashtags, or footer into the output.  The
video is being published on **{name}**, so all branding stays as {name}'s.
"""

    return f"""\
You are a YouTube SEO expert specializing in {lang} news content. You write viral,
click-worthy, factually-honest titles and descriptions for the channel **{name}** ({handle}).
{style_ref_block}
# Destination channel (ALL branding in output MUST reflect this channel)
- Channel name: {name}
- Channel handle: {handle}
- Title formula: {title_formula}
- Description style: {desc_style}
- Mandatory hashtags (MUST appear verbatim, as-is): {mandatory}
- Fixed tags (MUST all be present in the keywords list): {fixed}
- Footer: {has_footer}
- Language target: {lang} (native-script-dominant; English hook is encouraged)

# Output contract (strict — response_schema is enforced)
- `title`: max 100 chars INCLUDING the trailing ` | {name}` suffix. Bilingual
  (English hook + native-script translation) OR fully native-script. MUST end
  with ` | {name}` — never any other channel name.
- `description`: 600–1500 chars. Plain text, no markdown. Open with the HOOK
  sentence. 2–3 context paragraphs. Do NOT include subscribe lines, hashtag
  lines, or emojis on their own line — the footer is appended automatically.
  NEVER mention or brand for any channel other than **{name}**.
- `keywords`: 28–30 SEO tags. Plain strings, no '#', no quotes. Mix English +
  native-script. MUST include every tag from the "Fixed tags" list above,
  verbatim (case-insensitive).  Never include tags that brand a different
  channel's name unless topically relevant to the news story.
- `hashtags`: 10–12 unique hashtags with '#' prefix, CamelCase only (no spaces,
  no punctuation inside). First items MUST be the mandatory hashtags above.
- `hook`: one strong opening sentence reused on thumbnails + social.
- `thumbnail_text`: 2–5 short shouting words, no punctuation.
- `metadata.sentiment`: one of shock | breaking | political | emotional | analytical | positive
- `metadata.category`: one of politics | cinema | sports | crime | national | state | viral | other
- `metadata.viral_score`: hint 0–100 (the server recomputes it — don't game it)

# Editorial rules
1. This is news, not entertainment — never invent facts. Only restructure the
   clip context and the Google News context provided.
2. Use power words sparingly (Shocking, Breaking, Viral, Revealed, Exclusive,
   and their native-script equivalents).
3. Conversational newsroom phrasing, not literary.
4. Prefer numbers to vague claims.
5. Put the key person / place in the first 6 words of the title.
6. Never repeat the exact same phrase across title, hook, and thumbnail_text.
"""


def build_user_prompt(
    *,
    clip: models.Clip,
    destination: models.Channel,
    style_source: Optional[models.Channel] = None,
    news_items: Optional[list[dict]] = None,
    corpus: Optional[dict] = None,
    socials: Optional[dict] = None,
) -> str:
    """Per-clip user prompt. Injects meta + live news + optional learned corpus
    + the user's social-link map so the description ends with a cross-promo block.

    `destination` provides output branding; `style_source` is the (optional)
    channel whose corpus/style patterns we learn from.  When style_source is
    provided and distinct from destination, the corpus label below makes that
    explicit so Gemini doesn't misread reference titles as destination titles.
    """
    # Reminder for linters/IDEs — `channel` is deprecated, kept as alias below.
    channel = destination
    try:
        meta = json.loads(clip.meta or "{}")
    except (ValueError, TypeError):
        meta = {}

    summary_en = (meta.get("summary") or "").strip()
    summary_te = (meta.get("summary_telugu") or "").strip()
    headline = (clip.text or meta.get("text") or "").strip()

    key_people = meta.get("key_people") or meta.get("speakers") or []
    key_topics = meta.get("key_topics") or []
    key_locations = meta.get("key_locations") or []
    sentiment = (clip.sentiment or meta.get("mood") or "").strip()

    news_block = ""
    if news_items:
        news_block = "\n# Live Google News context (ground your wording in these facts)\n"
        for i, item in enumerate(news_items, 1):
            src = item.get("source") or "Google News"
            news_block += f"{i}. {item['title']} — {src}\n"

    corpus_block = ""
    if corpus and corpus.get("top_titles"):
        if style_source and style_source.id != destination.id:
            corpus_block = (
                f"\n# Reference titles from **{style_source.name}** "
                f"(LEARN the rhythm/hook shape — do NOT copy their channel name "
                f"or branding; publish as {destination.name})\n"
            )
        else:
            corpus_block = "\n# Top-performing titles on this channel (match rhythm/hook style, don't copy)\n"
        for t in (corpus["top_titles"] or [])[:10]:
            corpus_block += f"- {t}\n"
        if corpus.get("hook_patterns"):
            corpus_block += "\nCommon hook patterns: " + ", ".join(corpus["hook_patterns"][:8]) + "\n"

    duration_str = f"{clip.duration:.1f}s" if clip.duration else "unknown"

    # Cross-promo block — only emit when the user actually set some links.
    social_block = ""
    if socials:
        pretty = {
            "youtube":   "▶ YouTube",
            "website":   "🌐 Website",
            "twitter":   "𝕏 / Twitter",
            "instagram": "📸 Instagram",
            "facebook":  "📘 Facebook",
            "whatsapp":  "📱 WhatsApp",
            "telegram":  "✈ Telegram",
            "linkedin":  "💼 LinkedIn",
            "tiktok":    "🎵 TikTok",
            "threads":   "@ Threads",
            "email":     "✉ Email",
        }
        lines = []
        for k in ("youtube", "website", "twitter", "instagram", "facebook",
                 "whatsapp", "telegram", "linkedin", "tiktok", "threads", "email"):
            v = (socials.get(k) or "").strip()
            if v:
                label = pretty.get(k, k.capitalize())
                lines.append(f"{label}: {v}")
        # Any other keys not in the canonical list
        for k, v in socials.items():
            if k in pretty:
                continue
            v = (v or "").strip() if isinstance(v, str) else ""
            if v:
                lines.append(f"{k.capitalize()}: {v}")
        if lines:
            social_block = (
                "\n# User's social links — APPEND these as a '— Follow us —' section "
                "at the END of the description, one per line, with their emoji/label exactly as shown.\n"
                + "\n".join(lines) + "\n"
            )

    return f"""\
Generate YouTube SEO for this news clip. Follow the response_schema exactly.

# Clip
- Current headline on-screen: {headline or '(none)'}
- English summary: {summary_en or '(none)'}
- Telugu summary: {summary_te or '(none)'}
- Key people: {', '.join(key_people) if key_people else '(none)'}
- Key topics: {', '.join(key_topics) if key_topics else '(none)'}
- Key locations: {', '.join(key_locations) if key_locations else '(none)'}
- Sentiment / mood: {sentiment or '(unspecified)'}
- Clip duration: {duration_str}
{news_block}{corpus_block}{social_block}
Write the JSON now.
"""
