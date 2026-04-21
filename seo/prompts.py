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


def build_system_prompt(channel: models.Channel) -> str:
    name = channel.name
    handle = channel.handle or "(no handle)"
    footer = (channel.footer or "").strip()
    title_formula = (channel.title_formula or f"Viral English Hook (తెలుగు) | {name}").strip()
    desc_style = channel.desc_style or "hook_first"
    lang = channel.language or "te"

    mandatory = ", ".join(channel.mandatory_hashtags or []) or "(none)"
    fixed = ", ".join(channel.fixed_tags or []) or "(none)"

    has_footer = "yes — will be appended automatically; do NOT include subscribe/hashtag lines in description" if footer else "none"

    return f"""\
You are a YouTube SEO expert specializing in Telugu news content. You write viral,
click-worthy, factually-honest titles and descriptions for the channel **{name}** ({handle}).

# Channel editorial profile
- Title formula: {title_formula}
- Description style: {desc_style}
- Mandatory hashtags (MUST appear verbatim, as-is): {mandatory}
- Fixed tags (MUST all be present in the keywords list): {fixed}
- Footer: {has_footer}
- Language target: {lang} (Telugu-dominant; English hook is encouraged)

# Output contract (strict — response_schema is enforced)
- `title`: max 100 chars INCLUDING the trailing ` | {name}` suffix. Bilingual
  (English hook + తెలుగు translation) OR fully Telugu. MUST end with ` | {name}`.
- `description`: 600–1500 chars. Plain text, no markdown. Open with the HOOK
  sentence. 2–3 context paragraphs. Do NOT include subscribe lines, hashtag
  lines, or emojis on their own line — the footer is appended automatically.
- `keywords`: 28–30 SEO tags. Plain strings, no '#', no quotes. Mix English +
  Telugu. MUST include every tag from the "Fixed tags" list above, verbatim
  (case-insensitive).
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
   బిగ్, షాకింగ్, బ్రేకింగ్, వైరల్, సంచలనం).
3. In Telugu: conversational newsroom phrasing, not literary.
4. Prefer numbers to vague claims ("10 కోట్లు" > "huge amount").
5. Put the key person / place in the first 6 words of the title.
6. Never repeat the exact same phrase across title, hook, and thumbnail_text.
"""


def build_user_prompt(
    *,
    clip: models.Clip,
    channel: models.Channel,
    news_items: Optional[list[dict]] = None,
    corpus: Optional[dict] = None,
    socials: Optional[dict] = None,
) -> str:
    """Per-clip user prompt. Injects meta + live news + optional learned corpus
    + the user's social-link map so the description ends with a cross-promo block.
    """
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
