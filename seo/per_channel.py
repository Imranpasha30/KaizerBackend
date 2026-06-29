"""Per-channel SEO adaptation — Option B ("base + adapt").

Two SEO modes (operator decision 2026-06-16):
  * shared      — ONE SEO for all channels (current behaviour, FREE tier default).
                  Title/tags shared; per-channel social links injected by the
                  composer. Unchanged — free users are unaffected.
  * per_channel — PAID. Generate one strong BASE SEO once, then cheaply ADAPT it
                  per channel using that channel's feedback profile (winning
                  keywords from past performance), its voice (style-preference)
                  and its socials. No extra full AI generation per channel, so it
                  stays cheap; each channel's SEO can then improve from its own
                  results (the feedback loop).

`adapt_seo_for_channel` is a PURE function (no AI call, no network) so adapting
N channels costs ~nothing. It also returns a ``why`` breakdown that powers the
per-channel preview ("what is sent to each channel and why"). Until a channel
has feedback data it gracefully falls back to the base SEO (advisory, safe).
"""
from __future__ import annotations

from typing import Any, Optional


def _norm_tags(tags) -> list[str]:
    out, seen = [], set()
    for t in (tags or []):
        s = str(t).strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def adapt_seo_for_channel(
    *,
    base_seo: dict[str, Any],
    channel: Any = None,
    feedback: Optional[dict[str, Any]] = None,
    content_keywords: Optional[list[str]] = None,
    max_tags: int = 15,
) -> dict[str, Any]:
    """Tailor a base SEO to ONE channel. Returns
    ``{title, description, tags, why, source}`` where ``why`` is a list of
    human-readable reasons (for the preview) and ``source`` is 'base' |
    'adapted'. Never raises — on any issue it returns the base SEO verbatim."""
    try:
        base_title = (base_seo.get("title") or "").strip()
        base_desc = (base_seo.get("description") or "").strip()
        base_tags = _norm_tags(base_seo.get("tags"))
        fb = feedback or {}
        win_kw = [str(k).strip() for k in (fb.get("winning_keywords") or []) if str(k).strip()]
        content_keywords = [str(k).strip() for k in (content_keywords or []) if str(k).strip()]

        why: list[str] = []
        ch_name = getattr(channel, "name", None) or (channel.get("name") if isinstance(channel, dict) else None) or "this channel"

        # ── Relevance gate ──────────────────────────────────────────────
        # A channel's winning keyword is only carried onto THIS video when it
        # fits the current story. Generic/beat terms a news video already uses
        # ("telugu news", "breaking") pass; a stale topical term from an old
        # story (a person/place) is DROPPED unless this video is about it too —
        # so e.g. "pawan kalyan" never lands on an unrelated new clip.
        _blob = f"{base_title} {base_desc} {' '.join(content_keywords)}".lower()
        def _relevant(kw: str) -> bool:
            k = kw.lower()
            if " " in k:
                return k in _blob or all(p in _blob for p in k.split())
            return k in _blob
        relevant_win = [k for k in win_kw if _relevant(k)]

        # ── TAGS: base + the RELEVANT winning keywords + content keywords ──
        tags = _norm_tags(base_tags + relevant_win + content_keywords)[:max_tags]
        if relevant_win:
            why.append(
                f"Reinforced {min(len(relevant_win), 5)} of {ch_name}'s proven "
                f"keyword(s) that fit this video: {', '.join(relevant_win[:5])}"
            )
        elif win_kw:
            why.append(
                f"{ch_name}'s top terms ({', '.join(win_kw[:3])}) were from other "
                f"stories — skipped as off-topic for this video"
            )
        elif content_keywords and len(base_tags) < 8:
            why.append("Filled out tags from the video's own keywords")

        # ── TITLE: never inject a learned term. The headline stays driven by
        # THIS video's content; the channel's brand is appended by the composer
        # at publish. Stamping an old top term onto an unrelated headline is the
        # exact bug for news, so we keep the base title untouched.
        title = base_title
        description = base_desc

        adapted = (tags != base_tags) or bool(relevant_win)
        if not why:
            why.append(f"Using the base SEO for {ch_name} (no channel-specific signal yet)")

        return {
            "title": title,
            "description": description,
            "tags": tags,
            "why": why,
            "source": "adapted" if adapted else "base",
        }
    except Exception:
        # Fail-closed to the base SEO — per-channel adaptation can never break a publish.
        return {
            "title": (base_seo or {}).get("title", ""),
            "description": (base_seo or {}).get("description", ""),
            "tags": _norm_tags((base_seo or {}).get("tags")),
            "why": ["Base SEO (adaptation skipped)"],
            "source": "base",
        }


def generate_channel_seo(
    *,
    base_seo: dict[str, Any],
    channel: Any,
    content_text: str = "",
    language: str = "te",
    winning_keywords: Optional[list[str]] = None,
    content_keywords: Optional[list[str]] = None,
    max_tags: int = 15,
    variation_index: int = 0,
    avoid_titles: Optional[list[str]] = None,
) -> dict[str, Any]:
    """FULL per-channel SEO: write a DISTINCT title/description/tags for ONE
    channel using the real SEO generator (the channel's own voice + its proven
    winning keywords steer the headline), then score it for SEO/CTR quality.

    This is the "use all the SEO + CTR tools per channel" path: each channel
    gets a genuinely different headline (not the shared base title). FAIL-CLOSED
    on ANY error to ``adapt_seo_for_channel`` (the cheap tags-only adapt) so a
    preview/publish is never blocked.
    """
    winning_keywords = winning_keywords or []
    content_keywords = content_keywords or []
    # Safe baseline (relevance-gated tags + 'why') — also our fallback.
    base = adapt_seo_for_channel(
        base_seo=base_seo, channel=channel,
        feedback={"winning_keywords": winning_keywords},
        content_keywords=content_keywords, max_tags=max_tags,
    )
    try:
        from pipeline_v4.seo_provider import generate_seo, SeoInput
        from seo.score_checker import score_seo

        ch_name = (getattr(channel, "name", None)
                   or (channel.get("name") if isinstance(channel, dict) else None)
                   or "this channel")
        bt = (base_seo.get("title") or "")
        bd = (base_seo.get("description") or "")
        blob = f"{bt} {bd} {' '.join(content_keywords)}".lower()
        # Relevance gate (same rule as the tag adapter) so we only steer with a
        # channel's proven term when it actually fits THIS video.
        def _rel(kw: str) -> bool:
            k = kw.lower()
            return (k in blob) if " " not in k else (k in blob or all(p in blob for p in k.split()))
        rel = [k for k in winning_keywords if _rel(k)]
        steer = ""
        if rel:
            steer = ("\n\nThis channel's proven high-performing terms — weave the "
                     "ones that fit naturally into the title/description: "
                     + ", ".join(rel[:8]))
        # ── Anti-duplicate ANGLE (the dedup bypass that actually works) ──────
        # When two channels share the same base SEO + the same (empty) voice,
        # the generator converges to an IDENTICAL headline — YouTube then flags
        # the videos as duplicates across channels, which is the whole thing we
        # publish per-channel SEO to avoid. So we force a DISTINCT structural
        # angle per channel (deterministic by its position in the channel list)
        # plus an explicit "do not echo the reference title" instruction, so the
        # headlines diverge by construction, not by luck.
        _ANGLES = [
            "Lead with raw shock / emotion (a punchy 'Shocking:' / 'Sensational:' opener).",
            "Lead with the key PERSON's name + the bold action they took.",
            "Frame it as a provocative QUESTION the viewer must click to answer.",
            "Lead with the CONSEQUENCE / impact ('What this really means for…').",
            "Use a CURIOSITY-GAP teaser that withholds the punchline.",
            "Lead with the PLACE / event, then the twist.",
            "Open with a bold NUMBER or list framing ('3 things…').",
            "Lead with a direct QUOTE-style hook in the subject's voice.",
        ]
        angle = _ANGLES[variation_index % len(_ANGLES)]
        steer += (
            "\n\nIMPORTANT — this video is published to MULTIPLE channels, so this "
            "channel's title MUST be worded differently from the others to avoid "
            "YouTube duplicate-content flags. Write a headline with a DISTINCT "
            "structure unique to this channel using this angle: " + angle +
            " Rephrase substantially — do NOT echo the reference/base title verbatim, "
            "and vary the opening words. Keep it accurate and in the target language."
        )
        # The angle alone isn't enough when a strong base hook (e.g. 'Shocking:')
        # magnetises every channel to the same opening. So we also hand this
        # channel the titles ALREADY taken by sibling channels and forbid reusing
        # their opening — the only reliable way to guarantee different openers.
        _avoid = [str(t).strip() for t in (avoid_titles or []) if str(t).strip()]
        if _avoid:
            steer += (
                "\n\nThese titles are ALREADY used by other channels for this same "
                "video — yours must NOT start with the same word(s) and must read "
                "clearly differently from every one of them:\n- "
                + "\n- ".join(_avoid[:8])
            )
        inp = SeoInput(
            kind="bulletin",
            language=language or "te",
            title_native=bt[:200],
            summary=bd[:500],
            body=((content_text or "") + steer)[:1800],
        )
        # KEEP IMPROVING until this channel's SEO scores 85+ (bounded passes). The helper
        # re-generates feeding the score-checker's suggestions back in, returns the best
        # attempt + its seo_score/seo_attempts, and never raises (empty title -> fall back
        # to the safe adapted base, so a rate limit never collapses to the shared title).
        from pipeline_v4.seo_provider import generate_seo_to_score
        gen = generate_seo_to_score(inp, style_source=channel, target_score=85, max_attempts=3)
        gtitle = (gen.get("title") or "").strip()
        if not gtitle:
            return base
        gdesc = (gen.get("description") or "").strip() or base.get("description", "")
        gtags = _norm_tags(list(gen.get("keywords") or []) + list(base.get("tags") or []))[:max_tags]
        ghash = [str(h).strip() for h in (gen.get("hashtags") or []) if str(h).strip()][:12]
        seo_score = gen.get("seo_score")
        attempts = gen.get("seo_attempts") or 1

        why = [f"Wrote a distinct headline for {ch_name} with the full SEO engine"
               + (f" + its proven terms ({', '.join(rel[:3])})" if rel else "")]
        if seo_score is not None:
            why.append(f"SEO/CTR score {seo_score}/100"
                       + (f" — improved over {attempts} passes" if attempts > 1 else ""))
        return {
            "title": gtitle,
            "description": gdesc,
            "tags": gtags,
            "hashtags": ghash,
            "why": why,
            "source": "generated",
            "seo_score": seo_score,
            "seo_attempts": attempts,
        }
    except Exception:
        return base


def build_channel_previews(
    *,
    base_seo: dict[str, Any],
    channels: list[Any],
    feedback_by_channel: Optional[dict[Any, dict]] = None,
    content_keywords: Optional[list[str]] = None,
    mode: str = "per_channel",
    generate: bool = False,
    content_text: str = "",
    language: str = "te",
) -> list[dict[str, Any]]:
    """Produce the per-channel SEO preview list: for each channel, the exact
    title/description/tags that will be sent + why. In 'shared' mode every
    channel shows the identical base SEO (no adaptation) so the preview is
    honest about which mode is active."""
    feedback_by_channel = feedback_by_channel or {}
    out: list[dict[str, Any]] = []
    _generated_titles: list[str] = []   # titles already taken by earlier channels (dedup)
    for _idx, ch in enumerate(channels):
        cid = getattr(ch, "id", None) if not isinstance(ch, dict) else ch.get("id")
        name = getattr(ch, "name", None) if not isinstance(ch, dict) else ch.get("name")
        if mode == "shared":
            out.append({
                "channel_id": cid, "channel_name": name,
                "title": (base_seo.get("title") or ""),
                "description": (base_seo.get("description") or ""),
                "tags": _norm_tags(base_seo.get("tags")),
                "why": ["Shared mode: one SEO for all channels (free)"],
                "source": "base",
            })
        elif generate:
            a = generate_channel_seo(
                base_seo=base_seo, channel=ch,
                content_text=content_text, language=language,
                winning_keywords=(feedback_by_channel.get(cid) or {}).get("winning_keywords") or [],
                content_keywords=content_keywords,
                variation_index=_idx,   # distinct headline angle per channel → no cross-channel dupes
                avoid_titles=list(_generated_titles),   # forbid reusing siblings' openings
            )
            if a.get("source") == "generated" and (a.get("title") or "").strip():
                _generated_titles.append(a["title"].strip())
            a.update({"channel_id": cid, "channel_name": name})
            out.append(a)
        else:
            a = adapt_seo_for_channel(
                base_seo=base_seo, channel=ch,
                feedback=feedback_by_channel.get(cid),
                content_keywords=content_keywords,
            )
            a.update({"channel_id": cid, "channel_name": name})
            out.append(a)
    return out
