"""Anthropic Claude integration for Express Mode.

Port of the teammate's three Claude calls:
  - SEO writer       (title / description / tags)
  - Shorts plan      (3-5 best moments with hook + image prompt)
  - Long-form trim   (which segments to keep / drop)

We talk directly to the Anthropic ``/v1/messages`` HTTP endpoint
instead of pulling in the Anthropic SDK — kept this way to match the
teammate's wire calls exactly so prompts are 1:1 verifiable, and to
avoid an extra dep when httpx already covers it.

Session 1 ships SEO + helpers only. ``plan_shorts`` and
``plan_longform_trim`` are stubbed in this module as part of the same
file so Session 2 only has to fill in the prompt body — the request
mechanics and JSON salvage logic are already wired.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import os

import httpx


CLAUDE_URL = "https://api.anthropic.com/v1/messages"
# Default matches teammate's model id, but is overridable via env so
# operators can pin to a specific revision (e.g. claude-sonnet-4-5)
# without redeploying code. The override is resolved per-call to allow
# hot-rotation without a server restart.
_DEFAULT_MODEL = "claude-sonnet-4-6"


def _model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "").strip() or _DEFAULT_MODEL


def _env_key() -> str:
    """Server-side Anthropic key fallback. Used only when the caller
    didn't supply one via the per-request form (Express Mode UI keys
    win because they're per-user)."""
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


ANTHROPIC_VERSION = "2023-06-01"


class ClaudeError(RuntimeError):
    """Claude API call failed or response wasn't parseable JSON."""


def _post(
    *,
    api_key: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int = 120,
) -> str:
    """Single Claude call. Returns ``content[0].text`` or raises.

    Per-request ``api_key`` wins (Express Mode UI keys), with the
    server-side ``ANTHROPIC_API_KEY`` env as fallback for testing.
    """
    effective_key = api_key or _env_key()
    if not effective_key:
        raise ClaudeError("ANTHROPIC_API_KEY missing")
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=15)) as cli:
            r = cli.post(
                CLAUDE_URL,
                headers={
                    "x-api-key":         effective_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type":      "application/json",
                },
                json={
                    "model":      _model(),
                    "max_tokens": max_tokens,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
    except httpx.HTTPError as exc:
        raise ClaudeError(f"network error: {exc}") from exc

    if r.status_code >= 400:
        raise ClaudeError(f"claude HTTP {r.status_code}: {r.text[:600]}")

    try:
        data = r.json()
    except ValueError as exc:
        raise ClaudeError(f"non-JSON response: {r.text[:300]}") from exc

    blocks = data.get("content") or []
    if not blocks:
        raise ClaudeError("claude returned no content blocks")
    text = blocks[0].get("text") or ""
    if not text:
        raise ClaudeError("claude returned empty text")
    return text


# ─── JSON salvage helpers (ports of teammate's extractJsonObject /
#     extractJsonArray / salvageTruncatedKeepPlan) ──────────────────

def _extract_json_object(text: str) -> Optional[dict]:
    """Find the first ``{...}`` block in ``text`` and parse it. Handles
    cases where Claude wraps the JSON in prose/markdown despite the
    'JSON only' instruction."""
    if not text:
        return None
    # Quickest happy path — the text IS the JSON.
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Strip markdown fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Greedy match — first { to last }. Brittle but usually correct
    # for Claude's chatty wrappers.
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _extract_json_array(text: str) -> Optional[list]:
    """Same idea as ``_extract_json_object`` but for ``[...]``."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ─── Public planners ───────────────────────────────────────────────

def write_seo(
    *,
    api_key: str,
    transcript: str,
    brief: str = "",
    names_hint: str = "",
    style_guide: str = "",
    timeout_s: int = 90,
) -> dict:
    """Generate YouTube SEO (title / description / tags) for the video.

    Mirrors the inline SEO writer in ``runAutopubPipeline`` exactly —
    same system framing, same JSON-only contract, same style-block
    composition, same transcript cap (12k chars).
    """
    enriched_style = style_guide or ""
    if brief:
        enriched_style += (
            f"\n\n**AUTHORITATIVE BRIEF (this is what the user says the video is "
            f"about — trust this OVER the noisy transcript when there's a conflict):**"
            f"\n{brief}\n"
        )
    if names_hint:
        enriched_style += (
            f"\n\n**KNOWN SUBJECTS in this video (use these exact names — the "
            f"transcript may have transcription errors that look similar):**"
            f"\n{names_hint}\n"
        )

    style_block = (
        f"STYLE EXAMPLE — match this exactly:\n\"\"\"\n{enriched_style[:4000]}\n\"\"\"\n\n"
        if enriched_style else ""
    )

    prompt = (
        'You are a senior YouTube SEO editor for "Kaizer News".\n'
        "OUTPUT FORMAT: Single JSON object only. Start with { end with }. No prose, no fences.\n"
        'Schema: { "title": string ≤100 chars, "description": string, "tags": string[] }.\n'
        "Title and description MUST include specific named entities (people, places, events) "
        "from the transcript — never generic placeholders.\n\n"
        + style_block
        + f'Transcript:\n"""\n{transcript[:12000]}\n"""'
    )

    text = _post(api_key=api_key, prompt=prompt, max_tokens=2048, timeout_s=timeout_s)
    seo = _extract_json_object(text)
    if not seo:
        raise ClaudeError(
            "SEO response wasn't valid JSON. Raw response: " + text[:300]
        )

    # Defensive normalisation — Claude sometimes returns numbers or
    # missing keys. Downstream code expects strings/list.
    return {
        "title":       str(seo.get("title", "") or "")[:100],
        "description": str(seo.get("description", "") or ""),
        "tags":        [str(t) for t in (seo.get("tags") or []) if t],
    }


# ─── Target-window helpers (port of teammate's targetShortCount /
#     targetTrimWindow). Keep the same per-duration tiers so the
#     UX feels identical side-by-side. ─────────────────────────

def target_short_count(src_sec: float) -> int:
    """How many Shorts to cut for a source of given length."""
    m = float(src_sec) / 60.0
    if m < 3:    return 2
    if m <= 4:   return 3
    if m <= 6:   return 4
    return 5     # 6+ min: cap at 5 to keep production manageable


def target_trim_window(src_sec: float) -> dict:
    """Return ``{min_sec, max_sec, label}`` — the hard runtime cap
    Claude must hit when trimming the long-form."""
    m = float(src_sec) / 60.0
    if m < 3:
        return {"min_sec": round(src_sec * 0.7),
                "max_sec": round(src_sec * 0.9),
                "label":   "short clip — trim 10–30%"}
    if m <= 4:    return {"min_sec": 110, "max_sec": 130, "label": "3–4 min → ~2 min"}
    if m <= 6:    return {"min_sec": 120, "max_sec": 180, "label": "4–6 min → 2–3 min"}
    if m <= 10:   return {"min_sec": 180, "max_sec": 210, "label": "6–10 min → 3–3.5 min"}
    if m <= 20:   return {"min_sec": 240, "max_sec": 480, "label": "10–20 min → 4–8 min"}
    return        {"min_sec": 240, "max_sec": 600, "label": "20+ min → 4–10 min"}


# ─── plan_longform_trim — the Claude trim planner port ─────────────

def _salvage_truncated_keep_plan(text: str) -> Optional[dict]:
    """When Claude's response gets cut off mid-JSON (hitting max_tokens
    on a long transcript), the outer braces are gone but the partial
    ``keep`` array might still be parseable. Find every complete
    ``{start, end, reason}`` triple and assemble them into a plan.

    Direct port of teammate's salvageTruncatedKeepPlan."""
    if not text:
        return None
    # Match {"start": <n>, "end": <n>, "reason": "<...>"} in any order.
    # Use a non-greedy body match and tolerate extra whitespace.
    pattern = re.compile(
        r'\{\s*(?:"start"\s*:\s*(?P<s1>[-\d.]+)\s*,\s*"end"\s*:\s*(?P<e1>[-\d.]+)\s*,'
        r'\s*"reason"\s*:\s*"(?P<r1>[^"]*)"'
        r'|"end"\s*:\s*(?P<e2>[-\d.]+)\s*,\s*"start"\s*:\s*(?P<s2>[-\d.]+)\s*,'
        r'\s*"reason"\s*:\s*"(?P<r2>[^"]*)")\s*\}',
        re.DOTALL,
    )
    keeps: list[dict] = []
    for m in pattern.finditer(text):
        try:
            s = float(m.group("s1") or m.group("s2"))
            e = float(m.group("e1") or m.group("e2"))
            r = (m.group("r1") or m.group("r2") or "")[:80]
        except (TypeError, ValueError):
            continue
        if e > s and e - s >= 0.4:
            keeps.append({"start": s, "end": e, "reason": r})
    if not keeps:
        return None
    return {"keep": keeps, "removedSeconds": 0, "summary": "salvaged from truncated Claude response"}


def plan_longform_trim(
    *,
    api_key: str,
    segments: list[dict],
    language: str,
    duration: float,
    target_min: float,
    target_max: float,
    style_guide: str = "",
    timeout_s: int = 180,
) -> dict:
    """Ask Claude which transcript segments to KEEP for the trimmed
    long-form. Returns ``{keep, removedSeconds, summary}`` matching
    teammate's exactly.

    Post-processes:
      - clamps floats, drops <0.4s micro-fragments
      - sorts by start, merges overlapping/abutting (≤0.2s gap)
      - enforces target window (trims tail when too long, fallback
        expansion when too short)
    """
    if not segments:
        raise ClaudeError("no transcript segments to plan from")

    seg_lines = "\n".join(
        f"[{i}] [{float(s.get('start') or 0):.2f}-{float(s.get('end') or 0):.2f}] "
        f"{(s.get('text') or '').strip()}"
        for i, s in enumerate(segments)
    )

    style_block = (
        f"Channel style hint:\n\"\"\"\n{style_guide[:2000]}\n\"\"\"\n\n"
        if style_guide else ""
    )

    prompt = (
        'You are a video editor for a Telugu news channel ("Kaizer News"). Below is a numbered '
        f'list of transcript segments from a {round(duration or 0)}-second video, each with its '
        'start/end time. Your job: identify which segments to KEEP and which to CUT, producing '
        'a tight, engaging final cut.\n\n'
        '**REMOVE segments that are:**\n'
        '- Dead silence / non-speech\n'
        '- Filler words and pauses ("um", "uh", "ఆ...", "మంటే")\n'
        '- Repetitive content (the same point made twice)\n'
        '- Off-topic asides, trail-offs, mistakes, retakes\n'
        '- Pre-roll / post-roll throat-clearing\n\n'
        '**KEEP segments that are:**\n'
        '- Clear factual statements with named subjects\n'
        '- Strong quotes, reactions, key reveals\n'
        '- Natural transitions between topics\n\n'
        '**TARGET RUNTIME (hard rule):** the SUM of all kept segment durations must land between '
        f'{int(target_min)} and {int(target_max)} seconds (i.e. between '
        f'{target_min/60:.1f} and {target_max/60:.1f} minutes). '
        'If the source is already short, lean toward keeping more; if it\'s long, be aggressive '
        'about cutting. Group consecutive keep-segments into longer kept blocks where possible '
        "(don't split mid-sentence).\n\n"
        '**OUTPUT FORMAT (zero deviation):**\n'
        'Your entire response MUST be a single valid JSON object. No prose. No preamble. '
        'No markdown fences. Start with "{" end with "}".\n'
        'Schema:\n'
        '{\n'
        '  "keep": [{"start": <sec>, "end": <sec>, "reason": "<MAX 20 chars, e.g. \'reveal\' / \'key fact\' / \'transition\'>"}, ...],\n'
        '  "removedSeconds": <number>,\n'
        '  "summary": "<1-line description of what was cut, MAX 100 chars>"\n'
        '}\n'
        '\nKeep "reason" SHORT (≤20 characters). Don\'t describe — just label. '
        'The "keep" array must be SORTED by start ascending, with no overlaps. Times must come '
        'from the timestamps below.\n\n'
        + style_block
        + f'TRANSCRIPT (language={language or "auto"}, {len(segments)} segments, total '
          f'{round(duration or 0)}s):\n"""\n{seg_lines[:35000]}\n"""'
    )

    text = _post(api_key=api_key, prompt=prompt, max_tokens=8192, timeout_s=timeout_s)

    obj = _extract_json_object(text)
    if not obj or not isinstance(obj.get("keep"), list):
        obj = _salvage_truncated_keep_plan(text)
    if not obj or not isinstance(obj.get("keep"), list):
        raise ClaudeError(
            "Claude longform plan was not valid JSON: " + text[:400]
        )

    # ── Sanitise: clamp floats, drop tiny fragments, sort ──
    keep: list[dict] = []
    for s in obj["keep"]:
        try:
            start = max(0.0, float(s.get("start") or 0))
            end   = max(0.0, float(s.get("end") or 0))
        except (TypeError, ValueError):
            continue
        if end - start < 0.4:
            continue
        keep.append({
            "start":  start,
            "end":    end,
            "reason": str(s.get("reason") or "")[:80],
        })
    keep.sort(key=lambda x: x["start"])

    # ── Merge overlapping / abutting (≤0.2s gap) ──
    merged: list[dict] = []
    for s in keep:
        if merged and s["start"] <= merged[-1]["end"] + 0.2:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(dict(s))

    # ── Enforce the target window post-hoc ──
    total = sum(s["end"] - s["start"] for s in merged)
    enforcement = ""

    if target_max and total > target_max:
        while merged and total > target_max:
            last = merged[-1]
            dur = last["end"] - last["start"]
            if total - dur < (target_min or 0):
                shorten = total - target_max
                last["end"] = max(last["start"] + 0.5, last["end"] - shorten)
                total = total - shorten
                break
            merged.pop()
            total -= dur
        enforcement = f" Enforced runtime cap ({target_max/60:.1f} min)."

    elif target_min and total < target_min * 0.7:
        # Fail-safe: Claude over-cut. Add back the longest unused segments
        # farthest from existing kept blocks until we hit min.
        original_kept_sec = total
        all_segs = sorted(
            [{"start": max(0.0, float(s.get("start") or 0)),
              "end":   max(0.0, float(s.get("end") or 0))}
             for s in segments],
            key=lambda x: x["start"],
        )
        all_segs = [s for s in all_segs if s["end"] - s["start"] >= 0.4]

        safety = 0
        while total < target_min and safety < 200:
            safety += 1
            # Find segments not already covered by ``merged``.
            candidates: list[dict] = []
            for s in all_segs:
                covered = any(
                    s["start"] >= k["start"] - 0.2 and s["end"] <= k["end"] + 0.2
                    for k in merged
                )
                if not covered:
                    candidates.append(s)
            if not candidates:
                break
            # Score: longer + farther from existing kept (so they don't
            # re-merge in).
            scored: list[tuple[float, dict]] = []
            for s in candidates:
                nearest = float("inf")
                for k in merged:
                    nearest = min(nearest, abs(s["start"] - k["end"]),
                                  abs(s["end"] - k["start"]))
                score = (s["end"] - s["start"]) + min(nearest, 30) * 0.5
                scored.append((score, s))
            scored.sort(key=lambda t: -t[0])
            pick = scored[0][1]
            merged.append({"start": pick["start"], "end": pick["end"],
                           "reason": "fallback (auto-expanded)"})
            # Re-merge from scratch.
            merged.sort(key=lambda x: x["start"])
            re_merged: list[dict] = []
            for s in merged:
                if re_merged and s["start"] <= re_merged[-1]["end"] + 0.2:
                    re_merged[-1]["end"] = max(re_merged[-1]["end"], s["end"])
                else:
                    re_merged.append(dict(s))
            merged = re_merged
            total = sum(s["end"] - s["start"] for s in merged)

        enforcement = (
            f" Fallback: Claude under-cut to {original_kept_sec:.0f}s; "
            f"expanded to {total:.0f}s to meet {target_min/60:.1f}-min minimum."
        )

    elif target_min and total < target_min:
        enforcement = (
            f" Note: runtime {total/60:.1f} min is just under target "
            f"{target_min/60:.1f} min."
        )

    return {
        "keep":           merged,
        "removedSeconds": float(obj.get("removedSeconds") or 0),
        "summary":        (str(obj.get("summary") or "") + enforcement)[:300],
    }


# ─── plan_shorts — Claude picks the 3-5 best moments ──────────────

def plan_shorts(
    *,
    api_key: str,
    segments: list[dict],
    language: str,
    duration: float,
    count: int,
    style_guide: str = "",
    timeout_s: int = 180,
) -> list[dict]:
    """Direct port of teammate's planShortsWithClaude.

    Asks Claude to pick ``count`` engaging 15-60s moments. Each
    returned clip has:
      - start, end (sec, from the provided segments)
      - title, description, tags (YouTube SEO for that short)
      - subject (2-5 word English label naming the specific subject)
      - hook (2-line Telugu with ``*bomb*`` markers, total 6-9 words)
      - imagePrompt (English prompt for gpt-image-1)

    Post-processes:
      - drops items whose end-start is outside [15, 60]
      - clamps start/end to [0, duration]
      - normalises tags to a list of strings
    """
    if not segments:
        raise ClaudeError("no transcript segments to plan from")

    seg_lines = "\n".join(
        f"[{float(s.get('start') or 0):.1f}-{float(s.get('end') or 0):.1f}] "
        f"{(s.get('text') or '').strip()}"
        for s in segments
    )

    style_block = (
        'MATCH THIS STYLE for language, tone, hashtag usage, branding, '
        'channel-info layout:\n"""\n'
        + style_guide[:4000]
        + '\n"""\n\n'
        if style_guide else ""
    )

    prompt = (
        'You are a YouTube Shorts editor for a Telugu news channel '
        '("KAIZER NEWS"). The transcript below comes from a long video. '
        f'Pick the {count} most engaging, self-contained moments — each '
        'between 15 and 60 seconds long — that would each work as a YouTube Short.\n\n'
        '**CRITICAL — NAMED ENTITY EXTRACTION:**\n'
        'Read the transcript carefully and identify EVERY proper noun mentioned: people '
        '(e.g. "Preethi Reddy", "Malla Reddy", "Modi"), places ("Hyderabad", "Telangana"), '
        'events ("AI Initiative"), organizations ("BJP", "Congress"), specific projects.\n'
        'EVERY title, description, hook, subject, imagePrompt, and tag below MUST be '
        'specific to the actual named subjects. Never use generic placeholders like '
        '"ముఖ్యమైన పరిణామం" / "police matter" / "an Indian politician" — always name '
        'the specific person, place, or event from the transcript.\n\n'
        '\n**OUTPUT FORMAT (zero deviation allowed):**\n'
        'Your entire response must be a single valid JSON array. No prose. No '
        'explanation. No markdown fences. No "I will analyze..." preamble. No "Here '
        'is the JSON..." preamble. If the transcript looks garbled or wrong, STILL '
        'produce the JSON array using your best guess at the topic — never refuse. '
        'Start your output with the character "[" and end with "]". Anything else '
        'and the response is invalid.\n\n'
        'Each item in the JSON array must have:\n'
        '- "start": number (seconds, must come from the timestamps below)\n'
        '- "end": number (seconds, must come from the timestamps below; end - start '
        'MUST be between 15 and 60)\n'
        '- "title": string, max 100 chars, optimized for Shorts\n'
        '- "description": string, the YouTube description for this short\n'
        '- "tags": array of 8-15 string tags (no # symbol)\n'
        '- "subject": A 2-5 word English label naming the SPECIFIC central subject '
        'of this short — the actual person, place, event, brand, or thing the moment '
        'is about. If a named public figure is the focus, write their well-known '
        'English name (e.g. "Mangli, Telugu folk singer", "Pawan Kalyan", "KCR", '
        '"Sridevi", "Tirupati temple", "Hyderabad metro launch"). Do NOT write a '
        'generic placeholder like "a singer" or "a politician".\n'
        '- "hook": The Telugu thumbnail headline burned onto the short, formatted '
        'as EXACTLY 2 lines joined by "\\n". Strict layout rules (do NOT violate, '
        'the panel width is fixed):\n'
        '   • Total length: 6-9 Telugu words across both lines combined.\n'
        '   • Line 1: 3-5 words AND ≤ 18 characters including spaces.\n'
        '   • Line 2: 3-5 words AND ≤ 22 characters including spaces.\n'
        '   • Telugu script only — no English, no digits, no hashtags, no quotes.\n'
        '   • Line 1 = subject/setup ; Line 2 = punch/reveal '
        '(often ends with !, ?, or a strong verb).\n'
        '   • **EACH HOOK MUST BE UNIQUE PER SHORT** — never copy the same hook '
        'across multiple shorts. The hook must reflect the specific moment of THIS short.\n'
        '   • **VALID TELUGU ONLY** — only use real, dictionary-valid Telugu words. '
        'The transcript may contain Whisper transcription errors (garbled or non-words '
        'like "ప్రత్తను" instead of "ప్రియాంక"). If a transcribed word looks suspicious '
        'or isn\'t a real Telugu word, DO NOT copy it into the hook — substitute with '
        'a generic but topical Telugu news term: e.g. "నాయకురాలు" (leader), '
        '"నాయకుడు" (male leader), "మంత్రి" (minister), "ఎంపీ" (MP), "నేత" (politician), '
        '"వ్యవహారం" (matter), "వార్తలు" (news), "పరిణామం" (development), '
        '"ప్రకటన" (announcement). Better to use a generic descriptor than a garbled name.\n'
        '   • **BOMB WORD HIGHLIGHT**: wrap exactly 1-2 most impactful "punch" words '
        'with single asterisks like *word*. These will be coloured yellow on top of the '
        'otherwise-white text. Aim for ~60% white / 40% yellow word ratio. The bomb '
        'word is usually the dramatic/shocking/reveal word (e.g. *షాకింగ్*, *కౌంటర్*, '
        '*వార్నింగ్*, *ఎక్స్‌క్లూజివ్*) — never the connector/article words.\n'
        'Example (* marks the bomb word): "*షాకింగ్* నిజం\\nపోలీసుల ముందు *బయట*!"\n'
        'Example: "కేంద్ర మంత్రి కిషన్ రెడ్డికి\\nఎంపీ చామల కిరణ్ *కౌంటర్*"\n'
        '- "imagePrompt": string, an English-only prompt for OpenAI gpt-image-1. '
        'Carefully read the transcript to identify exactly WHO/WHAT this short is '
        'about, and write a vivid 1-2 sentence prompt that describes THAT specific '
        'subject — not a generic stand-in. If the central subject is a named public '
        'figure (singer, actor, politician, sportsperson, religious leader), name '
        'them by their well-known name and describe their distinctive recognizable '
        'features so the image actually looks like them: face shape, hair, typical '
        'attire, characteristic expression, the setting they\'re known for. '
        'If the subject is a place/event/object, describe it specifically with '
        'recognizable visual details (architecture, region, props). '
        'Style guidance: photorealistic Telugu-news-channel thumbnail look, dramatic '
        'saturated colors, intense expression, news-graphic energy. STRICT: do NOT '
        'request any text/captions/percentages/logos/watermarks inside the image — '
        'that\'s added separately. 1-3 sentences max.\n\n'
        + style_block
        + f'TRANSCRIPT (language={language or "auto"}, total {round(duration or 0)}s):\n'
          f'"""\n{seg_lines[:30000]}\n"""'
    )

    text = _post(api_key=api_key, prompt=prompt, max_tokens=8192, timeout_s=timeout_s)
    plan = _extract_json_array(text)
    if plan is None:
        raise ClaudeError(
            "Claude shorts plan was not valid JSON. Likely the transcript came back "
            "garbled — try Groq Whisper Large v3 or set language to 'te'. "
            "Raw: " + text[:300]
        )

    cleaned: list[dict] = []
    for it in plan:
        if not isinstance(it, dict):
            continue
        try:
            start = max(0.0, float(it.get("start") or 0))
            end   = max(0.0, float(it.get("end") or 0))
        except (TypeError, ValueError):
            continue
        if duration > 0:
            end = min(end, duration)
        dur = end - start
        if not (14.5 <= dur <= 60.5):    # tolerate small float drift
            continue
        cleaned.append({
            "start":       start,
            "end":         end,
            "title":       str(it.get("title") or "")[:100],
            "description": str(it.get("description") or ""),
            "tags":        [str(t) for t in (it.get("tags") or []) if t],
            "subject":     str(it.get("subject") or "")[:120],
            "hook":        str(it.get("hook") or "")[:240],
            "imagePrompt": str(it.get("imagePrompt") or "")[:1200],
        })
    if not cleaned:
        raise ClaudeError("Claude returned a JSON array but none of its clips passed validation")
    return cleaned
