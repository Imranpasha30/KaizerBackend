"""V4 SEO provider — generates YouTube-grade title / description /
hashtags / keywords per short and per bulletin.

Reuses V2's SEO system prompt (``seo/prompts.py:build_system_prompt``)
and the same Gemini 2.5 Flash model so the editorial voice matches
what the team is already shipping on V2 jobs. V4 doesn't have a
``Clip`` row to feed into ``seo/generator.generate_seo_for_clip``, so
this module builds the user-prompt and parses Gemini's reply directly.

The returned dict has the same shape as ``clip.seo`` so the publish
flow (``routers/youtube_upload.py::_compose_metadata``) can consume it
without changes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_MODEL = os.environ.get("KAIZER_SEO_MODEL", "gemini-2.5-flash")


@dataclass
class SeoInput:
    """One unit of content to SEO. Short or bulletin — same shape."""
    kind: str               # "short" | "bulletin"
    language: str           # ISO code
    title_native: str
    title_english: str = ""
    summary: str = ""
    body: str = ""          # extra context (joined headlines for bulletin)


def _build_user_prompt(inp: SeoInput) -> str:
    facts = [
        f"language: {inp.language}",
        f"output kind: {inp.kind}",
    ]
    if inp.title_native:
        facts.append(f"native-script title: {inp.title_native}")
    if inp.title_english:
        facts.append(f"English title: {inp.title_english}")
    if inp.summary:
        facts.append(f"summary: {inp.summary}")
    if inp.body:
        facts.append(f"additional context:\n{inp.body[:1500]}")
    return (
        "Generate channel-agnostic YouTube SEO for the news clip below.\n\n"
        + "\n".join(facts)
        + "\n\nRespond as ONE JSON object with keys: title, description, "
          "keywords (array of 28-30 strings), hashtags (array of 10-12 "
          "strings each starting with #), hook, thumbnail_text, "
          "metadata (object: sentiment, category, viral_score). "
          "No surrounding prose. Strict JSON."
    )


def _strip_json_fence(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        # ```json\n...\n```  or  ```\n...\n```
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _empty_seo(language: str) -> dict:
    """A safe-default SEO blob so downstream code never sees a None."""
    return {
        "title": "",
        "description": "",
        "keywords": [],
        "hashtags": [],
        "hook": "",
        "thumbnail_text": "",
        "metadata": {"sentiment": "", "category": "", "viral_score": 0},
        "language": language,
        "model": "",
        "edited_by_user": False,
    }


def generate_seo(inp: SeoInput, *, model_name: Optional[str] = None) -> dict:
    """Single-shot SEO generation. Returns the V2-compatible dict on
    success, an empty-but-valid skeleton on failure. Never raises so
    the caller's render path is never blocked on SEO."""
    try:
        from seo.generator import _gemini_client
        from seo.prompts import build_system_prompt
        from google.genai import types as genai_types
    except Exception as exc:
        print(f"[v4/seo] SDK unavailable, skipping: {exc}", flush=True)
        return _empty_seo(inp.language)

    try:
        client = _gemini_client()
    except Exception as exc:
        print(f"[v4/seo] Gemini key missing, skipping: {exc}", flush=True)
        return _empty_seo(inp.language)

    sys_prompt = build_system_prompt(language=inp.language)
    user_prompt = _build_user_prompt(inp)
    model = model_name or DEFAULT_MODEL

    try:
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
    except Exception as exc:
        print(f"[v4/seo] Gemini call failed: {exc}", flush=True)
        return _empty_seo(inp.language)

    raw = ""
    try:
        raw = (resp.text or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return _empty_seo(inp.language)

    try:
        data = json.loads(_strip_json_fence(raw))
    except Exception as exc:
        print(f"[v4/seo] JSON parse failed: {exc} | raw[:200]={raw[:200]}",
              flush=True)
        return _empty_seo(inp.language)

    seo = _empty_seo(inp.language)
    seo.update({
        "title":          str(data.get("title", "")).strip(),
        "description":    str(data.get("description", "")).strip(),
        "keywords":       [str(k).strip() for k in (data.get("keywords") or [])
                           if str(k).strip()][:30],
        "hashtags":       [str(h).strip() for h in (data.get("hashtags") or [])
                           if str(h).strip()][:12],
        "hook":           str(data.get("hook", "")).strip(),
        "thumbnail_text": str(data.get("thumbnail_text", "")).strip(),
        "metadata":       dict(data.get("metadata") or {}),
        "model":          model,
    })
    return seo
