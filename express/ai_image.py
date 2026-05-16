"""OpenAI gpt-image-1 wrapper.

Direct port of the teammate's ``openaiGenerateImage`` plus the
styled-prompt wrapper from ``/api/shorts/generate-image``.

Two public entry points:

  ``generate_short_inset(api_key, raw_prompt, output_path, ...)``
      → styled Telugu-news-channel thumbnail, 1536×1024 PNG. Used as
        the inset photo for a Shorts cut. Hard 60s timeout matches
        the teammate's autopub default — falls back to None on timeout
        so the caller can use a video frame instead.

  ``generate_thumbnail(api_key, raw_prompt, output_path, ...)``
      → styled long-form YouTube thumbnail, 1536×1024 PNG. Used for
        the AI-trimmed long-form video. Slightly longer timeout (90s)
        because thumbnails are more visible and we'd rather wait than
        ship a video without one.

Both return the output path on success, or None on failure. Errors
are logged but never raised — image generation is "best effort" in
the pipeline (Claude already wrote SEO and we have video frames as
fallback for shorts).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import httpx


_URL = "https://api.openai.com/v1/images/generations"
_MODEL = "gpt-image-1"


class AIImageError(RuntimeError):
    """OpenAI image generation failed or timed out."""


def _generate(
    *,
    api_key: str,
    prompt: str,
    size: str = "1536x1024",
    quality: str = "medium",
    timeout_s: int = 90,
) -> bytes:
    """Raw OpenAI call. Returns PNG bytes or raises AIImageError.

    Direct wire mirror of teammate's openaiGenerateImage. Per-request
    key wins; falls back to ``OPENAI_API_KEY`` env."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AIImageError("OPENAI_API_KEY missing")
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=15)) as cli:
            r = cli.post(
                _URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":   _MODEL,
                    "prompt":  prompt,
                    "n":       1,
                    "size":    size,
                    "quality": quality,
                },
            )
    except httpx.TimeoutException as exc:
        raise AIImageError(f"openai timed out after {timeout_s}s") from exc
    except httpx.HTTPError as exc:
        raise AIImageError(f"openai network error: {exc}") from exc

    if r.status_code >= 400:
        raise AIImageError(f"openai HTTP {r.status_code}: {r.text[:600]}")

    try:
        data = r.json()
    except ValueError as exc:
        raise AIImageError(f"openai returned non-JSON: {r.text[:300]}") from exc

    arr = data.get("data") or []
    if not arr:
        raise AIImageError("openai response had empty data array")
    b64 = arr[0].get("b64_json") or ""
    if not b64:
        raise AIImageError("openai response missing b64_json")
    try:
        return base64.b64decode(b64)
    except (ValueError, TypeError) as exc:
        raise AIImageError(f"could not decode b64 image: {exc}") from exc


# ─── Styled prompt wrapper (Shorts inset) ──────────────────────────

_SHORTS_INSET_STYLE = (
    "Generate a high-impact Telugu news channel thumbnail image inspired by the styling of "
    "BIG TV Telugu, NTV, ABN Andhra Jyothy, TV9, and Republic Bharat — bold, saturated, "
    "cinematic, attention-grabbing news graphic.\n\n"
    "SUBJECT / SCENE:\n{prompt}\n\n"
    "STYLE:\n"
    "- Vivid saturated palette (deep reds, oranges, yellows, blacks, whites). Strong dramatic contrast.\n"
    "- Photorealistic faces and props, sharp focus on subjects, professional cinematic studio lighting.\n"
    "- Dramatic vignette, slight rim light on subjects, faint motion-blur energy in the background.\n"
    "- Subjects feel intense, serious, news-anchor-style framing.\n"
    "- Background: simple, slightly out-of-focus, supports the subject (e.g. press conference flags, "
    "blurred newsroom, crowd silhouette, government building, microphones).\n\n"
    "COMPOSITION:\n"
    "- 16:9 widescreen framing assumed; subjects clearly visible in the centre.\n"
    "- Leave the BOTTOM 35% of the frame visually calmer / darker / less detailed so an external "
    "text overlay sits cleanly on top.\n\n"
    "STRICT RULES (must follow):\n"
    "- NO text inside the image. No captions, no headlines, no Telugu/English words painted in.\n"
    "- NO watermarks, NO channel logos, NO percentages, NO brand marks, NO on-screen graphics.\n"
    "- NO subtitles, NO chyrons, NO speech bubbles.\n"
    "- If a specific named subject is described above (a singer, actor, politician, place, event), "
    "produce an image that visually MATCHES that specific subject — render their recognizable features "
    "(face shape, hair, typical attire, characteristic expression) and the setting they're known for. "
    "Do NOT swap the subject for a random generic stand-in.\n"
    "- High color saturation, news-graphic energy, dramatic but tasteful."
)


def generate_short_inset(
    *,
    api_key: str,
    raw_prompt: str,
    output_path: str,
    size: str = "1536x1024",
    quality: str = "medium",
    timeout_s: int = 60,
) -> Optional[str]:
    """Generate a Shorts inset photo. Wraps ``raw_prompt`` with the
    BIG-TV-news styling so even minimal Claude prompts produce a
    dramatic news image. Returns the saved path or None on failure
    (caller falls back to video frame)."""
    if not (raw_prompt or "").strip():
        return None
    styled = _SHORTS_INSET_STYLE.format(prompt=(raw_prompt or "").strip())
    try:
        buf = _generate(
            api_key=api_key, prompt=styled,
            size=size, quality=quality, timeout_s=timeout_s,
        )
    except AIImageError as exc:
        print(f"[ai_image] inset gen failed: {exc}")
        return None
    try:
        Path(output_path).write_bytes(buf)
    except OSError as exc:
        print(f"[ai_image] inset save failed: {exc}")
        return None
    return output_path


# ─── Long-form thumbnail (AI Trim) ─────────────────────────────────

_THUMBNAIL_STYLE = (
    'Create a polished 16:9 YouTube thumbnail for a Telugu news video titled: "{title}".'
    "{brief_block}{names_block}\n\n"
    "STYLE: Bold dramatic Telugu news channel thumbnail. Vivid red/yellow/navy/white palette, "
    "cinematic lighting, photorealistic.\n\n"
    "STRICT: NO text/captions/logos painted in."
)


def generate_thumbnail(
    *,
    api_key: str,
    title: str,
    brief: str = "",
    names_hint: str = "",
    output_path: str,
    size: str = "1536x1024",
    quality: str = "medium",
    timeout_s: int = 90,
) -> Optional[str]:
    """Generate a long-form YouTube thumbnail for the trimmed video.
    Mirrors the prompt the teammate's autopub builds when
    ``thumbnailStrategy`` is 'same-as-inset' with no shared inset
    (the regenerate-fresh path)."""
    brief_block = ""
    if (brief or "").strip():
        brief_block = f"\n\nAuthoritative brief: {brief.strip()[:600]}"
    names_block = ""
    if (names_hint or "").strip():
        names_block = f"\n\nKnown subjects: {names_hint.strip()}"

    prompt = _THUMBNAIL_STYLE.format(
        title=(title or "").strip() or "Telugu news headline",
        brief_block=brief_block,
        names_block=names_block,
    )
    try:
        buf = _generate(
            api_key=api_key, prompt=prompt,
            size=size, quality=quality, timeout_s=timeout_s,
        )
    except AIImageError as exc:
        print(f"[ai_image] thumbnail gen failed: {exc}")
        return None
    try:
        Path(output_path).write_bytes(buf)
    except OSError as exc:
        print(f"[ai_image] thumbnail save failed: {exc}")
        return None
    return output_path
