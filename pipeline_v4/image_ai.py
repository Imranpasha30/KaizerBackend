"""V4 story-image generator — Nano Banana via Vertex AI.

Sister module to ``thumbnail_ai.py``. Where the thumbnail is a single
high-CTR click-bait composition for the YouTube thumbnail, the story
images here are the B-roll / sidebar / carousel imagery shown next to
the talking head while the news plays.

Priority chain for "give me a story image":
    1. User-uploaded image in the job's _pool/ (always wins)
    2. AI-generated via Nano Banana (this module — new)
    3. Web search (V1's CSE -> DDG -> Pexels chain in image_provider)
    4. Generated news-card fallback (V1's PIL renderer)

Why this module instead of always falling back to web search:
  - Free reign over composition / aspect ratio
  - No third-party rate limits
  - Bills to your GCP credits (same Vertex path as thumbnails)
  - Iterative tweak path: rewrite the previous prompt with a small
    change instead of starting from scratch on every retry
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_IMAGE_MODEL  = os.environ.get("KAIZER_NANO_BANANA_MODEL",       "gemini-2.5-flash-image")
DEFAULT_PROMPT_MODEL = os.environ.get("KAIZER_NANO_BANANA_PROMPT_MODEL", "gemini-2.5-flash")


# Sidebar images are NOT thumbnails — no big shout text, no garish red
# CTA palette. They're cinematic B-roll the viewer sees next to the
# anchor while the story plays. Documentary / news-magazine vibe.
_PROMPT_WRITER_SYSTEM = """\
You are a news-broadcast art director writing image prompts for the
B-roll / sidebar slot that plays next to the news anchor while a story
is read. Given a news clip's headline + summary, write ONE image
prompt for a single sidebar image.

Rules:
1. Output a single prompt, no preamble, no bullet points, no JSON.
2. 16:9 cinematic news B-roll aesthetic. Documentary photography vibe.
3. NO large overlay text. NO shout headlines. The sidebar is contextual
   imagery — text comes from the news strap/ticker, not the picture.
4. Faces of REAL public figures should be replaced with generic
   silhouettes, back-of-head shots, or environmental scene-setters.
   Never name a real person.
5. For crime / scam / fraud stories: prefer evocative symbols
   (handcuffs, evidence bags, court house, vault, gold bars) over
   trying to depict the criminal or victim directly.
6. For political / policy stories: prefer parliament buildings, voting
   booths, podium silhouettes, papers being signed.
7. For economic stories: market charts, currency stacks, factory
   floors, shipping ports.
8. Sharp focus, natural lighting, slightly desaturated like a real
   news photo — NOT a movie poster, NOT a vibrant social-media post.
9. No watermarks, no channel branding, no logos.

Output: ONE prompt string under 300 words. Nothing else.
"""

_PROMPT_TWEAKER_SYSTEM = """\
You are revising a news B-roll image prompt. The user is happy with
most of the previous prompt but wants ONE change. Rewrite the prompt
so that change is applied, keeping every other element intact.

Rules:
1. Output a SINGLE revised prompt, no preamble, no JSON.
2. Keep the cinematic news B-roll aesthetic and 16:9 format.
3. Apply the user's tweak verbatim wherever it overrides the original
   (e.g. "darker mood" -> dial down brightness; "courtroom instead of
   prison" -> swap the scene).
4. Never name real public figures; keep silhouettes/symbolic imagery
   if the previous prompt used them.
5. No watermarks, no channel branding.

Output: ONE prompt string under 300 words.
"""


def _gemini_client():
    """Reuse the Vertex client builder from thumbnail_ai so both image
    generators share the same auth + region setup. If the user later
    flips to AI Studio mode, both surfaces switch together."""
    from pipeline_v4.thumbnail_ai import _gemini_client as build
    return build()


def write_image_prompt(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    language: str = "te",
) -> str:
    """Ask Gemini for a sidebar-image brief from the story content.
    Falls back to a deterministic stock prompt on failure."""
    facts = []
    if title_native:  facts.append(f"native-script headline: {title_native}")
    if title_english: facts.append(f"English headline: {title_english}")
    if summary:       facts.append(f"summary: {summary[:600]}")
    facts.append(f"language: {language}")
    user_msg = (
        "Write the sidebar-image prompt for this news story.\n\n"
        + "\n".join(facts)
    )
    try:
        from google.genai import types as genai_types
        client = _gemini_client()
        resp = client.models.generate_content(
            model=DEFAULT_PROMPT_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=_PROMPT_WRITER_SYSTEM,
                temperature=0.8,
            ),
        )
        text = (resp.text or "").strip()
        if text:
            return text
    except Exception as exc:
        print(f"[v4/image-ai] prompt writer failed: {exc}", flush=True)

    headline = (title_native or title_english or "BREAKING").strip()
    return (
        f"Cinematic 16:9 news B-roll photograph about: {headline}. "
        f"Documentary style, natural lighting, slightly desaturated colour. "
        f"Generic silhouettes only (no recognisable real people). "
        f"Sharp focus, evocative composition. No overlay text, no logo."
    )


def tweak_image_prompt(*, previous_prompt: str, tweak: str) -> str:
    """One Gemini text call that rewrites the last prompt with a small
    change. Cheaper than writing a new prompt from scratch."""
    if not previous_prompt:
        return tweak or ""
    if not tweak:
        return previous_prompt
    try:
        from google.genai import types as genai_types
        client = _gemini_client()
        user_msg = (
            f"Previous prompt:\n{previous_prompt}\n\n"
            f"User tweak: {tweak}\n\n"
            f"Output the revised prompt."
        )
        resp = client.models.generate_content(
            model=DEFAULT_PROMPT_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=_PROMPT_TWEAKER_SYSTEM,
                temperature=0.6,
            ),
        )
        text = (resp.text or "").strip()
        if text:
            return text
    except Exception as exc:
        print(f"[v4/image-ai] tweak failed: {exc}", flush=True)
    return f"{previous_prompt}\n\nAdditional direction: {tweak}"


def generate_image(
    *,
    prompt: str,
    out_path: str,
    width: int = 1280,
    height: int = 720,
) -> Optional[str]:
    """Call Nano Banana with ``prompt`` and save the first image to
    ``out_path``. Stamps ``generate_image.last_error`` on failure so
    the route can surface the verbatim error to the UI."""
    generate_image.last_error = ""
    if not prompt:
        return None
    try:
        from google.genai import types as genai_types
    except Exception as exc:
        generate_image.last_error = f"SDK missing: {exc}"
        print(f"[v4/image-ai] google-genai missing: {exc}", flush=True)
        return None
    try:
        client = _gemini_client()
    except Exception as exc:
        generate_image.last_error = f"auth: {exc}"
        print(f"[v4/image-ai] auth: {exc}", flush=True)
        return None
    try:
        resp = client.models.generate_content(
            model=DEFAULT_IMAGE_MODEL,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.9,
            ),
        )
    except Exception as exc:
        msg = str(exc)
        generate_image.last_error = msg
        print(f"[v4/image-ai] image generate failed: {exc}", flush=True)
        return None

    try:
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                if not inline:
                    continue
                raw = getattr(inline, "data", None)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "wb") as fh:
                    fh.write(raw)
                _ensure_aspect(out_path, width, height)
                return out_path
    except Exception as exc:
        generate_image.last_error = f"parse: {exc}"
        print(f"[v4/image-ai] response parse failed: {exc}", flush=True)
    return None


generate_image.last_error = ""


def _ensure_aspect(path: str, width: int, height: int) -> None:
    """Cover-crop the saved JPG to exactly width x height."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGB")
            src_ratio = img.width / max(1, img.height)
            tgt_ratio = width / height
            if src_ratio > tgt_ratio:
                new_h = height
                new_w = int(img.width * (height / img.height))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                x = (new_w - width) // 2
                img = img.crop((x, 0, x + width, height))
            else:
                new_w = width
                new_h = int(img.height * (width / img.width))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                y = (new_h - height) // 2
                img = img.crop((0, y, width, y + height))
            img.save(path, "JPEG", quality=92)
    except Exception as exc:
        print(f"[v4/image-ai] aspect fix failed: {exc}", flush=True)


def make_image_for_story(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    language: str = "te",
    out_path: str,
    previous_prompt: str = "",
    tweak: str = "",
    width: int = 1280,
    height: int = 720,
) -> Tuple[Optional[str], str]:
    """Two-step: derive prompt (tweak path when previous_prompt+tweak
    supplied, otherwise fresh from SEO) -> render image -> save.
    Returns ``(saved_path or None, final_prompt)`` so the caller can
    persist the prompt for a future tweak iteration."""
    if previous_prompt and tweak:
        prompt = tweak_image_prompt(previous_prompt=previous_prompt, tweak=tweak)
    else:
        prompt = write_image_prompt(
            title_native=title_native,
            title_english=title_english,
            summary=summary,
            language=language,
        )
    print(f"[v4/image-ai] prompt: {prompt[:160]}…", flush=True)
    saved = generate_image(prompt=prompt, out_path=out_path, width=width, height=height)
    return saved, prompt
