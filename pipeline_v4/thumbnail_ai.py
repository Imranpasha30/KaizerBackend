"""AI thumbnail generator — "Nano Banana" image gen via Gemini.

Pipeline:
  1. AI writes the prompt itself from the canvas's SEO + title +
     summary. We don't ask the user to write image prompts; the
     thumbnail brief is derived from the news story.
  2. Gemini's image-gen model produces a 16:9 thumbnail PNG.
  3. We save it next to the rendered MP4 so the publisher can pick
     it up as the YouTube thumbnail at upload time.

Why Gemini instead of OpenAI/DALL-E:
  * Same ``GEMINI_API_KEY`` the rest of the pipeline already uses —
    no extra GCP setup, no separate billing
  * Free tier on Google AI Studio covers the small per-job volume V4
    produces (one thumbnail per bulletin + one per short)
  * Same ``google.genai`` SDK already loaded by ``seo/generator.py``

Model: ``gemini-2.5-flash-image`` (the public name for "Nano Banana").
Caller can override via ``KAIZER_NANO_BANANA_MODEL`` env to track
newer revisions without a code change.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional


DEFAULT_IMAGE_MODEL  = os.environ.get("KAIZER_NANO_BANANA_MODEL",  "gemini-2.5-flash-image")
DEFAULT_PROMPT_MODEL = os.environ.get("KAIZER_NANO_BANANA_PROMPT_MODEL", "gemini-2.5-flash")

# ── Auth modes ─────────────────────────────────────────────────────────
# The google.genai SDK supports two backends for the SAME generate_content
# call:
#
#   1) AI Studio (api_key=...)         → free tier + AI Studio prepay
#                                         bucket only. Does NOT see your
#                                         GCP billing credits.
#   2) Vertex AI (vertexai=True, ...)  → billed through the linked
#                                         project's regular Cloud billing
#                                         account → uses your GenAI App
#                                         Builder / Free Trial credits.
#
# When KAIZER_GCP_PROJECT is set we switch to Vertex mode. This is the
# right path when you have GCP credits but no AI Studio prepay balance.
# Auth in Vertex mode comes from Application Default Credentials — run
# `gcloud auth application-default login` once, OR point
# GOOGLE_APPLICATION_CREDENTIALS at a service-account JSON.
NANO_BANANA_ENV_KEY    = "KAIZER_NANO_BANANA_API_KEY"
GCP_PROJECT_ENV_KEY    = "KAIZER_GCP_PROJECT"
GCP_LOCATION_ENV_KEY   = "KAIZER_GCP_LOCATION"
DEFAULT_GCP_LOCATION   = "us-central1"

# Prompt the prompt-writer uses to turn news SEO into a thumbnail brief.
# It instructs Gemini to produce a SINGLE prompt string — no JSON, no
# preamble — so we can feed the response straight to the image model.
_PROMPT_WRITER_SYSTEM = """\
You are a YouTube thumbnail art director for a news channel. Given a
news clip's headline + summary, you write ONE image prompt that will
generate a high-CTR thumbnail.

Rules:
1. Output a single prompt, no preamble, no bullet points, no JSON.
2. 16:9 photographic news-broadcast aesthetic.
3. Bold colour palette (red, yellow, white accents) — feels urgent.
4. Includes a 2-3 word LARGE shout text overlay matching the hook,
   typeset bold. The text must be in the source language of the clip.
5. Faces of REAL public figures should be replaced with generic
   silhouettes / outlines / back-of-head shots — never name them.
   Generic symbols (handcuffs, scales of justice, money bags, broken
   glass, court hammer) for crime/scam stories.
6. No watermarks, no channel branding, no logos.
7. Aspect ratio: 16:9, sharp focus, cinematic lighting, slight vignette.

Output: ONE prompt string under 300 words. Nothing else.
"""

# Cheaper iterative tweak — takes the previous successful prompt and
# the user's tweak instruction, returns a revised prompt. Avoids
# starting from scratch from SEO each time the user nudges the image.
_PROMPT_TWEAKER_SYSTEM = """\
You are revising a YouTube thumbnail prompt. The user is happy with
most of the previous prompt but wants ONE change. Rewrite the prompt
so that change is applied, keeping every other element intact.

Rules:
1. Output a SINGLE revised prompt, no preamble, no JSON.
2. Keep the photographic news-broadcast aesthetic and the 16:9 format.
3. Apply the user's tweak verbatim wherever it overrides the original
   (e.g. "darker mood" -> tone down lighting; "money bags instead of
   handcuffs" -> swap the symbol).
4. Never name real public figures; keep silhouettes/outlines if the
   previous prompt used them.
5. No watermarks, no channel branding.

Output: ONE prompt string under 300 words.
"""


def _gemini_client():
    """Build a fresh google.genai client. Vertex AI mode wins when
    ``KAIZER_GCP_PROJECT`` is set — that's the path that spends your
    GCP billing credits (GenAI App Builder / Free Trial). Without
    that variable we fall back to AI Studio mode using whichever key
    is configured.

    Vertex mode auth picks up Application Default Credentials —
    typically created by ``gcloud auth application-default login`` —
    or a service-account JSON pointed to by
    ``GOOGLE_APPLICATION_CREDENTIALS``."""
    from google import genai

    project = (os.environ.get(GCP_PROJECT_ENV_KEY) or "").strip()
    if project:
        location = (os.environ.get(GCP_LOCATION_ENV_KEY) or DEFAULT_GCP_LOCATION).strip()
        # Build Service Account credentials from the dedicated Vertex
        # SA file and pass them EXPLICITLY to genai.Client. We don't
        # touch GOOGLE_APPLICATION_CREDENTIALS because the STT pipeline
        # has its own SA pointed at by that env var; if we let the SDK
        # auto-discover credentials it picks up the STT SA's default
        # project and Vertex returns 403 on the wrong-project resource.
        vertex_creds_path = (os.environ.get("KAIZER_VERTEX_CREDENTIALS") or "").strip()
        explicit_creds = None
        if vertex_creds_path and os.path.isfile(vertex_creds_path):
            try:
                from google.oauth2 import service_account
                explicit_creds = service_account.Credentials.from_service_account_file(
                    vertex_creds_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                print(f"[nano-banana] Vertex AI mode  project={project} location={location}  "
                      f"sa={os.path.basename(vertex_creds_path)}", flush=True)
            except Exception as exc:
                print(f"[nano-banana] failed to load Vertex SA from {vertex_creds_path}: {exc}",
                      flush=True)
        else:
            print(f"[nano-banana] Vertex AI mode  project={project} location={location}  "
                  f"sa=(ADC fallback — KAIZER_VERTEX_CREDENTIALS not set)", flush=True)
        return genai.Client(
            vertexai=True, project=project, location=location,
            credentials=explicit_creds,
        )

    key = (os.environ.get(NANO_BANANA_ENV_KEY) or "").strip()
    if not key:
        key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "No Gemini auth. Either:\n"
            "  - set KAIZER_GCP_PROJECT (recommended — uses GCP credits via Vertex AI), OR\n"
            "  - set KAIZER_NANO_BANANA_API_KEY / GEMINI_API_KEY (AI Studio mode)."
        )
    print(f"[nano-banana] AI Studio mode  key={key[:8]}…{key[-4:]}", flush=True)
    return genai.Client(api_key=key)


def write_thumbnail_prompt(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    seo_hook: str = "",
    seo_thumbnail_text: str = "",
    language: str = "te",
) -> str:
    """Ask Gemini for a thumbnail brief based on the clip's content.

    Returns the prompt string. Falls back to a deterministic stock
    prompt on failure so the image generation step never starves."""
    facts = []
    if title_native:       facts.append(f"native-script headline: {title_native}")
    if title_english:      facts.append(f"English headline: {title_english}")
    if summary:            facts.append(f"summary: {summary[:600]}")
    if seo_hook:           facts.append(f"hook: {seo_hook}")
    if seo_thumbnail_text: facts.append(f"shout text suggested by SEO: {seo_thumbnail_text}")
    facts.append(f"language: {language}")
    user_msg = (
        "Write the thumbnail prompt for the following news clip.\n\n"
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
        print(f"[nano-banana] prompt writer failed: {exc}", flush=True)

    # Deterministic fallback so the image step can still run.
    headline = (title_native or title_english or "BREAKING").strip()
    shout    = (seo_thumbnail_text or "BREAKING").strip()
    return (
        f"16:9 photographic YouTube thumbnail for breaking news about: {headline}. "
        f"Bold red and yellow palette, cinematic lighting, dramatic mood. "
        f"Large bold shout text overlay reading '{shout}' in white with red outline, "
        f"top-center placement. Generic silhouettes only (no recognisable real people). "
        f"Sharp focus, high contrast. No channel logo, no watermark."
    )


def generate_thumbnail_image(
    *,
    prompt: str,
    out_path: str,
    width: int = 1280,
    height: int = 720,
) -> Optional[str]:
    """Call Gemini's image-generation model with ``prompt`` and save
    the first returned image to ``out_path``.

    Returns the absolute output path on success, ``None`` on any
    failure (caller falls back to ffmpeg frame-grab thumbnails)."""
    if not prompt:
        return None
    try:
        from google.genai import types as genai_types
    except Exception as exc:
        print(f"[nano-banana] google-genai SDK missing: {exc}", flush=True)
        return None

    try:
        client = _gemini_client()
    except Exception as exc:
        print(f"[nano-banana] no GEMINI_API_KEY: {exc}", flush=True)
        return None

    # Image generation needs response_modalities=["IMAGE", "TEXT"] —
    # Gemini returns image bytes inside a Part's inline_data; the TEXT
    # half catches any safety / refusal notes the model may emit.
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
        # Stash the raw Gemini error string on the function so the
        # route can surface it back to the user verbatim instead of a
        # generic "check key and quota" hand-wave.
        msg = str(exc)
        generate_thumbnail_image.last_error = msg
        print(f"[nano-banana] image generate failed: {exc}", flush=True)
        return None

    # Walk the response for the first inline_data blob.
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
                # SDK may return raw bytes OR a base64 string depending
                # on version — handle both.
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "wb") as fh:
                    fh.write(raw)
                # Resize/letterbox to the requested aspect via PIL so the
                # downstream YouTube uploader can hand it to videos.insert
                # without surprises.
                _ensure_aspect(out_path, width, height)
                return out_path
    except Exception as exc:
        print(f"[nano-banana] response parse failed: {exc}", flush=True)
        return None
    return None


def _ensure_aspect(path: str, width: int, height: int) -> None:
    """Resize the saved image to exactly width x height with cover-
    crop semantics. Skips silently when PIL is unavailable."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGB")
            src_ratio = img.width / max(1, img.height)
            tgt_ratio = width / height
            if src_ratio > tgt_ratio:
                # source wider — scale by height then crop sides
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
        print(f"[nano-banana] aspect fix failed: {exc}", flush=True)


def tweak_thumbnail_prompt(*, previous_prompt: str, tweak: str) -> str:
    """Revise an existing thumbnail prompt with a small change instead
    of writing a new one from scratch. One Gemini text call instead of
    the full SEO-context rewrite — roughly half the prompt-side cost.
    Falls back to a naive append on failure."""
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
        print(f"[nano-banana] tweak failed: {exc}", flush=True)
    return f"{previous_prompt}\n\nAdditional direction: {tweak}"


def make_thumbnail_for_canvas(
    *,
    canvas,                              # pipeline_v4.canvas_schema.Canvas
    out_path: str,
    language: str = "te",
    previous_prompt: str = "",
    tweak: str = "",
) -> tuple[Optional[str], str]:
    """End-to-end helper: derive prompt → render image → save to
    ``out_path``. Returns ``(saved_path or None, final_prompt)`` so the
    caller can store the prompt and feed it back for a future tweak.

    Iterative path: when ``previous_prompt`` + ``tweak`` are supplied
    we call the tweaker instead of the full prompt-writer — one cheaper
    Gemini text call instead of the full SEO context rewrite. When
    neither is supplied we generate from scratch using SEO context.

    Caller is expected to update ``Clip.thumb_path`` after success so
    the publish flow uses the AI thumbnail instead of the ffmpeg
    frame-grab."""
    if previous_prompt and tweak:
        prompt = tweak_thumbnail_prompt(
            previous_prompt=previous_prompt, tweak=tweak,
        )
    else:
        story0 = canvas.stories[0] if canvas.stories else None
        seo    = canvas.seo
        prompt = write_thumbnail_prompt(
            title_native=(story0.title_native if story0 else "") or "",
            title_english=(story0.title_english if story0 else "") or "",
            summary=(story0.summary if story0 else "") or "",
            seo_hook=(seo.hook if seo else "") or "",
            seo_thumbnail_text=(seo.thumbnail_text if seo else "") or "",
            language=language,
        )
    print(f"[nano-banana] prompt: {prompt[:160]}…", flush=True)
    saved = generate_thumbnail_image(prompt=prompt, out_path=out_path)
    return saved, prompt
