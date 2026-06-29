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
news clip's transcript + SEO description + headline, you write ONE
image-generation prompt that produces a high-CTR thumbnail.

# ⚠ LANGUAGE CONTRACT (MOST IMPORTANT — PIPELINE FAILS WITHOUT THIS)
The clip's LANGUAGE is specified in the user message (full name +
script). The SHOUT TEXT in the thumbnail MUST be written in that
language's NATIVE SCRIPT — not romanised, not English.

For example:
- language=Telugu/Telugu  → shout text in Telugu script (e.g. "షాకింగ్!", "బ్రేకింగ్")
- language=Hindi/Devanagari → shout text in Devanagari (e.g. "बड़ी खबर", "ब्रेकिंग")
- language=Tamil/Tamil    → shout text in Tamil script (e.g. "முக்கியம்")
- language=English/Latin  → shout text in English (e.g. "SHOCKING!")

The prompt you write MUST embed the actual native-script characters
the image generator will render — don't ask the generator to "write
Telugu text", spell out the literal characters in the prompt so the
generator can rasterise them. Pick 2-4 words that match the SEO hook
or thumbnail_text suggestion, in the native script.

If you output a prompt with romanised text (e.g. "ShoCkInG" for a
Telugu clip), the verifier rejects the thumbnail and the operator has
to regenerate manually — a workflow failure we never want.

# Output rules
1. Output a single prompt, no preamble, no bullet points, no JSON.
2. 16:9 photographic news-broadcast aesthetic.
3. Bold colour palette (red, yellow, white accents) — feels urgent.
4. Includes a 2-4 word LARGE shout text overlay matching the SEO hook,
   typeset bold. The text MUST be in the native script per the contract
   above. Spell out the literal characters inside double-quotes in the
   prompt so the image model rasterises them verbatim.
5. Compose the visual scene from facts in the transcript / SEO
   description — pick the most arresting element (the heist tools,
   the courtroom, the protest crowd, the burning vehicle, the
   document). Do NOT invent details the source doesn't support.
6. Faces of REAL public figures should be replaced with generic
   silhouettes / outlines / back-of-head shots — never name them.
   Generic symbols (handcuffs, scales of justice, money bags, broken
   glass, court hammer) for crime/scam stories.
7. No watermarks, no channel branding, no logos.
8. Aspect ratio: 16:9, sharp focus, cinematic lighting, slight vignette.

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
    seo_description: str = "",
    transcript_excerpt: str = "",
    language: str = "te",
    style: str = "symbolic",          # one of pipeline_v4.thumbnail_styles.STYLES keys
    has_references: int = 0,          # how many reference images the operator supplied
) -> str:
    """Ask Gemini for a thumbnail brief based on the clip's content.

    Delegates to the ``thumbnail_director`` package — a 2-stage Gemini
    orchestration (Pass 1: structured ScenePlan, Pass 2: Nano Banana
    prompt) replaced the legacy single-pass prompt writer in 2026-06-09.

    Keeping this function as the public entry point so the route at
    ``routers/v4_editor.generate_thumbnail`` and any other caller
    doesn't change — the director module is the implementation detail.

    Returns the prompt string. Falls back to a deterministic stock
    prompt on total failure so the image step never starves."""
    try:
        from pipeline_v4.thumbnail_director import build_thumbnail_prompt
        return build_thumbnail_prompt(
            title_native=title_native,
            title_english=title_english,
            summary=summary,
            seo_hook=seo_hook,
            seo_thumbnail_text=seo_thumbnail_text,
            seo_description=seo_description,
            transcript_excerpt=transcript_excerpt,
            language=language,
            style=style,
            has_references=has_references,
        )
    except Exception as exc:
        print(f"[nano-banana] thumbnail_director failed, using legacy fallback: {exc}",
              flush=True)

    # Deterministic last-resort fallback so the image step can still run.
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
    reference_paths: Optional[list[str]] = None,
) -> Optional[str]:
    """Call Gemini's image-generation model with ``prompt`` + optional
    ``reference_paths`` (Nano Banana multi-image input) and save the
    first returned image to ``out_path``.

    ``reference_paths`` lets the operator anchor the generation on a
    real photograph — e.g. their reporter's face for a reporter-led
    style, or two politicians for a split-screen. Nano Banana preserves
    the identity from the reference while applying the text prompt's
    composition. Up to 2 references are passed through; more are
    truncated since the model's effective attention drops off fast.

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

    # Build the contents list. Reference images go FIRST so Nano Banana
    # treats them as the anchor; the text prompt then says how to use
    # them. Empty / unreadable files are skipped silently.
    contents: list = []
    refs = (reference_paths or [])[:2]
    for ref_path in refs:
        try:
            if not ref_path or not os.path.isfile(ref_path):
                continue
            with open(ref_path, "rb") as fh:
                data = fh.read()
            # Guess mime from extension — Gemini rejects mismatched
            # types. We standardise to image/jpeg / image/png based on
            # the file's signature byte rather than trusting the suffix.
            mime = "image/jpeg"
            if data[:8].startswith(b"\x89PNG"):
                mime = "image/png"
            elif data[:6] in (b"GIF87a", b"GIF89a"):
                mime = "image/gif"
            elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                mime = "image/webp"
            contents.append(genai_types.Part.from_bytes(data=data, mime_type=mime))
        except Exception as exc:
            print(f"[nano-banana] reference load failed for {ref_path}: {exc}", flush=True)
    contents.append(prompt)

    # Image generation needs response_modalities=["IMAGE", "TEXT"] —
    # Gemini returns image bytes inside a Part's inline_data; the TEXT
    # half catches any safety / refusal notes the model may emit.
    #
    # Temperature is reference-image-aware: when the operator passed
    # a face reference (reporter_led / subject_photo / split_screen)
    # we drop temperature to 0.35 so Nano Banana stays close to the
    # reference identity. The legacy 0.9 was creative enough to
    # swap a male reporter for a generic female anchor — observed
    # 2026-06-09 in production. Pure-symbolic / scene-photograph
    # styles still get 0.9 because creative composition matters and
    # there's no identity to preserve.
    has_ref_for_identity = bool(refs)
    temp = 0.35 if has_ref_for_identity else 0.9
    try:
        resp = client.models.generate_content(
            model=DEFAULT_IMAGE_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=temp,
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


# ─── Pillow text overlay (Phase 2: hybrid plate + perfect typography) ─

# Maps ISO language code → Noto Sans Bold (or equivalent) basename in
# resources/fonts/. The image model produces a text-LESS plate; we
# rasterise the actual native-script text here using a real .ttf so
# Telugu / Hindi / Tamil ligatures never come out garbled.
_LANG_TO_FONT_BOLD = {
    "te": "NotoSansTelugu-Bold.ttf",
    "hi": "NotoSansDevanagari-Bold.ttf",
    "ta": "NotoSansTamil-Bold.ttf",
    "kn": "NotoSansKannada-Bold.ttf",
    "ml": "NotoSansMalayalam-Bold.ttf",
    "bn": "NotoSansBengali-Bold.ttf",
    "mr": "NotoSansDevanagari-Bold.ttf",
    "gu": "NotoSansGujarati-Bold.ttf",
    "en": "NotoSans-Bold.ttf",
}


def _resolve_font_path(language: str) -> Optional[str]:
    """Resolve the absolute font path for the given ISO language code.
    Falls back to NotoSans-Bold then to None if nothing is on disk.
    """
    backend_root = Path(__file__).resolve().parent.parent
    fonts_dir = backend_root / "resources" / "fonts"
    candidate = fonts_dir / _LANG_TO_FONT_BOLD.get(
        language, _LANG_TO_FONT_BOLD["en"]
    )
    if candidate.is_file():
        return str(candidate)
    fallback = fonts_dir / "NotoSans-Bold.ttf"
    return str(fallback) if fallback.is_file() else None


def _pick_overlay_geometry(style: str, image_w: int, image_h: int) -> dict:
    """Decide WHERE the text strap lives based on style. Mirrors the
    "safe zone" guidance baked into each style's system prompt so
    the AI-generated plate's empty area matches where we paint.

    Returns ``{x, y, w, h, anchor, fg, bg}``:
      - x/y/w/h are pixel rectangles within the image
      - anchor is the text alignment ("mm" = middle/middle)
      - fg / bg are the overlay foreground and background hex strings
    """
    s = (style or "symbolic").lower()
    # Default: bottom 28% strap, white text on translucent dark.
    geom = {
        "x": 0,
        "y": int(image_h * 0.72),
        "w": image_w,
        "h": int(image_h * 0.28),
        "anchor": "mm",
        "fg": "#FFFFFF",
        "bg": (0, 0, 0, 200),       # rgba — slightly transparent black strap
    }
    if s in ("reporter_led",):
        # Right half — face is on the left. Text rectangle covers
        # the right ~45% from top to bottom (vertical centring).
        geom.update({
            "x": int(image_w * 0.50),
            "y": int(image_h * 0.10),
            "w": int(image_w * 0.45),
            "h": int(image_h * 0.80),
            "anchor": "mm",
            "fg": "#FFFFFF",
            "bg": (0, 0, 0, 180),
        })
    elif s == "subject_photo":
        # Left 40% column when the subject is right-ish, or
        # bottom-third strap otherwise. Default to bottom strap for
        # safety since we can't probe face position client-side.
        geom.update({
            "x": 0,
            "y": int(image_h * 0.70),
            "w": image_w,
            "h": int(image_h * 0.30),
            "fg": "#FFFFFF",
            "bg": (180, 0, 0, 200),  # red strap for urgency
        })
    elif s == "split_screen":
        # Centre 18% horizontal band — overlays the "VS" between the
        # two faces.
        geom.update({
            "x": int(image_w * 0.05),
            "y": int(image_h * 0.40),
            "w": int(image_w * 0.90),
            "h": int(image_h * 0.20),
            "fg": "#FFD600",
            "bg": (0, 0, 0, 220),
        })
    elif s == "live_breaking":
        # Centre 70% — huge yellow text on the red plate.
        geom.update({
            "x": int(image_w * 0.05),
            "y": int(image_h * 0.20),
            "w": int(image_w * 0.90),
            "h": int(image_h * 0.60),
            "fg": "#FFD600",
            "bg": (180, 0, 0, 0),    # transparent — plate is already red
        })
    elif s == "text_dominant":
        # Centre 70% — text is the focal point.
        geom.update({
            "x": int(image_w * 0.05),
            "y": int(image_h * 0.15),
            "w": int(image_w * 0.90),
            "h": int(image_h * 0.70),
            "fg": "#FFFFFF",
            "bg": (0, 0, 0, 0),
        })
    elif s == "festival_religion":
        # Soft golden strap at bottom.
        geom.update({
            "x": 0,
            "y": int(image_h * 0.72),
            "w": image_w,
            "h": int(image_h * 0.28),
            "fg": "#FFD700",         # gold
            "bg": (110, 30, 0, 200), # deep crimson
        })
    elif s == "sports_action":
        # Bottom third — high-contrast white-on-black title card.
        geom.update({
            "x": 0,
            "y": int(image_h * 0.70),
            "w": image_w,
            "h": int(image_h * 0.30),
            "fg": "#FFFFFF",
            "bg": (0, 0, 0, 220),
        })
    elif s == "entertainment_tollywood":
        # Bottom third — neon-cyan title card.
        geom.update({
            "x": 0,
            "y": int(image_h * 0.70),
            "w": image_w,
            "h": int(image_h * 0.30),
            "fg": "#00E5FF",
            "bg": (50, 0, 80, 200),  # deep purple
        })
    return geom


def _fit_text_to_box(*, draw, text: str, font_path: str,
                     max_w: int, max_h: int,
                     min_size: int = 28, max_size: int = 220) -> "ImageFont.FreeTypeFont":
    """Binary-search the largest font size that fits ``text`` inside a
    ``max_w × max_h`` rectangle. Pillow handles word wrapping by
    splitting on whitespace; we measure each candidate font size and
    pick the largest that still fits.
    """
    from PIL import ImageFont
    lo, hi = min_size, max_size
    best = ImageFont.truetype(font_path, lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        # Estimate wrapped text size — use multiline_textbbox so wide
        # words get measured at their full width.
        wrapped = _wrap_text_for_box(text, font, max_w, draw=draw)
        bbox = draw.multiline_textbbox(
            (0, 0), wrapped, font=font, spacing=mid // 6
        )
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= max_w and th <= max_h:
            best = font
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _wrap_text_for_box(text: str, font, max_w: int, draw) -> str:
    """Word-wrap ``text`` so each line fits within ``max_w`` pixels.
    Falls back to character-wrapping for languages without word
    boundaries (most Indic scripts use spaces but some segmentation
    is needed for very long compound words)."""
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        trial = " ".join(current + [w])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_w:
            current.append(w)
            continue
        if current:
            lines.append(" ".join(current))
            current = [w]
        else:
            # Single word longer than the box — character-split it.
            lines.append(w)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _plate_path_for(thumbnail_path: str) -> str:
    """Resolve the bare-plate path alongside ``thumbnail_path``.

    The bare plate is the AI-generated image WITHOUT the Pillow text
    overlay. We save a copy of it right after generation so the
    manual drag editor can re-paint the text without burning another
    AI call. Convention: ``<name>.plate.<ext>`` — keeps the file
    extension so the format round-trips losslessly (JPEG for V4 today).
    """
    p = Path(thumbnail_path)
    return str(p.with_suffix(f".plate{p.suffix}"))


# Manual-editor colour presets — surfaced in the V4 editor's
# drag-to-position panel. Each entry is {fg, bg, label}. Operator
# picks by preset key; backend looks up the RGB values here.
OVERLAY_COLOR_PRESETS: dict[str, dict] = {
    "white_on_red": {
        "fg": "#FFFFFF", "bg": (180, 0, 0, 200),
        "label": "White on Red",
    },
    "yellow_on_red": {
        "fg": "#FFD600", "bg": (180, 0, 0, 200),
        "label": "Yellow on Red",
    },
    "gold_on_crimson": {
        "fg": "#FFD700", "bg": (110, 30, 0, 200),
        "label": "Gold on Crimson",
    },
    "white_on_black": {
        "fg": "#FFFFFF", "bg": (0, 0, 0, 220),
        "label": "White on Black",
    },
}


def overlay_native_text(
    *,
    image_path: str,
    text: str,
    language: str = "te",
    style: str = "symbolic",
    geometry_override: Optional[dict] = None,
    color_preset: str = "",
    source_plate_path: str = "",
) -> bool:
    """Paint ``text`` onto ``image_path`` using the matching Noto Sans
    font and the style's safe-zone geometry. Saves over the input.

    Parameters
    ----------
    geometry_override
        When supplied, overrides the style-driven safe zone with
        explicit ``{x_pct, y_pct, w_pct, h_pct}`` (all 0-100). Used by
        the manual editor where the operator drags the text box.
    color_preset
        Key into ``OVERLAY_COLOR_PRESETS``. Overrides the style's
        default colours. Empty / unknown = use style defaults.
    source_plate_path
        When supplied AND a file exists at that path, copy it over
        ``image_path`` first so the overlay paints onto a clean
        plate instead of accumulating over a prior overlay. Used by
        the reoverlay endpoint when re-painting an existing thumbnail.

    Returns True on success, False on any failure so the caller can
    fall back to the bare AI plate. Best-effort by design — losing
    the overlay is better than losing the whole thumbnail."""
    # Restore a clean plate from disk before painting — lets the
    # reoverlay endpoint re-paint without overlay artifacts piling up.
    if source_plate_path and os.path.isfile(source_plate_path):
        try:
            import shutil
            shutil.copy2(source_plate_path, image_path)
        except Exception as exc:
            print(f"[nano-banana] plate restore failed (continuing on top of "
                  f"prior overlay): {exc}", flush=True)
    if not (text or "").strip():
        return False
    font_path = _resolve_font_path(language)
    if not font_path:
        print(f"[nano-banana] no font available for language {language!r} — skipping overlay",
              flush=True)
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"[nano-banana] Pillow unavailable: {exc} — skipping overlay", flush=True)
        return False

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            w, h = img.size
            geom = _pick_overlay_geometry(style, w, h)
            # Apply explicit geometry override from the manual editor
            # (operator drag-positioned the text). Pixel-level fields
            # come from percentages 0-100; clamp so a buggy frontend
            # can't write outside the canvas.
            if geometry_override:
                try:
                    def _clamp(v, lo, hi):
                        return max(lo, min(hi, v))
                    x_pct = float(geometry_override.get("x_pct", 0))
                    y_pct = float(geometry_override.get("y_pct", 0))
                    w_pct = float(geometry_override.get("w_pct", 100))
                    h_pct = float(geometry_override.get("h_pct", 28))
                    geom = {
                        **geom,
                        "x": int(_clamp(x_pct, 0, 100) / 100 * w),
                        "y": int(_clamp(y_pct, 0, 100) / 100 * h),
                        "w": int(_clamp(w_pct, 5, 100) / 100 * w),
                        "h": int(_clamp(h_pct, 5, 100) / 100 * h),
                    }
                    # Re-clamp width / height so the box stays inside the canvas.
                    geom["w"] = min(geom["w"], w - geom["x"])
                    geom["h"] = min(geom["h"], h - geom["y"])
                except Exception as exc:
                    print(f"[nano-banana] bad geometry_override (ignored): {exc}",
                          flush=True)
            # Apply colour preset override.
            if color_preset and color_preset in OVERLAY_COLOR_PRESETS:
                preset = OVERLAY_COLOR_PRESETS[color_preset]
                geom = {**geom, "fg": preset["fg"], "bg": preset["bg"]}

            # Build an overlay layer so the translucent strap can be
            # composited under the text crisply.
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Paint the strap background only if non-transparent.
            bg = geom["bg"]
            if bg[3] > 0:
                draw.rectangle(
                    (geom["x"], geom["y"],
                     geom["x"] + geom["w"], geom["y"] + geom["h"]),
                    fill=bg,
                )

            # Pick the largest font size that fits in the safe zone.
            text_pad = max(16, int(min(geom["w"], geom["h"]) * 0.06))
            usable_w = geom["w"] - text_pad * 2
            usable_h = geom["h"] - text_pad * 2
            font = _fit_text_to_box(
                draw=draw, text=text, font_path=font_path,
                max_w=usable_w, max_h=usable_h,
                min_size=28, max_size=min(180, usable_h),
            )
            wrapped = _wrap_text_for_box(text, font, usable_w, draw=draw)

            # Centre the wrapped block inside the safe zone.
            cx = geom["x"] + geom["w"] // 2
            cy = geom["y"] + geom["h"] // 2

            # Draw with stroke for legibility against any background.
            stroke_w = max(2, font.size // 24)
            fg = geom["fg"]
            stroke = "#000000" if fg.upper() != "#000000" else "#FFFFFF"
            draw.multiline_text(
                (cx, cy), wrapped, font=font,
                fill=fg, anchor="mm", align="center",
                stroke_width=stroke_w, stroke_fill=stroke,
                spacing=font.size // 6,
            )

            # Composite + save back over the same path (jpeg/png).
            out = Image.alpha_composite(img, overlay).convert("RGB")
            fmt = "JPEG" if image_path.lower().endswith((".jpg", ".jpeg")) else "PNG"
            save_kwargs = {"quality": 92} if fmt == "JPEG" else {}
            out.save(image_path, fmt, **save_kwargs)
            # Log with ASCII-safe repr — Windows cp1252 stdout chokes
            # on Telugu / Hindi / Tamil characters in repr() output.
            safe_preview = text[:40].encode("ascii", "backslashreplace").decode("ascii")
            print(f"[nano-banana] text overlay applied (style={style}, "
                  f"lang={language}, font_size={font.size}px, "
                  f"text_len={len(text)}, preview={safe_preview!r})",
                  flush=True)
        return True
    except Exception as exc:
        # ASCII-safe error message — exception messages may carry the
        # offending text too.
        safe_exc = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        print(f"[nano-banana] text overlay failed: {safe_exc} -- keeping plain plate",
              flush=True)
        return False


def make_thumbnail_for_canvas(
    *,
    canvas,                              # pipeline_v4.canvas_schema.Canvas
    out_path: str,
    language: str = "te",
    previous_prompt: str = "",
    tweak: str = "",
    transcript_excerpt: str = "",        # raw spoken content, when caller has it
    style: str = "symbolic",             # one of thumbnail_styles.STYLES keys
    reference_paths: Optional[list[str]] = None,  # absolute paths for Nano Banana
    engine: str = "gemini",              # director preset: gemini | hybrid | claude
    text_mode: str = "ai",               # "ai" (model renders text) | "overlay" (Pillow paints)
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
        # Build a "transcript-shaped" excerpt when the caller didn't
        # pass one explicitly: join every story's title + summary so
        # Gemini has the full editorial spread to draw a compelling
        # composition from, not just the first story. The verbatim
        # speech transcript would be richer, but stories carry the
        # important facts and live in the canvas the route reads.
        if not transcript_excerpt:
            chunks = []
            for s in (canvas.stories or []):
                hd = (s.title_native or s.title_english or "").strip()
                sm = (s.summary or "").strip()
                if hd or sm:
                    chunks.append(f"- {hd}\n  {sm}".strip())
            transcript_excerpt = "\n".join(chunks)
        # Filter the supplied references to the ones that actually
        # exist on disk — the route may have passed a list with
        # stale/deleted paths. Pass the COUNT (not the list) to the
        # prompt-writer so it knows whether to instruct face
        # preservation.
        usable_refs = [p for p in (reference_paths or []) if p and os.path.isfile(p)]
        # Use the director directly so we capture the ScenePlan and
        # can reuse plan.text_for_overlay for the Pillow step. The
        # legacy write_thumbnail_prompt() helper threw the plan away
        # — that's why Pillow was painting generic "BIG BREAKING"
        # instead of a story-specific shout.
        plan = None
        try:
            from pipeline_v4.thumbnail_director import (
                plan_scene, prompt_from_plan, _resolve_engine,
            )
            engine_pair = _resolve_engine(engine)
            plan = plan_scene(
                title_native=(story0.title_native if story0 else "") or "",
                title_english=(story0.title_english if story0 else "") or "",
                summary=(story0.summary if story0 else "") or "",
                seo_hook=(seo.hook if seo else "") or "",
                seo_thumbnail_text=(seo.thumbnail_text if seo else "") or "",
                seo_description=(seo.description if seo else "") or "",
                transcript_excerpt=transcript_excerpt,
                language=language,
                style=style,
                has_references=len(usable_refs),
                planner=engine_pair["planner"],
            )
            prompt = prompt_from_plan(
                plan=plan, language=language, style=style,
                has_references=len(usable_refs),
                prompter=engine_pair["prompter"],
                text_mode=text_mode,
            )
        except Exception as exc:
            print(f"[nano-banana] director path failed, falling back to legacy: {exc}",
                  flush=True)
            prompt = write_thumbnail_prompt(
                title_native=(story0.title_native if story0 else "") or "",
                title_english=(story0.title_english if story0 else "") or "",
                summary=(story0.summary if story0 else "") or "",
                seo_hook=(seo.hook if seo else "") or "",
                seo_thumbnail_text=(seo.thumbnail_text if seo else "") or "",
                seo_description=(seo.description if seo else "") or "",
                transcript_excerpt=transcript_excerpt,
                language=language,
                style=style,
                has_references=len(usable_refs),
            )
    print(f"[nano-banana] style={style} refs={len(reference_paths or [])} prompt: {prompt[:160]}…", flush=True)
    saved = generate_thumbnail_image(
        prompt=prompt,
        out_path=out_path,
        reference_paths=reference_paths,
    )

    # Save a bare-plate copy alongside the final so the reoverlay
    # endpoint can re-paint the text without burning another AI call.
    # ``<thumbnail>.plate.<ext>`` — same extension as the original so
    # the file format is preserved (JPEG for V4 thumbnails today).
    # Skipped for text_mode=ai because there's no clean plate to keep
    # (the model rendered the text inside the image already).
    if saved and text_mode == "overlay":
        try:
            import shutil as _shutil
            plate_path = _plate_path_for(saved)
            _shutil.copy2(saved, plate_path)
            print(f"[nano-banana] bare plate saved at {plate_path} "
                  f"(for the manual drag editor)", flush=True)
        except Exception as exc:
            print(f"[nano-banana] plate-save soft-failed: {exc}", flush=True)

    # ── Hybrid render: overlay perfect native-script text ──
    # The image model produced a text-LESS plate. Pillow now paints
    # the actual shout text using the matching Noto Sans font, so
    # Telugu / Hindi / Tamil ligatures are guaranteed correct.
    #
    # Text source priority (highest first):
    #   1. plan.text_for_overlay  ← Gemini-chosen story-specific shout
    #   2. seo.thumbnail_text     ← legacy SEO field (often generic)
    #   3. seo.hook               ← first-line attention grabber
    #   4. seo.title              ← last resort
    #
    # The plan field wins because the planner saw the SPECIFIC story
    # context and picked text grounded in it ("100 కోట్ల కుంభకోణం" not
    # "బిగ్ బ్రేకింగ్"). Best-effort — if overlay fails the bare plate
    # still ships rather than blocking publish.
    #
    # ``text_mode="ai"`` SKIPS this step — the image model rendered the
    # text inside the plate already, painting on top of it would double-
    # render. Operator picks via the API field; default is "ai" because
    # the current TV9-style aesthetic wants integrated typography.
    if text_mode == "ai":
        print(f"[nano-banana] text_mode=ai — skipping Pillow overlay "
              f"(image model already rendered the text inside the plate)",
              flush=True)
    elif saved:
        try:
            seo = getattr(canvas, "seo", None)
            plan_overlay = (getattr(plan, "text_for_overlay", "") if plan else "") or ""
            overlay_text = ""
            for cand in (
                plan_overlay,
                getattr(seo, "thumbnail_text", "") if seo else "",
                getattr(seo, "hook", "") if seo else "",
                getattr(seo, "title", "") if seo else "",
            ):
                cand = (cand or "").strip()
                if cand:
                    # Take only the first newline-separated line — SEO
                    # title may carry channel suffix on later lines.
                    overlay_text = cand.split("\n", 1)[0].strip()
                    break
            if overlay_text:
                src = "plan" if (plan and overlay_text == plan_overlay.split("\n", 1)[0].strip()) else "seo"
                print(f"[nano-banana] overlay text source: {src}", flush=True)
                overlay_native_text(
                    image_path=saved,
                    text=overlay_text,
                    language=language,
                    style=style,
                )
            else:
                print(f"[nano-banana] no overlay text on canvas SEO — keeping bare plate",
                      flush=True)
        except Exception as exc:
            print(f"[nano-banana] overlay step soft-failed: {exc}", flush=True)
    return saved, prompt
