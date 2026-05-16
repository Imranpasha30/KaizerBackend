"""Google Imagen wrapper for news-broadcast image generation.

Sister module to ``openai_images.py``. Same public surface, same prompt
construction, different backend — gives the operator a side-by-side A/B
between gpt-image-1 and Imagen on every compound job. Renders pick whichever
the operator settles on after a few jobs of comparison.

Design rules (same as openai_images.py):
1.  **Never raise.** A failure here must NOT block the pipeline. Return
    ``None`` so the caller can fall back to the OpenAI version of the
    same image_plan id.
2.  **No real public figures, no celebrities, no logos.** Imagen has its
    own safety filter that will refuse those — refusal is treated as
    "fall back to the OpenAI image," same as a network error.
3.  **One retry on transient errors only.** Network blips + rate-limit
    get one immediate retry. Anything else returns ``None``.
4.  **60-second timeout.** Matches the OpenAI wrapper so both backends
    run inside the same wall-clock budget.

Pricing (2026 estimates, Imagen 3 family)
-----------------------------------------
- ``imagen-3.0-generate-002`` 1024×1024 → ~$0.04 per image (similar to
  gpt-image-1 medium).
- Per-image cost shown in the stdout log line for operator awareness.

Configuration
-------------
- ``KAIZER_IMAGEN_MODEL``  — model id, default ``imagen-3.0-generate-002``.
- ``KAIZER_IMAGEN_SIZE``   — image dim, default ``1024x1024``.
- ``KAIZER_IMAGEN``        — kill switch (set to ``0`` to disable without
                              code change).
- ``GEMINI_API_KEY``       — reused; Imagen lives on the same Google API
                              key as Gemini.
"""
from __future__ import annotations

import io
import os
import time
from typing import Optional


# ─── Public knobs ─────────────────────────────────────────────────────

_MODEL   = os.environ.get("KAIZER_IMAGEN_MODEL",   "imagen-3.0-generate-002")
_SIZE    = os.environ.get("KAIZER_IMAGEN_SIZE",    "1024x1024")
_TIMEOUT = float(os.environ.get("KAIZER_IMAGEN_TIMEOUT", "60"))

# Per-image cost estimate for the stdout log line. Actual billing is
# whatever Google reports on the dashboard.
_COST_USD = float(os.environ.get("KAIZER_IMAGEN_COST_USD", "0.04"))


# ─── Prompt construction ──────────────────────────────────────────────
# Mirror of the OpenAI wrapper so an operator can compare apples-to-apples.
# We accept a free-form ``description`` (Gemini-authored in compound mode)
# AND the legacy entity/topic/query path used by single-platform callers.

_LANG_TO_LOCATION = {
    "te": "Andhra Pradesh / Telangana, India",
    "hi": "North India",
    "ta": "Tamil Nadu, India",
    "kn": "Karnataka, India",
    "ml": "Kerala, India",
    "bn": "West Bengal, India",
    "mr": "Maharashtra, India",
    "gu": "Gujarat, India",
    "pa": "Punjab, India",
    "or": "Odisha, India",
    "en": "India",
}

_PROMPT_TEMPLATE = (
    "Photorealistic news photograph in the visual style of an Indian "
    "news channel (TV9, NDTV, ETV, Zee News). "
    "The image MUST be directly relevant to this news subject: {subject}. "
    "Setting: {location}. "
    "Style: documentary news photography, natural daylight, broadcast "
    "TV quality, slight motion-blur for authenticity, 16:9 horizontal "
    "composition with the subject in the right two-thirds. "
    "No text overlays, no logos, no watermarks, no banners, no on-screen "
    "channel branding. "
    "Realistic anonymous Indian faces only — no celebrities, no politicians, "
    "no real public figures by name. Image must be safe for general "
    "audiences."
)


def _build_prompt(
    description: Optional[str],
    query: str,
    entities: list,
    topics: list,
    language: str,
) -> str:
    """Build the prompt string fed to Imagen.

    Preference order for the subject text (highest first):
      1. ``description`` — Gemini-authored one-liner. Used in compound mode.
      2. First concrete entity (e.g. a person or place name).
      3. First topic.
      4. ``query`` fallback.
    """
    if description and isinstance(description, str) and description.strip():
        subject = description.strip()
    else:
        subject = ""
        for ent in (entities or []):
            if ent and isinstance(ent, str) and ent.strip():
                subject = ent.strip()
                break
        if not subject:
            for tp in (topics or []):
                if tp and isinstance(tp, str) and tp.strip():
                    subject = tp.strip()
                    break
        if not subject:
            subject = (query or "general news scene").strip()

    location = _LANG_TO_LOCATION.get((language or "").lower(), "India")
    return _PROMPT_TEMPLATE.format(subject=subject, location=location)


# ─── Public API ───────────────────────────────────────────────────────

def is_enabled() -> bool:
    """True iff Imagen generation should be attempted.

    Two ways to disable: missing ``GEMINI_API_KEY`` (silent — same key
    pool as Gemini) or env-var kill switch ``KAIZER_IMAGEN=0``.
    """
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return False
    if os.environ.get("KAIZER_IMAGEN", "1").strip() in ("0", "false", "no"):
        return False
    return True


def generate_news_image(
    *,
    query: str = "",
    description: Optional[str] = None,
    entities: Optional[list] = None,
    topics: Optional[list] = None,
    language: str = "en",
    out_path: str,
) -> Optional[str]:
    """Generate a news image via Google Imagen and write it to ``out_path``.

    Public surface mirrors ``openai_images.generate_news_image`` with one
    addition: ``description``. In compound mode the caller passes the
    Gemini-authored per-image description; we use that verbatim as the
    subject (Gemini already wrote a tight prompt). In single mode the
    caller leaves ``description`` empty and we fall back to the
    entity/topic/query stack.

    Returns ``out_path`` on success; ``None`` on any failure (caller
    typically falls back to the OpenAI image of the same id).
    """
    if not is_enabled():
        return None

    prompt = _build_prompt(description, query, entities or [], topics or [], language)

    # Local import — keeps non-pipeline callers from paying the cost.
    try:
        from google import genai
        from google.genai import types as genai_types  # noqa: F401
    except ImportError as exc:
        print(f"    [imagen] google-genai not installed: {exc}")
        return None

    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    except Exception as exc:
        print(f"    [imagen] client init failed: {exc}")
        return None

    # Pull job / user out of env so any logged DB row gets attribution.
    try:
        _job_id  = int(os.environ.get("KAIZER_JOB_ID")  or 0) or None
        _user_id = int(os.environ.get("KAIZER_USER_ID") or 0) or None
    except (TypeError, ValueError):
        _job_id = _user_id = None

    # Optional DB logging mirroring openai_images.
    try:
        from learning.openai_log import log_openai_call as _log_call
    except Exception:
        _log_call = None

    last_err: Optional[BaseException] = None
    started = time.monotonic()
    response = None

    for attempt in (1, 2):
        try:
            if _log_call is not None:
                with _log_call(
                    db=None, user_id=_user_id, job_id=_job_id, clip_id=None,
                    model=_MODEL, purpose="bulletin-image-imagen",
                    image_size=_SIZE, image_quality="standard", image_count=1,
                ) as _call:
                    response = client.models.generate_images(
                        model=_MODEL,
                        prompt=prompt,
                    )
                    if hasattr(_call, "record_image"):
                        _call.record_image(response)
            else:
                response = client.models.generate_images(
                    model=_MODEL,
                    prompt=prompt,
                )
            break  # success
        except Exception as exc:
            last_err = exc
            msg = type(exc).__name__
            transient = msg in {"DeadlineExceeded", "ServiceUnavailable", "ResourceExhausted"} \
                        or "429" in str(exc) or "503" in str(exc) or "timeout" in str(exc).lower()
            if attempt == 1 and transient:
                print(f"    [imagen] {msg}, retrying in 2s…")
                time.sleep(2)
                continue
            print(f"    [imagen] generation failed ({msg}): {str(exc)[:200]}")
            return None
    else:
        print(f"    [imagen] generation failed after retry: {last_err}")
        return None

    # Decode and save. google-genai's GenerateImagesResponse carries the
    # rendered bytes under ``generated_images[i].image.image_bytes`` (PNG).
    try:
        gen_images = getattr(response, "generated_images", None) or []
        if not gen_images:
            print("    [imagen] response carried no images")
            return None
        img = gen_images[0]
        # SDK returns either an Image object with .image_bytes OR raw bytes.
        png_bytes = (
            getattr(getattr(img, "image", None), "image_bytes", None)
            or getattr(img, "image_bytes", None)
        )
        if not png_bytes:
            print("    [imagen] response missing image_bytes")
            return None
    except Exception as exc:
        print(f"    [imagen] response decode failed: {exc}")
        return None

    # Convert PNG → JPG so the file name + size match the OpenAI pool's
    # convention. Pillow is already installed for the rest of the pipeline.
    try:
        from PIL import Image
        with Image.open(io.BytesIO(png_bytes)) as im:
            im = im.convert("RGB")
            im.save(out_path, "JPEG", quality=88, optimize=True)
    except Exception as exc:
        # Fall back to the raw PNG if Pillow chokes — the carousel /
        # overlay code accepts both extensions.
        try:
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            print(f"    [imagen] PIL convert failed ({exc}), wrote raw PNG instead")
        except Exception as write_exc:
            print(f"    [imagen] image write failed: {write_exc}")
            return None

    elapsed_ms = int((time.monotonic() - started) * 1000)
    print(
        f"    [imagen] generated {os.path.basename(out_path)} in {elapsed_ms}ms "
        f"(${_COST_USD:.3f}, model={_MODEL})"
    )
    return out_path
