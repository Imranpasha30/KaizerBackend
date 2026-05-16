"""OpenAI image generation for the per-clip editorial image.

Why this exists
---------------
The pipeline's ``search_news_images()`` (in ``pipeline.py``) fills the
broadcast-layout image slot beside each clip. Today it queries Pexels
+ Google CSE + DuckDuckGo, which return generic stock photos that don't
match Telugu / Indian news context. A Telangana scandal segment ends up
overlaid with a stock "woman with microphone" from California.

This module adds an OpenAI ``images.generate`` path so the pipeline can
ask ``gpt-image-1`` for a contextually-appropriate news-broadcast-style
image, prompted with the per-clip subject, key entity, and language
hint.

Design rules
------------
1.  **Never raise.** A failure here must NOT block the pipeline. Return
    ``None`` and let the caller fall back to Pexels / Google / cards.
2.  **No real public figures, no celebrities, no logos.** The prompt
    template explicitly forbids them. OpenAI will refuse the request
    if a celebrity name leaks in — we treat that refusal as "fall back
    to Pexels," same as a network error.
3.  **One retry on transient errors only.** Network blips +
    rate-limit (429) get one immediate retry. Anything else returns
    ``None`` immediately.
4.  **60-second timeout.** Image gen is slow (~15-30s per image); the
    timeout sits well above p99 latency without hanging the pipeline
    forever on a stuck request.
5.  **Stdout logging** matching the Gemini pattern:
    ``[openai] generated news_03.jpg in 18432ms ($0.040)``.
    First iteration is stdout-only — admin-panel DB rows can come in a
    follow-up commit (see plan "Out of scope").

Pricing (2026-04, OpenAI image API)
-----------------------------------
- ``gpt-image-1`` size=1024×1024 quality=medium → $0.040 per image.
- ``gpt-image-1`` size=1024×1024 quality=high   → $0.190 per image.
We default to ``medium`` — visually indistinguishable from ``high`` in
the broadcast layout where the image renders at 540×960px max.
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Optional

# openai is already imported by pipeline.py:237; this module imports it
# locally only when actually called so unit tests of unrelated pipeline
# code don't pay the import cost.


# ─── Public knobs ─────────────────────────────────────────────────────

# Model + quality. gpt-image-1 medium is the sweet spot for cost / quality
# at our render size. Tweakable via env if a future iteration wants high.
_MODEL    = os.environ.get("KAIZER_OPENAI_IMAGE_MODEL",   "gpt-image-1")
_SIZE     = os.environ.get("KAIZER_OPENAI_IMAGE_SIZE",    "1024x1024")
_QUALITY  = os.environ.get("KAIZER_OPENAI_IMAGE_QUALITY", "medium")
_TIMEOUT  = float(os.environ.get("KAIZER_OPENAI_IMAGE_TIMEOUT", "60"))

# Estimated cost per image at the configured quality. Used for the log
# line only — actual billing is whatever OpenAI reports on its dashboard.
_COST_USD_BY_QUALITY = {
    "low":      0.011,
    "medium":   0.040,
    "high":     0.190,
    "standard": 0.040,   # legacy DALL·E 3 alias
    "hd":       0.080,   # legacy DALL·E 3 alias
}


# ─── Prompt construction ──────────────────────────────────────────────

# Language → location hint. Determines what country / region the image
# should depict. Telugu → Andhra Pradesh / Telangana; everything else
# falls through to a generic "India" hint.
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
    # ── Style anchor ───────────────────────────────────────────────────
    # Explicitly grounded in Indian regional news-channel aesthetics so
    # gpt-image-1 leans on the TV9 / NDTV / ETV / Zee News reference set
    # in its training data rather than stock Western broadcast imagery.
    "Photorealistic news photograph in the visual style of an Indian "
    "news channel (TV9, NDTV, ETV, Zee News). "
    # ── Relevance hook ─────────────────────────────────────────────────
    # Reinforced because the model otherwise drifts to generic newsroom
    # B-roll. The subject text is built from real entities + topics
    # extracted by Gemini from the actual transcript.
    "The image MUST be directly relevant to this news subject: {subject}. "
    "Setting: {location}. "
    # ── Composition + technical ────────────────────────────────────────
    "Style: documentary news photography, natural daylight, broadcast "
    "TV quality, slight motion-blur for authenticity, 16:9 horizontal "
    "composition with the subject in the right two-thirds. "
    # ── Hard rules ────────────────────────────────────────────────────
    "No text overlays, no logos, no watermarks, no banners, no on-screen "
    "channel branding. "
    "Realistic anonymous Indian faces only — no celebrities, no politicians, "
    "no real public figures by name. Image must be safe for general "
    "audiences."
)


def _build_prompt(query: str, entities: list, topics: list, language: str) -> str:
    """Assemble the per-image prompt from clip context."""
    # Subject preference: first concrete entity, then first topic, then
    # the raw search query as last resort.
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
    """True iff OpenAI image generation should be attempted.

    Two ways to disable: missing ``OPENAI_API_KEY`` (silent — same as
    not having a Pexels key today) or env-var kill switch
    ``KAIZER_OPENAI_IMAGES=0`` (rollback without a code change).
    """
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return False
    if os.environ.get("KAIZER_OPENAI_IMAGES", "1").strip() in ("0", "false", "no"):
        return False
    return True


def generate_news_image(
    *,
    query: str,
    entities: Optional[list] = None,
    topics: Optional[list] = None,
    language: str = "en",
    out_path: str,
) -> Optional[str]:
    """Generate a news-broadcast-style image and write it to ``out_path``.

    Parameters
    ----------
    query : str
        Search query the pipeline would have sent to Pexels — used as the
        subject fallback when no ``entities`` / ``topics`` are provided.
    entities : list[str] | None
        Gemini-extracted key entities (people, organisations, places).
        First non-empty entry becomes the prompt subject.
    topics : list[str] | None
        Gemini-extracted key topics. Used as subject fallback after
        ``entities``.
    language : str
        Two-letter language code (te, hi, en, …). Drives the location
        hint embedded in the prompt.
    out_path : str
        Absolute path where the JPG should be written. Parent directory
        must already exist.

    Returns
    -------
    str | None
        ``out_path`` on success; ``None`` on any failure (caller falls
        back to Pexels / Google / synthetic card).
    """
    if not is_enabled():
        return None

    prompt = _build_prompt(query, entities or [], topics or [], language)

    # Local import — keeps non-pipeline callers (tests, admin panel) from
    # paying the openai SDK init cost.
    try:
        import openai  # type: ignore
    except ImportError as exc:
        print(f"    [openai] SDK not installed: {exc}")
        return None

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=_TIMEOUT)

    # Pull the owning job/user out of the env the runner sets when it
    # spawns the pipeline subprocess.  Either being absent (e.g. ad-hoc
    # CLI run) just leaves the corresponding column NULL on the logged
    # OpenAiCall row — aggregate-by-day still works.
    try:
        _job_id  = int(os.environ.get("KAIZER_JOB_ID")  or 0) or None
        _user_id = int(os.environ.get("KAIZER_USER_ID") or 0) or None
    except (TypeError, ValueError):
        _job_id = _user_id = None

    # One retry on transient errors only.
    last_err: Optional[BaseException] = None
    started = time.monotonic()
    resp = None
    # Local import — keeps test envs that lack the DB happy.
    try:
        from learning.openai_log import log_openai_call as _log_openai
    except Exception:
        _log_openai = None

    for attempt in (1, 2):
        try:
            # Wrap each call so we capture even transient retries as
            # separate rows — admin can see "X rate-limited attempts
            # before success" in the timeline.
            if _log_openai is not None:
                with _log_openai(
                    db=None,
                    user_id=_user_id,
                    job_id=_job_id,
                    clip_id=None,
                    model=_MODEL,
                    purpose="bulletin-image",
                    image_size=_SIZE,
                    image_quality=_QUALITY,
                    image_count=1,
                ) as _call:
                    resp = client.images.generate(
                        model=_MODEL,
                        prompt=prompt,
                        size=_SIZE,
                        quality=_QUALITY,
                        n=1,
                    )
                    _call.record_image(resp)
            else:
                resp = client.images.generate(
                    model=_MODEL,
                    prompt=prompt,
                    size=_SIZE,
                    quality=_QUALITY,
                    n=1,
                )
            break  # success
        except Exception as exc:
            last_err = exc
            err_name = type(exc).__name__
            # Retry only on transient classes; everything else is fatal.
            transient = err_name in {
                "APIConnectionError",
                "APITimeoutError",
                "RateLimitError",
            }
            if attempt == 1 and transient:
                print(f"    [openai] {err_name}, retrying in 2s...")
                time.sleep(2)
                continue
            print(f"    [openai] generation failed ({err_name}): {str(exc)[:200]}")
            return None
    else:
        # Loop exhausted without break → both attempts errored.
        print(f"    [openai] generation failed after retry: {last_err}")
        return None

    # ── Decode the base64 PNG response and write as JPG ─────────────
    try:
        data = resp.data[0]  # type: ignore[attr-defined]
        b64 = getattr(data, "b64_json", None)
        if not b64:
            print("    [openai] response missing b64_json field")
            return None
        png_bytes = base64.b64decode(b64)
    except Exception as exc:
        print(f"    [openai] response decode failed: {exc}")
        return None

    # Convert PNG → JPG to match the existing news_NN.jpg convention and
    # keep R2 storage smaller. Pillow is already installed.
    try:
        from PIL import Image
        with Image.open(io.BytesIO(png_bytes)) as im:
            im = im.convert("RGB")
            im.save(out_path, "JPEG", quality=88, optimize=True)
    except Exception as exc:
        # Fall back to writing the raw PNG if Pillow chokes — caller's
        # downstream code accepts both extensions.
        try:
            with open(out_path, "wb") as f:
                f.write(png_bytes)
        except Exception as write_exc:
            print(f"    [openai] image write failed: {write_exc}")
            return None
        print(f"    [openai] PIL convert failed ({exc}), wrote raw PNG instead")

    elapsed_ms = int((time.monotonic() - started) * 1000)
    cost = _COST_USD_BY_QUALITY.get(_QUALITY, _COST_USD_BY_QUALITY["medium"])
    print(
        f"    [openai] generated {os.path.basename(out_path)} in {elapsed_ms}ms "
        f"(${cost:.3f}, model={_MODEL}, quality={_QUALITY})"
    )
    return out_path


# ─── Bulletin-level pool generator ────────────────────────────────────
#
# Generate ONE small set of diverse images for an ENTIRE bulletin instead
# of per-story. The carousel cycles through this pool across all stories.
#
# Why job-level beats per-story:
#  1. 6 images per job  ≈ 75% cheaper than 25 per job (~$0.24 vs $0.96)
#  2. ~75% faster (OpenAI Tier 1 caps at 5 images/min — 6 fits in 1 min)
#  3. Less repetitive: we vary the subject across the 6 calls so the
#     pool actually looks different, whereas per-story prompts the same
#     topic 4–6 times in a row and gets near-duplicates.

def generate_bulletin_image_pool(
    *,
    pool_size: int,
    entities: list,
    topics: list,
    queries: list,
    language: str,
    out_dir: str,
) -> list[str]:
    """Generate ``pool_size`` visually distinct, news-relevant images
    using the FULL bulletin context (everything Gemini extracted across
    every story). Each call uses a different subject so the pool is
    actually varied.

    Returns the list of image file paths written under ``out_dir``.
    Empty list when OpenAI is disabled or every attempt failed — caller
    falls back to existing per-story flow.
    """
    if not is_enabled():
        return []

    os.makedirs(out_dir, exist_ok=True)

    # ── Build a deduped pool of subject candidates (entities first,
    # topics next, queries last). Each pool slot picks a different
    # subject so the resulting 6 images cover the full story range
    # rather than 6 variants of the same prompt.
    seen: set[str] = set()
    subjects: list[str] = []

    def _push(s: str) -> None:
        s = (s or "").strip()
        if not s:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        subjects.append(s)

    for e in (entities or []):
        if isinstance(e, str): _push(e)
    for t in (topics or []):
        if isinstance(t, str): _push(t)
    for q in (queries or []):
        if isinstance(q, str): _push(q[:80])

    if not subjects:
        subjects = ["general Indian news scene"]

    out: list[str] = []
    for i in range(pool_size):
        # Rotate through the subject pool so consecutive images differ.
        subject = subjects[i % len(subjects)]
        out_path = os.path.join(out_dir, f"news_{i+1:02d}.jpg")
        # Skip if already exists from a prior cache (resume safety).
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 5000:
            print(f"    [openai-pool] reuse cached {os.path.basename(out_path)}")
            out.append(out_path)
            continue
        # Reuse the single-image generator. Each call has its own
        # built-in retry on rate-limit (one retry, 2s wait).
        result = generate_news_image(
            query=subject,
            entities=[subject],     # forces _build_prompt to pick this subject
            topics=topics or [],
            language=language,
            out_path=out_path,
        )
        if result:
            out.append(result)

    if out:
        print(f"  [openai-pool] generated {len(out)}/{pool_size} job-level "
              f"images in {os.path.basename(out_dir)}")
    return out
