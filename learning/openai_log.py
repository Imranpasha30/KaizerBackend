"""OpenAI API call accounting wrapper — mirror of ``gemini_log.py``.

Every site that calls the OpenAI SDK should wrap the call in
``log_openai_call(...)``.  The wrapper writes one ``OpenAiCall`` row per
call (success or failure, with latency + estimated $ cost) so the
admin Usage dashboard can sum spend by user / job / video / model.

Usage
-----

    from learning.openai_log import log_openai_call

    with log_openai_call(
        db=None, user_id=user_id, job_id=job_id, clip_id=clip_id,
        model="gpt-image-1", purpose="bulletin-image",
        image_size="1024x1024", image_quality="medium", image_count=1,
    ) as call:
        resp = client.images.generate(...)
        call.record_image(resp)         # picks up output[0].b64_json count etc.
        return resp

Rules
-----
*   Exceptions from the wrapped call propagate — we do NOT swallow them.
*   Exceptions from the logger itself ARE swallowed — the OpenAI call
    must succeed regardless of whether bookkeeping works.
*   Cost is estimated from ``IMAGE_COST_USD`` and ``TEXT_COST_PER_1K_TOKENS``
    tables maintained in this file.  Unknown sizes/qualities fall back to a
    conservative medium-1024 rate.
*   If the caller doesn't pass a ``db`` session we open a short-lived one
    via ``SessionLocal`` so background jobs (compose pipeline, runner.py
    subprocess) are also tracked.
"""
from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Optional

from sqlalchemy.orm import Session

import models


logger = logging.getLogger("kaizer.openai_log")


# ─── Public OpenAI pricing (USD, as of 2026-04) ─────────────────────
# Source: https://openai.com/pricing  +  gpt-image-1 page.
# Maintenance: when OpenAI changes a price, bump the number here.
# Historical rows keep their old cost_usd (we don't rewrite history).
#
# gpt-image-1 prices are per IMAGE.  Other text/audio models price per
# 1K tokens — same shape as gemini_log.COST_PER_1K_TOKENS.
IMAGE_COST_USD: dict[tuple[str, str, str], float] = {
    # (model, size, quality)
    ("gpt-image-1", "1024x1024", "low"):    0.011,
    ("gpt-image-1", "1024x1024", "medium"): 0.042,
    ("gpt-image-1", "1024x1024", "high"):   0.167,
    ("gpt-image-1", "1024x1536", "low"):    0.016,
    ("gpt-image-1", "1024x1536", "medium"): 0.063,
    ("gpt-image-1", "1024x1536", "high"):   0.25,
    ("gpt-image-1", "1536x1024", "low"):    0.016,
    ("gpt-image-1", "1536x1024", "medium"): 0.063,
    ("gpt-image-1", "1536x1024", "high"):   0.25,
}
_DEFAULT_IMAGE_COST = 0.042   # fall back to medium-1024 if unknown size/quality

# Text / audio call pricing (per 1K tokens).
TEXT_COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o":        {"input": 0.0025,  "output": 0.010},
    "gpt-4o-mini":   {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":   {"input": 0.010,   "output": 0.030},
    "gpt-3.5-turbo": {"input": 0.0005,  "output": 0.0015},
    # Whisper bills per audio minute, not tokens — we don't track those
    # here.  Add a separate column if/when we ship transcription.
}
_DEFAULT_TEXT_RATE = {"input": 0.002, "output": 0.006}


def _image_cost(model: str, size: str, quality: str, count: int) -> float:
    """Look up per-image cost × count.  Unknown combos → default rate."""
    key = (model or "", size or "", (quality or "").lower())
    unit = IMAGE_COST_USD.get(key, _DEFAULT_IMAGE_COST)
    return float(unit) * max(0, int(count or 0))


def _text_cost(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """Look up per-token cost.  Unknown model → default rate."""
    rate = TEXT_COST_PER_1K_TOKENS.get(model)
    if rate is None:
        for known, r in TEXT_COST_PER_1K_TOKENS.items():
            if model.startswith(known):
                rate = r
                break
    if rate is None:
        rate = _DEFAULT_TEXT_RATE
    return (
        (int(prompt_tokens or 0) / 1000.0) * rate.get("input", 0.0)
        + (int(output_tokens or 0) / 1000.0) * rate.get("output", 0.0)
    )


class _Call:
    """Handle returned from ``log_openai_call(...)`` — the caller feeds the
    OpenAI SDK response into ``.record_image()`` / ``.record_text()`` to
    capture usage metadata."""

    def __init__(
        self,
        model: str,
        purpose: str,
        image_size: str = "",
        image_quality: str = "",
        image_count: int = 0,
    ):
        self.model           = model
        self.purpose         = purpose
        self.image_size      = image_size or ""
        self.image_quality   = (image_quality or "").lower()
        self.image_count     = int(image_count or 0)
        self.prompt_tokens   = 0
        self.output_tokens   = 0
        self.total_tokens    = 0
        self.cost_usd        = 0.0
        self.status          = "ok"
        self.error           = ""

    def record_image(self, resp: Any) -> None:
        """For image-gen calls.  We trust the constructor's ``image_count``
        when set; otherwise we try to read ``resp.data`` length."""
        try:
            if not self.image_count:
                data = getattr(resp, "data", None) or []
                self.image_count = len(data) if hasattr(data, "__len__") else 0
            self.cost_usd = _image_cost(
                self.model, self.image_size, self.image_quality, self.image_count,
            )
        except Exception as e:
            logger.warning("openai_log.record_image failed: %s", e)

    def record_text(self, resp: Any) -> None:
        """For chat / completion calls.  Reads ``resp.usage`` (prompt_tokens,
        completion_tokens, total_tokens)."""
        try:
            usage = getattr(resp, "usage", None)
            if usage is None:
                return

            def _pull(*names):
                for n in names:
                    v = getattr(usage, n, None)
                    if v is None and isinstance(usage, dict):
                        v = usage.get(n)
                    if v is not None:
                        try:
                            return int(v)
                        except (TypeError, ValueError):
                            continue
                return 0

            self.prompt_tokens = _pull("prompt_tokens", "promptTokens")
            self.output_tokens = _pull(
                "completion_tokens", "completionTokens",
                "output_tokens",     "outputTokens",
            )
            total = _pull("total_tokens", "totalTokens")
            self.total_tokens = total or (self.prompt_tokens + self.output_tokens)
            self.cost_usd = _text_cost(self.model, self.prompt_tokens, self.output_tokens)
        except Exception as e:
            logger.warning("openai_log.record_text failed: %s", e)

    def mark_rate_limited(self) -> None:
        """Optional helper for callers who detect 429 before the wrapper
        sees the exception."""
        self.status = "rate_limited"


@contextmanager
def log_openai_call(
    db: Optional[Session],
    *,
    user_id: Optional[int] = None,
    job_id: Optional[int] = None,
    clip_id: Optional[int] = None,
    model: str,
    purpose: str = "",
    image_size: str = "",
    image_quality: str = "",
    image_count: int = 0,
):
    """Context manager that writes an ``OpenAiCall`` row when the ``with``
    block exits — whether it succeeded or raised.

    ``image_size`` / ``image_quality`` / ``image_count`` apply to
    gpt-image-1 calls; leave them blank for text / chat / completion
    calls (which fill the token columns via ``.record_text(resp)``).
    """
    call = _Call(
        model=model, purpose=purpose,
        image_size=image_size, image_quality=image_quality, image_count=image_count,
    )
    start = time.monotonic()
    try:
        yield call
    except BaseException as e:
        call.status = "error"
        msg = str(e)
        low = msg.lower()
        if "429" in msg or "rate limit" in low or "quota" in low:
            call.status = "rate_limited"
        call.error = (msg or e.__class__.__name__)[:1000]
        raise
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        # Pre-compute cost for image calls even when record_image() wasn't
        # called — we already know size/quality/count from the constructor.
        if call.cost_usd == 0.0 and call.image_count and call.model == "gpt-image-1":
            call.cost_usd = _image_cost(
                call.model, call.image_size, call.image_quality, call.image_count,
            )

        _owned_session = False
        session: Optional[Session] = db
        try:
            if session is None:
                try:
                    from database import SessionLocal
                    session = SessionLocal()
                    _owned_session = True
                except Exception:
                    session = None

            if session is not None:
                row = models.OpenAiCall(
                    user_id=user_id,
                    job_id=job_id,
                    clip_id=clip_id,
                    model=model or "",
                    purpose=purpose or "",
                    image_size=call.image_size,
                    image_quality=call.image_quality,
                    image_count=int(call.image_count or 0),
                    prompt_tokens=int(call.prompt_tokens or 0),
                    output_tokens=int(call.output_tokens or 0),
                    total_tokens=int(call.total_tokens or 0),
                    cost_usd=float(call.cost_usd or 0.0),
                    latency_ms=latency_ms,
                    status=call.status,
                    error=call.error,
                )
                session.add(row)
                session.commit()
        except Exception as log_err:
            logger.warning(
                "openai_log: failed to persist OpenAiCall row "
                "(model=%s purpose=%s): %s\n%s",
                model, purpose, log_err, traceback.format_exc(limit=3),
            )
            try:
                if session is not None:
                    session.rollback()
            except Exception:
                pass
        finally:
            if _owned_session and session is not None:
                try:
                    session.close()
                except Exception:
                    pass
