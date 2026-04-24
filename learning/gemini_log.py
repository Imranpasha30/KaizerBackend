"""Gemini call accounting wrapper.

Exposes a single context manager — `log_gemini_call(...)` — every Gemini
call site wraps so the admin panel can sum tokens + cost + per-user /
per-job usage.

Usage
-----

    from learning.gemini_log import log_gemini_call

    with log_gemini_call(
        db=db, user_id=user.id, job_id=job.id, clip_id=None,
        model="gemini-2.0-flash", purpose="seo",
    ) as call:
        resp = gemini_model.generate_content(prompt)
        call.record(resp)       # extracts usage_metadata
        return resp

Rules
-----
*   Exceptions from the wrapped call propagate — we do NOT swallow them.
*   Exceptions from the logger itself ARE swallowed — the caller's Gemini
    call must succeed regardless of whether bookkeeping works.
*   Cost estimates are a best-effort lookup against `COST_PER_1K_TOKENS`.
    Unknown models fall back to a conservative flash-tier rate.
*   Call `record(resp)` before the `with` block exits to capture usage
    metadata.  Forgetting is fine — we still write a row with zeroes so
    the call count is accurate.
"""
from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Optional

from sqlalchemy.orm import Session

import models


logger = logging.getLogger("kaizer.gemini_log")


# ─── Public published Gemini pricing per 1K tokens (USD) ──────────────────
# Source: Google AI pricing page (2026-04).  Kept in-line rather than in
# config because rates rarely change and exfiltrating them from env adds
# no security value.  Any unknown model falls back to _DEFAULT_RATE.
#
# Maintenance: when Google bumps a price, bump the number here.  Historical
# rows keep their old cost_usd (we don't rewrite history).
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # Gemini 2.5 family
    "gemini-2.5-pro":         {"input": 0.00125,  "output": 0.010},
    "gemini-2.5-flash":       {"input": 0.000075, "output": 0.0003},
    "gemini-2.5-flash-lite":  {"input": 0.000038, "output": 0.00015},
    # Gemini 2.0 family
    "gemini-2.0-flash":       {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash-exp":   {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash-lite":  {"input": 0.000038, "output": 0.00015},
    # Gemini 1.5 family (retiring, kept for historical completeness)
    "gemini-1.5-pro":         {"input": 0.00125,  "output": 0.005},
    "gemini-1.5-flash":       {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-flash-8b":    {"input": 0.0000375,"output": 0.00015},
}

# Fallback when the model name isn't in the table (new models, typos).
_DEFAULT_RATE = {"input": 0.0001, "output": 0.0004}


def _rate_for(model: str) -> dict[str, float]:
    # Exact match first; then prefix match (so "gemini-2.0-flash-001" maps
    # onto "gemini-2.0-flash").
    if model in COST_PER_1K_TOKENS:
        return COST_PER_1K_TOKENS[model]
    for known, rates in COST_PER_1K_TOKENS.items():
        if model.startswith(known):
            return rates
    return _DEFAULT_RATE


class _Call:
    """Handle returned from `log_gemini_call(...)` — the caller feeds the
    Gemini SDK response into `.record()` to capture usage metadata."""

    def __init__(self, model: str, purpose: str,
                 file_bytes: int = 0, video_duration_s: float = 0.0):
        self.model = model
        self.purpose = purpose
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        # Video accounting — set by the caller for video upload calls.
        # Mutable so the caller can also update via .add_file(...) if their
        # call uses multiple file parts.
        self.file_bytes = int(file_bytes or 0)
        self.video_duration_s = float(video_duration_s or 0.0)
        self.cost_usd = 0.0
        self.status = "ok"     # overwritten to "error" if the with-block raises
        self.error = ""
        self._recorded = False

    def add_file(self, bytes_count: int, duration_s: float = 0.0) -> None:
        """Accumulate file-part accounting for calls that upload multiple
        files (e.g. frame montages, chapter splits)."""
        self.file_bytes += int(bytes_count or 0)
        self.video_duration_s += float(duration_s or 0.0)

    def record(self, resp: Any) -> None:
        """Extract token counts from a Gemini SDK response object.

        Supports both google-generativeai (legacy) responses that expose
        `usage_metadata` with snake_case keys, and the newer google-genai
        SDK which exposes `.usage_metadata.prompt_token_count` etc.
        Silently ignores missing attributes — partial data is better
        than no row."""
        try:
            meta = getattr(resp, "usage_metadata", None)
            if meta is None:
                return

            def _pull(*names):
                for n in names:
                    v = getattr(meta, n, None)
                    if v is None and isinstance(meta, dict):
                        v = meta.get(n)
                    if v is not None:
                        try:
                            return int(v)
                        except (TypeError, ValueError):
                            continue
                return 0

            self.prompt_tokens = _pull("prompt_token_count", "promptTokenCount")
            self.output_tokens = _pull(
                "candidates_token_count", "candidatesTokenCount",
                "output_token_count",     "outputTokenCount",
            )
            total = _pull("total_token_count", "totalTokenCount")
            self.total_tokens = total or (self.prompt_tokens + self.output_tokens)
            rate = _rate_for(self.model)
            self.cost_usd = (
                (self.prompt_tokens / 1000.0) * rate.get("input", 0.0)
                + (self.output_tokens / 1000.0) * rate.get("output", 0.0)
            )
            self._recorded = True
        except Exception as e:      # never raise — logging must never break callers
            logger.warning("gemini_log.record failed: %s", e)

    def mark_rate_limited(self) -> None:
        """Optional helper for callers who detect 429 before the wrapper sees
        the exception.  Stamps the row with status='rate_limited' instead
        of 'error'."""
        self.status = "rate_limited"


@contextmanager
def log_gemini_call(
    db: Optional[Session],
    *,
    user_id: Optional[int] = None,
    job_id: Optional[int] = None,
    clip_id: Optional[int] = None,
    model: str,
    purpose: str = "",
    file_bytes: int = 0,
    video_duration_s: float = 0.0,
):
    """Context manager that writes a `GeminiCall` row when the `with` block
    exits — whether it succeeded, raised, or the caller forgot `.record()`.

    Parameters
    ----------
    db : Session | None
        SQLAlchemy session.  If None we skip the write (useful for CLI scripts
        that don't have a DB handy but still share Gemini calling code).
    user_id, job_id, clip_id : optional foreign keys for drill-down views.
    model : model name exactly as sent to the SDK — used for pricing lookup.
    purpose : short tag like "seo" / "script" / "thumbnail" / "translation".

    Yields
    ------
    _Call object — call `.record(resp)` on the Gemini SDK response.
    """
    call = _Call(
        model=model, purpose=purpose,
        file_bytes=file_bytes, video_duration_s=video_duration_s,
    )
    start = time.monotonic()
    exc: Optional[BaseException] = None
    try:
        yield call
    except BaseException as e:
        exc = e
        call.status = "error"
        # Classify 429s as rate_limited so admin panel shows them separately.
        msg = str(e)
        low = msg.lower()
        if "429" in msg or "resourceexhausted" in low or "rate limit" in low or "quota" in low:
            call.status = "rate_limited"
        call.error = (msg or e.__class__.__name__)[:1000]
        raise
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        # If the caller didn't supply a session we open a short-lived one via
        # SessionLocal so background jobs (learning corpus, trending radar,
        # upload worker) also get their calls tracked.  If SessionLocal isn't
        # importable for any reason we silently skip the write.
        _owned_session = False
        session: Optional[Session] = db
        try:
            if session is None:
                try:
                    from database import SessionLocal  # local import → avoid cycles
                    session = SessionLocal()
                    _owned_session = True
                except Exception:
                    session = None

            if session is not None:
                row = models.GeminiCall(
                    user_id=user_id,
                    job_id=job_id,
                    clip_id=clip_id,
                    model=model or "",
                    purpose=purpose or "",
                    prompt_tokens=int(call.prompt_tokens or 0),
                    output_tokens=int(call.output_tokens or 0),
                    total_tokens=int(call.total_tokens or 0),
                    file_bytes=int(call.file_bytes or 0),
                    video_duration_s=float(call.video_duration_s or 0.0),
                    cost_usd=float(call.cost_usd or 0.0),
                    latency_ms=latency_ms,
                    status=call.status,
                    error=call.error,
                )
                session.add(row)
                session.commit()
        except Exception as log_err:
            # Logging must NEVER break the caller — swallow + log warning.
            logger.warning(
                "gemini_log: failed to persist GeminiCall row (model=%s purpose=%s): %s\n%s",
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
    # If an exception was raised inside the with-block, the `raise` above
    # already propagated it before `finally` ran.  Nothing more to do.
