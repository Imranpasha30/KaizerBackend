"""Claude / Anthropic call accounting wrapper.

Single context manager — ``log_anthropic_call(...)`` — every Claude call
site wraps so the admin Usage panel can sum tokens + cost + per-user /
per-job Claude spend alongside Gemini and OpenAI.

Usage
-----
    from learning.claude_log import log_anthropic_call

    with log_anthropic_call(
        db=db, user_id=uid, job_id=jid, clip_id=None,
        model="claude-opus-4-7", purpose="cut-plan",
    ) as call:
        resp = client.messages.create(...)
        call.record(resp)        # extracts resp.usage.{input,output,cache_*}_tokens
        return resp

Rules (identical contract to gemini_log)
----------------------------------------
* Exceptions from the wrapped call propagate; exceptions from the logger
  itself are swallowed — Claude calls must succeed regardless of bookkeeping.
* ``db=None`` → the wrapper opens a short-lived SessionLocal so background
  jobs (V4 render subprocess, Express) still get tracked; ``user_id``/
  ``job_id`` may be None (the row is still written for global cost totals).
* Cost is a best-effort lookup against ``COST_PER_1K_TOKENS``; unknown models
  fall back to a conservative Sonnet-tier rate.
"""
from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Optional

from sqlalchemy.orm import Session

import models


logger = logging.getLogger("kaizer.claude_log")


# ─── Published Anthropic pricing per 1K tokens (USD) ──────────────────────
# Source: Anthropic pricing (per-1M ÷ 1000). Cache-read is ~10% of input;
# cache-write (5-min) is ~1.25× input. Bump here when Anthropic changes
# prices; historical rows keep their old cost_usd.
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # Opus tier (~$15 in / $75 out per 1M)
    "claude-opus-4-8":   {"input": 0.015,  "output": 0.075, "cache_read": 0.0015,  "cache_write": 0.01875},
    "claude-opus-4-7":   {"input": 0.015,  "output": 0.075, "cache_read": 0.0015,  "cache_write": 0.01875},
    "claude-opus-4":     {"input": 0.015,  "output": 0.075, "cache_read": 0.0015,  "cache_write": 0.01875},
    # Sonnet tier (~$3 in / $15 out per 1M)
    "claude-sonnet-4-6": {"input": 0.003,  "output": 0.015, "cache_read": 0.0003,  "cache_write": 0.00375},
    "claude-sonnet-4":   {"input": 0.003,  "output": 0.015, "cache_read": 0.0003,  "cache_write": 0.00375},
    # Haiku tier (~$1 in / $5 out per 1M)
    "claude-haiku-4-5":  {"input": 0.001,  "output": 0.005, "cache_read": 0.0001,  "cache_write": 0.00125},
}

_DEFAULT_RATE = {"input": 0.003, "output": 0.015, "cache_read": 0.0003, "cache_write": 0.00375}


def _rate_for(model: str) -> dict[str, float]:
    if model in COST_PER_1K_TOKENS:
        return COST_PER_1K_TOKENS[model]
    for known, rates in COST_PER_1K_TOKENS.items():
        if model.startswith(known):
            return rates
    return _DEFAULT_RATE


class _Call:
    """Handle from ``log_anthropic_call(...)`` — feed the Anthropic SDK
    response into ``.record()`` to capture usage."""

    def __init__(self, model: str, purpose: str):
        self.model = model
        self.purpose = purpose
        self.prompt_tokens = 0        # Anthropic input_tokens
        self.output_tokens = 0
        self.total_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.cost_usd = 0.0
        self.status = "ok"
        self.error = ""
        self._recorded = False

    def record(self, resp: Any) -> None:
        """Extract token counts from an Anthropic ``messages.create`` response
        (``resp.usage`` with input_tokens / output_tokens /
        cache_read_input_tokens / cache_creation_input_tokens). Also accepts a
        plain dict. Never raises."""
        try:
            usage = getattr(resp, "usage", None)
            if usage is None and isinstance(resp, dict):
                usage = resp.get("usage")
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

            self.prompt_tokens = _pull("input_tokens", "prompt_tokens")
            self.output_tokens = _pull("output_tokens", "completion_tokens")
            self.cache_read_tokens = _pull("cache_read_input_tokens")
            self.cache_write_tokens = _pull("cache_creation_input_tokens")
            self.total_tokens = self.prompt_tokens + self.output_tokens
            r = _rate_for(self.model)
            self.cost_usd = (
                (self.prompt_tokens / 1000.0) * r.get("input", 0.0)
                + (self.output_tokens / 1000.0) * r.get("output", 0.0)
                + (self.cache_read_tokens / 1000.0) * r.get("cache_read", 0.0)
                + (self.cache_write_tokens / 1000.0) * r.get("cache_write", 0.0)
            )
            self._recorded = True
        except Exception as e:
            logger.warning("claude_log.record failed: %s", e)

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0,
                      cost_usd: Optional[float] = None) -> None:
        """For call sites that already computed usage/cost (e.g. V2 stage 2):
        set values directly instead of parsing a response object."""
        try:
            self.prompt_tokens = int(input_tokens or 0)
            self.output_tokens = int(output_tokens or 0)
            self.total_tokens = self.prompt_tokens + self.output_tokens
            if cost_usd is not None:
                self.cost_usd = float(cost_usd)
            else:
                r = _rate_for(self.model)
                self.cost_usd = (
                    (self.prompt_tokens / 1000.0) * r.get("input", 0.0)
                    + (self.output_tokens / 1000.0) * r.get("output", 0.0)
                )
            self._recorded = True
        except Exception as e:
            logger.warning("claude_log.record_tokens failed: %s", e)


@contextmanager
def log_anthropic_call(
    db: Optional[Session],
    *,
    user_id: Optional[int] = None,
    job_id: Optional[int] = None,
    clip_id: Optional[int] = None,
    model: str,
    purpose: str = "",
):
    """Write an ``AnthropicCall`` row when the ``with`` block exits — whether
    it succeeded, raised, or the caller forgot ``.record()``."""
    call = _Call(model=model, purpose=purpose)
    start = time.monotonic()
    try:
        yield call
    except BaseException as e:
        call.status = "error"
        msg = str(e)
        low = msg.lower()
        if "429" in msg or "overloaded" in low or "rate limit" in low or "quota" in low:
            call.status = "rate_limited"
        call.error = (msg or e.__class__.__name__)[:1000]
        raise
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        _owned = False
        session: Optional[Session] = db
        try:
            if session is None:
                try:
                    from database import SessionLocal
                    session = SessionLocal()
                    _owned = True
                except Exception:
                    session = None
            if session is not None:
                row = models.AnthropicCall(
                    user_id=user_id,
                    job_id=job_id,
                    clip_id=clip_id,
                    model=model or "",
                    purpose=purpose or "",
                    prompt_tokens=int(call.prompt_tokens or 0),
                    output_tokens=int(call.output_tokens or 0),
                    total_tokens=int(call.total_tokens or 0),
                    cache_read_tokens=int(call.cache_read_tokens or 0),
                    cache_write_tokens=int(call.cache_write_tokens or 0),
                    cost_usd=float(call.cost_usd or 0.0),
                    latency_ms=latency_ms,
                    status=call.status,
                    error=call.error,
                )
                session.add(row)
                session.commit()
        except Exception as log_err:
            logger.warning(
                "claude_log: failed to persist AnthropicCall row (model=%s purpose=%s): %s\n%s",
                model, purpose, log_err, traceback.format_exc(limit=3),
            )
            try:
                if session is not None:
                    session.rollback()
            except Exception:
                pass
        finally:
            if _owned and session is not None:
                try:
                    session.close()
                except Exception:
                    pass
