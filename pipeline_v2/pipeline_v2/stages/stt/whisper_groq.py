"""Whisper-Groq STT provider (Step 4.2).

Hosted Whisper large-v3 via Groq's OpenAI-compatible API. Same code
path our V1 production has been using successfully on Telugu / Hindi /
Telugu-English / Hindi-English code-mixed content.

Why hosted Whisper instead of local faster-whisper or Chirp 3:
  - Faster-whisper local hit hallucination loops on Telugu (see
    ``tests/fixtures/step4_diag/`` for the empirical evidence)
  - Chirp 3 returned INTERNAL on Telugu (Preview status, no SLA)
  - Groq's hosted Whisper, by contrast, is V1-proven on the same audio

Vendor feature gaps (document so Stage 2 logic accounts for them):
  - **No per-word confidence** on Groq's verbose_json response.
    ``Word.confidence`` will always be None for transcripts from this
    provider. Stage 2 should treat None as "unknown confidence", not
    "zero confidence".

File-size handling:
  - Groq's per-file cap is 25 MB on free tier, 100 MB on dev tier
    (verified 2026-05-18 from console.groq.com/docs/speech-to-text).
  - Files > 100 MB: raise ValueError with a clear mitigation hint --
    "Stage 0 should extract audio at 64kbps mono for the Groq path
    (30 min at that bitrate is <16 MB)".
  - Files 25-100 MB on free tier: log a warning but attempt the
    request anyway. The API will reject with 413 if free tier is
    actually in use; user must switch to dev tier.
  - 0-byte files: raise ValueError.

Initial-prompt biasing:
  - Whisper takes a free-text ``prompt`` (224-token cap). We construct
    it from ``brief`` + comma-joined ``names``, mirroring V1's
    production pattern.
  - No explicit mixed-script support in Groq's docs, but Whisper
    itself handles UTF-8 freely.

Tier-aware cost ledger:
  - Free tier: cost_usd = 0.0
  - Dev tier:  cost_usd = audio_duration_sec / 60 * (0.04 / 60)
  - Selected via ``GROQ_TIER=free|dev`` env var (default free).
  - For Step 4.2 acceptance we're on free tier. The dev-tier promotion
    decision happens between Step 12 and Step 13 (see project
    roadmap). One-line env change when ready.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

# Eager SDK import. If the groq package isn't installed in this env,
# this raises ImportError at module load -- caught by the auto-loader
# in pipeline_v2/stages/stt/__init__.py so the rest of the stt
# package still works without Groq. Importing eagerly (vs lazily
# inside _get_client) keeps AsyncGroq in this module's namespace so
# unit tests can patch it directly.
from groq import AsyncGroq

from pipeline_v2.models import Word, WordLevelTranscript
from pipeline_v2.stages.stt import ProviderResponse, register

logger = logging.getLogger("pipeline_v2.stt.whisper_groq")


# --- Constants ---------------------------------------------------------

# Per-file caps. Last verified 2026-05-18 from
# console.groq.com/docs/speech-to-text.
GROQ_FREE_TIER_MB = 25
GROQ_DEV_TIER_MB = 100

# Pricing. Last verified 2026-05-18 from console.groq.com/docs/rate-limits.
# Re-verify quarterly.
GROQ_FREE_USD_PER_HOUR = 0.0
GROQ_DEV_USD_PER_HOUR = 0.04


# --- Provider ---------------------------------------------------------


@register("whisper-groq")
class WhisperGroqProvider:
    """Whisper large-v3 via Groq's OpenAI-compatible API."""

    name: str = "whisper-groq"   # also set by @register (belt + suspenders)
    DEFAULT_MODEL = "whisper-large-v3"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        tier: Optional[str] = None,
    ):
        # Allow constructor overrides for tests / future per-call control.
        # In production the dispatcher instantiates with no args; env
        # vars drive behaviour.
        self.model = (
            model
            or os.environ.get("GROQ_MODEL", "").strip()
            or self.DEFAULT_MODEL
        )
        resolved_tier = (
            tier
            or os.environ.get("GROQ_TIER", "free").strip().lower()
        )
        if resolved_tier not in ("free", "dev"):
            raise ValueError(
                f"GROQ_TIER must be 'free' or 'dev', got {resolved_tier!r}"
            )
        self.tier = resolved_tier
        self._usd_per_hour = (
            GROQ_FREE_USD_PER_HOUR if resolved_tier == "free"
            else GROQ_DEV_USD_PER_HOUR
        )
        self._client = None  # lazy-init; AsyncGroq() needs the API key

    def _get_client(self):
        """Lazy AsyncGroq client. API key fetched at first use, not at
        provider instantiation -- the dispatcher constructs the
        provider eagerly and we want a clear error if the key is
        missing at call time, not at registration."""
        if self._client is not None:
            return self._client
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY env var is empty/unset. Add it to "
                "KaizerBackend/.env -- generate one at console.groq.com."
            )
        # AsyncGroq is imported at module top so test files can patch
        # it via ``pipeline_v2.stages.stt.whisper_groq.AsyncGroq``.
        self._client = AsyncGroq(api_key=api_key)
        return self._client

    def _validate_file(self, audio_path: str) -> int:
        """Pre-flight checks before the network call. Returns size_bytes.

        Permanent failures (per Step 10 D-10.3 refinement): raise
        ``PermanentSTTError`` so the orchestrator translates to
        Inngest's ``NonRetriableError``. Both conditions below are
        deterministic from the file itself -- retrying changes nothing.
        """
        # Lazy-import to avoid a circular import at provider-module-load
        # (the stt package's __init__ imports providers at the bottom).
        from pipeline_v2.stages.stt import PermanentSTTError

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        size = path.stat().st_size
        if size == 0:
            raise PermanentSTTError(
                f"empty_file: audio file {audio_path!r} is 0 bytes"
            )
        size_mb = size / (1024 * 1024)
        if size_mb > GROQ_DEV_TIER_MB:
            raise PermanentSTTError(
                f"file_too_large: audio file is {size_mb:.1f}MB which "
                f"exceeds Groq's largest per-file cap "
                f"({GROQ_DEV_TIER_MB}MB on dev tier). Mitigation: have "
                f"Stage 0 extract audio at 64kbps mono for the Groq path "
                f"-- 30 min at that bitrate is <16 MB. Path: {audio_path}"
            )
        if size_mb > GROQ_FREE_TIER_MB and self.tier == "free":
            logger.warning(
                "audio file %s is %.1fMB which exceeds the free-tier "
                "%dMB cap. Groq will reject with 413 if free tier is "
                "actually in use. Switch to dev tier (GROQ_TIER=dev) or "
                "extract audio at lower bitrate.",
                audio_path, size_mb, GROQ_FREE_TIER_MB,
            )
        return size

    def _build_prompt(self, brief: str, names: list[str]) -> Optional[str]:
        """Construct Whisper's initial_prompt -- ``<brief>. <name1>, <name2>``.

        Matches our V1 production pattern. Returns None if both inputs
        are empty (don't send an empty prompt -- it adds no biasing
        and burns tokens uselessly).
        """
        parts: list[str] = []
        brief = (brief or "").strip()
        if brief:
            parts.append(brief)
        names_clean = [n.strip() for n in (names or []) if n.strip()]
        if names_clean:
            parts.append(", ".join(names_clean))
        if not parts:
            return None
        prompt = ". ".join(parts)
        # 224-token cap. ~3 chars/token for multi-script content; 600
        # chars is the practical safety threshold.
        if len(prompt) > 600:
            logger.warning(
                "initial_prompt is %d chars; may exceed Groq's "
                "224-token cap. If the API rejects with 400 about "
                "prompt length, trim brief / names.", len(prompt),
            )
        return prompt

    async def transcribe(
        self,
        *,
        audio_path: str,
        language: Optional[str],
        brief: str = "",
        names: Optional[list[str]] = None,
    ) -> ProviderResponse:
        """Transcribe via Groq's hosted Whisper large-v3.

        Raises:
            FileNotFoundError: audio_path doesn't exist.
            ValueError: file is 0 bytes or >100 MB.
            RuntimeError: GROQ_API_KEY missing at call time.
            (Groq SDK exceptions propagate unchanged.)
        """
        self._validate_file(audio_path)
        prompt = self._build_prompt(brief, names or [])

        # Build kwargs conditionally so we omit (rather than pass None)
        # for language and prompt -- some SDK versions treat None
        # differently than absent.
        kwargs: dict = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        # Read the file synchronously -- typical audio (<50 MB) reads in
        # well under a second. We DON'T offload to a thread because the
        # network call below dominates wall time.
        with open(audio_path, "rb") as f:
            kwargs["file"] = (Path(audio_path).name, f.read())

        client = self._get_client()

        # ``with_raw_response`` exposes the HTTP request_id header for
        # the ledger. If the SDK doesn't surface it (or returns ""),
        # we generate a client-side ID so the field is never empty.
        request_id = ""
        transcription = None
        try:
            raw = await client.audio.transcriptions.with_raw_response.create(**kwargs)
            request_id = getattr(raw, "request_id", "") or ""
            transcription = raw.parse()
        except AttributeError:
            # Older SDK without with_raw_response support.
            transcription = await client.audio.transcriptions.create(**kwargs)

        if not request_id:
            request_id = f"groq-client-{uuid.uuid4().hex[:12]}"

        audio_duration_sec = float(
            getattr(transcription, "duration", 0.0) or 0.0
        )
        detected_language = getattr(transcription, "language", "") or ""
        raw_words = getattr(transcription, "words", None) or []

        # Map Groq word objects to our Word schema. Per-word confidence
        # is NOT returned by Groq (vendor gap); confidence stays None.
        words: list[Word] = []
        for w in raw_words:
            text = getattr(w, "word", "") or ""
            words.append(Word(
                w=text.strip(),
                s=float(getattr(w, "start", 0.0) or 0.0),
                e=float(getattr(w, "end", 0.0) or 0.0),
                speaker=None,
                confidence=None,                     # vendor gap
            ))

        # detected_languages list: include Whisper's detected code; if
        # the user hinted a different code, surface both so Stage 2
        # has the full signal.
        detected_languages = [detected_language] if detected_language else []
        if language and language != detected_language:
            detected_languages.append(language)

        transcript = WordLevelTranscript(
            words=words,
            duration_sec=audio_duration_sec,
            detected_languages=detected_languages,
            provider=self.name,
        )

        cost_usd = audio_duration_sec * (self._usd_per_hour / 3600.0)

        return ProviderResponse(
            transcript=transcript,
            cost_usd=cost_usd,
            request_id=request_id,
            audio_duration_sec=audio_duration_sec,
        )
