"""Stage 1 STT abstraction layer.

What this module provides:

  - ``TranscriptionProvider`` Protocol  -- the contract every backend
    (Chirp 3 / Whisper-Groq / Deepgram / AssemblyAI / Sarvam / future)
    must satisfy.
  - ``ProviderResponse`` dataclass      -- what providers return.
  - ``PROVIDERS`` registry              -- ``dict[str, type[TranscriptionProvider]]``
    populated by the ``@register("name")`` decorator at provider-module
    import time.
  - ``run_stage_1(...)`` dispatcher     -- looks up the provider, calls
    it, validates the word-level contract, wraps in ``Stage1Output``,
    optionally persists transcript + metadata to ``out_dir``.

Architectural contract -- DO NOT DEGRADE
========================================
Every provider MUST emit word-level timestamps. A response with an
empty word list OR with words missing ``s``/``e`` is an architectural
violation and the dispatcher raises. Stage 2 cuts on word boundaries
and Stage 3c snaps images to words -- segment-level fallback breaks
the downstream pipeline. There is no graceful degradation here.

Provider modules in this package register themselves at import time:

  @register("chirp3")
  class Chirp3Provider:
      async def transcribe(self, *, audio_path, language, brief, names): ...

Concrete provider modules are NOT imported here -- that's done by the
provider files themselves at the bottom of their module (so missing
vendor SDKs fail at provider-import, not at this package's import).
The dispatcher will surface a clear error if a requested provider name
isn't in the registry.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from pipeline_v2.models import Stage1Output, WordLevelTranscript

logger = logging.getLogger("pipeline_v2.stt")


# ---------------------------------------------------------------------- #
# Contract                                                               #
# ---------------------------------------------------------------------- #


@dataclass
class ProviderResponse:
    """What a provider's ``transcribe()`` method returns to the dispatcher.

    ``transcript`` is the typed payload (the dispatcher does not modify
    it after receiving). ``cost_usd``, ``request_id``,
    ``audio_duration_sec`` are provider-computed telemetry that flow
    into the per-job STT ledger.
    """

    transcript: WordLevelTranscript
    cost_usd: float
    request_id: str
    audio_duration_sec: float


class PermanentSTTError(Exception):
    """Raised by STT providers for KNOWN-PERMANENT failures.

    The Step 10 orchestrator catches this at the Stage 1 step
    boundary and translates it to Inngest's ``NonRetriableError`` so
    Inngest skips the retry burn and routes straight to the
    dead-letter path. The job is marked ``failed`` with a clear
    ``Job.error = "permanent: <reason>"`` row.

    Conditions providers SHOULD raise this for (per Step 10 D-10.3
    refinement):

      * Audio file too short (under provider minimum, typically 1-3 s)
      * Unsupported language (not in provider's locale list)
      * Audio format corrupt / unparseable
      * File size exceeds provider's hard cap
      * 0-byte / empty audio file

    Conditions providers MUST NOT raise this for (these should be
    regular exceptions so Inngest retries):

      * Transient HTTP errors (5xx, timeouts, rate limits)
      * Auth failures (operator-fixable, often transient)
      * Provider-internal errors with no documented permanence
      * Network errors

    Convention: the exception message starts with a short slug
    identifying the failure category, then a colon, then human-
    readable detail. The orchestrator surfaces this verbatim in
    ``Job.error`` via the ``permanent: <message>`` prefix it adds.

    Example slugs:
      * "audio_too_short: 0.6s under provider min 1.0s"
      * "unsupported_language: 'xyz' not in {en, hi, te, ...}"
      * "audio_corrupt: ffprobe could not decode"
      * "file_too_large: 600MB exceeds Deepgram free-tier 25MB"
      * "empty_file: audio is 0 bytes"
    """


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Provider contract.

    Implementations are concrete classes decorated with ``@register(...)``.
    The ``name`` attribute is set by the decorator -- providers don't
    populate it themselves.
    """

    name: str  # set by @register("...")

    async def transcribe(
        self,
        *,
        audio_path: str,
        language: Optional[str],
        brief: str = "",
        names: Optional[list[str]] = None,
    ) -> ProviderResponse:
        ...


# ---------------------------------------------------------------------- #
# Registry                                                               #
# ---------------------------------------------------------------------- #


PROVIDERS: dict[str, type[TranscriptionProvider]] = {}


def register(name: str):
    """Class decorator: add a TranscriptionProvider class to PROVIDERS.

    Sets ``cls.name = name`` so the registry key and the class's name
    attribute cannot diverge.

    Raises:
        ValueError: if ``name`` is empty or already registered.
    """
    if not name or not name.strip():
        raise ValueError("provider name must be non-empty")

    def _decorator(cls: type[TranscriptionProvider]) -> type[TranscriptionProvider]:
        if name in PROVIDERS:
            raise ValueError(
                f"provider name {name!r} is already registered to "
                f"{PROVIDERS[name].__qualname__}"
            )
        cls.name = name        # source-of-truth for the registry key
        PROVIDERS[name] = cls
        logger.debug("registered STT provider %r -> %s", name, cls.__qualname__)
        return cls

    return _decorator


# ---------------------------------------------------------------------- #
# Dispatcher                                                             #
# ---------------------------------------------------------------------- #


def _validate_word_level_contract(
    transcript: WordLevelTranscript,
    *,
    provider: str,
) -> None:
    """Enforce the architectural contract: word-level timestamps only.

    Raises ``RuntimeError`` with a clear message on any violation. This
    runs in production -- segment-level data must not propagate.
    """
    if not transcript.words:
        raise RuntimeError(
            f"STT provider {provider!r} returned a transcript with zero "
            f"words. Word-level timestamps are mandatory; segment-only or "
            f"empty responses cannot flow into Stage 2."
        )
    if transcript.provider != provider:
        raise RuntimeError(
            f"STT provider {provider!r} returned transcript.provider="
            f"{transcript.provider!r}. Provider must set the field to match "
            f"its registry key."
        )
    # Sanity-check the first few words. Pydantic already enforces s and e
    # are floats (not None), so what we're guarding against here is
    # zero-duration fakes or implausible negatives.
    for i, w in enumerate(transcript.words[: min(5, len(transcript.words))]):
        if w.e < w.s:
            raise RuntimeError(
                f"STT provider {provider!r} returned word {i} with "
                f"end < start ({w.e} < {w.s}); aborting before this "
                f"propagates to Stage 2."
            )


def _avg_confidence(transcript: WordLevelTranscript) -> Optional[float]:
    """Mean of per-word confidences; None if no word has confidence."""
    confs = [w.confidence for w in transcript.words if w.confidence is not None]
    if not confs:
        return None
    return sum(confs) / len(confs)


def _detected_language(
    transcript: WordLevelTranscript,
    language_hint: Optional[str],
) -> str:
    """First detected language, or the hint, or "unknown" as a last resort."""
    if transcript.detected_languages:
        return transcript.detected_languages[0]
    return language_hint or "unknown"


def _write_outputs(
    out_dir: Path,
    transcript: WordLevelTranscript,
    metadata: dict,
) -> tuple[str, str]:
    """Persist transcript.json + stt_metadata.json. Returns (paths)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.json"
    transcript_path.write_text(
        transcript.model_dump_json(indent=2),
        encoding="utf-8",
    )
    metadata_path = out_dir / "stt_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(transcript_path), str(metadata_path)


async def run_stage_1(
    audio_path: str,
    *,
    provider: str,
    language_hint: Optional[str] = None,
    brief: str = "",
    names: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
) -> Stage1Output:
    """Dispatch a transcription request to the named provider.

    Args:
        audio_path: Path to a local audio file (mp3 / wav / m4a / any
            ffmpeg-decodable container the provider supports).
        provider: Registry key from ``PROVIDERS`` (e.g. ``"chirp3"``,
            ``"whisper-groq"``). Required -- no default. The Step 11
            UI will let users pick; until then callers explicitly
            specify.
        language_hint: ISO-639-1 code if known (``"te"``, ``"hi"``,
            ``"en"``). None for auto-detect.
        brief: Free-text description of the audio. Providers map this
            to their model-adaptation / initial-prompt mechanism.
        names: Proper nouns (people, places, organisations) for biasing.
            Each provider maps to its native phrase-set / word-boost
            API.
        out_dir: Where to persist ``transcript.json`` +
            ``stt_metadata.json``. None keeps everything in-memory.

    Returns:
        Stage1Output with the typed transcript + per-job ledger fields.

    Raises:
        FileNotFoundError: audio_path doesn't exist.
        ValueError: provider name not in PROVIDERS registry.
        RuntimeError: provider returned data that violates the word-
            level contract (empty word list, or transcript.provider
            mismatch, or negative-duration word).
    """
    src = Path(audio_path)
    if not src.is_file():
        raise FileNotFoundError(f"stage_1: audio file not found: {src}")

    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown STT provider {provider!r}. "
            f"Available: {sorted(PROVIDERS)!r}"
        )

    provider_cls = PROVIDERS[provider]
    provider_instance = provider_cls()  # type: ignore[call-arg]

    names_list: list[str] = list(names) if names else []

    wall_start = time.perf_counter()
    response = await provider_instance.transcribe(
        audio_path=str(src),
        language=language_hint,
        brief=brief,
        names=names_list,
    )
    wall_seconds = time.perf_counter() - wall_start

    _validate_word_level_contract(response.transcript, provider=provider)

    avg_conf = _avg_confidence(response.transcript)
    detected_lang = _detected_language(response.transcript, language_hint)
    word_count = len(response.transcript.words)

    transcript_json_path: Optional[str] = None
    metadata_json_path: Optional[str] = None
    if out_dir is not None:
        ledger = {
            "stt_provider": provider,
            "stt_audio_duration_sec": response.audio_duration_sec,
            "stt_wall_seconds": wall_seconds,
            "stt_cost_usd": response.cost_usd,
            "stt_word_count": word_count,
            "stt_avg_confidence": avg_conf,
            "stt_language_detected": detected_lang,
            "stt_request_id": response.request_id,
            "stt_language_hint": language_hint,
            "stt_brief": brief,
            "stt_names": names_list,
        }
        transcript_json_path, metadata_json_path = _write_outputs(
            Path(out_dir), response.transcript, ledger,
        )

    realtime = (
        response.audio_duration_sec / wall_seconds
        if wall_seconds > 0 else float("inf")
    )
    logger.info(
        "stage_1: provider=%s audio=%.1fs wall=%.1fs realtime=%.1fx "
        "words=%d avg_conf=%s detected=%s cost=$%.4f req_id=%s",
        provider, response.audio_duration_sec, wall_seconds, realtime,
        word_count, f"{avg_conf:.3f}" if avg_conf is not None else "n/a",
        detected_lang, response.cost_usd, response.request_id,
    )

    return Stage1Output(
        transcript=response.transcript,
        transcript_json_path=transcript_json_path,
        metadata_json_path=metadata_json_path,
        stt_provider=provider,
        stt_audio_duration_sec=response.audio_duration_sec,
        stt_wall_seconds=wall_seconds,
        stt_cost_usd=response.cost_usd,
        stt_word_count=word_count,
        stt_avg_confidence=avg_conf,
        stt_language_detected=detected_lang,
        stt_request_id=response.request_id,
        stt_language_hint=language_hint,
        stt_brief=brief,
        stt_names=names_list,
    )


__all__ = [
    "ProviderResponse",
    "PermanentSTTError",
    "TranscriptionProvider",
    "PROVIDERS",
    "register",
    "run_stage_1",
]


# ---------------------------------------------------------------------- #
# Concrete provider auto-load                                             #
# ---------------------------------------------------------------------- #
#
# Each provider module's ``@register("name")`` decorator runs at import
# time, populating ``PROVIDERS``. We import them HERE (not at the top
# of the file) so each provider can lazy-import its vendor SDK inside
# its own module without triggering chains of heavy imports just to
# call into the dispatcher.
#
# Graceful degradation: if a provider's vendor SDK isn't installed,
# the import below fails with ImportError. We log a warning and keep
# going -- the provider just isn't available in PROVIDERS, and the
# dispatcher's "unknown provider" error path catches any later call.

def _autoload_providers() -> None:
    for mod_name in ("whisper_groq", "deepgram", "assemblyai"):    # extended per provider step
        try:
            __import__(f"{__name__}.{mod_name}")
        except ImportError as exc:
            logger.warning(
                "STT provider module %r failed to import: %s. Install "
                "its vendor SDK to enable this provider; the others "
                "still work.", mod_name, exc,
            )


_autoload_providers()
