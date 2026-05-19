"""Deepgram Nova-3 STT provider (Step 4.3).

Hosted Nova-3 via Deepgram's REST API. Mid-premium tier in the
4-provider lineup (Whisper-Groq free / Deepgram PAYG / AssemblyAI
mid / Sarvam Indian-specialized).

Telugu strategy: Nova-3 multilingual (``language="multi"``) does NOT
include Telugu (set is en/es/fr/de/hi/ru/pt/ja/it/nl, verified
2026-05-18). For our Telugu workload we use Nova-3 SINGLE-language
mode via ``language="te"``. This disables Deepgram's native
code-switching to English, which is acceptable because:
  - Code-switching to English is a minority of our content
  - Whisper-Groq covers code-mixed audio better; users can route
    code-mixed jobs there via the Step 11 dropdown
  - Nova-3 single-language Telugu is GA, not Preview (unlike Chirp 3)

Vendor specifics (document so Stage 2 logic accounts for them):
  - **Word-level timestamps**: always present in the response by
    default with model="nova-3" -- no flag required. Per-word
    confidence IS returned (unlike Chirp 3 and Whisper-Groq, both of
    which leave Word.confidence=None). This is the FIRST provider in
    the lineup with per-word confidence -- Stage 2 should use it.
  - **Diarization**: enabled by default in this provider so Stage 2
    has speaker labels for ``crew_talk`` detection. Adds $0.0020/min
    on top of the base $0.0077.
  - **Adaptation**: uses Nova-3's ``keyterm`` parameter (the newer
    replacement for ``keywords``). Names + tokenised brief flow into
    a single keyterm list; Deepgram weights internally.
  - **Pricing model**: pay-as-you-go (PAYG) -- $0.0097/min total for
    nova-3 + diarize + smart_format. Growth tier is $0.0082/min if
    negotiated. We default to PAYG; flip the constant if/when growth
    tier kicks in.

File-size policy:
  - Deepgram's historical cap is 2 GB / multi-hour. Our files are
    5-30 min mp3s (under 30 MB). We enforce a 500 MB upper bound here
    as a runaway-protection -- not because Deepgram rejects it.
  - 0-byte files: raise ValueError.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

# Eager SDK import. If the deepgram-sdk package isn't installed in this
# env, this raises ImportError at module load -- caught by the
# auto-loader in pipeline_v2/stages/stt/__init__.py so the rest of the
# stt package still works without Deepgram. Importing eagerly (vs
# lazily inside _get_client) keeps AsyncDeepgramClient in this module's
# namespace so unit tests can patch it directly.
from deepgram import AsyncDeepgramClient

from pipeline_v2.models import Word, WordLevelTranscript
from pipeline_v2.stages.stt import (
    PermanentSTTError,
    ProviderResponse,
    register,
)

logger = logging.getLogger("pipeline_v2.stt.deepgram")


# --- Constants ---------------------------------------------------------

# Runaway-protection upper bound. Deepgram's own cap is multi-GB; our
# audio is always <50 MB. Raise immediately if something's wrong upstream.
DEEPGRAM_MAX_FILE_MB = 500

# Pricing (PAYG tier). Last verified 2026-05-18 from deepgram.com/pricing.
# Re-verify quarterly.
#   nova-3 base prerecorded:    $0.0077/min
#   speaker diarization:        $0.0020/min
#   smart formatting:           included
# Growth tier (negotiated): $0.0082/min total. Update the constant if
# the project graduates to growth tier.
DEEPGRAM_NOVA3_USD_PER_MIN = 0.0077 + 0.0020   # $0.0097


# --- Provider ----------------------------------------------------------


@register("deepgram")
class DeepgramNova3Provider:
    """Nova-3 + Telugu single-language via Deepgram's REST API."""

    name: str = "deepgram"   # also set by @register
    DEFAULT_MODEL = "nova-3"

    def __init__(self, *, model: Optional[str] = None):
        # Constructor override for tests and for future A/B (e.g.
        # "nova-2" or "enhanced"). In production the dispatcher
        # instantiates with no args.
        self.model = (
            model
            or os.environ.get("DEEPGRAM_MODEL", "").strip()
            or self.DEFAULT_MODEL
        )
        self._client: Optional[AsyncDeepgramClient] = None

    def _get_client(self) -> AsyncDeepgramClient:
        """Lazy AsyncDeepgramClient. API key fetched at first use so
        the dispatcher's eager provider instantiation doesn't fail on
        a missing key at registration time."""
        if self._client is not None:
            return self._client
        api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY env var is empty/unset. Add it to "
                "KaizerBackend/.env -- generate one at console.deepgram.com."
            )
        self._client = AsyncDeepgramClient(api_key=api_key)
        return self._client

    def _validate_file(self, audio_path: str) -> int:
        # Permanent failure conditions (per Step 10 D-10.3 refinement):
        # raise PermanentSTTError so the orchestrator converts these to
        # Inngest NonRetriableError -- there's no point retrying a
        # 0-byte file or a file that exceeds the size cap.
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        size = path.stat().st_size
        if size == 0:
            raise PermanentSTTError(
                f"empty_file: audio file {audio_path!r} is 0 bytes"
            )
        size_mb = size / (1024 * 1024)
        if size_mb > DEEPGRAM_MAX_FILE_MB:
            raise PermanentSTTError(
                f"file_too_large: audio file is {size_mb:.1f}MB which "
                f"exceeds runaway-protection cap of {DEEPGRAM_MAX_FILE_MB}MB. "
                f"Deepgram itself accepts up to 2GB; bump "
                f"DEEPGRAM_MAX_FILE_MB if you have a legitimate use "
                f"case. Path: {audio_path}"
            )
        return size

    def _build_keyterm(self, brief: str, names: list[str]) -> list[str]:
        """Construct the Nova-3 ``keyterm`` list.

        Combines proper-noun ``names`` with notable tokens from
        ``brief``. Deepgram weights internally; we don't pass boost
        values (Nova-3 keyterm doesn't take :N suffixes the way the
        legacy ``keywords`` param did).

        Returns an empty list if nothing to bias on (the SDK accepts
        ``keyterm=None`` to mean "no biasing").
        """
        out: list[str] = []
        seen: set[str] = set()
        for n in names or []:
            n = n.strip()
            if n and n.lower() not in seen:
                out.append(n)
                seen.add(n.lower())
        # Tokenise brief: keep words >= 4 chars to skip articles /
        # prepositions / particles. Cap total keyterms at 64 (Nova-3
        # doesn't publish a hard cap but more than this dilutes the
        # signal).
        import re
        for tok in re.findall(r"[\wऀ-ॿఀ-౿]+", brief or "", flags=re.UNICODE):
            if len(tok) < 4:
                continue
            if tok.lower() in seen:
                continue
            out.append(tok)
            seen.add(tok.lower())
            if len(out) >= 64:
                break
        return out

    async def transcribe(
        self,
        *,
        audio_path: str,
        language: Optional[str],
        brief: str = "",
        names: Optional[list[str]] = None,
    ) -> ProviderResponse:
        """Transcribe via Deepgram Nova-3.

        Raises:
            FileNotFoundError: audio_path doesn't exist.
            ValueError: file is 0 bytes or exceeds DEEPGRAM_MAX_FILE_MB.
            RuntimeError: DEEPGRAM_API_KEY missing at call time.
            (Deepgram SDK exceptions propagate unchanged.)
        """
        self._validate_file(audio_path)
        keyterm = self._build_keyterm(brief, names or [])

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Build kwargs conditionally so we send absent (not None) for
        # optional parameters -- some SDK versions distinguish.
        kwargs: dict = {
            "request": audio_bytes,
            "model": self.model,
            "punctuate": True,
            "diarize": True,
            "smart_format": True,
        }
        if language:
            kwargs["language"] = language
        if keyterm:
            kwargs["keyterm"] = keyterm

        client = self._get_client()

        # with_raw_response exposes the HTTP request_id header in
        # addition to Deepgram's body-level metadata.request_id.
        # Both should match but we prefer the body-level value as the
        # canonical Deepgram identifier.
        http_request_id = ""
        response = None
        try:
            raw = await client.listen.v1.media.with_raw_response.transcribe_file(**kwargs)
            http_request_id = getattr(raw, "request_id", "") or ""
            response = raw.parse()
        except AttributeError:
            response = await client.listen.v1.media.transcribe_file(**kwargs)

        metadata = getattr(response, "metadata", None)
        audio_duration_sec = float(
            getattr(metadata, "duration", 0.0) or 0.0
        )
        body_request_id = (getattr(metadata, "request_id", "") or "").strip()
        request_id = body_request_id or http_request_id or (
            f"deepgram-client-{uuid.uuid4().hex[:12]}"
        )

        # Walk the response: results.channels[0].alternatives[0].words
        words: list[Word] = []
        detected_language = ""
        results = getattr(response, "results", None)
        channels = getattr(results, "channels", None) or []
        if channels:
            ch = channels[0]
            alts = getattr(ch, "alternatives", None) or []
            if alts:
                alt = alts[0]
                # Some responses include a per-channel detected_language
                # when detect_language=True. We didn't set that flag so
                # fall back to whatever the alternative reports, or the
                # language hint passed in.
                detected_language = (
                    getattr(ch, "detected_language", None)
                    or getattr(alt, "language", None)
                    or (language or "")
                )
                for w in (getattr(alt, "words", None) or []):
                    # punctuated_word is the casing/punctuation-aware
                    # form; word is the raw token. Prefer punctuated
                    # since punctuate=True is set.
                    text = (
                        getattr(w, "punctuated_word", None)
                        or getattr(w, "word", "")
                        or ""
                    )
                    speaker_val = getattr(w, "speaker", None)
                    confidence_val = getattr(w, "confidence", None)
                    words.append(Word(
                        w=text.strip(),
                        s=float(getattr(w, "start", 0.0) or 0.0),
                        e=float(getattr(w, "end", 0.0) or 0.0),
                        speaker=int(speaker_val) if speaker_val is not None else None,
                        confidence=float(confidence_val) if confidence_val is not None else None,
                    ))

        detected_languages: list[str] = []
        if detected_language:
            detected_languages.append(detected_language)
        if language and language != detected_language:
            detected_languages.append(language)

        transcript = WordLevelTranscript(
            words=words,
            duration_sec=audio_duration_sec,
            detected_languages=detected_languages,
            provider=self.name,
        )

        cost_usd = (audio_duration_sec / 60.0) * DEEPGRAM_NOVA3_USD_PER_MIN

        return ProviderResponse(
            transcript=transcript,
            cost_usd=cost_usd,
            request_id=request_id,
            audio_duration_sec=audio_duration_sec,
        )
