"""AssemblyAI Universal-2 STT provider (Step 4.4).

Hosted Universal-class transcription via AssemblyAI's API. Mid-tier
position in the 4-provider lineup (Whisper-Groq free / Deepgram PAYG /
AssemblyAI free-185h-then-paid / Sarvam Indian-specialized).

Telugu strategy: AssemblyAI's Universal-2 supports Telugu under
``language_code="te"`` (lowercase, ISO-639-1, verified 2026-05-18 from
assemblyai.com/docs/pre-recorded-audio/supported-languages). Listed in
the standard supported-languages table with no beta/preview asterisk.

Vendor specifics:
  - **SDK is sync-only**. ``Transcriber().transcribe(path)`` blocks
    until done (uploads + polls internally). We wrap in
    ``asyncio.to_thread`` so the async dispatcher's event loop stays
    responsive.
  - **Timestamps are in milliseconds**, not seconds. Provider
    converts to seconds before populating Word.s / Word.e.
  - **Speaker labels are letters** ("A", "B", "C"...) when
    ``speaker_labels=True``. Our Word.speaker is ``int | None``;
    provider converts via ``ord(letter) - ord('A')``, capped to a
    sensible range (returns None for non-letter or >26-speaker
    edge cases).
  - **Per-word confidence: YES**, like Deepgram, unlike Whisper-Groq.
  - **Adaptation API**: ``keyterms_prompt`` (the modern Universal-2+
    surface; older ``word_boost`` with boost levels still works for
    legacy models but is being deprecated). Cap is 200 phrases for
    Universal-2, 1000 for Universal-3 Pro.
  - **speech_model**: SDK 0.64.2 only exposes the enum values
    {``best``, ``nano``, ``slam_1``, ``universal``}. We pin to
    ``SpeechModel.universal`` so AssemblyAI's routing layer picks the
    Universal-class model appropriate for ``language_code="te"``.

Pricing (last verified 2026-05-18 from assemblyai.com/pricing):
  - Universal-2 prerecorded: $0.15/hr base
  - Speaker labels add-on:   $0.02/hr
  - Combined:                $0.17/hr  =  $0.00283/min
  - Free tier:               185 lifetime hours, no card required
    (per their pricing page; treat as a development quota)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional

# Eager SDK import. If `assemblyai` isn't installed, ImportError fires
# here -- caught by the auto-loader in
# pipeline_v2/stages/stt/__init__.py so the rest of the stt package
# still works without AssemblyAI. Importing eagerly keeps Transcriber
# in this module's namespace so unit tests can patch it directly.
import assemblyai as aai

from pipeline_v2.models import Word, WordLevelTranscript
from pipeline_v2.stages.stt import ProviderResponse, register

logger = logging.getLogger("pipeline_v2.stt.assemblyai")


# --- Constants ---------------------------------------------------------

# Runaway-protection upper bound. AssemblyAI itself accepts very large
# files via upload; this cap exists just to catch obvious upstream
# bugs (e.g. accidental video instead of audio).
ASSEMBLYAI_MAX_FILE_MB = 500

# Pricing. Last verified 2026-05-18 from assemblyai.com/pricing.
# Re-verify quarterly.
#   universal-2 prerecorded:   $0.15/hr  ($0.0025/min)
#   speaker_labels add-on:     $0.02/hr  (~$0.00033/min)
#   total at our config:       $0.17/hr  ($0.00283/min)
ASSEMBLYAI_USD_PER_MIN = 0.17 / 60.0

# Universal-2's keyterms_prompt cap (1000 for Universal-3 Pro, 200 for
# Universal-2). We're routing via SpeechModel.universal which the
# vendor may resolve to either; 200 is the safe ceiling.
KEYTERMS_PROMPT_MAX = 200


# --- Provider ---------------------------------------------------------


def _speaker_letter_to_int(s) -> Optional[int]:
    """Convert AssemblyAI's "A"/"B"/"C" speaker label to int.

    Returns None for non-letter input or alphabet overflow (>26
    speakers, which doesn't happen in our podcast workload).
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip().upper()
    if len(s) == 1 and "A" <= s <= "Z":
        return ord(s) - ord("A")
    return None


@register("assemblyai")
class AssemblyAIUniversal2Provider:
    """AssemblyAI Universal-class transcription with Telugu support."""

    name: str = "assemblyai"   # also set by @register

    def __init__(self):
        # Constructor takes no args; the SDK's TranscriptionConfig is
        # built per-call inside transcribe() so language hint /
        # keyterms can be set per-job.
        self._configured = False

    def _ensure_api_key(self) -> None:
        """Set aai.settings.api_key from env on first transcribe call.

        AssemblyAI's SDK is module-level configured (singleton-ish);
        we set the key lazily so the dispatcher's eager provider
        instantiation doesn't fail on a missing key at registration
        time. Tests can override by setting aai.settings.api_key
        directly before invoking the provider.
        """
        if self._configured:
            return
        api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY env var is empty/unset. Add it to "
                "KaizerBackend/.env -- generate one at "
                "assemblyai.com/dashboard."
            )
        aai.settings.api_key = api_key
        self._configured = True

    def _validate_file(self, audio_path: str) -> int:
        """Pre-flight permanent-failure checks. Raises
        ``PermanentSTTError`` per Step 10 D-10.3 refinement so the
        orchestrator skips Inngest retry burn on 0-byte / too-large
        files (both deterministic from file metadata; retrying changes
        nothing).
        """
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
        if size_mb > ASSEMBLYAI_MAX_FILE_MB:
            raise PermanentSTTError(
                f"file_too_large: audio file is {size_mb:.1f}MB which "
                f"exceeds runaway-protection cap of {ASSEMBLYAI_MAX_FILE_MB}MB. "
                f"AssemblyAI itself accepts larger files; bump "
                f"ASSEMBLYAI_MAX_FILE_MB if you have a legitimate use "
                f"case. Path: {audio_path}"
            )
        return size

    def _build_keyterms(self, brief: str, names: list[str]) -> list[str]:
        """Construct AssemblyAI's ``keyterms_prompt`` list.

        Names first (most important for proper-noun accuracy),
        followed by tokens from the brief (>= 4 chars to skip
        articles/particles). Deduped case-insensitive. Capped at
        KEYTERMS_PROMPT_MAX (200 for Universal-2).
        """
        out: list[str] = []
        seen: set[str] = set()
        for n in names or []:
            n = n.strip()
            if n and n.lower() not in seen:
                out.append(n)
                seen.add(n.lower())
            if len(out) >= KEYTERMS_PROMPT_MAX:
                return out
        for tok in re.findall(r"[\wऀ-ॿఀ-౿]+", brief or "", flags=re.UNICODE):
            if len(tok) < 4:
                continue
            if tok.lower() in seen:
                continue
            out.append(tok)
            seen.add(tok.lower())
            if len(out) >= KEYTERMS_PROMPT_MAX:
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
        """Transcribe via AssemblyAI Universal-2.

        Raises:
            FileNotFoundError: audio_path doesn't exist.
            ValueError: file is 0 bytes or > ASSEMBLYAI_MAX_FILE_MB.
            RuntimeError: ASSEMBLYAI_API_KEY missing at call time, OR
                AssemblyAI's transcription failed (transcript.error
                surfaced).
            (AssemblyAI SDK exceptions propagate unchanged.)
        """
        self._validate_file(audio_path)
        self._ensure_api_key()

        keyterms = self._build_keyterms(brief, names or [])

        # Build TranscriptionConfig. language_code is omitted (None)
        # to enable auto-detection when caller passes None; otherwise
        # pin to the hint (e.g. "te" for Telugu).
        config_kwargs: dict = {
            "speech_model": aai.SpeechModel.universal,
            "speaker_labels": True,
            "punctuate": True,
            "format_text": True,
        }
        if language:
            config_kwargs["language_code"] = language
        if keyterms:
            config_kwargs["keyterms_prompt"] = keyterms

        config = aai.TranscriptionConfig(**config_kwargs)
        transcriber = aai.Transcriber()

        # The SDK transcribe() is sync + blocks for many seconds
        # (uploads + polls). Wrap in to_thread so the asyncio event
        # loop stays responsive (Inngest heartbeats keep flowing).
        transcript = await asyncio.to_thread(
            transcriber.transcribe, audio_path, config=config,
        )

        # AssemblyAI surfaces errors via transcript.status == "error" +
        # transcript.error message; the call doesn't raise. We do.
        status = getattr(transcript, "status", None)
        if status is not None and str(status).lower() == "error":
            err = getattr(transcript, "error", "unknown error")
            raise RuntimeError(
                f"AssemblyAI transcription failed: {err}. "
                f"Transcript id: {getattr(transcript, 'id', '?')}"
            )

        # Pull request_id (= transcript.id in AssemblyAI's API).
        request_id = (getattr(transcript, "id", "") or "").strip()
        if not request_id:
            request_id = f"assemblyai-client-{uuid.uuid4().hex[:12]}"

        # AssemblyAI returns audio_duration in seconds (already, not
        # milliseconds like the per-word timestamps).
        audio_duration_sec = float(
            getattr(transcript, "audio_duration", 0.0) or 0.0
        )

        # Walk transcript.words[]. Note: start/end are MILLISECONDS in
        # AssemblyAI's response -- divide by 1000 for our seconds-based
        # Word schema.
        words: list[Word] = []
        for w in (getattr(transcript, "words", None) or []):
            text = (getattr(w, "text", "") or "").strip()
            start_ms = float(getattr(w, "start", 0.0) or 0.0)
            end_ms = float(getattr(w, "end", 0.0) or 0.0)
            speaker_label = getattr(w, "speaker", None)
            confidence = getattr(w, "confidence", None)
            words.append(Word(
                w=text,
                s=start_ms / 1000.0,
                e=end_ms / 1000.0,
                speaker=_speaker_letter_to_int(speaker_label),
                confidence=float(confidence) if confidence is not None else None,
            ))

        # detected_language: AssemblyAI returns `language_code` on the
        # transcript object (matches what was sent if pinned, or what
        # was auto-detected).
        detected_language = (getattr(transcript, "language_code", "") or "").strip()

        detected_languages: list[str] = []
        if detected_language:
            detected_languages.append(detected_language)
        if language and language != detected_language:
            detected_languages.append(language)

        wlt = WordLevelTranscript(
            words=words,
            duration_sec=audio_duration_sec,
            detected_languages=detected_languages,
            provider=self.name,
        )

        cost_usd = (audio_duration_sec / 60.0) * ASSEMBLYAI_USD_PER_MIN

        return ProviderResponse(
            transcript=wlt,
            cost_usd=cost_usd,
            request_id=request_id,
            audio_duration_sec=audio_duration_sec,
        )
