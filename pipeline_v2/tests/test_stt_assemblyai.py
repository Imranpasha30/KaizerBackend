"""Unit tests for the AssemblyAI Universal-2 STT provider.

Mocks the assemblyai SDK end-to-end -- no real network calls. Accuracy
testing on a real Telugu fixture is deferred to Step 4.6 (combined
cross-provider comparison).

Covers:
  - Registration: provider in PROVIDERS, all 3 providers coexist
  - File validation: missing / empty / oversized
  - API key: lazy (not at __init__) and surfaced clearly when missing
  - Happy path: provider name, words populated, per-word confidence
  - Critical unit-conversions:
    * milliseconds -> seconds for word timestamps
    * "A"/"B"/"C" letters -> 0/1/2 int for Word.speaker
    * Non-letter / multi-char speaker label -> None (graceful)
  - keyterms_prompt build: names first, brief tokens follow, dedup,
    cap at KEYTERMS_PROMPT_MAX
  - TranscriptionConfig flags: speech_model, speaker_labels,
    punctuate, format_text, language_code (omitted when None)
  - Status="error" raises RuntimeError with the SDK's error message
  - Cost: $0.17/hr split
  - API exception propagation
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline_v2.models import WordLevelTranscript
from pipeline_v2.stages.stt import PROVIDERS, ProviderResponse
from pipeline_v2.stages.stt.assemblyai import (
    ASSEMBLYAI_USD_PER_MIN,
    AssemblyAIUniversal2Provider,
    KEYTERMS_PROMPT_MAX,
    _speaker_letter_to_int,
)


# ---------------------------------------------------------------------- #
# Fixtures                                                               #
# ---------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key-not-real")
    yield


def _fake_word(text, start_ms, end_ms, *, speaker="A", confidence=0.9):
    return SimpleNamespace(
        text=text, start=start_ms, end=end_ms,
        speaker=speaker, confidence=confidence,
    )


def _fake_transcript(
    *,
    words=None,
    duration_sec: float = 12.5,
    transcript_id: str = "transcript_abc123",
    language_code: str = "te",
    status: str = "completed",
    error: str = "",
):
    if words is None:
        words = [
            _fake_word("Namaste", 100, 450, speaker="A", confidence=0.94),
            _fake_word("world", 500, 800, speaker="B", confidence=0.88),
        ]
    return SimpleNamespace(
        id=transcript_id,
        status=status,
        error=error,
        audio_duration=duration_sec,
        language_code=language_code,
        words=words,
    )


def _fake_transcriber(transcript):
    """Build a mock Transcriber whose transcribe() returns ``transcript``."""
    t = MagicMock()
    t.transcribe = MagicMock(return_value=transcript)
    return t


@pytest.fixture
def audio_file(tmp_path):
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


# ====================================================================== #
# Registration                                                           #
# ====================================================================== #


class TestRegistration:
    def test_registered(self):
        assert "assemblyai" in PROVIDERS
        assert PROVIDERS["assemblyai"] is AssemblyAIUniversal2Provider

    def test_class_name_attribute_matches_key(self):
        assert AssemblyAIUniversal2Provider.name == "assemblyai"

    def test_all_three_providers_coexist(self):
        # Auto-load runs each provider's @register; check the full set.
        for name in ("whisper-groq", "deepgram", "assemblyai"):
            assert name in PROVIDERS


# ====================================================================== #
# _speaker_letter_to_int helper                                          #
# ====================================================================== #


class TestSpeakerLetterToInt:
    @pytest.mark.parametrize("letter,expected", [
        ("A", 0), ("B", 1), ("C", 2), ("Z", 25),
        ("a", 0), ("b", 1),   # case insensitive
    ])
    def test_letters_map_to_ints(self, letter, expected):
        assert _speaker_letter_to_int(letter) == expected

    @pytest.mark.parametrize("bad", [
        None, "", "AA", "AB", "1", "α", 0, 5, True,
    ])
    def test_non_single_letter_returns_none(self, bad):
        assert _speaker_letter_to_int(bad) is None


# ====================================================================== #
# File validation                                                        #
# ====================================================================== #


class TestFileValidation:
    @pytest.mark.asyncio
    async def test_missing_file_raises(self):
        p = AssemblyAIUniversal2Provider()
        with pytest.raises(FileNotFoundError):
            await p.transcribe(audio_path="/no/such.mp3", language="te")

    @pytest.mark.asyncio
    async def test_empty_file_raises(self, tmp_path):
        # Step 10 D-10.3: empty-file is a permanent failure ->
        # PermanentSTTError (was ValueError pre-Step-10).
        from pipeline_v2.stages.stt import PermanentSTTError
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        p = AssemblyAIUniversal2Provider()
        with pytest.raises(PermanentSTTError, match=r"empty_file"):
            await p.transcribe(audio_path=str(empty), language="te")

    @pytest.mark.asyncio
    async def test_oversized_file_raises(self, tmp_path):
        from pipeline_v2.stages.stt import PermanentSTTError
        big = tmp_path / "big.mp3"
        big.write_bytes(b"")
        os.truncate(big, 501 * 1024 * 1024)
        p = AssemblyAIUniversal2Provider()
        with pytest.raises(PermanentSTTError, match="500MB"):
            await p.transcribe(audio_path=str(big), language="te")


# ====================================================================== #
# API key                                                                #
# ====================================================================== #


class TestApiKey:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_runtime_error(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        # Also clear any settings.api_key the SDK might have cached
        # from a previous test (the SDK is singleton-configured).
        import assemblyai as aai
        aai.settings.api_key = None
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"\x00" * 1024)
        p = AssemblyAIUniversal2Provider()
        with pytest.raises(RuntimeError, match="ASSEMBLYAI_API_KEY"):
            await p.transcribe(audio_path=str(audio), language="te")

    def test_init_does_not_require_api_key(self, monkeypatch):
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        AssemblyAIUniversal2Provider()         # must not raise


# ====================================================================== #
# Happy path -- timestamps, confidence, speaker                          #
# ====================================================================== #


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_provider_response(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert isinstance(r, ProviderResponse)
        assert isinstance(r.transcript, WordLevelTranscript)
        assert r.transcript.provider == "assemblyai"

    @pytest.mark.asyncio
    async def test_ms_to_seconds_conversion(self, audio_file):
        # AssemblyAI returns timestamps in milliseconds; provider must
        # convert to seconds for our Word schema.
        transcript = _fake_transcript(words=[
            _fake_word("hi", 1500, 1800),   # 1.5s..1.8s
            _fake_word("there", 1900, 2300),
        ])
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        w = r.transcript.words
        assert w[0].s == pytest.approx(1.5)
        assert w[0].e == pytest.approx(1.8)
        assert w[1].s == pytest.approx(1.9)
        assert w[1].e == pytest.approx(2.3)

    @pytest.mark.asyncio
    async def test_speaker_letter_to_int(self, audio_file):
        transcript = _fake_transcript(words=[
            _fake_word("first", 0, 300, speaker="A"),
            _fake_word("second", 400, 700, speaker="B"),
            _fake_word("third", 800, 1000, speaker="C"),
        ])
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        speakers = [w.speaker for w in r.transcript.words]
        assert speakers == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_non_letter_speaker_falls_back_to_none(self, audio_file):
        transcript = _fake_transcript(words=[
            _fake_word("oops", 0, 300, speaker=None),
            _fake_word("again", 400, 700, speaker="AB"),  # 2-char
        ])
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert all(w.speaker is None for w in r.transcript.words)

    @pytest.mark.asyncio
    async def test_per_word_confidence_populated(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        # AssemblyAI returns per-word confidence, like Deepgram.
        confidences = [w.confidence for w in r.transcript.words]
        assert all(c is not None for c in confidences)
        assert confidences[0] == pytest.approx(0.94)
        assert confidences[1] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_audio_duration_already_in_seconds(self, audio_file):
        # AssemblyAI's `audio_duration` is in SECONDS already (unlike
        # word.start/end which are in ms). Provider must not double-
        # convert.
        transcript = _fake_transcript(duration_sec=720.5)
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.audio_duration_sec == 720.5
        assert r.transcript.duration_sec == 720.5

    @pytest.mark.asyncio
    async def test_request_id_from_transcript_id(self, audio_file):
        transcript = _fake_transcript(transcript_id="aai_transcript_xyz")
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.request_id == "aai_transcript_xyz"

    @pytest.mark.asyncio
    async def test_request_id_fallback_when_empty(self, audio_file):
        transcript = _fake_transcript(transcript_id="")
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.request_id.startswith("assemblyai-client-")
        assert len(r.request_id) == len("assemblyai-client-") + 12


# ====================================================================== #
# TranscriptionConfig kwargs                                             #
# ====================================================================== #


class TestConfigKwargs:
    def _captured_config_kwargs(self, fake_config_cls):
        """Get the kwargs the provider passed to TranscriptionConfig."""
        return fake_config_cls.call_args.kwargs

    @pytest.mark.asyncio
    async def test_required_kwargs(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        fake_config_cls = MagicMock(return_value=MagicMock())
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ), patch(
            "pipeline_v2.stages.stt.assemblyai.aai.TranscriptionConfig",
            fake_config_cls,
        ):
            p = AssemblyAIUniversal2Provider()
            await p.transcribe(audio_path=audio_file, language="te")
        kw = self._captured_config_kwargs(fake_config_cls)
        assert kw["speaker_labels"] is True
        assert kw["punctuate"] is True
        assert kw["format_text"] is True
        assert "speech_model" in kw          # routes to Universal-class

    @pytest.mark.asyncio
    async def test_language_code_passed_when_given(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        fake_config_cls = MagicMock(return_value=MagicMock())
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ), patch(
            "pipeline_v2.stages.stt.assemblyai.aai.TranscriptionConfig",
            fake_config_cls,
        ):
            p = AssemblyAIUniversal2Provider()
            await p.transcribe(audio_path=audio_file, language="te")
        assert self._captured_config_kwargs(fake_config_cls)["language_code"] == "te"

    @pytest.mark.asyncio
    async def test_language_code_omitted_when_none(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        fake_config_cls = MagicMock(return_value=MagicMock())
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ), patch(
            "pipeline_v2.stages.stt.assemblyai.aai.TranscriptionConfig",
            fake_config_cls,
        ):
            p = AssemblyAIUniversal2Provider()
            await p.transcribe(audio_path=audio_file, language=None)
        assert "language_code" not in self._captured_config_kwargs(fake_config_cls)


# ====================================================================== #
# keyterms_prompt                                                        #
# ====================================================================== #


class TestKeytermsPrompt:
    def _captured_kwargs(self, fake_config_cls):
        return fake_config_cls.call_args.kwargs

    @pytest.mark.asyncio
    async def test_keyterms_built_from_names_and_brief(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        fake_config_cls = MagicMock(return_value=MagicMock())
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ), patch(
            "pipeline_v2.stages.stt.assemblyai.aai.TranscriptionConfig",
            fake_config_cls,
        ):
            p = AssemblyAIUniversal2Provider()
            await p.transcribe(
                audio_path=audio_file, language="te",
                brief="Telangana election results",
                names=["Modi", "Reddy"],
            )
        kt = self._captured_kwargs(fake_config_cls)["keyterms_prompt"]
        assert kt[0] == "Modi"
        assert kt[1] == "Reddy"
        for tok in kt[2:]:
            assert len(tok) >= 4
        assert "Telangana" in kt
        assert "election" in kt
        assert "results" in kt

    @pytest.mark.asyncio
    async def test_keyterms_omitted_when_empty(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        fake_config_cls = MagicMock(return_value=MagicMock())
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ), patch(
            "pipeline_v2.stages.stt.assemblyai.aai.TranscriptionConfig",
            fake_config_cls,
        ):
            p = AssemblyAIUniversal2Provider()
            await p.transcribe(audio_path=audio_file, language="te")
        assert "keyterms_prompt" not in self._captured_kwargs(fake_config_cls)

    @pytest.mark.asyncio
    async def test_keyterms_dedup_case_insensitive(self, audio_file):
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        fake_config_cls = MagicMock(return_value=MagicMock())
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ), patch(
            "pipeline_v2.stages.stt.assemblyai.aai.TranscriptionConfig",
            fake_config_cls,
        ):
            p = AssemblyAIUniversal2Provider()
            await p.transcribe(
                audio_path=audio_file, language="te",
                brief="Modi government policy",
                names=["Modi", "modi", "MODI"],
            )
        kt = self._captured_kwargs(fake_config_cls)["keyterms_prompt"]
        assert sum(1 for t in kt if t.lower() == "modi") == 1

    @pytest.mark.asyncio
    async def test_keyterms_cap(self, audio_file):
        # 250 unique names should clamp to KEYTERMS_PROMPT_MAX (200).
        transcript = _fake_transcript()
        transcriber = _fake_transcriber(transcript)
        fake_config_cls = MagicMock(return_value=MagicMock())
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ), patch(
            "pipeline_v2.stages.stt.assemblyai.aai.TranscriptionConfig",
            fake_config_cls,
        ):
            many_names = [f"Name{i}" for i in range(250)]
            p = AssemblyAIUniversal2Provider()
            await p.transcribe(
                audio_path=audio_file, language="te",
                brief="", names=many_names,
            )
        kt = self._captured_kwargs(fake_config_cls)["keyterms_prompt"]
        assert len(kt) == KEYTERMS_PROMPT_MAX


# ====================================================================== #
# Error path                                                             #
# ====================================================================== #


class TestErrorPath:
    @pytest.mark.asyncio
    async def test_transcript_status_error_raises(self, audio_file):
        # AssemblyAI doesn't raise on transcribe(); errors come back
        # in transcript.status == "error" + transcript.error.
        transcript = _fake_transcript(
            status="error",
            error="audio_duration_too_short",
        )
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            with pytest.raises(RuntimeError, match="audio_duration_too_short"):
                await p.transcribe(audio_path=audio_file, language="te")

    @pytest.mark.asyncio
    async def test_sdk_exception_propagates(self, audio_file):
        transcriber = MagicMock()
        transcriber.transcribe = MagicMock(
            side_effect=RuntimeError("HTTP 401 unauthorized"),
        )
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            with pytest.raises(RuntimeError, match="unauthorized"):
                await p.transcribe(audio_path=audio_file, language="te")


# ====================================================================== #
# Cost                                                                   #
# ====================================================================== #


class TestCost:
    @pytest.mark.asyncio
    async def test_cost_computed_from_duration(self, audio_file):
        # 30 min at $0.17/hr = $0.085
        transcript = _fake_transcript(duration_sec=1800.0)
        transcriber = _fake_transcriber(transcript)
        with patch(
            "pipeline_v2.stages.stt.assemblyai.aai.Transcriber",
            return_value=transcriber,
        ):
            p = AssemblyAIUniversal2Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        expected = 30.0 * ASSEMBLYAI_USD_PER_MIN
        assert r.cost_usd == pytest.approx(expected, rel=1e-6)
        assert ASSEMBLYAI_USD_PER_MIN == pytest.approx(0.17 / 60.0)
