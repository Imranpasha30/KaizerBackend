"""Unit tests for the Whisper-Groq STT provider.

Mocks the groq SDK end-to-end -- no real network calls. The accuracy
test against a real Telugu fixture is the cross-provider comparison
script in Step 4.6, NOT a per-provider unit test (deferred per user
direction).

Covers:
  - Registration: provider lands in PROVIDERS via the @register decorator
  - Constructor: defaults, env-var overrides, tier validation
  - File validation: missing file, empty file, oversize, free-tier warn
  - API key missing at call time raises RuntimeError (not at init)
  - Happy path: words mapped correctly, provider name set, prompt built
  - Prompt + language: omitted from kwargs when not provided
  - Request ID: groq's id passed through; client-side fallback when empty
  - Cost: free tier == 0, dev tier computed from per-hour rate
  - API exceptions propagate (auth fail, rate limit) for Inngest retry
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline_v2.models import Word, WordLevelTranscript
from pipeline_v2.stages.stt import PROVIDERS, ProviderResponse
from pipeline_v2.stages.stt.whisper_groq import WhisperGroqProvider


# ---------------------------------------------------------------------- #
# Fixtures                                                               #
# ---------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Default test env: real-shaped API key, free tier, no model override.

    Individual tests override these with monkeypatch as needed.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real-for-tests-only")
    monkeypatch.delenv("GROQ_TIER", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    yield


def _fake_word(word: str, start: float, end: float):
    return SimpleNamespace(word=word, start=start, end=end)


def _fake_transcription(
    *,
    words=None,
    duration: float = 12.0,
    language: str = "te",
):
    if words is None:
        words = [
            _fake_word("నమస్తే", 0.10, 0.45),
            _fake_word(" hello", 0.50, 0.80),   # leading space to test strip()
        ]
    return SimpleNamespace(
        words=words,
        duration=duration,
        language=language,
        text=" ".join(w.word for w in words),
    )


def _fake_async_groq_client(
    transcription,
    *,
    request_id: str = "groq_req_test_001",
):
    """Build a MagicMock AsyncGroq client that returns ``transcription``
    when ``audio.transcriptions.with_raw_response.create()`` is awaited."""
    raw = MagicMock()
    raw.request_id = request_id
    raw.parse.return_value = transcription

    client = MagicMock()
    client.audio = MagicMock()
    client.audio.transcriptions = MagicMock()
    client.audio.transcriptions.with_raw_response = MagicMock()
    client.audio.transcriptions.with_raw_response.create = AsyncMock(
        return_value=raw,
    )
    return client


@pytest.fixture
def audio_file(tmp_path):
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


# ====================================================================== #
# Registration                                                           #
# ====================================================================== #


class TestRegistration:
    def test_registered_in_providers_dict(self):
        # The @register decorator runs at module import; the auto-load
        # in stages/stt/__init__.py triggers it.
        assert "whisper-groq" in PROVIDERS
        assert PROVIDERS["whisper-groq"] is WhisperGroqProvider

    def test_class_name_attribute_matches_registry_key(self):
        assert WhisperGroqProvider.name == "whisper-groq"


# ====================================================================== #
# Constructor                                                            #
# ====================================================================== #


class TestProviderInit:
    def test_default_model_and_tier(self):
        p = WhisperGroqProvider()
        assert p.model == "whisper-large-v3"
        assert p.tier == "free"

    def test_constructor_overrides(self):
        p = WhisperGroqProvider(model="whisper-large-v3-turbo", tier="dev")
        assert p.model == "whisper-large-v3-turbo"
        assert p.tier == "dev"

    def test_invalid_tier_rejected(self):
        with pytest.raises(ValueError, match="GROQ_TIER"):
            WhisperGroqProvider(tier="enterprise")

    def test_tier_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_TIER", "dev")
        p = WhisperGroqProvider()
        assert p.tier == "dev"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "whisper-large-v3-turbo")
        p = WhisperGroqProvider()
        assert p.model == "whisper-large-v3-turbo"

    def test_constructor_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_TIER", "dev")
        p = WhisperGroqProvider(tier="free")
        assert p.tier == "free"

    def test_init_does_not_require_api_key(self, monkeypatch):
        # API key is fetched lazily on first transcribe() call, not at
        # construction. The dispatcher constructs providers eagerly;
        # we don't want a missing key to crash registration.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        p = WhisperGroqProvider()             # must not raise
        assert p.tier == "free"


# ====================================================================== #
# File validation                                                        #
# ====================================================================== #


class TestFileValidation:
    @pytest.mark.asyncio
    async def test_missing_file_raises(self):
        p = WhisperGroqProvider()
        with pytest.raises(FileNotFoundError):
            await p.transcribe(audio_path="/no/such.mp3", language="te")

    @pytest.mark.asyncio
    async def test_empty_file_raises(self, tmp_path):
        # Step 10 D-10.3: empty-file is permanent -> PermanentSTTError.
        from pipeline_v2.stages.stt import PermanentSTTError
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        p = WhisperGroqProvider()
        with pytest.raises(PermanentSTTError, match=r"empty_file"):
            await p.transcribe(audio_path=str(empty), language="te")

    @pytest.mark.asyncio
    async def test_oversized_file_raises_with_mitigation_hint(self, tmp_path):
        # Step 10 D-10.3: oversize-file is permanent -> PermanentSTTError.
        from pipeline_v2.stages.stt import PermanentSTTError
        big = tmp_path / "big.mp3"
        big.write_bytes(b"")
        # 101 MB sparse file -- os.path.getsize() reports truncated size
        # without writing real bytes (fast).
        os.truncate(big, 101 * 1024 * 1024)
        p = WhisperGroqProvider()
        with pytest.raises(PermanentSTTError) as excinfo:
            await p.transcribe(audio_path=str(big), language="te")
        msg = str(excinfo.value)
        assert "100MB" in msg
        assert "dev tier" in msg
        assert "64kbps" in msg                 # mitigation hint present

    @pytest.mark.asyncio
    async def test_freetier_warning_on_25_to_100mb(self, tmp_path, caplog):
        # 30 MB sparse: passes the >100MB check, but exceeds free-tier
        # cap. Provider should log a warning AND proceed (Groq itself
        # may or may not be on free tier; the provider can't know
        # authoritatively, only warn).
        medium = tmp_path / "medium.mp3"
        medium.write_bytes(b"")
        os.truncate(medium, 30 * 1024 * 1024)

        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)
        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider(tier="free")
            with caplog.at_level("WARNING"):
                await p.transcribe(audio_path=str(medium), language="te")
        assert any("25MB" in r.message or "free-tier" in r.message
                   for r in caplog.records), (
            "expected a free-tier warning in log records"
        )

    @pytest.mark.asyncio
    async def test_dev_tier_no_warning_at_30mb(self, tmp_path, caplog):
        medium = tmp_path / "medium.mp3"
        medium.write_bytes(b"")
        os.truncate(medium, 30 * 1024 * 1024)

        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)
        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider(tier="dev")
            with caplog.at_level("WARNING"):
                await p.transcribe(audio_path=str(medium), language="te")
        free_tier_warnings = [
            r for r in caplog.records
            if "free-tier" in r.message.lower() or "25mb" in r.message.lower()
        ]
        assert not free_tier_warnings


# ====================================================================== #
# API key missing at call time                                           #
# ====================================================================== #


class TestApiKeyMissing:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_runtime_error(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"\x00" * 1024)
        p = WhisperGroqProvider()
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            await p.transcribe(audio_path=str(audio), language="te")


# ====================================================================== #
# Happy path                                                             #
# ====================================================================== #


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_provider_response(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            response = await p.transcribe(
                audio_path=audio_file,
                language="te",
                brief="A test podcast",
                names=["Modi"],
            )

        assert isinstance(response, ProviderResponse)
        assert isinstance(response.transcript, WordLevelTranscript)
        assert response.transcript.provider == "whisper-groq"

    @pytest.mark.asyncio
    async def test_words_mapped_with_strip_and_no_confidence(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            response = await p.transcribe(audio_path=audio_file, language="te")

        words = response.transcript.words
        assert len(words) == 2
        assert words[0].w == "నమస్తే"          # already trimmed in fixture
        assert words[1].w == "hello"            # leading space stripped
        assert all(w.speaker is None for w in words)
        # Critical: Groq does not return per-word confidence (vendor gap)
        assert all(w.confidence is None for w in words)

    @pytest.mark.asyncio
    async def test_audio_duration_from_groq_response(self, audio_file):
        transcription = _fake_transcription(duration=720.5)
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            response = await p.transcribe(audio_path=audio_file, language="te")

        assert response.audio_duration_sec == 720.5
        assert response.transcript.duration_sec == 720.5


# ====================================================================== #
# Prompt + language kwargs                                               #
# ====================================================================== #


class TestRequestKwargs:
    def _call_kwargs(self, fake_client) -> dict:
        return (
            fake_client.audio.transcriptions.with_raw_response
            .create.call_args.kwargs
        )

    @pytest.mark.asyncio
    async def test_initial_prompt_built_from_brief_and_names(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            await p.transcribe(
                audio_path=audio_file, language="te",
                brief="Telugu news podcast",
                names=["Modi", "Telangana"],
            )

        kwargs = self._call_kwargs(fake_client)
        # Format mirrors teammate's V1 production pattern:
        #   "<brief>. <name1>, <name2>"
        assert kwargs["prompt"] == "Telugu news podcast. Modi, Telangana"

    @pytest.mark.asyncio
    async def test_prompt_omitted_when_no_brief_no_names(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            await p.transcribe(audio_path=audio_file, language="te")

        kwargs = self._call_kwargs(fake_client)
        assert "prompt" not in kwargs

    @pytest.mark.asyncio
    async def test_prompt_omitted_when_only_empty_strings(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            await p.transcribe(
                audio_path=audio_file, language="te",
                brief="   ",                # whitespace only
                names=["", "  "],           # empty after strip
            )

        kwargs = self._call_kwargs(fake_client)
        assert "prompt" not in kwargs

    @pytest.mark.asyncio
    async def test_language_omitted_when_none(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            await p.transcribe(audio_path=audio_file, language=None)

        kwargs = self._call_kwargs(fake_client)
        assert "language" not in kwargs

    @pytest.mark.asyncio
    async def test_language_passed_through(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            await p.transcribe(audio_path=audio_file, language="hi")

        kwargs = self._call_kwargs(fake_client)
        assert kwargs["language"] == "hi"

    @pytest.mark.asyncio
    async def test_required_kwargs_always_set(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            await p.transcribe(audio_path=audio_file, language="te")

        kwargs = self._call_kwargs(fake_client)
        assert kwargs["model"] == "whisper-large-v3"
        assert kwargs["response_format"] == "verbose_json"
        assert kwargs["timestamp_granularities"] == ["word"]
        # file is a (name, bytes) tuple
        assert "file" in kwargs


# ====================================================================== #
# Request ID                                                             #
# ====================================================================== #


class TestRequestId:
    @pytest.mark.asyncio
    async def test_groq_request_id_passed_through(self, audio_file):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(
            transcription, request_id="groq_actual_id_xyz",
        )

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            response = await p.transcribe(audio_path=audio_file, language="te")

        assert response.request_id == "groq_actual_id_xyz"

    @pytest.mark.asyncio
    async def test_fallback_client_side_id_when_groq_returns_empty(
        self, audio_file,
    ):
        transcription = _fake_transcription()
        fake_client = _fake_async_groq_client(transcription, request_id="")

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            response = await p.transcribe(audio_path=audio_file, language="te")

        assert response.request_id.startswith("groq-client-")
        # 12-char hex suffix
        assert len(response.request_id) == len("groq-client-") + 12


# ====================================================================== #
# Cost                                                                   #
# ====================================================================== #


class TestCost:
    @pytest.mark.asyncio
    async def test_free_tier_zero_cost_regardless_of_duration(self, audio_file):
        # 5 minutes of audio
        transcription = _fake_transcription(duration=300.0)
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider(tier="free")
            response = await p.transcribe(audio_path=audio_file, language="te")

        assert response.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_dev_tier_cost_from_per_hour_rate(self, audio_file):
        # 30 min at $0.04/hr = $0.02
        transcription = _fake_transcription(duration=1800.0)
        fake_client = _fake_async_groq_client(transcription)

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider(tier="dev")
            response = await p.transcribe(audio_path=audio_file, language="te")

        assert response.cost_usd == pytest.approx(0.02, rel=1e-6)


# ====================================================================== #
# API errors propagate                                                   #
# ====================================================================== #


class TestApiErrors:
    @pytest.mark.asyncio
    async def test_api_exception_propagates(self, audio_file):
        # Provider does NOT catch SDK exceptions -- the dispatcher
        # surfaces them and Inngest's retry layer handles backoff.
        fake_client = MagicMock()
        fake_client.audio = MagicMock()
        fake_client.audio.transcriptions = MagicMock()
        fake_client.audio.transcriptions.with_raw_response = MagicMock()
        fake_client.audio.transcriptions.with_raw_response.create = AsyncMock(
            side_effect=RuntimeError("HTTP 429 rate_limit_exceeded"),
        )

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
                await p.transcribe(audio_path=audio_file, language="te")

    @pytest.mark.asyncio
    async def test_auth_error_propagates(self, audio_file):
        fake_client = MagicMock()
        fake_client.audio = MagicMock()
        fake_client.audio.transcriptions = MagicMock()
        fake_client.audio.transcriptions.with_raw_response = MagicMock()
        fake_client.audio.transcriptions.with_raw_response.create = AsyncMock(
            side_effect=RuntimeError("HTTP 401 invalid_api_key"),
        )

        with patch(
            "pipeline_v2.stages.stt.whisper_groq.AsyncGroq",
            return_value=fake_client,
        ):
            p = WhisperGroqProvider()
            with pytest.raises(RuntimeError, match="invalid_api_key"):
                await p.transcribe(audio_path=audio_file, language="te")
