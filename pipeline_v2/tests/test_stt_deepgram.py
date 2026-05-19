"""Unit tests for the Deepgram Nova-3 STT provider.

Mocks the deepgram SDK end-to-end -- no real network calls. The accuracy
test against a real Telugu fixture is the cross-provider comparison
script in Step 4.6, NOT a per-provider unit test (deferred per user
direction).

Covers:
  - Registration: provider lands in PROVIDERS via the @register decorator
  - Constructor: defaults, env-var overrides
  - File validation: missing, empty, oversize (>DEEPGRAM_MAX_FILE_MB)
  - API key missing at call time raises RuntimeError (not at init)
  - Happy path: words mapped correctly, provider name set, keyterm built
  - Per-word confidence + speaker labels DO populate (unlike Groq)
  - punctuated_word preferred over raw word
  - kwargs: language + keyterm conditionally included
  - Request ID: prefer Deepgram's body-level metadata.request_id; fall
    back to HTTP header; final fallback to client-side UUID
  - Cost: $0.0097/min on PAYG tier
  - API exceptions propagate
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline_v2.models import Word, WordLevelTranscript
from pipeline_v2.stages.stt import PROVIDERS, ProviderResponse
from pipeline_v2.stages.stt.deepgram import (
    DEEPGRAM_NOVA3_USD_PER_MIN,
    DeepgramNova3Provider,
)


# ---------------------------------------------------------------------- #
# Fixtures                                                               #
# ---------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-real-for-tests")
    monkeypatch.delenv("DEEPGRAM_MODEL", raising=False)
    yield


def _fake_word(word, start, end, *, speaker=0, confidence=0.93,
               punctuated=None):
    """Build a Deepgram-shaped Word object."""
    return SimpleNamespace(
        word=word,
        start=start,
        end=end,
        confidence=confidence,
        speaker=speaker,
        punctuated_word=punctuated if punctuated is not None else word,
    )


def _fake_response(
    *,
    words=None,
    duration: float = 12.5,
    request_id: str = "deepgram-body-req-001",
    detected_language: Optional[str] = None,
):
    if words is None:
        words = [
            _fake_word("namaste", 0.10, 0.45, speaker=0, confidence=0.94,
                       punctuated="Namaste"),
            _fake_word("world", 0.50, 0.80, speaker=1, confidence=0.88,
                       punctuated="world."),
        ]
    metadata = SimpleNamespace(
        duration=duration,
        request_id=request_id,
    )
    alt = SimpleNamespace(
        transcript="Namaste world.",
        confidence=0.92,
        words=words,
        language=detected_language,
    )
    channel = SimpleNamespace(
        alternatives=[alt],
        detected_language=detected_language,
    )
    results = SimpleNamespace(channels=[channel])
    return SimpleNamespace(metadata=metadata, results=results)


def _fake_async_deepgram_client(response, *, http_request_id="dg-http-001"):
    """Build a MagicMock AsyncDeepgramClient whose
    ``listen.v1.media.with_raw_response.transcribe_file`` returns the
    given response wrapped with an HTTP-level request_id."""
    raw = MagicMock()
    raw.request_id = http_request_id
    raw.parse.return_value = response

    client = MagicMock()
    client.listen = MagicMock()
    client.listen.v1 = MagicMock()
    client.listen.v1.media = MagicMock()
    client.listen.v1.media.with_raw_response = MagicMock()
    client.listen.v1.media.with_raw_response.transcribe_file = AsyncMock(
        return_value=raw,
    )
    return client


# Local import for Optional inside _fake_response above
from typing import Optional   # noqa: E402  (must come after fixtures use it)


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
        assert "deepgram" in PROVIDERS
        assert PROVIDERS["deepgram"] is DeepgramNova3Provider

    def test_class_name_attribute_matches_registry_key(self):
        assert DeepgramNova3Provider.name == "deepgram"

    def test_both_providers_coexist(self):
        # whisper-groq and deepgram should both be present after the
        # auto-load. This guards against module-import ordering bugs.
        assert "whisper-groq" in PROVIDERS
        assert "deepgram" in PROVIDERS


# ====================================================================== #
# Constructor                                                            #
# ====================================================================== #


class TestProviderInit:
    def test_default_model(self):
        p = DeepgramNova3Provider()
        assert p.model == "nova-3"

    def test_constructor_override(self):
        p = DeepgramNova3Provider(model="nova-2")
        assert p.model == "nova-2"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_MODEL", "enhanced")
        p = DeepgramNova3Provider()
        assert p.model == "enhanced"

    def test_init_does_not_require_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        # Construction must succeed without a key (dispatcher
        # instantiates providers eagerly; key checked at call time).
        DeepgramNova3Provider()


# ====================================================================== #
# File validation                                                        #
# ====================================================================== #


class TestFileValidation:
    @pytest.mark.asyncio
    async def test_missing_file_raises(self):
        p = DeepgramNova3Provider()
        with pytest.raises(FileNotFoundError):
            await p.transcribe(audio_path="/no/such.mp3", language="te")

    @pytest.mark.asyncio
    async def test_empty_file_raises(self, tmp_path):
        # Step 10 D-10.3 refinement: empty-file failure is permanent,
        # so the provider raises PermanentSTTError (not ValueError).
        from pipeline_v2.stages.stt import PermanentSTTError
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        p = DeepgramNova3Provider()
        with pytest.raises(PermanentSTTError, match=r"empty_file"):
            await p.transcribe(audio_path=str(empty), language="te")

    @pytest.mark.asyncio
    async def test_oversized_file_raises(self, tmp_path):
        # Step 10 D-10.3: oversize-file failure is permanent ->
        # PermanentSTTError (not ValueError).
        from pipeline_v2.stages.stt import PermanentSTTError
        big = tmp_path / "big.mp3"
        big.write_bytes(b"")
        os.truncate(big, 501 * 1024 * 1024)
        p = DeepgramNova3Provider()
        with pytest.raises(PermanentSTTError) as excinfo:
            await p.transcribe(audio_path=str(big), language="te")
        msg = str(excinfo.value)
        assert "500MB" in msg
        assert "file_too_large" in msg

    @pytest.mark.asyncio
    async def test_size_just_under_cap_is_ok(self, tmp_path):
        # 100 MB sparse file -- under cap, should proceed to API call.
        ok = tmp_path / "ok.mp3"
        ok.write_bytes(b"")
        os.truncate(ok, 100 * 1024 * 1024)
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=str(ok), language="te")
        assert r is not None


# ====================================================================== #
# API key missing                                                        #
# ====================================================================== #


class TestApiKeyMissing:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_runtime_error(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"\x00" * 1024)
        p = DeepgramNova3Provider()
        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
            await p.transcribe(audio_path=str(audio), language="te")


# ====================================================================== #
# Happy path -- words, confidence, speaker, punctuation                 #
# ====================================================================== #


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_provider_response(self, audio_file):
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert isinstance(r, ProviderResponse)
        assert isinstance(r.transcript, WordLevelTranscript)
        assert r.transcript.provider == "deepgram"

    @pytest.mark.asyncio
    async def test_per_word_confidence_populated(self, audio_file):
        # Critical differentiator vs Whisper-Groq / Chirp 3: Deepgram
        # DOES return per-word confidence. Stage 2 should use it.
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        words = r.transcript.words
        assert all(w.confidence is not None for w in words)
        assert words[0].confidence == pytest.approx(0.94)
        assert words[1].confidence == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_speaker_labels_populated_from_diarization(self, audio_file):
        # diarize=True is set by the provider -> speaker labels arrive
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        words = r.transcript.words
        assert words[0].speaker == 0
        assert words[1].speaker == 1

    @pytest.mark.asyncio
    async def test_punctuated_word_preferred(self, audio_file):
        # Provider should use punctuated_word over the raw word so
        # downstream stages see proper casing/punctuation.
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.transcript.words[0].w == "Namaste"
        assert r.transcript.words[1].w == "world."

    @pytest.mark.asyncio
    async def test_falls_back_to_word_when_no_punctuated(self, audio_file):
        words = [_fake_word("hi", 0.0, 0.3, punctuated=None)]
        words[0].punctuated_word = None             # explicit None
        response = _fake_response(words=words)
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.transcript.words[0].w == "hi"

    @pytest.mark.asyncio
    async def test_audio_duration_from_metadata(self, audio_file):
        response = _fake_response(duration=720.5)
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.audio_duration_sec == 720.5
        assert r.transcript.duration_sec == 720.5


# ====================================================================== #
# Request kwargs                                                         #
# ====================================================================== #


class TestRequestKwargs:
    def _call_kwargs(self, client):
        return (
            client.listen.v1.media.with_raw_response
            .transcribe_file.call_args.kwargs
        )

    @pytest.mark.asyncio
    async def test_required_kwargs(self, audio_file):
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            await p.transcribe(audio_path=audio_file, language="te")
        kw = self._call_kwargs(client)
        assert kw["model"] == "nova-3"
        assert kw["punctuate"] is True
        assert kw["diarize"] is True
        assert kw["smart_format"] is True
        assert "request" in kw                       # the audio bytes

    @pytest.mark.asyncio
    async def test_language_passed_through(self, audio_file):
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            await p.transcribe(audio_path=audio_file, language="te")
        assert self._call_kwargs(client)["language"] == "te"

    @pytest.mark.asyncio
    async def test_language_omitted_when_none(self, audio_file):
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            await p.transcribe(audio_path=audio_file, language=None)
        assert "language" not in self._call_kwargs(client)

    @pytest.mark.asyncio
    async def test_keyterm_built_from_names_and_brief(self, audio_file):
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            await p.transcribe(
                audio_path=audio_file, language="te",
                brief="Telangana election results",
                names=["Modi", "Reddy"],
            )
        kw = self._call_kwargs(client)
        assert "keyterm" in kw
        # names come first
        assert kw["keyterm"][0] == "Modi"
        assert kw["keyterm"][1] == "Reddy"
        # tokenised brief follows (4+ char words only)
        brief_tokens = kw["keyterm"][2:]
        for tok in brief_tokens:
            assert len(tok) >= 4
        assert "Telangana" in brief_tokens
        assert "election" in brief_tokens
        assert "results" in brief_tokens

    @pytest.mark.asyncio
    async def test_keyterm_omitted_when_no_names_no_brief(self, audio_file):
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            await p.transcribe(audio_path=audio_file, language="te")
        assert "keyterm" not in self._call_kwargs(client)

    @pytest.mark.asyncio
    async def test_keyterm_dedupes_case_insensitive(self, audio_file):
        response = _fake_response()
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            await p.transcribe(
                audio_path=audio_file, language="te",
                brief="Modi government policy",
                names=["Modi", "modi", "MODI"],
            )
        kw = self._call_kwargs(client)
        modi_count = sum(1 for t in kw["keyterm"] if t.lower() == "modi")
        assert modi_count == 1


# ====================================================================== #
# Request ID                                                             #
# ====================================================================== #


class TestRequestId:
    @pytest.mark.asyncio
    async def test_prefers_body_request_id(self, audio_file):
        response = _fake_response(request_id="body-canonical-id")
        client = _fake_async_deepgram_client(
            response, http_request_id="http-header-id",
        )
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        # Body-level metadata.request_id is the canonical Deepgram id.
        assert r.request_id == "body-canonical-id"

    @pytest.mark.asyncio
    async def test_falls_back_to_http_when_body_empty(self, audio_file):
        response = _fake_response(request_id="")
        client = _fake_async_deepgram_client(
            response, http_request_id="http-fallback-id",
        )
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.request_id == "http-fallback-id"

    @pytest.mark.asyncio
    async def test_falls_back_to_uuid_when_both_empty(self, audio_file):
        response = _fake_response(request_id="")
        client = _fake_async_deepgram_client(response, http_request_id="")
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        assert r.request_id.startswith("deepgram-client-")
        assert len(r.request_id) == len("deepgram-client-") + 12


# ====================================================================== #
# Cost                                                                   #
# ====================================================================== #


class TestCost:
    @pytest.mark.asyncio
    async def test_cost_computed_from_duration(self, audio_file):
        # 30 min at $0.0097/min = $0.291
        response = _fake_response(duration=1800.0)
        client = _fake_async_deepgram_client(response)
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            r = await p.transcribe(audio_path=audio_file, language="te")
        expected = 30.0 * DEEPGRAM_NOVA3_USD_PER_MIN
        assert r.cost_usd == pytest.approx(expected, rel=1e-6)
        # Sanity-check the constant matches our pricing comment
        assert DEEPGRAM_NOVA3_USD_PER_MIN == pytest.approx(0.0097)


# ====================================================================== #
# API errors propagate                                                   #
# ====================================================================== #


class TestApiErrors:
    @pytest.mark.asyncio
    async def test_api_exception_propagates(self, audio_file):
        client = MagicMock()
        client.listen = MagicMock()
        client.listen.v1 = MagicMock()
        client.listen.v1.media = MagicMock()
        client.listen.v1.media.with_raw_response = MagicMock()
        client.listen.v1.media.with_raw_response.transcribe_file = AsyncMock(
            side_effect=RuntimeError("HTTP 429 rate_limit_exceeded"),
        )
        with patch(
            "pipeline_v2.stages.stt.deepgram.AsyncDeepgramClient",
            return_value=client,
        ):
            p = DeepgramNova3Provider()
            with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
                await p.transcribe(audio_path=audio_file, language="te")
