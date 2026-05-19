"""Tests for the Stage 1 STT abstraction layer.

Covers:
  - Registry: ``@register`` decorator, duplicate names, empty names,
    cls.name attribute set correctly.
  - Dispatcher: provider lookup, missing audio file, unknown provider.
  - Word-level contract enforcement: empty word list rejected,
    transcript.provider mismatch rejected, negative-duration word
    rejected.
  - Stage1Output construction: dispatcher wraps provider response
    into the typed output correctly.
  - On-disk persistence: transcript.json + stt_metadata.json get
    written to out_dir.
  - Telemetry: stt_avg_confidence is mean of populated, None if all
    None; realtime_factor computed correctly.

NO real provider is invoked. We hand-build ``ProviderResponse`` objects
and register fake provider classes for each test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from pipeline_v2.models import Stage1Output, Word, WordLevelTranscript
from pipeline_v2.stages import stt as stt_module
from pipeline_v2.stages.stt import (
    PROVIDERS,
    ProviderResponse,
    register,
    run_stage_1,
)


# ---------------------------------------------------------------------- #
# Fixtures / helpers                                                     #
# ---------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_registry():
    """Snapshot + restore the global registry around every test.

    Tests register fake providers; we don't want one test's
    registration leaking into the next, and we don't want to clobber
    any real providers a future provider-module import added.
    """
    snapshot = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(snapshot)


@pytest.fixture
def audio_file(tmp_path):
    p = tmp_path / "src.mp3"
    p.write_bytes(b"\x00" * 16)        # content doesn't matter; providers are mocked
    return str(p)


def _transcript(provider: str = "fake", *, words=None) -> WordLevelTranscript:
    """Build a small valid WordLevelTranscript for fake providers."""
    if words is None:
        words = [
            Word(w="hi", s=0.10, e=0.40, confidence=0.95),
            Word(w="there", s=0.42, e=0.80, confidence=0.91),
        ]
    return WordLevelTranscript(
        words=words,
        duration_sec=2.0,
        detected_languages=["en"],
        provider=provider,
    )


def _make_fake_provider(
    response_factory,
    name: str = "fake",
):
    """Create + register a fake provider class that returns ``response``."""

    @register(name)
    class _FakeProvider:
        # name set by @register
        async def transcribe(self, *, audio_path, language, brief="", names=None):
            return response_factory(
                audio_path=audio_path, language=language,
                brief=brief, names=names or [],
            )

    return _FakeProvider


# ====================================================================== #
# Registry                                                               #
# ====================================================================== #


class TestRegistry:
    def test_register_adds_to_dict_and_sets_name(self):
        @register("alpha")
        class A:
            async def transcribe(self, **kw): ...
        assert "alpha" in PROVIDERS
        assert PROVIDERS["alpha"] is A
        assert A.name == "alpha"

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            register("")(type("X", (), {}))

    def test_register_rejects_duplicate_name(self):
        @register("dup")
        class A: ...
        with pytest.raises(ValueError, match="already registered"):
            @register("dup")
            class B: ...

    def test_provider_class_self_attribute(self):
        @register("zeta")
        class Z: ...
        # cls.name and registry key match by construction -- they can't
        # diverge.
        assert Z.name == "zeta"
        assert PROVIDERS["zeta"].name == "zeta"


# ====================================================================== #
# Dispatcher input validation                                            #
# ====================================================================== #


class TestDispatcherInput:
    @pytest.mark.asyncio
    async def test_missing_audio_file_raises(self):
        with pytest.raises(FileNotFoundError, match="audio file not found"):
            await run_stage_1("/no/such/audio.mp3", provider="anything")

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_with_available_list(self, audio_file):
        with pytest.raises(ValueError, match="unknown STT provider"):
            await run_stage_1(audio_file, provider="not-a-real-provider-name")

    @pytest.mark.asyncio
    async def test_provider_required_kwarg(self, audio_file):
        with pytest.raises(TypeError):                # no default
            await run_stage_1(audio_file)             # type: ignore[call-arg]


# ====================================================================== #
# Word-level contract enforcement                                        #
# ====================================================================== #


class TestWordLevelContract:
    @pytest.mark.asyncio
    async def test_empty_word_list_rejected(self, audio_file):
        empty = WordLevelTranscript(
            words=[], duration_sec=2.0, detected_languages=["en"],
            provider="fake",
        )
        _make_fake_provider(lambda **kw: ProviderResponse(
            transcript=empty, cost_usd=0.0, request_id="x",
            audio_duration_sec=2.0,
        ))
        with pytest.raises(RuntimeError, match="zero.*words"):
            await run_stage_1(audio_file, provider="fake")

    @pytest.mark.asyncio
    async def test_provider_name_mismatch_rejected(self, audio_file):
        # Provider returns transcript.provider="impostor" but is registered
        # under "fake". Dispatcher must catch the mismatch.
        bad = _transcript(provider="impostor")
        _make_fake_provider(lambda **kw: ProviderResponse(
            transcript=bad, cost_usd=0.0, request_id="x",
            audio_duration_sec=2.0,
        ))
        with pytest.raises(RuntimeError, match="transcript.provider"):
            await run_stage_1(audio_file, provider="fake")

    @pytest.mark.asyncio
    async def test_negative_duration_word_rejected(self, audio_file):
        # Pydantic won't catch this -- start and end are both floats,
        # the relationship is the contract.
        bad_words = [
            Word(w="oops", s=5.0, e=4.0),     # end before start
        ]
        bad = _transcript(words=bad_words)
        _make_fake_provider(lambda **kw: ProviderResponse(
            transcript=bad, cost_usd=0.0, request_id="x",
            audio_duration_sec=2.0,
        ))
        with pytest.raises(RuntimeError, match="end < start"):
            await run_stage_1(audio_file, provider="fake")


# ====================================================================== #
# Stage1Output wrapping                                                  #
# ====================================================================== #


class TestStage1OutputWrapping:
    @pytest.mark.asyncio
    async def test_dispatcher_wraps_into_stage1_output(self, audio_file):
        resp = ProviderResponse(
            transcript=_transcript(),
            cost_usd=0.0123,
            request_id="req_fake_001",
            audio_duration_sec=2.0,
        )
        _make_fake_provider(lambda **kw: resp)

        out = await run_stage_1(
            audio_file, provider="fake",
            language_hint="en", brief="a test brief", names=["Alice"],
        )

        assert isinstance(out, Stage1Output)
        assert out.stt_provider == "fake"
        assert out.stt_audio_duration_sec == 2.0
        assert out.stt_cost_usd == 0.0123
        assert out.stt_word_count == 2
        assert out.stt_request_id == "req_fake_001"
        assert out.stt_language_hint == "en"
        assert out.stt_brief == "a test brief"
        assert out.stt_names == ["Alice"]
        assert out.stt_wall_seconds >= 0
        # avg_confidence: mean of [0.95, 0.91] = 0.93
        assert out.stt_avg_confidence == pytest.approx(0.93)

    @pytest.mark.asyncio
    async def test_avg_confidence_none_when_no_words_have_it(self, audio_file):
        words_no_conf = [
            Word(w="x", s=0.0, e=0.3),
            Word(w="y", s=0.31, e=0.5),
        ]
        resp = ProviderResponse(
            transcript=_transcript(words=words_no_conf),
            cost_usd=0.0, request_id="x", audio_duration_sec=1.0,
        )
        _make_fake_provider(lambda **kw: resp)

        out = await run_stage_1(audio_file, provider="fake")
        assert out.stt_avg_confidence is None

    @pytest.mark.asyncio
    async def test_detected_language_falls_back_to_hint(self, audio_file):
        # Provider returned a transcript with detected_languages=[] but the
        # caller passed a hint. Dispatcher should use the hint.
        t = WordLevelTranscript(
            words=[Word(w="x", s=0, e=0.3)],
            duration_sec=1.0, detected_languages=[],   # empty
            provider="fake",
        )
        _make_fake_provider(lambda **kw: ProviderResponse(
            transcript=t, cost_usd=0.0, request_id="x", audio_duration_sec=1.0,
        ))
        out = await run_stage_1(audio_file, provider="fake", language_hint="te")
        assert out.stt_language_detected == "te"

    @pytest.mark.asyncio
    async def test_detected_language_unknown_when_no_signal(self, audio_file):
        t = WordLevelTranscript(
            words=[Word(w="x", s=0, e=0.3)],
            duration_sec=1.0, detected_languages=[],
            provider="fake",
        )
        _make_fake_provider(lambda **kw: ProviderResponse(
            transcript=t, cost_usd=0.0, request_id="x", audio_duration_sec=1.0,
        ))
        out = await run_stage_1(audio_file, provider="fake")  # no hint
        assert out.stt_language_detected == "unknown"


# ====================================================================== #
# Persistence (transcript.json + stt_metadata.json)                      #
# ====================================================================== #


class TestPersistence:
    @pytest.mark.asyncio
    async def test_writes_both_files_when_out_dir(self, audio_file, tmp_path):
        resp = ProviderResponse(
            transcript=_transcript(),
            cost_usd=0.05, request_id="r1", audio_duration_sec=2.0,
        )
        _make_fake_provider(lambda **kw: resp)

        out_dir = tmp_path / "stage1_out"
        out = await run_stage_1(
            audio_file, provider="fake", language_hint="en",
            brief="b", names=["N"], out_dir=str(out_dir),
        )

        assert Path(out.transcript_json_path).is_file()
        assert Path(out.metadata_json_path).is_file()

        # transcript.json roundtrips through the same Pydantic model
        t = WordLevelTranscript.model_validate_json(
            Path(out.transcript_json_path).read_text(encoding="utf-8")
        )
        assert t.provider == "fake"

        # stt_metadata.json has all ledger fields
        meta = json.loads(Path(out.metadata_json_path).read_text(encoding="utf-8"))
        for key in (
            "stt_provider", "stt_audio_duration_sec", "stt_wall_seconds",
            "stt_cost_usd", "stt_word_count", "stt_avg_confidence",
            "stt_language_detected", "stt_request_id",
            "stt_language_hint", "stt_brief", "stt_names",
        ):
            assert key in meta, f"missing {key!r} in stt_metadata.json"
        assert meta["stt_provider"] == "fake"
        assert meta["stt_brief"] == "b"
        assert meta["stt_names"] == ["N"]

    @pytest.mark.asyncio
    async def test_no_out_dir_means_no_files(self, audio_file):
        resp = ProviderResponse(
            transcript=_transcript(), cost_usd=0.0,
            request_id="x", audio_duration_sec=1.0,
        )
        _make_fake_provider(lambda **kw: resp)

        out = await run_stage_1(audio_file, provider="fake", out_dir=None)
        assert out.transcript_json_path is None
        assert out.metadata_json_path is None


# ====================================================================== #
# Dispatcher passes inputs through correctly                             #
# ====================================================================== #


class TestDispatcherForwarding:
    @pytest.mark.asyncio
    async def test_passes_all_kwargs_to_provider(self, audio_file):
        captured: dict = {}

        def _capture(**kw):
            captured.update(kw)
            return ProviderResponse(
                transcript=_transcript(), cost_usd=0.0,
                request_id="x", audio_duration_sec=1.0,
            )
        _make_fake_provider(_capture)

        await run_stage_1(
            audio_file, provider="fake",
            language_hint="te", brief="news podcast",
            names=["Modi", "Telangana"],
        )

        assert captured["language"] == "te"
        assert captured["brief"] == "news podcast"
        assert captured["names"] == ["Modi", "Telangana"]
        # audio_path normalised to str
        assert isinstance(captured["audio_path"], str)
        assert captured["audio_path"].endswith(".mp3")

    @pytest.mark.asyncio
    async def test_names_default_to_empty_list(self, audio_file):
        captured: dict = {}

        def _capture(**kw):
            captured.update(kw)
            return ProviderResponse(
                transcript=_transcript(), cost_usd=0.0,
                request_id="x", audio_duration_sec=1.0,
            )
        _make_fake_provider(_capture)

        await run_stage_1(audio_file, provider="fake")   # no names=
        assert captured["names"] == []
