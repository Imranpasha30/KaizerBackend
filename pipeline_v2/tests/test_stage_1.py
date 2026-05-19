"""Model-shape tests for Stage 1.

After Step 4's expansion to a multi-provider STT layer, the bulk of
the dispatcher / contract tests moved to ``test_stt_dispatcher.py``.
This file keeps only the schema-shape tests for the public Pydantic
models (``Word`` / ``WordLevelTranscript`` / ``Stage1Output``) and the
regression guards for the Step 4 renames.

The 14 faster-whisper-specific tests that used to live here were
deleted along with the faster-whisper implementation. Their git
history records the failure mode (Telugu hallucination loops) that
prompted the multi-provider switch.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_v2.models import Stage1Output, Word, WordLevelTranscript


class TestWordModel:
    def test_minimal_construct(self):
        w = Word(w="hello", s=0.0, e=0.3)
        assert w.w == "hello"
        assert w.speaker is None
        assert w.confidence is None         # new field defaults to None

    def test_confidence_accepts_float(self):
        # Pydantic stores the float as-is; we deliberately don't clamp
        # to [0,1] because providers expose confidence on different
        # scales.
        w = Word(w="hi", s=0.0, e=0.3, confidence=0.87)
        assert w.confidence == 0.87

    def test_confidence_optional(self):
        w = Word(w="hi", s=0.0, e=0.3, confidence=None)
        assert w.confidence is None


class TestWordLevelTranscriptModel:
    def test_construct_and_roundtrip(self):
        t = WordLevelTranscript(
            words=[Word(w="hi", s=0.0, e=0.3, confidence=0.9)],
            duration_sec=1.0,
            detected_languages=["en"],
            provider="chirp3",
        )
        # JSON roundtrip preserves all fields including the new provider
        # key and per-word confidence.
        t2 = WordLevelTranscript.model_validate_json(t.model_dump_json())
        assert t2.provider == "chirp3"
        assert t2.words[0].confidence == 0.9

    def test_provider_field_required(self):
        # provider is a required field as of Step 4.0 -- providers must
        # set it explicitly.
        with pytest.raises(ValidationError):
            WordLevelTranscript(
                words=[Word(w="hi", s=0.0, e=0.3)],
                duration_sec=1.0,
                detected_languages=["en"],
                # provider intentionally omitted
            )

    def test_old_name_removed(self):
        # Step 4 rename regression guard: DeepgramTranscript was
        # renamed in an earlier sub-step; ensure nobody adds it back.
        from pipeline_v2 import models
        assert not hasattr(models, "DeepgramTranscript")
        assert hasattr(models, "WordLevelTranscript")


class TestStage1OutputModel:
    def _make_transcript(self) -> WordLevelTranscript:
        return WordLevelTranscript(
            words=[Word(w="hi", s=0.0, e=0.3)],
            duration_sec=1.0,
            detected_languages=["en"],
            provider="chirp3",
        )

    def test_construct_with_all_metadata_fields(self):
        out = Stage1Output(
            transcript=self._make_transcript(),
            stt_provider="chirp3",
            stt_audio_duration_sec=120.5,
            stt_wall_seconds=15.3,
            stt_cost_usd=0.064,
            stt_word_count=1,
            stt_avg_confidence=0.87,
            stt_language_detected="te",
            stt_request_id="req_abc123",
            stt_language_hint="te",
            stt_brief="A Telugu news podcast about election results.",
            stt_names=["Modi", "Telangana"],
        )
        assert out.stt_provider == "chirp3"
        assert out.stt_names == ["Modi", "Telangana"]

    def test_realtime_factor_property(self):
        out = Stage1Output(
            transcript=self._make_transcript(),
            stt_provider="chirp3",
            stt_audio_duration_sec=300.0,
            stt_wall_seconds=30.0,
            stt_cost_usd=0.0,
            stt_word_count=1,
            stt_language_detected="te",
            stt_request_id="x",
        )
        assert out.realtime_factor == 10.0

    def test_realtime_factor_handles_zero_wall(self):
        out = Stage1Output(
            transcript=self._make_transcript(),
            stt_provider="chirp3",
            stt_audio_duration_sec=300.0,
            stt_wall_seconds=0.0,             # degenerate
            stt_cost_usd=0.0,
            stt_word_count=1,
            stt_language_detected="te",
            stt_request_id="x",
        )
        assert out.realtime_factor == float("inf")

    def test_avg_confidence_optional(self):
        # Providers without per-word confidence pass None.
        out = Stage1Output(
            transcript=self._make_transcript(),
            stt_provider="whisper-groq",
            stt_audio_duration_sec=1.0,
            stt_wall_seconds=0.1,
            stt_cost_usd=0.0,
            stt_word_count=1,
            stt_avg_confidence=None,
            stt_language_detected="en",
            stt_request_id="x",
        )
        assert out.stt_avg_confidence is None


class TestStage1TranscribeRedirect:
    def test_redirect_module_reexports_dispatcher(self):
        # stage_1_transcribe.py is a thin redirect after Step 4. Anything
        # that imports run_stage_1 from there must still work.
        from pipeline_v2.stages import stage_1_transcribe
        from pipeline_v2.stages.stt import run_stage_1 as canonical

        assert stage_1_transcribe.run_stage_1 is canonical
        assert hasattr(stage_1_transcribe, "PROVIDERS")
        assert hasattr(stage_1_transcribe, "register")
        assert hasattr(stage_1_transcribe, "ProviderResponse")
        assert hasattr(stage_1_transcribe, "TranscriptionProvider")
