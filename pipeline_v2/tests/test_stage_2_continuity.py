"""Unit tests for Stage 2 ContinuityEditor (Step 5.2).

All Gemini calls are mocked. No real API hits. The Step 5.3+ flow
covers real-API testing separately.

Coverage:
  - Constructor defaults + overrides
  - Lazy client init (no API key required to instantiate)
  - Prompt loading from disk + override via constructor
  - Word-array serialisation (Deepgram-style, 3-decimal timestamps,
    speaker/confidence dropped)
  - Markdown-fence stripping (defends against the 1.x bug residual)
  - Manual ``model_validate`` path — do NOT trust ``response.parsed``
  - Corrective retry: first fails -> second succeeds
  - Both attempts fail -> RuntimeError wrapping both
  - Empty response.text raises JSONDecodeError (triggers retry, not
    a silent None)
  - SDK exceptions propagate (auth, rate limit)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline_v2.models import (
    Stage1Output,
    Stage2Output,
    Word,
    WordLevelTranscript,
)
from pipeline_v2.stages.stage_2_continuity import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    Stage2ContinuityEditor,
    _serialize_words_for_prompt,
    _strip_markdown_fences,
)


# ---------------------------------------------------------------------- #
# Fixtures                                                               #
# ---------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _gemini_key(monkeypatch):
    """Provide a fake API key so _get_client() doesn't raise on
    construction. Tests still mock the actual Gemini call."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    yield


@pytest.fixture
def prompt_file(tmp_path):
    """Write a minimal prompt template to a temp file. Each test can
    pass this via ``prompt_path=`` so they're independent of the
    Step 5.3 file on disk."""
    p = tmp_path / "test_prompt.md"
    p.write_text(
        "# Test prompt\n\nEmit Stage2Output JSON.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def stage1_output():
    """Build a minimal Stage1Output suitable as Stage 2 input."""
    transcript = WordLevelTranscript(
        words=[
            Word(w="hi",    s=0.10, e=0.40, confidence=0.95, speaker=0),
            Word(w="there", s=0.42, e=0.80, confidence=0.91, speaker=0),
            Word(w="cut",   s=1.00, e=1.20, confidence=0.55, speaker=1),
        ],
        duration_sec=1.5,
        detected_languages=["te"],
        provider="whisper-groq",
    )
    return Stage1Output(
        transcript=transcript,
        stt_provider="whisper-groq",
        stt_audio_duration_sec=1.5,
        stt_wall_seconds=0.5,
        stt_cost_usd=0.0,
        stt_word_count=3,
        stt_avg_confidence=0.80,
        stt_language_detected="te",
        stt_request_id="r1",
    )


@pytest.fixture
def valid_stage2_json():
    """A well-formed Stage2Output JSON string."""
    return json.dumps({
        "full_video_cuts": [
            {
                "index": 0, "start_word_idx": 0, "end_word_idx": 1,
                "start_sec": 0.10, "end_sec": 0.80, "importance": 7,
            },
        ],
        "skipped_segments": [
            {
                "start_word_idx": 2, "end_word_idx": 2,
                "start_sec": 1.00, "end_sec": 1.20,
                "category": "crew_talk",
                "reason": "Off-camera voice says 'cut'.",
            },
        ],
        "retake_audit":
            "Skipped 1 crew_talk segment ('cut' at 1.00s). No retakes.",
    })


def _mock_response(text: str):
    """A mock GenerateContentResponse with the given ``.text`` value."""
    return SimpleNamespace(text=text, parsed=None)


# ====================================================================== #
# Pure helpers                                                           #
# ====================================================================== #


class TestStripMarkdownFences:
    def test_no_fences_unchanged(self):
        s = '{"a": 1}'
        assert _strip_markdown_fences(s) == s

    def test_json_fence_stripped(self):
        s = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fences(s) == '{"a": 1}'

    def test_bare_triple_backtick_stripped(self):
        s = '```\n{"a": 1}\n```'
        assert _strip_markdown_fences(s) == '{"a": 1}'

    def test_leading_whitespace_tolerated(self):
        s = '   ```json\n{"a": 1}\n```   '
        assert _strip_markdown_fences(s) == '{"a": 1}'

    def test_no_closing_fence_still_strips_opening(self):
        # Defensive: if the model truncates mid-fence, the opening
        # fence is still stripped so json.loads has a fighting chance
        # on the partial JSON.
        s = '```json\n{"a": 1}'
        assert _strip_markdown_fences(s) == '{"a": 1}'


class TestSerializeWordsForPrompt:
    def test_deepgram_shape(self, stage1_output):
        s = _serialize_words_for_prompt(stage1_output)
        arr = json.loads(s)
        assert len(arr) == 3
        assert set(arr[0].keys()) == {"w", "s", "e"}
        # Speaker / confidence intentionally NOT included.
        assert "speaker" not in arr[0]
        assert "confidence" not in arr[0]

    def test_3_decimal_rounding(self, stage1_output):
        # Inject a word with messy float to verify rounding.
        stage1_output.transcript.words.append(
            Word(w="x", s=2.123456789, e=2.987654321)
        )
        arr = json.loads(_serialize_words_for_prompt(stage1_output))
        last = arr[-1]
        assert last["s"] == 2.123
        assert last["e"] == 2.988

    def test_unicode_preserved(self, stage1_output):
        # Telugu glyphs should survive json.dumps via ensure_ascii=False
        stage1_output.transcript.words = [Word(w="నమస్తే", s=0, e=0.5)]
        s = _serialize_words_for_prompt(stage1_output)
        assert "నమస్తే" in s
        assert "\\u" not in s   # not ASCII-escaped


# ====================================================================== #
# Constructor                                                            #
# ====================================================================== #


class TestConstructor:
    def test_defaults(self, prompt_file):
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        assert e.model == DEFAULT_MODEL == "gemini-2.5-pro"
        assert e.temperature == DEFAULT_TEMPERATURE == 0.2
        assert e.thinking_budget == DEFAULT_THINKING_BUDGET == 2048
        assert e.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS == 16384

    def test_overrides(self, prompt_file):
        e = Stage2ContinuityEditor(
            model="gemini-2.5-flash",
            temperature=0.5,
            thinking_budget=0,
            max_output_tokens=8000,
            prompt_path=prompt_file,
        )
        assert e.model == "gemini-2.5-flash"
        assert e.temperature == 0.5
        assert e.thinking_budget == 0
        assert e.max_output_tokens == 8000

    def test_no_api_key_at_construction_ok(self, monkeypatch, prompt_file):
        # Construction should be free of side effects -- the dispatcher
        # instantiates Stage 2 eagerly and we don't want missing key to
        # break that. The key is checked when transcribe_to_decisions
        # actually runs.
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        Stage2ContinuityEditor(prompt_path=prompt_file)  # must not raise

    def test_missing_prompt_file_raises_lazily(self, tmp_path):
        # File missing at construction is FINE -- it's only read on
        # first transcribe call.
        missing = tmp_path / "nope.md"
        e = Stage2ContinuityEditor(prompt_path=missing)
        # Loading the prompt directly raises with the helpful message:
        with pytest.raises(FileNotFoundError, match="Step 5.3"):
            e._load_prompt()


# ====================================================================== #
# Gemini call + config                                                   #
# ====================================================================== #


class TestGeminiCallConfig:
    @pytest.mark.asyncio
    async def test_config_has_all_required_fields(
        self, stage1_output, prompt_file, valid_stage2_json,
    ):
        """The Gemini call must set response_mime_type, response_schema,
        temperature, max_output_tokens, thinking_config."""
        e = Stage2ContinuityEditor(prompt_path=prompt_file)

        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            return_value=_mock_response(valid_stage2_json),
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            await e.transcribe_to_decisions(stage1_output)

        kwargs = fake_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-pro"
        config = kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is Stage2Output
        assert config.temperature == 0.2
        assert config.max_output_tokens == 16384
        assert config.thinking_config.thinking_budget == 2048


# ====================================================================== #
# Happy path                                                             #
# ====================================================================== #


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_stage2_output(
        self, stage1_output, prompt_file, valid_stage2_json,
    ):
        e = Stage2ContinuityEditor(prompt_path=prompt_file)

        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            return_value=_mock_response(valid_stage2_json),
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            result = await e.transcribe_to_decisions(stage1_output)

        assert isinstance(result, Stage2Output)
        assert len(result.full_video_cuts) == 1
        assert result.full_video_cuts[0].importance == 7
        assert len(result.skipped_segments) == 1
        assert result.skipped_segments[0].category == "crew_talk"
        assert result.retake_audit.startswith("Skipped 1 crew_talk")

    @pytest.mark.asyncio
    async def test_markdown_fence_wrapped_response_still_parses(
        self, stage1_output, prompt_file, valid_stage2_json,
    ):
        # Defends against the 1.x markdown-fence bug residue.
        wrapped = f"```json\n{valid_stage2_json}\n```"
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            return_value=_mock_response(wrapped),
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            result = await e.transcribe_to_decisions(stage1_output)
        assert isinstance(result, Stage2Output)


# ====================================================================== #
# Manual validate path -- does NOT trust response.parsed                 #
# ====================================================================== #


class TestManualValidatePath:
    @pytest.mark.asyncio
    async def test_invented_category_caught_by_manual_validate(
        self, stage1_output, prompt_file, valid_stage2_json,
    ):
        """Critical regression guard: even though google-genai 2.4.0's
        ``response.parsed`` would SILENTLY swallow a ValidationError
        from an invented category and return None, our manual
        ``Stage2Output.model_validate()`` path raises explicitly.

        Without this, an invented category like 'redundancy' would
        slip through to Stage 2 output unnoticed -- the prompt's
        forbid-invention rule would be undermined.
        """
        # Inject 'redundancy' (the prototype invented category from
        # the Step 0 V1 incident) into the response.
        bad_json = json.dumps({
            "full_video_cuts": [
                {"index": 0, "start_word_idx": 0, "end_word_idx": 1,
                 "start_sec": 0.10, "end_sec": 0.80, "importance": 7},
            ],
            "skipped_segments": [
                {"start_word_idx": 2, "end_word_idx": 2,
                 "start_sec": 1.00, "end_sec": 1.20,
                 "category": "redundancy",   # INVENTED -- must raise
                 "reason": "x"},
            ],
            "retake_audit": "x",
        })
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        # Both calls return the same bad JSON so the corrective retry
        # also fails -- we expect RuntimeError, NOT a silently-None
        # Stage2Output.
        fake_client.aio.models.generate_content = AsyncMock(
            return_value=_mock_response(bad_json),
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await e.transcribe_to_decisions(stage1_output)

    @pytest.mark.asyncio
    async def test_empty_response_text_treated_as_decode_error(
        self, stage1_output, prompt_file,
    ):
        """If Gemini returns an empty body (MAX_TOKENS truncation,
        thinking exhausted the budget, etc.), don't silently produce
        None. Surface as a JSONDecodeError so the corrective-retry
        layer engages or the second-failure RuntimeError fires."""
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            return_value=_mock_response(""),
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await e.transcribe_to_decisions(stage1_output)


# ====================================================================== #
# Corrective retry                                                       #
# ====================================================================== #


class TestCorrectiveRetry:
    @pytest.mark.asyncio
    async def test_first_fails_second_succeeds(
        self, stage1_output, prompt_file, valid_stage2_json,
    ):
        bad_json = '{"this is not valid Stage2Output": true}'
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=[
                _mock_response(bad_json),         # first attempt: invalid
                _mock_response(valid_stage2_json),  # retry: valid
            ],
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            result = await e.transcribe_to_decisions(stage1_output)

        assert isinstance(result, Stage2Output)
        # Confirm BOTH calls happened (1 attempt + 1 retry = 2 calls).
        assert fake_client.aio.models.generate_content.await_count == 2

        # The retry prompt should include the validation error.
        retry_prompt = (
            fake_client.aio.models.generate_content
            .await_args_list[1].kwargs["contents"][0]
        )
        assert "previous response failed validation" in retry_prompt.lower()

    @pytest.mark.asyncio
    async def test_both_attempts_fail_raises_runtimeerror(
        self, stage1_output, prompt_file,
    ):
        bad_json = '{"wrong": "shape"}'
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            return_value=_mock_response(bad_json),
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await e.transcribe_to_decisions(stage1_output)
        # Both attempts were made; we don't double-retry beyond 1.
        assert fake_client.aio.models.generate_content.await_count == 2

    @pytest.mark.asyncio
    async def test_corrective_note_appended_to_prompt(
        self, stage1_output, prompt_file, valid_stage2_json,
    ):
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=[
                _mock_response('{"bad": true}'),
                _mock_response(valid_stage2_json),
            ],
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            await e.transcribe_to_decisions(stage1_output)

        first_call_prompt = (
            fake_client.aio.models.generate_content
            .await_args_list[0].kwargs["contents"][0]
        )
        retry_prompt = (
            fake_client.aio.models.generate_content
            .await_args_list[1].kwargs["contents"][0]
        )
        # First prompt has no correction note; retry prompt does.
        assert "Correction note" not in first_call_prompt
        assert "Correction note" in retry_prompt
        # Forbid-invention reminder is included in the correction note.
        assert "NEVER invent new categories" in retry_prompt


# ====================================================================== #
# Exception propagation                                                  #
# ====================================================================== #


class TestExceptionPropagation:
    @pytest.mark.asyncio
    async def test_sdk_auth_error_propagates(
        self, stage1_output, prompt_file,
    ):
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        fake_client = MagicMock()
        fake_client.aio = MagicMock()
        fake_client.aio.models = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("HTTP 401 unauthorized"),
        )
        with patch(
            "pipeline_v2.stages.stage_2_providers.genai.Client",
            return_value=fake_client,
        ):
            # The provider does NOT catch SDK exceptions -- only
            # JSONDecodeError / ValidationError. Auth errors etc.
            # propagate to the dispatcher / Inngest.
            with pytest.raises(RuntimeError, match="unauthorized"):
                await e.transcribe_to_decisions(stage1_output)

    @pytest.mark.asyncio
    async def test_missing_api_key_at_call_time_raises(
        self, stage1_output, prompt_file, monkeypatch,
    ):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            await e.transcribe_to_decisions(stage1_output)


# ====================================================================== #
# Prompt loading                                                         #
# ====================================================================== #


class TestPromptLoading:
    def test_default_prompt_path_resolves_under_stages(self):
        e = Stage2ContinuityEditor()
        assert e.prompt_path.name == "stage_2_prompt.md"
        # Co-located with the module.
        assert "stages" in e.prompt_path.parts

    def test_prompt_template_cached_after_first_load(self, prompt_file):
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        a = e._load_prompt()
        b = e._load_prompt()
        assert a is b   # cached, not re-read

    def test_prompt_path_override_honored(self, prompt_file):
        e = Stage2ContinuityEditor(prompt_path=prompt_file)
        loaded = e._load_prompt()
        assert "Test prompt" in loaded
