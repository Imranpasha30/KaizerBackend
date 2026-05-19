"""Unit tests for Stage 3a (Shorts Generator) -- Step 7.2.

Same testing pattern as Stage 2 / 2.5:

  - All Gemini calls mocked.
  - Manual model_validate path (NEVER response.parsed).
  - Corrective retry on JSONDecodeError / ValidationError; second
    failure raises RuntimeError.
  - The 15-60s duration validator on ShortsCut IS part of the
    Pydantic validation -- corrective retry catches violations.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    Stage3aOutput,
    Word,
)
from pipeline_v2.stages.stage_3a_shorts import (
    Stage3aShortsGenerator,
    _strip_markdown_fences,
)


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _word(text: str, s: float, e: float) -> Word:
    return Word(w=text, s=s, e=e)


def _clean(n: int = 60) -> CleanTranscript:
    """Synthetic clean transcript -- n sequential 0.5s words spanning
    n*0.55 seconds. Long enough that 15-60s shorts can fit.
    """
    words = [_word(f"w{i}", i * 0.55, i * 0.55 + 0.5) for i in range(n)]
    return CleanTranscript(
        words=words,
        clip_boundaries={0: (0, n - 1)},
        source_word_map=list(range(n)),
    )


def _entity(name: str = "X", mentions=None, type_: str = "PERSON") -> Entity:
    return Entity(
        canonical_name=name,
        native_name=name,
        first_mention_word_idx=min(mentions) if mentions else 0,
        type=type_,
        mentions=mentions or [0],
    )


def _shorts_cut(idx: int = 0, start: float = 10.0, end: float = 28.0,
                hook: str = "Test hook", importance: int = 7) -> dict:
    return {
        "index": idx,
        "start_sec": start,
        "end_sec": end,
        "hook": hook,
        "importance": importance,
    }


def _good_response_body(n: int = 3) -> dict:
    return {
        "shorts_cuts": [
            _shorts_cut(idx=i, start=10.0 + i * 30, end=30.0 + i * 30,
                        hook=f"Hook {i}", importance=5 + i)
            for i in range(n)
        ],
    }


def _mock_response(json_text: str) -> MagicMock:
    r = MagicMock()
    r.text = json_text
    return r


def _writable_prompt(tmp_path: Path, body: str = "STAGE 3a STUB PROMPT") -> Path:
    p = tmp_path / "stage_3a_prompt.md"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def generator(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    prompt_path = _writable_prompt(tmp_path, "TEMPLATE")
    return Stage3aShortsGenerator(prompt_path=prompt_path)


# ====================================================================== #
# _strip_markdown_fences                                                  #
# ====================================================================== #


class TestStripMarkdownFences:
    def test_passthrough_clean_json(self):
        assert _strip_markdown_fences('{"x": 1}') == '{"x": 1}'

    def test_strip_json_fence(self):
        assert _strip_markdown_fences('```json\n{"x": 1}\n```') == '{"x": 1}'

    def test_strip_plain_fence(self):
        assert _strip_markdown_fences('```\n{"x": 1}\n```') == '{"x": 1}'


# ====================================================================== #
# Constructor + lazy resources                                            #
# ====================================================================== #


class TestConstructor:
    def test_defaults(self):
        g = Stage3aShortsGenerator()
        assert g.model == "gemini-2.5-flash"
        assert g.temperature == 0.7
        assert g.thinking_budget == 512
        assert g.max_output_tokens == 4096

    def test_kwargs_override(self):
        g = Stage3aShortsGenerator(
            model="gemini-2.5-flash-lite",
            temperature=0.5,
            thinking_budget=1024,
            max_output_tokens=2048,
        )
        assert g.model == "gemini-2.5-flash-lite"
        assert g.temperature == 0.5
        assert g.thinking_budget == 1024
        assert g.max_output_tokens == 2048

    def test_init_is_side_effect_free(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        g = Stage3aShortsGenerator(prompt_path=Path("/nope/missing.md"))
        assert g._client is None
        assert g._prompt_template is None

    def test_missing_api_key_raises_on_first_use(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        g = Stage3aShortsGenerator()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            g._get_client()

    def test_missing_prompt_raises_on_first_use(self):
        g = Stage3aShortsGenerator(prompt_path=Path("/nope/missing.md"))
        with pytest.raises(FileNotFoundError, match="Stage 3a prompt"):
            g._load_prompt()


# ====================================================================== #
# Prompt construction                                                     #
# ====================================================================== #


class TestPromptConstruction:
    def test_prompt_contains_metadata_and_entities(self, generator):
        clean = _clean(40)
        entities = [_entity("Revanth Reddy", [5, 10, 20], "PERSON")]
        prompt = generator._build_prompt(clean, entities)
        assert "TEMPLATE" in prompt
        assert "Word count:        40" in prompt
        # Entities appear in context section (NOT in output)
        assert "Revanth Reddy" in prompt
        # Word array surfaces
        assert '"idx": 0' in prompt
        assert '"w": "w0"' in prompt

    def test_empty_entities_works(self, generator):
        # 0 entities is legitimate -- the bulletin might have no
        # recognized named entities. Stage 3a should still produce
        # shorts.
        clean = _clean(20)
        prompt = generator._build_prompt(clean, [])
        assert "[]" in prompt  # entities serializes to []

    def test_correction_note_appended(self, generator):
        clean = _clean(20)
        prompt = generator._build_prompt(clean, [], correction_note="FIX X")
        assert "## Correction note" in prompt
        assert "FIX X" in prompt


# ====================================================================== #
# _parse_response: manual validate path                                   #
# ====================================================================== #


class TestParseResponse:
    def test_happy_path(self, generator):
        body = _good_response_body(n=4)
        resp = _mock_response(json.dumps(body))
        out = generator._parse_response(resp)
        assert isinstance(out, Stage3aOutput)
        assert len(out.shorts_cuts) == 4

    def test_strips_fences(self, generator):
        body = _good_response_body(n=3)
        resp = _mock_response(f"```json\n{json.dumps(body)}\n```")
        out = generator._parse_response(resp)
        assert len(out.shorts_cuts) == 3

    def test_empty_text_raises_json_decode_error(self, generator):
        resp = _mock_response("")
        with pytest.raises(json.JSONDecodeError):
            generator._parse_response(resp)

    def test_invalid_json_raises_json_decode_error(self, generator):
        resp = _mock_response("not json")
        with pytest.raises(json.JSONDecodeError):
            generator._parse_response(resp)

    def test_duration_violation_caught_by_manual_validate(self, generator):
        # CRITICAL REGRESSION: 14s short violates 15-60s validator.
        # Pydantic raises ValidationError; corrective-retry catches.
        body = {
            "shorts_cuts": [
                _shorts_cut(idx=0, start=0.0, end=14.0),   # 14s < 15s
                _shorts_cut(idx=1, start=30.0, end=50.0),
                _shorts_cut(idx=2, start=60.0, end=80.0),
            ],
        }
        resp = _mock_response(json.dumps(body))
        with pytest.raises(ValidationError, match="15-60s"):
            generator._parse_response(resp)

    def test_too_few_shorts_rejected(self, generator):
        # min_length=3 -- 2 shorts must raise.
        body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 20.0),
                _shorts_cut(1, 30.0, 50.0),
            ],
        }
        resp = _mock_response(json.dumps(body))
        with pytest.raises(ValidationError):
            generator._parse_response(resp)

    def test_too_many_shorts_rejected(self, generator):
        # max_length=10 -- 11 shorts must raise.
        body = {
            "shorts_cuts": [
                _shorts_cut(i, i * 30.0, i * 30.0 + 20.0)
                for i in range(11)
            ],
        }
        resp = _mock_response(json.dumps(body))
        with pytest.raises(ValidationError):
            generator._parse_response(resp)


# ====================================================================== #
# async generate: end-to-end with mocked Gemini                           #
# ====================================================================== #


class TestGenerateHappyPath:
    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(self, generator):
        clean = _clean(40)
        body = _good_response_body(n=4)
        mock_call = AsyncMock(return_value=_mock_response(json.dumps(body)))
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert len(out.shorts_cuts) == 4
        assert mock_call.await_count == 1


class TestGenerateCorrectiveRetry:
    @pytest.mark.asyncio
    async def test_duration_violation_triggers_retry(self, generator):
        clean = _clean(40)
        bad_body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 12.0),    # 12s < 15s, retry trigger
                _shorts_cut(1, 30.0, 50.0),
                _shorts_cut(2, 60.0, 80.0),
            ],
        }
        good_body = _good_response_body(n=3)
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(bad_body)),
            _mock_response(json.dumps(good_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert mock_call.await_count == 2
        assert len(out.shorts_cuts) == 3

    @pytest.mark.asyncio
    async def test_json_decode_triggers_retry(self, generator):
        clean = _clean(40)
        good_body = _good_response_body(n=3)
        mock_call = AsyncMock(side_effect=[
            _mock_response("not json"),
            _mock_response(json.dumps(good_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_double_failure_raises_runtime_error(self, generator):
        clean = _clean(40)
        mock_call = AsyncMock(side_effect=[
            _mock_response("not json"),
            _mock_response("still not json"),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await generator.generate(clean, [])
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_other_exceptions_propagate(self, generator):
        clean = _clean(40)
        mock_call = AsyncMock(side_effect=ConnectionError("network down"))
        with patch.object(generator, "_call_gemini", new=mock_call):
            with pytest.raises(ConnectionError, match="network down"):
                await generator.generate(clean, [])
        assert mock_call.await_count == 1
