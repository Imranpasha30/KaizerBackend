"""Unit tests for Stage 2.5 (Entity Canonicalizer) -- Step 6.2.

Same testing pattern as test_stage_2_continuity.py:

  - All Gemini calls mocked. We never hit the live API.
  - Manual model_validate path -- do NOT trust ``response.parsed``.
  - Corrective retry on JSONDecodeError / ValidationError; second
    failure raises RuntimeError.
  - Post-validate 6-cap truncation (D3 layer 3) with deterministic
    sort: mention count DESC, tiebreak first_mention_word_idx ASC.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    EntityType,
    Stage2_5Output,
    Word,
)
from pipeline_v2.stages.stage_2_5_entity_canonicalizer import (
    Stage2_5EntityCanonicalizer,
    _strip_markdown_fences,
    _truncate_to_cap,
)


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _word(text: str, s: float, e: float) -> Word:
    return Word(w=text, s=s, e=e)


def _clean(n: int = 30) -> CleanTranscript:
    """Synthetic clean transcript -- n sequential words, 1 cut covers
    the whole thing, identity source_word_map.
    """
    words = [_word(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(n)]
    return CleanTranscript(
        words=words,
        clip_boundaries={0: (0, n - 1)},
        source_word_map=list(range(n)),
    )


def _entity(name: str, mentions: list[int], type_: str = "PERSON") -> Entity:
    return Entity(
        canonical_name=name,
        native_name=name,
        first_mention_word_idx=min(mentions),
        type=type_,
        mentions=mentions,
    )


def _stage_2_5_output(entities: list[Entity]) -> Stage2_5Output:
    return Stage2_5Output(entities=entities)


def _mock_response(json_text: str) -> MagicMock:
    """Construct a mock response.* object whose .text returns json_text."""
    r = MagicMock()
    r.text = json_text
    return r


def _writable_prompt(tmp_path: Path, body: str = "STAGE 2.5 STUB PROMPT") -> Path:
    p = tmp_path / "stage_2_5_prompt.md"
    p.write_text(body, encoding="utf-8")
    return p


# ====================================================================== #
# _strip_markdown_fences                                                  #
# ====================================================================== #


class TestStripMarkdownFences:
    def test_passthrough_clean_json(self):
        assert _strip_markdown_fences('{"x": 1}') == '{"x": 1}'

    def test_strip_json_fence(self):
        wrapped = '```json\n{"x": 1}\n```'
        assert _strip_markdown_fences(wrapped) == '{"x": 1}'

    def test_strip_plain_fence(self):
        wrapped = '```\n{"x": 1}\n```'
        assert _strip_markdown_fences(wrapped) == '{"x": 1}'

    def test_strip_with_surrounding_whitespace(self):
        wrapped = '  \n```json\n{"x": 1}\n```\n  '
        assert _strip_markdown_fences(wrapped) == '{"x": 1}'


# ====================================================================== #
# _truncate_to_cap -- the third safety net (D3 layer 3)                   #
# ====================================================================== #


class TestTruncateToCapSortAlgorithm:
    """Sort algorithm per Step 6 D3-ADDITIONAL:

      1. Primary: mention count DESC
      2. Tiebreak: first_mention_word_idx ASC

    Reasoning: more-mentioned = more screen time in renderer;
    earlier-introduced wins ties to favor the story's primary thread.
    """

    def test_no_truncation_under_cap(self):
        ents = [_entity("A", [0]), _entity("B", [10])]
        assert _truncate_to_cap(ents, cap=6) == ents

    def test_no_truncation_at_cap(self):
        ents = [_entity(f"E{i}", [i * 10]) for i in range(6)]
        assert _truncate_to_cap(ents, cap=6) == ents

    def test_truncate_keeps_most_mentioned(self):
        # 7 entities; e3 has 5 mentions, e5 has 4, others have 1-2.
        # Keep top 6 by mention count.
        ents = [
            _entity("E0", [0]),                     # 1 mention
            _entity("E1", [10, 50]),                # 2 mentions
            _entity("E2", [20]),                    # 1 mention
            _entity("E3", [30, 40, 60, 70, 90]),    # 5 mentions
            _entity("E4", [45]),                    # 1 mention
            _entity("E5", [55, 65, 75, 80]),        # 4 mentions
            _entity("E6", [85]),                    # 1 mention
        ]
        kept = _truncate_to_cap(ents, cap=6)
        assert len(kept) == 6
        names = [e.canonical_name for e in kept]
        # E3 (5 mentions) and E5 (4 mentions) must survive
        assert "E3" in names
        assert "E5" in names
        # One of the 1-mention entities must be dropped. Tiebreak:
        # latest first_mention_word_idx loses. E6 (idx=85) loses.
        assert "E6" not in names

    def test_tiebreak_earliest_mention_wins(self):
        # All 7 entities have exactly 1 mention; tiebreak on
        # first_mention_word_idx ASC.
        ents = [_entity(f"E{i}", [i * 10]) for i in range(7)]
        kept = _truncate_to_cap(ents, cap=6)
        # E0..E5 (first_mention_word_idx = 0..50) survive; E6 (60) drops.
        assert [e.canonical_name for e in kept] == [
            "E0", "E1", "E2", "E3", "E4", "E5",
        ]

    def test_truncation_logs_warning(self, caplog):
        ents = [_entity(f"E{i}", [i]) for i in range(7)]
        with caplog.at_level(logging.WARNING, logger="pipeline_v2.stage_2_5"):
            _truncate_to_cap(ents, cap=6)
        warnings = [
            r for r in caplog.records
            if "truncating" in r.message.lower()
            or "emitted" in r.message.lower()
        ]
        assert warnings, "truncation should emit a warning"
        # The warning should name the dropped entity
        warning_text = warnings[0].message
        assert "E6" in str(warnings[0].args) or "E6" in warning_text

    def test_custom_cap(self):
        # Custom cap (e.g. for tests or future tuning).
        ents = [_entity(f"E{i}", [i * 10]) for i in range(5)]
        kept = _truncate_to_cap(ents, cap=3)
        assert len(kept) == 3

    def test_returns_new_list_not_mutates_input(self):
        ents = [_entity(f"E{i}", [i * 10]) for i in range(8)]
        original_first = ents[0]
        kept = _truncate_to_cap(ents, cap=6)
        # Input list unchanged
        assert len(ents) == 8
        assert ents[0] is original_first
        # Kept list is a NEW list (not a slice of input -- it's sorted)
        assert kept is not ents


# ====================================================================== #
# Stage2_5EntityCanonicalizer constructor + lazy resources                #
# ====================================================================== #


class TestConstructor:
    def test_defaults(self):
        c = Stage2_5EntityCanonicalizer()
        assert c.model == "gemini-2.5-flash"
        assert c.temperature == 0.2
        assert c.thinking_budget == 512
        assert c.max_output_tokens == 4096
        assert c.entity_cap == 6

    def test_kwargs_override(self):
        c = Stage2_5EntityCanonicalizer(
            model="gemini-2.5-flash-lite",
            temperature=0.0,
            thinking_budget=0,
            max_output_tokens=2048,
            entity_cap=4,
        )
        assert c.model == "gemini-2.5-flash-lite"
        assert c.temperature == 0.0
        assert c.thinking_budget == 0
        assert c.max_output_tokens == 2048
        assert c.entity_cap == 4

    def test_init_is_side_effect_free(self, monkeypatch):
        # No env-var read, no file read, no SDK construction at __init__
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        c = Stage2_5EntityCanonicalizer(prompt_path=Path("/nope/missing.md"))
        assert c._client is None
        assert c._prompt_template is None

    def test_missing_api_key_raises_on_first_use(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        c = Stage2_5EntityCanonicalizer()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            c._get_client()

    def test_missing_prompt_raises_on_first_use(self):
        c = Stage2_5EntityCanonicalizer(prompt_path=Path("/nope/missing.md"))
        with pytest.raises(FileNotFoundError, match="Stage 2.5 prompt"):
            c._load_prompt()

    def test_prompt_cached_after_first_load(self, tmp_path):
        prompt_path = _writable_prompt(tmp_path, "BODY-XYZ")
        c = Stage2_5EntityCanonicalizer(prompt_path=prompt_path)
        first = c._load_prompt()
        # Delete the file -- second call should still succeed (cached)
        prompt_path.unlink()
        second = c._load_prompt()
        assert first == second == "BODY-XYZ"


# ====================================================================== #
# Prompt construction                                                     #
# ====================================================================== #


class TestPromptConstruction:
    def test_prompt_contains_metadata(self, tmp_path):
        prompt_path = _writable_prompt(tmp_path, "TEMPLATE BODY")
        c = Stage2_5EntityCanonicalizer(prompt_path=prompt_path)
        clean = _clean(20)
        prompt = c._build_prompt(clean)
        assert "TEMPLATE BODY" in prompt
        assert "Word count:      20" in prompt
        assert "Clip count:      1" in prompt
        assert "Entity cap:      6" in prompt
        # Word array surfaces
        assert '"idx": 0' in prompt
        assert '"w": "w0"' in prompt

    def test_correction_note_appended_when_provided(self, tmp_path):
        prompt_path = _writable_prompt(tmp_path)
        c = Stage2_5EntityCanonicalizer(prompt_path=prompt_path)
        clean = _clean(5)
        prompt = c._build_prompt(clean, correction_note="FIX X")
        assert "## Correction note" in prompt
        assert "FIX X" in prompt

    def test_no_correction_section_by_default(self, tmp_path):
        prompt_path = _writable_prompt(tmp_path)
        c = Stage2_5EntityCanonicalizer(prompt_path=prompt_path)
        prompt = c._build_prompt(_clean(5))
        assert "## Correction note" not in prompt


# ====================================================================== #
# _parse_response: manual validate path                                   #
# ====================================================================== #


class TestParseResponse:
    """response.parsed would SILENTLY swallow ValidationError /
    JSONDecodeError and return None. The manual model_validate path
    surfaces those errors so the corrective-retry layer can catch
    them. These tests pin the manual path against future "cleanup".
    """

    def test_happy_path(self):
        c = Stage2_5EntityCanonicalizer()
        body = {
            "entities": [{
                "canonical_name": "Revanth Reddy",
                "native_name": "రేవంత్ రెడ్డి",
                "first_mention_word_idx": 5,
                "type": "PERSON",
                "mentions": [5, 12, 30],
            }],
        }
        resp = _mock_response(json.dumps(body))
        out = c._parse_response(resp)
        assert isinstance(out, Stage2_5Output)
        assert len(out.entities) == 1
        assert out.entities[0].type == EntityType.PERSON

    def test_strips_fences(self):
        c = Stage2_5EntityCanonicalizer()
        body = {"entities": []}
        resp = _mock_response(f"```json\n{json.dumps(body)}\n```")
        out = c._parse_response(resp)
        assert out.entities == []

    def test_empty_text_raises_json_decode_error(self):
        c = Stage2_5EntityCanonicalizer()
        resp = _mock_response("")
        with pytest.raises(json.JSONDecodeError):
            c._parse_response(resp)

    def test_invalid_json_raises_json_decode_error(self):
        c = Stage2_5EntityCanonicalizer()
        resp = _mock_response("this is not json")
        with pytest.raises(json.JSONDecodeError):
            c._parse_response(resp)

    def test_invented_type_caught_by_manual_validate(self):
        # CRITICAL REGRESSION TEST: response.parsed would SILENTLY
        # swallow this ValidationError; manual model_validate raises.
        c = Stage2_5EntityCanonicalizer()
        body = {
            "entities": [{
                "canonical_name": "X", "native_name": "X",
                "first_mention_word_idx": 0,
                "type": "LOCATION",       # invented (real is PLACE)
                "mentions": [0],
            }],
        }
        resp = _mock_response(json.dumps(body))
        with pytest.raises(ValidationError):
            c._parse_response(resp)


# ====================================================================== #
# _finalize: post-validate truncation gate                                #
# ====================================================================== #


class TestFinalize:
    def test_passes_through_under_cap(self):
        c = Stage2_5EntityCanonicalizer()
        out = _stage_2_5_output([_entity("A", [0]), _entity("B", [5])])
        result = c._finalize(out)
        assert result is out  # same object; no truncation needed

    def test_truncates_over_cap_via_custom_cap(self):
        # With cap=3, build a Stage2_5Output with 5 entities by
        # bypassing the Pydantic max_length (which only enforces
        # cap=6 globally). We construct via direct field-set: the
        # Pydantic validator allows up to 6 entities, and we want to
        # test what happens when our class-level cap is LOWER.
        c = Stage2_5EntityCanonicalizer(entity_cap=3)
        ents = [
            _entity("E0", [0]),
            _entity("E1", [10, 20, 30]),
            _entity("E2", [40]),
            _entity("E3", [50, 60]),
            _entity("E4", [70]),
        ]
        out = _stage_2_5_output(ents)
        result = c._finalize(out)
        assert len(result.entities) == 3
        # Top 3 by mention count: E1 (3), E3 (2), then tiebreak among
        # E0/E2/E4 (all 1) -> earliest first_mention_word_idx wins
        # -> E0 (idx 0).
        names = [e.canonical_name for e in result.entities]
        assert names[0] == "E1"
        assert "E3" in names
        assert "E0" in names


# ====================================================================== #
# async classify: end-to-end with mocked Gemini                           #
# ====================================================================== #


@pytest.fixture
def good_response_body():
    return {
        "entities": [{
            "canonical_name": "Revanth Reddy",
            "native_name": "రేవంత్ రెడ్డి",
            "first_mention_word_idx": 5,
            "type": "PERSON",
            "mentions": [5, 12, 25],
        }],
    }


@pytest.fixture
def canonicalizer(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    prompt_path = _writable_prompt(tmp_path, "TEMPLATE")
    return Stage2_5EntityCanonicalizer(prompt_path=prompt_path)


class TestClassifyHappyPath:
    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(
        self, canonicalizer, good_response_body,
    ):
        clean = _clean(30)
        mock_call = AsyncMock(
            return_value=_mock_response(json.dumps(good_response_body)),
        )
        with patch.object(canonicalizer, "_call_gemini", new=mock_call):
            out = await canonicalizer.classify(clean)
        assert len(out.entities) == 1
        assert out.entities[0].canonical_name == "Revanth Reddy"
        # _call_gemini called exactly once
        assert mock_call.await_count == 1


class TestClassifyCorrectiveRetry:
    @pytest.mark.asyncio
    async def test_validation_error_triggers_retry(
        self, canonicalizer, good_response_body,
    ):
        # First response: invented type -> ValidationError
        bad_body = {
            "entities": [{
                "canonical_name": "X", "native_name": "X",
                "first_mention_word_idx": 0,
                "type": "LOCATION",
                "mentions": [0],
            }],
        }
        clean = _clean(30)
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(bad_body)),
            _mock_response(json.dumps(good_response_body)),
        ])
        with patch.object(canonicalizer, "_call_gemini", new=mock_call):
            out = await canonicalizer.classify(clean)
        assert mock_call.await_count == 2
        assert out.entities[0].canonical_name == "Revanth Reddy"

    @pytest.mark.asyncio
    async def test_json_decode_error_triggers_retry(
        self, canonicalizer, good_response_body,
    ):
        clean = _clean(10)
        mock_call = AsyncMock(side_effect=[
            _mock_response("not valid json"),
            _mock_response(json.dumps(good_response_body)),
        ])
        with patch.object(canonicalizer, "_call_gemini", new=mock_call):
            out = await canonicalizer.classify(clean)
        assert mock_call.await_count == 2
        assert len(out.entities) == 1

    @pytest.mark.asyncio
    async def test_double_failure_raises_runtime_error(self, canonicalizer):
        clean = _clean(10)
        mock_call = AsyncMock(side_effect=[
            _mock_response("not valid json"),
            _mock_response("still not valid json"),
        ])
        with patch.object(canonicalizer, "_call_gemini", new=mock_call):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await canonicalizer.classify(clean)
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_other_exceptions_propagate(self, canonicalizer):
        clean = _clean(10)
        mock_call = AsyncMock(side_effect=ConnectionError("network down"))
        with patch.object(canonicalizer, "_call_gemini", new=mock_call):
            with pytest.raises(ConnectionError, match="network down"):
                await canonicalizer.classify(clean)
        # No retry on non-validation errors
        assert mock_call.await_count == 1


class TestClassifyTruncationIntegration:
    """Confirm the post-validate cap fires through the full classify()
    path -- not just the unit-tested _finalize() / _truncate_to_cap()
    helpers.
    """

    @pytest.mark.asyncio
    async def test_response_at_cap_passes_through(self, canonicalizer):
        # 6 entities (the Pydantic max) -> no truncation needed.
        body = {
            "entities": [{
                "canonical_name": f"E{i}",
                "native_name": f"E{i}",
                "first_mention_word_idx": i * 10,
                "type": "PERSON",
                "mentions": [i * 10],
            } for i in range(6)],
        }
        clean = _clean(60)
        with patch.object(
            canonicalizer, "_call_gemini",
            new=AsyncMock(return_value=_mock_response(json.dumps(body))),
        ):
            out = await canonicalizer.classify(clean)
        assert len(out.entities) == 6
