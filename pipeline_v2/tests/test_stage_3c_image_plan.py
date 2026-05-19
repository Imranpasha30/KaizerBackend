"""Unit tests for Stage 3c (Image Plan Generator) -- Step 7.4.

Two test surfaces:

  1. Standard SDK-pattern tests (constructor, lazy resources, parse,
     corrective retry, exception passthrough). Same shape as 3a / 3b.

  2. **CRITICAL: post-validate filter tests.** This is the heart of
     Stage 3c (D-7.10). Test all three invariants + the >50% drop
     hard-fail guardrail. These are the tests that fire when Gemini
     emits an image_plan with orphan entity names, invalid clip
     indices, or boundary violations -- exactly the failure modes
     V1 exhibited (Instagram Reel "?" id bug etc.).
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
    FullVideoCut,
    ImagePlan,
    ImagePlanEntry,
    Word,
)
from pipeline_v2.stages.stage_3c_image_plan import (
    Stage3cImagePlanner,
    _strip_markdown_fences,
    _validate_and_filter,
)


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _clean(n: int = 60) -> CleanTranscript:
    words = [Word(w=f"w{i}", s=i * 1.0, e=i * 1.0 + 0.8) for i in range(n)]
    return CleanTranscript(
        words=words,
        clip_boundaries={0: (0, n - 1)},
        source_word_map=list(range(n)),
    )


def _entity(name: str, type_: str = "PERSON") -> Entity:
    return Entity(
        canonical_name=name,
        native_name=name,
        first_mention_word_idx=0,
        type=type_,
        mentions=[0],
    )


def _cut(index: int, start_sec: float, end_sec: float,
         importance: int = 5) -> FullVideoCut:
    return FullVideoCut(
        index=index,
        start_word_idx=int(start_sec),
        end_word_idx=int(end_sec),
        start_sec=start_sec,
        end_sec=end_sec,
        importance=importance,
    )


def _entry(entity_name: str = "X", clip_index: int = 0,
           show_at: float = 5.0, duration: float = 3.0,
           description: str = "test") -> ImagePlanEntry:
    return ImagePlanEntry(
        entity_name=entity_name,
        entity_name_native=entity_name,
        description=description,
        clip_index=clip_index,
        show_at_sec=show_at,
        duration_sec=duration,
    )


def _good_response_body() -> dict:
    return {
        "entries": [
            {
                "entity_name": "Revanth Reddy",
                "entity_name_native": "రేవంత్ రెడ్డి",
                "description": "Revanth Reddy at press meet",
                "clip_index": 0,
                "show_at_sec": 5.0,
                "duration_sec": 4.0,
            }
        ],
    }


def _mock_response(json_text: str) -> MagicMock:
    r = MagicMock()
    r.text = json_text
    return r


def _writable_prompt(tmp_path: Path, body: str = "STAGE 3c STUB") -> Path:
    p = tmp_path / "stage_3c_prompt.md"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def planner(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    prompt_path = _writable_prompt(tmp_path, "TEMPLATE")
    return Stage3cImagePlanner(prompt_path=prompt_path)


# ====================================================================== #
# _validate_and_filter -- D-7.10 HEART OF STAGE 3c                        #
# ====================================================================== #


class TestValidateAndFilterInvariant1OrphanEntity:
    """Invariant 1: entity_name MUST match a canonical_name. Drop
    orphans with warning.
    """

    def test_valid_entity_kept(self):
        plan = ImagePlan(entries=[
            _entry(entity_name="Modi", clip_index=0, show_at=5, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Modi")]
        out = _validate_and_filter(plan, cuts, entities)
        assert len(out.entries) == 1

    def test_orphan_entity_dropped(self, caplog):
        # drop_ratio_threshold=1.1 -- isolate the invariant from the
        # guardrail. The guardrail is tested separately in
        # TestValidateAndFilterDropRatioGuardrail.
        plan = ImagePlan(entries=[
            _entry(entity_name="UnknownPerson", clip_index=0,
                   show_at=5, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Modi")]
        with caplog.at_level(logging.WARNING, logger="pipeline_v2.stage_3c"):
            out = _validate_and_filter(plan, cuts, entities,
                                       drop_ratio_threshold=1.1)
        assert len(out.entries) == 0
        assert any("orphan" in r.message.lower()
                   or "not in canonical" in r.message.lower()
                   for r in caplog.records)

    def test_case_sensitive_match(self):
        # "modi" != "Modi" -- case-sensitive
        plan = ImagePlan(entries=[
            _entry(entity_name="modi", clip_index=0, show_at=5, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Modi")]
        out = _validate_and_filter(plan, cuts, entities,
                                   drop_ratio_threshold=1.1)
        assert len(out.entries) == 0


class TestValidateAndFilterInvariant2InvalidClipIndex:
    """Invariant 2: clip_index MUST be one of the cut indices."""

    def test_valid_clip_index_kept(self):
        plan = ImagePlan(entries=[
            _entry(entity_name="X", clip_index=2, show_at=55, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0), _cut(1, 30.0, 50.0),
                _cut(2, 50.0, 100.0)]
        entities = [_entity("X")]
        out = _validate_and_filter(plan, cuts, entities)
        assert len(out.entries) == 1
        assert out.entries[0].clip_index == 2

    def test_invalid_clip_index_dropped(self, caplog):
        plan = ImagePlan(entries=[
            _entry(entity_name="X", clip_index=99, show_at=5, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X")]
        with caplog.at_level(logging.WARNING, logger="pipeline_v2.stage_3c"):
            out = _validate_and_filter(plan, cuts, entities,
                                       drop_ratio_threshold=1.1)
        assert len(out.entries) == 0
        assert any("clip_index" in r.message for r in caplog.records)


class TestValidateAndFilterInvariant3Boundary:
    """Invariant 3: [show_at, show_at + duration] inside clip range."""

    def test_window_inside_clip_kept(self):
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=10.0, duration=5.0),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X")]
        out = _validate_and_filter(plan, cuts, entities)
        assert len(out.entries) == 1

    def test_window_at_exact_clip_boundaries_kept(self):
        # Edge case: window exactly equals clip range
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=0.0, duration=30.0),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X")]
        out = _validate_and_filter(plan, cuts, entities)
        assert len(out.entries) == 1

    def test_window_start_before_clip_start_dropped(self, caplog):
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=5.0, duration=3.0),
        ])
        cuts = [_cut(0, 10.0, 30.0)]   # clip starts at 10, overlay at 5
        entities = [_entity("X")]
        with caplog.at_level(logging.WARNING, logger="pipeline_v2.stage_3c"):
            out = _validate_and_filter(plan, cuts, entities,
                                       drop_ratio_threshold=1.1)
        assert len(out.entries) == 0
        assert any("boundary" in r.message.lower() for r in caplog.records)

    def test_window_end_past_clip_end_dropped(self, caplog):
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=28.0, duration=5.0),
        ])
        cuts = [_cut(0, 0.0, 30.0)]   # window ends at 33, clip ends at 30
        entities = [_entity("X")]
        with caplog.at_level(logging.WARNING, logger="pipeline_v2.stage_3c"):
            out = _validate_and_filter(plan, cuts, entities,
                                       drop_ratio_threshold=1.1)
        assert len(out.entries) == 0
        assert any("boundary" in r.message.lower() for r in caplog.records)

    def test_window_slightly_past_end_dropped(self):
        # 0.1s overflow still drops
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=27.0, duration=3.1),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X")]
        out = _validate_and_filter(plan, cuts, entities,
                                   drop_ratio_threshold=1.1)
        assert len(out.entries) == 0


class TestValidateAndFilterDropRatioGuardrail:
    """D-7.10: >50% dropped → hard fail (RuntimeError)."""

    def test_zero_dropped_no_raise(self):
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=5, duration=3),
            _entry("Y", clip_index=0, show_at=10, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X"), _entity("Y")]
        out = _validate_and_filter(plan, cuts, entities)
        assert len(out.entries) == 2

    def test_below_threshold_no_raise(self):
        # 4 entries, 1 dropped (25%) -- under 50% threshold
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=5, duration=3),
            _entry("Y", clip_index=0, show_at=10, duration=3),
            _entry("Z", clip_index=0, show_at=15, duration=3),
            _entry("Unknown", clip_index=0, show_at=20, duration=3),  # orphan
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X"), _entity("Y"), _entity("Z")]
        out = _validate_and_filter(plan, cuts, entities)
        assert len(out.entries) == 3

    def test_exactly_50_percent_dropped_no_raise(self):
        # 4 entries, 2 dropped (50%) -- AT threshold, NOT over.
        # Implementation: > 0.5, so 0.5 itself does NOT raise.
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=5, duration=3),
            _entry("Y", clip_index=0, show_at=10, duration=3),
            _entry("Bad1", clip_index=0, show_at=15, duration=3),    # orphan
            _entry("Bad2", clip_index=0, show_at=20, duration=3),    # orphan
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X"), _entity("Y")]
        out = _validate_and_filter(plan, cuts, entities)
        assert len(out.entries) == 2

    def test_over_50_percent_dropped_raises(self):
        # 4 entries, 3 dropped (75%) -- ABOVE threshold
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=5, duration=3),
            _entry("Bad1", clip_index=0, show_at=10, duration=3),
            _entry("Bad2", clip_index=0, show_at=15, duration=3),
            _entry("Bad3", clip_index=0, show_at=20, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X")]
        with pytest.raises(RuntimeError, match=r"75%"):
            _validate_and_filter(plan, cuts, entities)

    def test_custom_threshold(self):
        # Lower threshold: 25% (drop 1 of 4)
        plan = ImagePlan(entries=[
            _entry("X", clip_index=0, show_at=5, duration=3),
            _entry("Y", clip_index=0, show_at=10, duration=3),
            _entry("Z", clip_index=0, show_at=15, duration=3),
            _entry("Bad", clip_index=0, show_at=20, duration=3),  # orphan
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X"), _entity("Y"), _entity("Z")]
        with pytest.raises(RuntimeError):
            _validate_and_filter(plan, cuts, entities,
                                 drop_ratio_threshold=0.2)

    def test_empty_plan_no_raise(self):
        # 0 entries / 0 total = no division-by-zero
        plan = ImagePlan(entries=[])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X")]
        out = _validate_and_filter(plan, cuts, entities)
        assert out.entries == []

    def test_all_dropped_raises(self):
        # 2 entries, both bad
        plan = ImagePlan(entries=[
            _entry("Bad1", clip_index=0, show_at=5, duration=3),
            _entry("Bad2", clip_index=0, show_at=10, duration=3),
        ])
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("X")]
        with pytest.raises(RuntimeError, match=r"100%"):
            _validate_and_filter(plan, cuts, entities)


# ====================================================================== #
# _strip_markdown_fences                                                  #
# ====================================================================== #


class TestStripMarkdownFences:
    def test_passthrough(self):
        assert _strip_markdown_fences('{"x": 1}') == '{"x": 1}'

    def test_strip_json_fence(self):
        assert _strip_markdown_fences('```json\n{"x": 1}\n```') == '{"x": 1}'


# ====================================================================== #
# Constructor                                                             #
# ====================================================================== #


class TestConstructor:
    def test_defaults(self):
        p = Stage3cImagePlanner()
        assert p.model == "gemini-2.5-flash"
        assert p.temperature == 0.2
        assert p.thinking_budget == 0
        assert p.max_output_tokens == 4096
        assert p.drop_ratio_threshold == 0.5

    def test_kwargs_override(self):
        p = Stage3cImagePlanner(
            temperature=0.0, drop_ratio_threshold=0.3,
        )
        assert p.temperature == 0.0
        assert p.drop_ratio_threshold == 0.3

    def test_init_is_side_effect_free(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        p = Stage3cImagePlanner(prompt_path=Path("/nope.md"))
        assert p._client is None
        assert p._prompt_template is None

    def test_missing_api_key_raises_on_first_use(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        p = Stage3cImagePlanner()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            p._get_client()

    def test_missing_prompt_raises_on_first_use(self):
        p = Stage3cImagePlanner(prompt_path=Path("/nope.md"))
        with pytest.raises(FileNotFoundError, match="Stage 3c prompt"):
            p._load_prompt()


# ====================================================================== #
# Prompt construction                                                     #
# ====================================================================== #


class TestPromptConstruction:
    def test_includes_all_three_input_sections(self, planner):
        clean = _clean(20)
        cuts = [_cut(0, 0.0, 15.0), _cut(1, 15.0, 25.0)]
        entities = [_entity("Modi", "PERSON")]
        prompt = planner._build_prompt(clean, cuts, entities)
        assert "Canonical entities" in prompt
        assert "Full video cuts" in prompt
        assert "Clean transcript word array" in prompt
        assert "Modi" in prompt
        assert '"index": 0' in prompt

    def test_correction_note_appended(self, planner):
        prompt = planner._build_prompt(_clean(5), [_cut(0, 0, 4)], [_entity("X")],
                                       correction_note="FIX X")
        assert "## Correction note" in prompt
        assert "FIX X" in prompt


# ====================================================================== #
# _parse_response: manual validate path                                   #
# ====================================================================== #


class TestParseResponse:
    def test_happy_path(self, planner):
        resp = _mock_response(json.dumps(_good_response_body()))
        out = planner._parse_response(resp)
        assert isinstance(out, ImagePlan)
        assert len(out.entries) == 1

    def test_strips_fences(self, planner):
        body = _good_response_body()
        resp = _mock_response(f"```json\n{json.dumps(body)}\n```")
        out = planner._parse_response(resp)
        assert len(out.entries) == 1

    def test_empty_text_raises(self, planner):
        with pytest.raises(json.JSONDecodeError):
            planner._parse_response(_mock_response(""))

    def test_duration_under_2_caught(self, planner):
        # ImagePlanEntry.duration_sec has Field(ge=2.0)
        body = {
            "entries": [{
                "entity_name": "X", "entity_name_native": "X",
                "description": "d", "clip_index": 0,
                "show_at_sec": 5.0, "duration_sec": 1.5,  # < 2.0
            }],
        }
        with pytest.raises(ValidationError):
            planner._parse_response(_mock_response(json.dumps(body)))


# ====================================================================== #
# async plan: end-to-end                                                  #
# ====================================================================== #


class TestPlanEndToEnd:
    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(self, planner):
        clean = _clean(30)
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Revanth Reddy")]
        mock_call = AsyncMock(
            return_value=_mock_response(json.dumps(_good_response_body())),
        )
        with patch.object(planner, "_call_gemini", new=mock_call):
            out = await planner.plan(clean, cuts, entities)
        assert len(out.entries) == 1
        assert mock_call.await_count == 1

    @pytest.mark.asyncio
    async def test_orphan_entries_filtered_during_plan(self, planner):
        # Body has 4 entries: 2 valid, 2 with orphan entity_name.
        # 50% dropped (at threshold, not over) -> no raise; 2 kept.
        clean = _clean(30)
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Modi"), _entity("Reddy")]
        body = {
            "entries": [
                {"entity_name": "Modi", "entity_name_native": "Modi",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 5.0, "duration_sec": 3.0},
                {"entity_name": "Reddy", "entity_name_native": "Reddy",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 10.0, "duration_sec": 3.0},
                {"entity_name": "Unknown1", "entity_name_native": "U",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 15.0, "duration_sec": 3.0},
                {"entity_name": "Unknown2", "entity_name_native": "U",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 20.0, "duration_sec": 3.0},
            ],
        }
        mock_call = AsyncMock(return_value=_mock_response(json.dumps(body)))
        with patch.object(planner, "_call_gemini", new=mock_call):
            out = await planner.plan(clean, cuts, entities)
        assert len(out.entries) == 2
        assert {e.entity_name for e in out.entries} == {"Modi", "Reddy"}

    @pytest.mark.asyncio
    async def test_excess_drops_raise_runtime_error(self, planner):
        # 4 entries, 3 orphans -> 75% drop -> RuntimeError from filter
        clean = _clean(30)
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Modi")]
        body = {
            "entries": [
                {"entity_name": "Modi", "entity_name_native": "Modi",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 5.0, "duration_sec": 3.0},
                {"entity_name": "X", "entity_name_native": "X",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 10.0, "duration_sec": 3.0},
                {"entity_name": "Y", "entity_name_native": "Y",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 15.0, "duration_sec": 3.0},
                {"entity_name": "Z", "entity_name_native": "Z",
                 "description": "d", "clip_index": 0,
                 "show_at_sec": 20.0, "duration_sec": 3.0},
            ],
        }
        mock_call = AsyncMock(return_value=_mock_response(json.dumps(body)))
        with patch.object(planner, "_call_gemini", new=mock_call):
            with pytest.raises(RuntimeError, match=r"75%"):
                await planner.plan(clean, cuts, entities)

    @pytest.mark.asyncio
    async def test_validation_error_triggers_retry(self, planner):
        clean = _clean(30)
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Modi")]
        bad = {
            "entries": [{
                "entity_name": "Modi", "entity_name_native": "Modi",
                "description": "d", "clip_index": 0,
                "show_at_sec": 5.0, "duration_sec": 1.5,  # under 2.0
            }],
        }
        good = _good_response_body()
        good["entries"][0]["entity_name"] = "Modi"
        good["entries"][0]["entity_name_native"] = "Modi"
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(bad)),
            _mock_response(json.dumps(good)),
        ])
        with patch.object(planner, "_call_gemini", new=mock_call):
            out = await planner.plan(clean, cuts, entities)
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_double_failure_raises(self, planner):
        clean = _clean(30)
        cuts = [_cut(0, 0.0, 30.0)]
        entities = [_entity("Modi")]
        mock_call = AsyncMock(side_effect=[
            _mock_response("not json"),
            _mock_response("still not json"),
        ])
        with patch.object(planner, "_call_gemini", new=mock_call):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await planner.plan(clean, cuts, entities)
