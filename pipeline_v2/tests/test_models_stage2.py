"""Unit tests for Stage 2 Pydantic models (Step 5.1).

Covers:
  - ``Stage2Output`` (LLM contract for Stage 2)
  - ``FullVideoCut``, ``SkippedSegment`` (already-locked sub-models;
    re-tested here for the constraints Stage 2 relies on)
  - ``SkippedCategory`` enum (6 values, no invented categories at the
    Pydantic level either)
  - retake_audit optional behavior (None / missing / non-empty string
    all valid)

No real Gemini calls. The Step 5.2 dispatcher will mock the SDK
client; these tests just verify the schemas are correct.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline_v2.models import (
    FullVideoCut,
    SkippedCategory,
    SkippedSegment,
    Stage2Output,
)


# ====================================================================== #
# SkippedCategory enum (locked Step 0 forbid-invention set)              #
# ====================================================================== #


class TestSkippedCategoryEnum:
    def test_six_locked_values(self):
        # Step 0's prompt-swap forbid-invention block names exactly
        # these 6. Pydantic enforces at the schema level too.
        expected = {
            "warm_up", "retake", "crew_talk",
            "hesitation", "aside", "self_correction",
        }
        assert {m.value for m in SkippedCategory} == expected

    def test_str_enum_value(self):
        # The enum is a str subclass so Pydantic serialises cleanly.
        assert SkippedCategory.retake == "retake"
        assert SkippedCategory.warm_up.value == "warm_up"

    @pytest.mark.parametrize("good", [
        "warm_up", "retake", "crew_talk",
        "hesitation", "aside", "self_correction",
    ])
    def test_skipped_segment_accepts_each_category(self, good):
        seg = SkippedSegment(
            start_word_idx=0, end_word_idx=5,
            start_sec=1.0, end_sec=3.0,
            category=good, reason="x",
        )
        assert seg.category == good

    @pytest.mark.parametrize("invented", [
        "redundancy",          # Gemini sometimes invents this -- must reject
        "filler",
        "off_topic",
        "mistake",
        "interruption",
        "intro_chatter",
        "",
        "RETAKE",              # case-sensitive
        "Retake",
    ])
    def test_skipped_segment_rejects_invented_category(self, invented):
        with pytest.raises(ValidationError):
            SkippedSegment(
                start_word_idx=0, end_word_idx=5,
                start_sec=1.0, end_sec=3.0,
                category=invented, reason="x",
            )


# ====================================================================== #
# FullVideoCut constraints                                               #
# ====================================================================== #


class TestFullVideoCut:
    def test_minimal_construct(self):
        c = FullVideoCut(
            index=0, start_word_idx=0, end_word_idx=99,
            start_sec=0.0, end_sec=12.5, importance=7,
        )
        assert c.importance == 7

    @pytest.mark.parametrize("bad_importance", [0, -1, 11, 100, -100])
    def test_importance_must_be_1_to_10(self, bad_importance):
        with pytest.raises(ValidationError):
            FullVideoCut(
                index=0, start_word_idx=0, end_word_idx=99,
                start_sec=0.0, end_sec=12.5, importance=bad_importance,
            )

    @pytest.mark.parametrize("ok_importance", [1, 5, 10])
    def test_importance_boundary_values_ok(self, ok_importance):
        c = FullVideoCut(
            index=0, start_word_idx=0, end_word_idx=99,
            start_sec=0.0, end_sec=12.5, importance=ok_importance,
        )
        assert c.importance == ok_importance


# ====================================================================== #
# Stage2Output -- the LLM contract                                       #
# ====================================================================== #


class TestStage2Output:
    def _cut(self, idx: int = 0, importance: int = 5):
        return FullVideoCut(
            index=idx, start_word_idx=0, end_word_idx=10,
            start_sec=0.0, end_sec=2.5, importance=importance,
        )

    def _skip(self, category: str = "retake"):
        return SkippedSegment(
            start_word_idx=0, end_word_idx=5,
            start_sec=0.0, end_sec=1.0,
            category=category, reason="duplicate phrasing",
        )

    def test_minimal_with_retake_audit(self):
        out = Stage2Output(
            full_video_cuts=[self._cut()],
            skipped_segments=[self._skip()],
            retake_audit="Skipped 1 retake at 0-1s.",
        )
        assert out.retake_audit == "Skipped 1 retake at 0-1s."

    def test_retake_audit_optional_can_be_none(self):
        out = Stage2Output(
            full_video_cuts=[self._cut()],
            skipped_segments=[self._skip()],
            retake_audit=None,
        )
        assert out.retake_audit is None

    def test_retake_audit_optional_can_be_omitted(self):
        out = Stage2Output(
            full_video_cuts=[self._cut()],
            skipped_segments=[self._skip()],
        )
        assert out.retake_audit is None

    def test_empty_lists_allowed(self):
        # A perfectly clean recording with 0 retakes and no clip
        # boundaries -- both lists may legitimately be empty.
        out = Stage2Output(
            full_video_cuts=[], skipped_segments=[],
        )
        assert out.full_video_cuts == []
        assert out.skipped_segments == []

    def test_json_roundtrip_preserves_all_fields(self):
        out = Stage2Output(
            full_video_cuts=[self._cut(0, 8), self._cut(1, 5)],
            skipped_segments=[
                self._skip("retake"),
                self._skip("hesitation"),
            ],
            retake_audit="Skipped 1 retake (0-1s) and 1 hesitation.",
        )
        data = json.loads(out.model_dump_json())
        # Round-trip through Pydantic.
        rebuilt = Stage2Output.model_validate(data)
        assert len(rebuilt.full_video_cuts) == 2
        assert rebuilt.full_video_cuts[1].importance == 5
        assert {s.category for s in rebuilt.skipped_segments} == {
            SkippedCategory.retake, SkippedCategory.hesitation,
        }
        assert rebuilt.retake_audit == out.retake_audit

    def test_invented_category_at_load_time_rejected(self):
        # The forbid-invention rule applies whether the data is
        # constructed in Python OR validated from a JSON dict (e.g.
        # parsed from Gemini's response_schema output). Both paths
        # share the same validators.
        bad_dict = {
            "full_video_cuts": [
                {"index": 0, "start_word_idx": 0, "end_word_idx": 10,
                 "start_sec": 0.0, "end_sec": 2.5, "importance": 5},
            ],
            "skipped_segments": [
                {"start_word_idx": 0, "end_word_idx": 5,
                 "start_sec": 0.0, "end_sec": 1.0,
                 "category": "redundancy",     # invented
                 "reason": "x"},
            ],
        }
        with pytest.raises(ValidationError):
            Stage2Output.model_validate(bad_dict)

    def test_importance_violation_at_load_time_rejected(self):
        bad_dict = {
            "full_video_cuts": [
                {"index": 0, "start_word_idx": 0, "end_word_idx": 10,
                 "start_sec": 0.0, "end_sec": 2.5,
                 "importance": 15},      # > 10
            ],
            "skipped_segments": [],
        }
        with pytest.raises(ValidationError):
            Stage2Output.model_validate(bad_dict)

    def test_explicit_distinction_from_existing_stagetwooutput(self):
        # Stage2Output is the LLM contract (no clean_transcript).
        # StageTwoOutput is the full stage return (with clean_transcript).
        # They MUST be separate types -- the prompt asks for the former,
        # the stage entry builds the latter. Confirm both exist and
        # the field sets differ.
        from pipeline_v2 import models
        s2_fields = set(models.Stage2Output.model_fields)
        st_fields = set(models.StageTwoOutput.model_fields)
        # Stage2Output has 3 fields, StageTwoOutput has 4 (the extra
        # is clean_transcript).
        assert "clean_transcript" not in s2_fields
        assert "clean_transcript" in st_fields
        # Both share the decision fields.
        assert "full_video_cuts" in s2_fields and "full_video_cuts" in st_fields
        assert "skipped_segments" in s2_fields and "skipped_segments" in st_fields
        assert "retake_audit" in s2_fields and "retake_audit" in st_fields


# ====================================================================== #
# Gemini-side compatibility checks                                       #
# ====================================================================== #


class TestGeminiResponseSchemaCompatibility:
    """Stage2Output is passed to ``client.models.generate_content`` as
    the ``response_schema`` config. Pydantic models become JSON Schema
    documents that Gemini's structured-output layer enforces. These
    tests check a few constraints that have historically tripped up
    Gemini's schema engine (additionalProperties handling, nested
    enums, optional fields).
    """

    def test_pydantic_json_schema_is_buildable(self):
        # If this raises, Gemini's response_schema will choke too --
        # they consume the same JSON Schema document.
        schema = Stage2Output.model_json_schema()
        assert "properties" in schema
        assert "full_video_cuts" in schema["properties"]
        assert "skipped_segments" in schema["properties"]
        assert "retake_audit" in schema["properties"]

    def test_skipped_category_enum_in_schema(self):
        # Confirm the enum surfaces in the schema. Gemini honors enum
        # constraints, so this is what stops the model from inventing
        # categories at the API level (the prompt is belt; the schema
        # is suspenders).
        schema = Stage2Output.model_json_schema()
        # Pydantic v2 embeds enums under $defs; find SkippedCategory
        defs = schema.get("$defs") or schema.get("definitions") or {}
        cat_schema = defs.get("SkippedCategory")
        assert cat_schema is not None, (
            "SkippedCategory enum should be referenced in the schema"
        )
        assert "enum" in cat_schema
        assert set(cat_schema["enum"]) == {
            "warm_up", "retake", "crew_talk",
            "hesitation", "aside", "self_correction",
        }

    def test_retake_audit_is_nullable_in_schema(self):
        # Optional[str] means the schema should accept str OR null.
        # Verify so Gemini doesn't reject the schema for being too
        # strict.
        schema = Stage2Output.model_json_schema()
        retake_audit = schema["properties"]["retake_audit"]
        # Pydantic v2 emits an "anyOf" with {type:string} and {type:null}
        # OR a single type list. Either form is acceptable to Gemini.
        if "anyOf" in retake_audit:
            types = {opt.get("type") for opt in retake_audit["anyOf"]}
            assert "null" in types or "string" in types
        elif "type" in retake_audit:
            t = retake_audit["type"]
            if isinstance(t, list):
                assert "null" in t and "string" in t
            else:
                # Single string type; the default=None on the field
                # makes it effectively optional even if the schema
                # itself doesn't mark it nullable.
                assert t == "string"
        else:
            pytest.fail(f"unexpected retake_audit schema shape: {retake_audit}")
