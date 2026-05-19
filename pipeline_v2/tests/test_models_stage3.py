"""Unit tests for Stage 3 Pydantic models (Step 7.1).

Covers:
  - ``ImagePlanEntry`` trim from 10 fields → 6 (Step 7.1 / D-7.2)
  - Confirm the 5 removed fields are gone (id, topic_clue,
    search_query, search_query_native, reason)
  - Confirm the new field entity_name_native is required non-None
  - ``ShortsCut`` unchanged behavior (15-60s duration validator)
  - ``Metadata`` unchanged behavior (13-field top-level shape)
  - JSON Schema buildability (Gemini's response_schema= consumes
    the same JSON Schema document)

No Gemini calls. Pure schema tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_v2 import models
from pipeline_v2.models import (
    ImagePlan,
    ImagePlanEntry,
    Metadata,
    ShortsCut,
)


# ====================================================================== #
# ImagePlanEntry: 6-field trimmed shape                                   #
# ====================================================================== #


class TestImagePlanEntryTrimmedShape:
    def _entry(self, **overrides):
        defaults = {
            "entity_name": "Narendra Modi",
            "entity_name_native": "నరేంద్ర మోదీ",
            "description": "PM Modi at the press conference",
            "clip_index": 0,
            "show_at_sec": 12.5,
            "duration_sec": 4.0,
        }
        defaults.update(overrides)
        return ImagePlanEntry(**defaults)

    def test_minimal_valid(self):
        e = self._entry()
        assert e.entity_name == "Narendra Modi"
        assert e.entity_name_native == "నరేంద్ర మోదీ"
        assert e.clip_index == 0
        assert e.show_at_sec == 12.5
        assert e.duration_sec == 4.0

    def test_six_fields_only(self):
        # ImagePlanEntry MUST have exactly 6 fields per D-7.2.
        # Adding a field later requires an explicit decision -- this
        # test fires if someone re-adds id/topic_clue/etc.
        fields = set(ImagePlanEntry.model_fields.keys())
        assert fields == {
            "entity_name",
            "entity_name_native",
            "description",
            "clip_index",
            "show_at_sec",
            "duration_sec",
        }

    @pytest.mark.parametrize("removed_field", [
        "id",
        "topic_clue",
        "search_query",
        "search_query_native",
        "reason",
    ])
    def test_removed_fields_gone(self, removed_field):
        # Pin the removal -- a future "compatibility shim" that re-adds
        # one of these should fail this suite and force an explicit
        # decision.
        assert removed_field not in ImagePlanEntry.model_fields

    def test_duration_sec_min_2_constraint_preserved(self):
        # The pre-existing Field(ge=2.0) constraint on duration_sec
        # MUST survive the trim -- the renderer assumes >=2s display.
        with pytest.raises(ValidationError):
            self._entry(duration_sec=1.5)

    def test_duration_sec_exactly_2_allowed(self):
        e = self._entry(duration_sec=2.0)
        assert e.duration_sec == 2.0

    def test_entity_name_native_required(self):
        # Per D6 (no defaults on response_schema fields), all 6 fields
        # are required at construction.
        with pytest.raises(ValidationError):
            ImagePlanEntry(   # type: ignore[call-arg]
                entity_name="X",
                # entity_name_native missing
                description="d",
                clip_index=0,
                show_at_sec=1.0,
                duration_sec=2.0,
            )

    def test_no_field_has_default(self):
        # D-7.8: confirm no Field(default=...) on response_schema
        # fields. ImagePlanEntry is part of ImagePlan which is part of
        # the Stage 3c response_schema -- defaults would raise at API
        # time (google-genai #699).
        for name, info in ImagePlanEntry.model_fields.items():
            assert info.is_required(), (
                f"ImagePlanEntry.{name} has a default; D-7.8 forbids "
                f"this on response_schema fields"
            )

    def test_json_roundtrip(self):
        e = self._entry()
        rebuilt = ImagePlanEntry.model_validate_json(e.model_dump_json())
        assert rebuilt == e


# ====================================================================== #
# ImagePlan wrapper                                                       #
# ====================================================================== #


class TestImagePlan:
    def _entry(self, idx: int = 0, name: str = "X") -> ImagePlanEntry:
        return ImagePlanEntry(
            entity_name=name,
            entity_name_native=name,
            description=f"desc{idx}",
            clip_index=idx,
            show_at_sec=1.0,
            duration_sec=2.0,
        )

    def test_empty_entries_allowed(self):
        # A bulletin with no image overlays is legitimate (rare but
        # possible -- e.g. anchor-only segment with no recognized
        # entities). The Stage 3c dispatcher's >50%-dropped guardrail
        # is the safety check for systemic failure, not "0 entries".
        plan = ImagePlan(entries=[])
        assert plan.entries == []

    def test_multiple_entries(self):
        plan = ImagePlan(entries=[
            self._entry(0, "A"), self._entry(1, "B"), self._entry(2, "C"),
        ])
        assert len(plan.entries) == 3


# ====================================================================== #
# ShortsCut: 15-60s duration validator unchanged                          #
# ====================================================================== #


class TestShortsCutDurationValidatorUnchanged:
    """D-7.1: ShortsCut shape and validator are preserved. The
    @field_validator("end_sec") raising on out-of-range duration is
    the single hardest invariant of Stage 3a (renderer assumes 15-60s
    shorts). This test pins that the validator is still in place.
    """

    @pytest.mark.parametrize("duration", [15.0, 30.0, 60.0])
    def test_valid_durations(self, duration):
        c = ShortsCut(
            index=0, start_sec=0.0, end_sec=duration,
            hook="A short hook", importance=7,
        )
        assert c.end_sec == duration

    @pytest.mark.parametrize("duration", [10.0, 14.9, 60.1, 90.0, 0.0])
    def test_invalid_durations_rejected(self, duration):
        with pytest.raises(ValidationError, match=r"shorts must be 15-60s"):
            ShortsCut(
                index=0, start_sec=0.0, end_sec=duration,
                hook="A short hook", importance=7,
            )

    def test_importance_1_to_10_constraint(self):
        c = ShortsCut(
            index=0, start_sec=0.0, end_sec=30.0,
            hook="x", importance=10,
        )
        assert c.importance == 10
        with pytest.raises(ValidationError):
            ShortsCut(
                index=0, start_sec=0.0, end_sec=30.0,
                hook="x", importance=11,
            )

    def test_field_set_unchanged_from_plan_a(self):
        # D-7.1: pin the field set so future "improvements" require
        # explicit decision.
        assert set(ShortsCut.model_fields.keys()) == {
            "index", "start_sec", "end_sec", "hook", "importance",
        }


# ====================================================================== #
# Metadata: 13 top-level fields unchanged                                 #
# ====================================================================== #


class TestMetadataShapeUnchanged:
    def test_field_set_unchanged_from_plan_a(self):
        # D-7.6: pin Metadata fields. V1's Step-0 evolution moved
        # `shorts_headline_native` and `bulletin_marquee_points` into
        # shorts_cuts items, but V2 (per current schema) keeps them
        # top-level. The adapter (Step 8) bridges to V1's editor_meta.
        assert set(Metadata.model_fields.keys()) == {
            "video_type",
            "language",
            "total_speakers",
            "overall_summary",
            "overall_summary_native",
            "shorts_headline_native",
            "bulletin_marquee_points",
            "image_search_queries",
            "key_people",
            "key_people_native",
            "key_topics",
            "key_locations",
        }

    def test_video_type_literal_locked(self):
        m = Metadata(
            video_type="SOLO", language="te", total_speakers=1,
            overall_summary="x", overall_summary_native="x",
            shorts_headline_native="x", bulletin_marquee_points=[],
            image_search_queries=[], key_people=[], key_people_native=[],
            key_topics=[], key_locations=[],
        )
        assert m.video_type == "SOLO"
        with pytest.raises(ValidationError):
            Metadata(
                video_type="UNKNOWN",   # invented
                language="te", total_speakers=1,
                overall_summary="x", overall_summary_native="x",
                shorts_headline_native="x", bulletin_marquee_points=[],
                image_search_queries=[], key_people=[],
                key_people_native=[], key_topics=[], key_locations=[],
            )


# ====================================================================== #
# JobOutput: still references the (now-trimmed) ImagePlanEntry            #
# ====================================================================== #


class TestJobOutputStillWiresStage3:
    """JobOutput composes Stage 3's outputs (shorts_cuts, metadata,
    image_plan). After trimming ImagePlanEntry, JobOutput must still
    type-check and reference the new shape.
    """

    def test_image_plan_field_type(self):
        from typing import get_args
        field = models.JobOutput.model_fields["image_plan"]
        # annotation is ImagePlan, not list[ImagePlan]
        assert field.annotation is ImagePlan

    def test_shorts_cuts_field_type(self):
        from typing import get_args, get_origin
        field = models.JobOutput.model_fields["shorts_cuts"]
        ann = field.annotation
        assert get_origin(ann) is list
        assert get_args(ann) == (ShortsCut,)

    def test_metadata_field_type(self):
        field = models.JobOutput.model_fields["metadata"]
        assert field.annotation is Metadata


# ====================================================================== #
# Gemini-side compatibility (response_schema = ImagePlan)                 #
# ====================================================================== #


class TestImagePlanGeminiResponseSchemaCompatibility:
    """ImagePlan is passed to Gemini Flash as response_schema= in
    Stage 3c. Pin the JSON Schema shape so Flash's structured-output
    layer doesn't see anything it can't handle.
    """

    def test_pydantic_json_schema_buildable(self):
        schema = ImagePlan.model_json_schema()
        assert "properties" in schema
        assert "entries" in schema["properties"]

    def test_image_plan_entry_required_fields(self):
        schema = ImagePlan.model_json_schema()
        defs = schema.get("$defs") or schema.get("definitions") or {}
        entry = defs.get("ImagePlanEntry")
        assert entry is not None
        required = set(entry.get("required", []))
        # All 6 fields required (no defaults per D-7.8)
        assert required == {
            "entity_name", "entity_name_native", "description",
            "clip_index", "show_at_sec", "duration_sec",
        }

    def test_image_plan_entry_no_legacy_fields_in_schema(self):
        schema = ImagePlan.model_json_schema()
        defs = schema.get("$defs") or schema.get("definitions") or {}
        entry = defs.get("ImagePlanEntry")
        props = set(entry.get("properties", {}).keys())
        # Confirm legacy fields don't leak into the JSON Schema either
        for legacy in ("id", "topic_clue", "search_query",
                       "search_query_native", "reason"):
            assert legacy not in props, (
                f"Legacy field '{legacy}' leaked into JSON Schema -- "
                f"would prompt Gemini to emit it, wasting tokens."
            )
