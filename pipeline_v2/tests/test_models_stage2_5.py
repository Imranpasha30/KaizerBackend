"""Unit tests for Stage 2.5 Pydantic models (Step 6.1).

Covers:
  - ``EntityType`` enum (5 locked values; forbid-invention at Pydantic
    level, mirroring the SkippedCategory pattern)
  - ``Entity`` schema (canonical_name, native_name, first_mention_word_idx,
    type, mentions)
  - ``Stage2_5Output`` schema with the Pydantic-level 6-cap
    (``Field(max_length=6)`` enforces at the schema layer; the
    Stage2_5EntityCanonicalizer adds a post-validate truncation
    safety net for when Gemini ignores the soft hint -- tested in
    test_stage_2_5_entity_canonicalizer.py)
  - JSON Schema buildability (Gemini consumes the same JSON Schema
    via response_schema=)
  - Removal of legacy CanonicalEntity / StageTwoFiveOutput / EntityMention
    classes (D1 decision: clean replacement, zero call sites)

No Gemini calls. Pure schema-level tests.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline_v2 import models
from pipeline_v2.models import (
    Entity,
    EntityType,
    Stage2_5Output,
)


# ====================================================================== #
# EntityType enum (locked Step 6 D2 taxonomy)                            #
# ====================================================================== #


class TestEntityTypeEnum:
    def test_five_locked_values(self):
        expected = {"PERSON", "ORG", "PLACE", "EVENT", "OTHER"}
        assert {m.value for m in EntityType} == expected

    def test_str_enum_value(self):
        # str subclass -> Pydantic serialises cleanly to JSON.
        assert EntityType.PERSON == "PERSON"
        assert EntityType.ORG.value == "ORG"

    @pytest.mark.parametrize("good", ["PERSON", "ORG", "PLACE", "EVENT", "OTHER"])
    def test_entity_accepts_each_type(self, good):
        e = Entity(
            canonical_name="X", native_name="X",
            first_mention_word_idx=0, type=good, mentions=[0],
        )
        assert e.type == good

    @pytest.mark.parametrize("invented", [
        "GROUP",           # invented
        "LOCATION",        # close to PLACE but not allowed
        "ORGANIZATION",    # close to ORG
        "PERSON_NAME",     # case-strict match required
        "Person",          # case-sensitive
        "person",
        "",
        "OTHER ",          # trailing space
    ])
    def test_entity_rejects_invented_type(self, invented):
        with pytest.raises(ValidationError):
            Entity(
                canonical_name="X", native_name="X",
                first_mention_word_idx=0, type=invented, mentions=[0],
            )


# ====================================================================== #
# Entity field-level constraints                                          #
# ====================================================================== #


class TestEntity:
    def _entity(self, **overrides):
        defaults = {
            "canonical_name": "Revanth Reddy",
            "native_name": "రేవంత్ రెడ్డి",
            "first_mention_word_idx": 12,
            "type": "PERSON",
            "mentions": [12, 45, 67],
        }
        defaults.update(overrides)
        return Entity(**defaults)

    def test_minimal_valid(self):
        e = self._entity()
        assert e.canonical_name == "Revanth Reddy"
        assert e.type == EntityType.PERSON

    def test_native_name_can_equal_canonical_for_english_only(self):
        # D4: English-only entities (BRS, RTI) set native_name ==
        # canonical_name. Never blank.
        e = self._entity(canonical_name="BRS", native_name="BRS", type="ORG")
        assert e.native_name == "BRS"

    def test_native_name_empty_string_allowed_at_pydantic_level(self):
        # Per D4 the prompt forbids blank native_name, but Pydantic
        # itself doesn't enforce min_length -- the prompt is the
        # contract. This test pins the current behavior; if we
        # later add min_length validation, this test will flip.
        e = self._entity(native_name="")
        assert e.native_name == ""

    def test_mentions_can_be_single_element(self):
        e = self._entity(mentions=[0])
        assert e.mentions == [0]

    def test_mentions_empty_rejected_at_business_logic_layer(self):
        # Pydantic accepts mentions=[] at the schema level; the prompt
        # forbids it ("length >= 1"). The post-validate layer in the
        # Stage 2.5 class can enforce. Pin the Pydantic behavior here.
        e = self._entity(mentions=[])
        assert e.mentions == []

    def test_first_mention_word_idx_can_match_mentions_zero(self):
        e = self._entity(first_mention_word_idx=12, mentions=[12, 45])
        assert e.first_mention_word_idx == e.mentions[0]


# ====================================================================== #
# Stage2_5Output -- the LLM contract                                      #
# ====================================================================== #


class TestStage2_5Output:
    def _entity(self, idx: int, mentions=None, type_: str = "PERSON"):
        return Entity(
            canonical_name=f"Person{idx}",
            native_name=f"Native{idx}",
            first_mention_word_idx=idx * 10,
            type=type_,
            mentions=mentions if mentions is not None else [idx * 10],
        )

    def test_minimal_empty_entities_allowed(self):
        # D6: no default. Caller must explicitly pass `entities=[]`.
        # Empty list is legitimate (transcript with zero named entities).
        out = Stage2_5Output(entities=[])
        assert out.entities == []

    def test_six_entities_is_max(self):
        ents = [self._entity(i) for i in range(6)]
        out = Stage2_5Output(entities=ents)
        assert len(out.entities) == 6

    def test_seven_entities_rejected_by_pydantic(self):
        ents = [self._entity(i) for i in range(7)]
        # Pydantic enforces max_length=6 at validation. Gemini's
        # schema engine treats this as a SOFT hint -- the post-
        # validate truncation in the canonicalizer class is the
        # third safety net, but Pydantic alone catches this here.
        with pytest.raises(ValidationError):
            Stage2_5Output(entities=ents)

    def test_field_default_not_present(self):
        # D6: confirm `entities` is required (no default).
        # Per google-genai #699, having `default=...` on a
        # response_schema field raises at API time. Pin no-default.
        with pytest.raises(ValidationError):
            Stage2_5Output()  # type: ignore[call-arg]

    def test_json_roundtrip_preserves_all_fields(self):
        ents = [
            self._entity(0, type_="PERSON"),
            self._entity(1, type_="ORG"),
            self._entity(2, type_="PLACE", mentions=[20, 30, 40]),
        ]
        out = Stage2_5Output(entities=ents)
        data = json.loads(out.model_dump_json())
        rebuilt = Stage2_5Output.model_validate(data)
        assert len(rebuilt.entities) == 3
        assert {e.type for e in rebuilt.entities} == {
            EntityType.PERSON, EntityType.ORG, EntityType.PLACE,
        }
        assert rebuilt.entities[2].mentions == [20, 30, 40]

    def test_invented_type_at_load_time_rejected(self):
        bad_dict = {
            "entities": [{
                "canonical_name": "X", "native_name": "X",
                "first_mention_word_idx": 0,
                "type": "ORGANIZATION",         # invented
                "mentions": [0],
            }],
        }
        with pytest.raises(ValidationError):
            Stage2_5Output.model_validate(bad_dict)


# ====================================================================== #
# Legacy class removal (D1)                                               #
# ====================================================================== #


class TestLegacyClassesRemoved:
    """The pre-Step-6 shapes (CanonicalEntity, StageTwoFiveOutput,
    EntityMention) have ZERO call sites, so D1 confirmed clean
    replacement. These tests pin the removal -- a future
    'compatibility shim' that re-introduces them should fail this
    suite and force an explicit decision.
    """

    def test_canonical_entity_removed(self):
        assert not hasattr(models, "CanonicalEntity")

    def test_stage_two_five_output_removed(self):
        assert not hasattr(models, "StageTwoFiveOutput")

    def test_entity_mention_removed(self):
        assert not hasattr(models, "EntityMention")


# ====================================================================== #
# JobOutput field type updated to list[Entity] (D12)                     #
# ====================================================================== #


class TestJobOutputFieldType:
    def test_canonical_entities_field_type(self):
        # D12: field name 'canonical_entities' (process meaning) +
        # class name 'Entity' (identity meaning).
        from typing import get_args, get_origin
        field = models.JobOutput.model_fields["canonical_entities"]
        # annotation is list[Entity]
        ann = field.annotation
        assert get_origin(ann) is list
        assert get_args(ann) == (Entity,)


# ====================================================================== #
# Gemini-side compatibility (response_schema = Stage2_5Output)            #
# ====================================================================== #


class TestGeminiResponseSchemaCompatibility:
    """Stage2_5Output is passed to Gemini Flash as response_schema=.
    Same JSON Schema doc that Pydantic builds is what Gemini's
    structured-output layer enforces. Pin a few constraints that
    have historically tripped up Flash's schema engine.
    """

    def test_pydantic_json_schema_is_buildable(self):
        schema = Stage2_5Output.model_json_schema()
        assert "properties" in schema
        assert "entities" in schema["properties"]

    def test_entities_max_items_in_schema(self):
        # Pydantic v2 emits maxItems for Field(max_length=6) on a
        # list field. Gemini honors maxItems as a soft hint
        # (issue #699) -- but the schema MUST surface it for the
        # belt-and-suspenders strategy.
        schema = Stage2_5Output.model_json_schema()
        ents = schema["properties"]["entities"]
        assert ents.get("maxItems") == 6

    def test_entity_type_enum_in_schema(self):
        # EntityType enum must surface as a JSON Schema enum so
        # Gemini's structured-output layer can reject invented values
        # at the API level (prompt is belt; schema is suspenders).
        schema = Stage2_5Output.model_json_schema()
        defs = schema.get("$defs") or schema.get("definitions") or {}
        et = defs.get("EntityType")
        assert et is not None
        assert "enum" in et
        assert set(et["enum"]) == {
            "PERSON", "ORG", "PLACE", "EVENT", "OTHER",
        }

    def test_entity_required_fields_in_schema(self):
        # All Entity fields are required (no defaults, per D6).
        schema = Stage2_5Output.model_json_schema()
        defs = schema.get("$defs") or schema.get("definitions") or {}
        entity_schema = defs.get("Entity")
        assert entity_schema is not None
        required = set(entity_schema.get("required", []))
        assert required == {
            "canonical_name", "native_name", "first_mention_word_idx",
            "type", "mentions",
        }
