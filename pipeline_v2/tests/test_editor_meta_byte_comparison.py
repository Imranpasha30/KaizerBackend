"""Byte-comparison test: V1 production output -> V2 adapter -> V1
editor_meta dict (Step 8.3).

The test constructs a synthetic ``JobOutput`` from the REAL V1 Gemini
output captured during Step 5.0's SDK regression
(``tests/fixtures/step5_diag/v1_regression_output.json``, run
2026-05-18 against ``test.mp4`` -- the Bandi Bhagirath case
bulletin). It then runs the V2 shorts adapter and asserts every
**content field** matches what V1's pipeline.py would have produced
given the same Gemini output.

Scope (per Step 8 D-list approval -- explicitly documented here so
future maintainers don't get confused):

  COMPARED (content fields):
    - top-level: video_type, language, language_english,
      language_native, script, title_native, title_telugu,
      title_english, summary, summary_native, summary_telugu,
      people, topics, keywords
    - per-clip: start, end, duration, importance

  NOT COMPARED:
    - render-time paths (clip_path, raw_path, thumb_path, image_path)
      -- caller-supplied via ClipRenderArtifacts, not deriveable
      from V1's pre-render output
    - storage URLs / keys / backend -- uploaded at render time
    - per-clip "summary" -- V1 has both a long per-cut summary AND a
      shorts_headline_native (headline); V2's ShortsCut has only
      ``hook``. The mapping V2.hook -> clip.summary is correct per
      D-8.9 but doesn't byte-match V1's per-clip summary text.
    - per-clip "text" -- V1's text is the global headline at the
      ROW-level. V2 also emits global headline. They match in
      structure but the source data here (regression output's
      per-cut headlines) doesn't have a single global headline.
      Tested implicitly via title_native matching.
    - per-clip preset/card_params/split_params/follow_params --
      render config, caller-supplied
    - created (timestamp) -- caller-supplied

Format note (D-8.5): V1's shorts_cuts in the regression output use
mixed-precision time strings ("01:12.00" = 2 decimals). V2's adapter
emits "01:12.000" (3 decimals) per the locked D-8.5 format. The
"expected V1" reference values in this test use V2's 3-decimal
format -- V1 inherited Gemini's whim; V2 standardizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_v2.editor_meta_adapter import (
    ClipRenderArtifacts,
    build_v1_shorts_editor_meta,
)
from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    FullVideoCut,
    ImagePlan,
    JobOutput,
    Metadata,
    ShortsCut,
    SkippedSegment,
    StageTwoOutput,
    Word,
)

# Path to the V1 production regression output
V1_REGRESSION_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "step5_diag"
    / "v1_regression_output.json"
)


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #


def _parse_mmss(ts: str) -> float:
    """Parse V1's 'MM:SS.mm' string -> float seconds.

    V1's regression output uses inconsistent precision:
      - "00:53.00" -- 2 decimals
      - "01:12.00" -- 2 decimals
      - "00:53.300" -- 3 decimals (some entries)
    Strip leading minutes, multiply by 60, add seconds part.
    """
    minutes_str, secs_str = ts.split(":", 1)
    return int(minutes_str) * 60 + float(secs_str)


@pytest.fixture(scope="module")
def v1_regression() -> dict:
    """Load the V1 production output (read-only). Module-scoped --
    parsed once across all tests in this file.
    """
    return json.loads(V1_REGRESSION_PATH.read_text(encoding="utf-8"))


def _v1_shorts_passing_v2_validator(v1: dict) -> list[dict]:
    """Filter V1's shorts_cuts to those that satisfy V2's
    ShortsCut 15-60s duration validator (D-7.11).

    **Real-data finding**: V1's regression output has 5 shorts; 2
    fall outside V2's [15, 60]s window (one at 13s, one at 102s).
    V1's prompt asked for 15-60s but didn't enforce -- Gemini
    occasionally returned out-of-range. V2's stricter validation
    (D-7.11) catches these at the ShortsCut layer.

    For the byte-comparison test, we filter to the conforming
    subset (3 of 5) and rebuild indices to be 0-based contiguous
    (V2 also requires this, D-8.12). The test still validates
    every content field on the kept shorts.
    """
    kept = []
    for sc in v1["shorts_cuts"]:
        dur = _parse_mmss(sc["end"]) - _parse_mmss(sc["start"])
        if 15.0 <= dur <= 60.0:
            kept.append(sc)
    return kept


def _synth_job_output_from_v1(v1: dict) -> JobOutput:
    """Construct a V2 JobOutput from V1's Gemini-output fields.

    V1's output has fields V2 doesn't (e.g. per-shorts-cut
    shorts_headline_native, bulletin_marquee_points). V2's
    ShortsCut has fewer fields (index, start_sec, end_sec, hook,
    importance). We map:

      - shorts_cuts[i].start / .end / .summary / .shorts_headline_native
        -> V2 ShortsCut(index=i_kept, start_sec=..., end_sec=...,
                        hook=shorts_headline_native, importance=7)
      - clips[i].importance -> V2 FullVideoCut.importance (for the
        bulletin path)
      - All metadata fields -> Metadata directly
      - top-level shorts_headline_native: V1 doesn't have one (only
        per-cut). Pick first KEPT shorts_cut's headline for the V2
        global headline.

    importance=7 is synthetic -- V1's shorts_cuts don't have an
    importance field; V2 requires it. Pinning to a known constant
    makes the test deterministic.

    Note: V1 shorts violating V2's 15-60s validator are filtered
    out (see ``_v1_shorts_passing_v2_validator`` docstring). Index
    is re-numbered 0-based on the kept subset to satisfy D-8.12.
    """
    # Build ShortsCut list ONLY from V1 shorts that pass V2's
    # 15-60s duration validator. Index re-numbered 0-based.
    kept_shorts = _v1_shorts_passing_v2_validator(v1)
    shorts_cuts = []
    for i, sc in enumerate(kept_shorts):
        shorts_cuts.append(ShortsCut(
            index=i,
            start_sec=_parse_mmss(sc["start"]),
            end_sec=_parse_mmss(sc["end"]),
            hook=sc["shorts_headline_native"],
            importance=7,   # synthetic; V1 omits
        ))

    # Build FullVideoCut list from V1's clips (the bulletin cuts).
    full_video_cuts = []
    for c in v1["clips"]:
        full_video_cuts.append(FullVideoCut(
            index=c["index"] - 1,  # V1 is 1-based; V2 is 0-based
            start_word_idx=0,      # not used for editor_meta
            end_word_idx=0,        # not used for editor_meta
            start_sec=_parse_mmss(c["start"]),
            end_sec=_parse_mmss(c["end"]),
            importance=c["importance"],
        ))

    # Build SkippedSegment list (just count matters for bulletin
    # shape's `skipped` field).
    skipped = []
    for s in v1["skipped_segments"]:
        skipped.append(SkippedSegment(
            start_word_idx=0, end_word_idx=0,
            start_sec=_parse_mmss(s["start"]),
            end_sec=_parse_mmss(s["end"]),
            category=s["category"],
            reason=s["reason"],
        ))

    # CleanTranscript: minimal synthetic words. The editor_meta
    # adapter doesn't read these for shorts; only used by Stage 9
    # render. Use 1 dummy word so model_validate passes.
    clean = CleanTranscript(
        words=[Word(w="dummy", s=0.0, e=1.0)],
        clip_boundaries={i: (0, 0) for i in range(len(full_video_cuts))},
        source_word_map=[0],
    )

    # Build Metadata. V2 expects ISO codes; V1 has "Telugu" --
    # map to "te". (V2 normally would have emitted "te" or "te-en";
    # we use "te" for canonical V1 compat.)
    # Use FIRST KEPT shorts_cut's headline as the canonical bulletin
    # headline (V1 doesn't have a top-level shorts_headline_native;
    # only per-cut. Pick the first one that passed validation.)
    first_kept = kept_shorts[0] if kept_shorts else v1["shorts_cuts"][0]
    metadata = Metadata(
        video_type=v1["video_type"],
        language="te",
        total_speakers=v1["total_speakers"],
        overall_summary=v1["overall_summary"],
        overall_summary_native=v1["overall_summary_native"],
        shorts_headline_native=first_kept["shorts_headline_native"],
        bulletin_marquee_points=first_kept["bulletin_marquee_points"],
        image_search_queries=v1["image_search_queries"],
        key_people=v1["key_people"],
        key_people_native=v1["key_people_native"],
        key_topics=v1["key_topics"],
        key_locations=v1["key_locations"],
    )

    # canonical_entities: synthetic Entity objects from key_people +
    # key_people_native (aligned by index). Type defaults to PERSON
    # since key_people is the PERSON subset of canonical entities.
    entities = []
    for i, (name, native) in enumerate(
        zip(v1["key_people"], v1["key_people_native"])
    ):
        entities.append(Entity(
            canonical_name=name,
            native_name=native,
            first_mention_word_idx=0,
            type="PERSON",
            mentions=[0],
        ))

    return JobOutput(
        stage_two=StageTwoOutput(
            full_video_cuts=full_video_cuts,
            skipped_segments=skipped,
            clean_transcript=clean,
            retake_audit=v1["retake_audit"],
        ),
        canonical_entities=entities,
        shorts_cuts=shorts_cuts,
        metadata=metadata,
        image_plan=ImagePlan(entries=[]),   # not relevant to test scope
    )


# ====================================================================== #
# Test surface                                                            #
# ====================================================================== #


@pytest.fixture(scope="module")
def v1_kept_shorts(v1_regression: dict) -> list[dict]:
    """V1 shorts_cuts that pass V2's 15-60s validator. See
    ``_v1_shorts_passing_v2_validator`` docstring -- 3 of 5 pass on
    the Bandi Bhagirath regression fixture.
    """
    return _v1_shorts_passing_v2_validator(v1_regression)


@pytest.fixture(scope="module")
def v2_adapter_output(v1_regression: dict,
                      v1_kept_shorts: list[dict]) -> dict:
    """Run the V2 adapter on a synthetic JobOutput derived from V1
    regression. Cached module-scope. Artifact count must match the
    KEPT shorts count (post-validator filter), not the raw V1 count.
    """
    job_output = _synth_job_output_from_v1(v1_regression)
    return build_v1_shorts_editor_meta(
        job_output,
        video_path="/abs/test.mp4",
        platform="youtube_short",
        frame_layout="torn_card",
        preset={
            "label": "YouTube Short", "width": 1080, "height": 1920,
            "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
            "vertical": True,
        },
        timestamp="20260518_140000",
        clip_artifacts=[
            ClipRenderArtifacts(
                clip_path=f"/abs/clip_{i+1:02d}.mp4",
                raw_path=f"/abs/raw_{i+1:02d}.mp4",
            )
            for i in range(len(v1_kept_shorts))
        ],
    )


class TestTopLevelContentFieldsMatchV1:
    """Each test compares ONE top-level content field. Single-purpose
    asserts so failure messages point directly at the broken field.
    """

    def test_video_type(self, v1_regression, v2_adapter_output):
        assert v2_adapter_output["video_type"] == v1_regression["video_type"]

    def test_language_iso(self, v2_adapter_output):
        # V1 says "Telugu"; V2 stores ISO. The adapter's output
        # canonicalises to ISO regardless.
        assert v2_adapter_output["language"] == "te"

    def test_language_english_resolved(self, v2_adapter_output):
        assert v2_adapter_output["language_english"] == "Telugu"

    def test_language_native_resolved(self, v2_adapter_output):
        assert v2_adapter_output["language_native"] == "తెలుగు"

    def test_script_resolved(self, v2_adapter_output):
        assert v2_adapter_output["script"] == "Telugu"

    def test_title_native_from_first_kept_shorts_cut_headline(
        self, v1_kept_shorts, v2_adapter_output,
    ):
        # Headline comes from the first short that PASSED V2's
        # validator (not the first raw V1 short, which may have been
        # filtered).
        expected = v1_kept_shorts[0]["shorts_headline_native"]
        assert v2_adapter_output["title_native"] == expected

    def test_title_telugu_legacy_alias_matches_title_native(
        self, v2_adapter_output,
    ):
        assert v2_adapter_output["title_telugu"] == v2_adapter_output["title_native"]

    def test_title_english_default_empty(self, v2_adapter_output):
        assert v2_adapter_output["title_english"] == ""

    def test_summary_matches_v1(self, v1_regression, v2_adapter_output):
        assert v2_adapter_output["summary"] == v1_regression["overall_summary"]

    def test_summary_native_matches_v1(
        self, v1_regression, v2_adapter_output,
    ):
        assert (
            v2_adapter_output["summary_native"]
            == v1_regression["overall_summary_native"]
        )

    def test_summary_telugu_legacy_alias(self, v2_adapter_output):
        assert (
            v2_adapter_output["summary_telugu"]
            == v2_adapter_output["summary_native"]
        )

    def test_people_matches_v1_key_people(
        self, v1_regression, v2_adapter_output,
    ):
        assert v2_adapter_output["people"] == v1_regression["key_people"]

    def test_topics_matches_v1_key_topics(
        self, v1_regression, v2_adapter_output,
    ):
        assert v2_adapter_output["topics"] == v1_regression["key_topics"]

    def test_keywords_always_empty(self, v2_adapter_output):
        assert v2_adapter_output["keywords"] == []


class TestPerClipContentFieldsMatchV1:
    """Per-clip comparison. Tests the four content fields the user
    explicitly scoped: start, end, duration, importance. Iterates
    over V1 shorts that passed V2's validator (kept subset).
    """

    def test_clip_count_matches_kept_shorts(
        self, v1_kept_shorts, v2_adapter_output,
    ):
        assert len(v2_adapter_output["clips"]) == len(v1_kept_shorts)

    def test_clip_start_times_are_v2_formatted(
        self, v1_kept_shorts, v2_adapter_output,
    ):
        # V1 uses "MM:SS.mm" (2 decimals); V2 emits "MM:SS.mmm"
        # (3 decimals). We compare numeric equivalence via the same
        # MM:SS.mmm format applied to V1's parsed value.
        from pipeline_v2.editor_meta_adapter import _format_mmss_mmm
        for i, (v1_sc, v2_clip) in enumerate(zip(
            v1_kept_shorts, v2_adapter_output["clips"],
        )):
            v1_secs = _parse_mmss(v1_sc["start"])
            expected = _format_mmss_mmm(v1_secs)
            assert v2_clip["start"] == expected, (
                f"Clip {i}: start mismatch. V2 emitted {v2_clip['start']!r}, "
                f"expected {expected!r} (parsed from V1 {v1_sc['start']!r})"
            )

    def test_clip_end_times_are_v2_formatted(
        self, v1_kept_shorts, v2_adapter_output,
    ):
        from pipeline_v2.editor_meta_adapter import _format_mmss_mmm
        for v1_sc, v2_clip in zip(
            v1_kept_shorts, v2_adapter_output["clips"],
        ):
            v1_secs = _parse_mmss(v1_sc["end"])
            expected = _format_mmss_mmm(v1_secs)
            assert v2_clip["end"] == expected

    def test_clip_durations_match(
        self, v1_kept_shorts, v2_adapter_output,
    ):
        for i, (v1_sc, v2_clip) in enumerate(zip(
            v1_kept_shorts, v2_adapter_output["clips"],
        )):
            v1_dur = round(
                _parse_mmss(v1_sc["end"]) - _parse_mmss(v1_sc["start"]),
                2,
            )
            assert v2_clip["duration"] == v1_dur, (
                f"Clip {i}: duration mismatch. "
                f"V2={v2_clip['duration']}, expected={v1_dur}"
            )

    def test_clip_importance_synthetic_constant(self, v2_adapter_output):
        # V1 omits per-shorts-cut importance; we synthesised 7 in the
        # test setup. Pin that here so a future change to the
        # synthetic value is intentional.
        for v2_clip in v2_adapter_output["clips"]:
            assert v2_clip["importance"] == 7

    def test_kept_shorts_count_is_3_of_5(self, v1_kept_shorts):
        # Pin the kept-count finding: V1's regression has 5 shorts, 3
        # pass V2's 15-60s validator (the other 2 are 13s and 102s).
        # If this number changes, the fixture has drifted or V2's
        # validator was relaxed -- both warrant re-evaluation.
        assert len(v1_kept_shorts) == 3


class TestRenderTimePathsExcludedFromComparison:
    """The render-time paths are NOT compared against V1 (per user
    spec). This test family CONFIRMS they exist in the output (so
    the adapter shape is complete) but doesn't assert their values.
    """

    def test_clip_paths_present_but_synthetic(
        self, v1_kept_shorts, v2_adapter_output,
    ):
        # Synthetic paths constructed in the fixture; verify they
        # surface intact (the adapter must not mutate caller-supplied
        # paths). Iterates over the KEPT count (post-filter).
        assert len(v2_adapter_output["clips"]) == len(v1_kept_shorts)
        for i, v2_clip in enumerate(v2_adapter_output["clips"]):
            assert v2_clip["clip_path"] == f"/abs/clip_{i+1:02d}.mp4"
            assert v2_clip["raw_path"] == f"/abs/raw_{i+1:02d}.mp4"

    def test_storage_fields_present_but_empty(self, v2_adapter_output):
        # Default empty -- no upload happened in this test.
        for v2_clip in v2_adapter_output["clips"]:
            assert v2_clip["storage_url"] == ""
            assert v2_clip["storage_key"] == ""
            assert v2_clip["storage_backend"] == ""


class TestV1RegressionFixtureIntegrity:
    """Sanity-check the fixture itself hasn't drifted. If
    v1_regression_output.json is modified, these tests fire and we
    re-evaluate the byte-comparison expectations.
    """

    def test_fixture_loads(self, v1_regression):
        assert v1_regression["video_type"] == "SOLO"
        assert v1_regression["language"] == "Telugu"

    def test_fixture_has_expected_shorts_count(self, v1_regression):
        # If V1 ever re-generates with different content, this number
        # will change -- a clear signal that byte-comparison needs
        # re-validation.
        assert len(v1_regression["shorts_cuts"]) == 5

    def test_fixture_has_expected_clips_count(self, v1_regression):
        assert len(v1_regression["clips"]) == 6

    def test_fixture_has_key_people(self, v1_regression):
        # Bandi Bhagirath case bulletin -- pin the known people set.
        assert "Bandi Bhagirath" in v1_regression["key_people"]
        assert "Revanth Reddy" in v1_regression["key_people"]
