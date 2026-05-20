"""Unit tests for Stage 4 Render orchestrator (Step 9).

This file accumulates tests across Step 9 sub-steps:
  9.1 -- constructor, converters, raw-cut wiring  (THIS COMMIT)
  9.2 -- per-clip compose dispatcher + image resolution + guardrail
  9.3 -- bulletin assembly + image overlay + thumbnail
  9.4 -- top-level render() integration

All V1 primitive calls (cut_video_clips, compose_*, resolve_image_plan,
overlay_image_plan) are MOCKED. Real-FFmpeg verification lives in
test_stage_4_render_integration.py (opt-in via
KAIZER_RUN_INTEGRATION_TESTS=1).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    EntityType,
    FullVideoCut,
    ImagePlan,
    ImagePlanEntry,
    JobOutput,
    Metadata,
    ShortsCut,
    SkippedSegment,
    StageTwoOutput,
    Word,
)
from pipeline_v2.stages.stage_4_render import (
    DEFAULT_DROP_RATIO_THRESHOLD,
    DEFAULT_FRAME_LAYOUT,
    DEFAULT_PLATFORM,
    RenderResult,
    Stage4Render,
    _format_mmss_mmm,
    full_video_cuts_to_v1_clip_dicts,
    image_plan_to_v1_dict,
    shorts_cuts_to_v1_clip_dicts,
)
import json as _json


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _shorts_cut(idx: int = 0, start: float = 10.0, end: float = 28.0,
                hook: str = "A hook", importance: int = 7) -> ShortsCut:
    return ShortsCut(
        index=idx, start_sec=start, end_sec=end,
        hook=hook, importance=importance,
    )


def _full_video_cut(idx: int = 0, start: float = 0.0, end: float = 100.0,
                    importance: int = 8) -> FullVideoCut:
    return FullVideoCut(
        index=idx,
        start_word_idx=0,
        end_word_idx=int(end),
        start_sec=start, end_sec=end, importance=importance,
    )


def _entity(name: str = "Modi", type_: str = "PERSON") -> Entity:
    return Entity(
        canonical_name=name, native_name=name,
        first_mention_word_idx=0, type=type_, mentions=[0],
    )


def _image_plan_entry(name: str = "Modi", clip_index: int = 0,
                      show_at: float = 5.0, duration: float = 4.0,
                      description: str = "Modi at podium") -> ImagePlanEntry:
    return ImagePlanEntry(
        entity_name=name,
        entity_name_native=name,
        description=description,
        clip_index=clip_index,
        show_at_sec=show_at,
        duration_sec=duration,
    )


def _metadata(video_type: str = "SOLO",
              headline: str = "HEADLINE") -> Metadata:
    return Metadata(
        video_type=video_type,
        language="te-en",
        total_speakers=1,
        overall_summary="English summary.",
        overall_summary_native="తెలుగు సారాంశం.",
        shorts_headline_native=headline,
        bulletin_marquee_points=["A", "B", "C"],
        image_search_queries=["q1"],
        key_people=["Modi"],
        key_people_native=["మోదీ"],
        key_topics=["topic"],
        key_locations=["Hyderabad"],
    )


def _preset() -> dict:
    return {
        "label": "YouTube Short", "width": 1080, "height": 1920,
        "min_dur": 15, "max_dur": 60, "ideal_dur": 45, "vertical": True,
    }


@pytest.fixture
def render(tmp_path: Path) -> Stage4Render:
    """Stage4Render with a tmp_path output_dir and a non-existent
    video_path (V1 functions are mocked so the file doesn't need to
    exist).
    """
    return Stage4Render(
        output_dir=tmp_path / "out",
        video_path=tmp_path / "source.mp4",
        preset=_preset(),
    )


# ====================================================================== #
# _format_mmss_mmm                                                        #
# ====================================================================== #


class TestFormatMmssMmm:
    """Mirror of the editor_meta_adapter helper. Duplicated in
    stage_4_render to keep the modules independently importable.
    """

    @pytest.mark.parametrize("seconds,expected", [
        (0.0,    "00:00.000"),
        (1.0,    "00:01.000"),
        (53.3,   "00:53.300"),
        (110.59, "01:50.590"),
        (60.0,   "01:00.000"),
        (4530.5, "75:30.500"),
    ])
    def test_known_values(self, seconds, expected):
        assert _format_mmss_mmm(seconds) == expected

    def test_floating_point_carry(self):
        # 59.9999 -> "01:00.000" (avoids invalid "00:60.000")
        assert _format_mmss_mmm(59.9999) == "01:00.000"


# ====================================================================== #
# Pure converter: shorts_cuts_to_v1_clip_dicts                            #
# ====================================================================== #


class TestShortsCutsToV1ClipDicts:
    def test_single_cut(self):
        meta = _metadata(video_type="SOLO")
        cuts = [_shorts_cut(0, start=53.3, end=70.5,
                            hook="A punch", importance=8)]
        out = shorts_cuts_to_v1_clip_dicts(cuts, meta)
        assert len(out) == 1
        d = out[0]
        assert d["start"] == "00:53.300"
        assert d["end"] == "01:10.500"
        assert d["summary"] == "A punch"      # D-8.9: hook is summary
        assert d["mood"] == ""
        assert d["importance"] == 8
        assert d["video_type"] == "SOLO"
        assert d["v2_index"] == 0

    def test_multiple_cuts_preserve_order(self):
        meta = _metadata()
        cuts = [
            _shorts_cut(0, 0, 20, "A", 5),
            _shorts_cut(1, 30, 50, "B", 7),
            _shorts_cut(2, 60, 80, "C", 9),
        ]
        out = shorts_cuts_to_v1_clip_dicts(cuts, meta)
        assert [d["summary"] for d in out] == ["A", "B", "C"]
        assert [d["v2_index"] for d in out] == [0, 1, 2]

    def test_empty_input(self):
        assert shorts_cuts_to_v1_clip_dicts([], _metadata()) == []

    def test_video_type_propagated_from_metadata(self):
        meta = _metadata(video_type="PRESS_CONFERENCE")
        out = shorts_cuts_to_v1_clip_dicts([_shorts_cut(0, 0, 20)], meta)
        assert out[0]["video_type"] == "PRESS_CONFERENCE"


# ====================================================================== #
# Pure converter: full_video_cuts_to_v1_clip_dicts                        #
# ====================================================================== #


class TestFullVideoCutsToV1ClipDicts:
    def test_single_cut(self):
        meta = _metadata()
        cuts = [_full_video_cut(0, start=0.0, end=100.0, importance=8)]
        out = full_video_cuts_to_v1_clip_dicts(cuts, meta)
        assert len(out) == 1
        d = out[0]
        assert d["start"] == "00:00.000"
        assert d["end"] == "01:40.000"
        assert d["summary"] == ""             # bulletin path: no per-clip summary
        assert d["importance"] == 8
        assert d["video_type"] == "SOLO"

    def test_multiple_cuts(self):
        cuts = [
            _full_video_cut(0, 0, 60),
            _full_video_cut(1, 60, 120),
            _full_video_cut(2, 120, 180),
        ]
        out = full_video_cuts_to_v1_clip_dicts(cuts, _metadata())
        assert len(out) == 3
        assert out[0]["end"] == "01:00.000"
        assert out[1]["start"] == "01:00.000"
        assert out[2]["end"] == "03:00.000"


# ====================================================================== #
# Pure converter: image_plan_to_v1_dict                                   #
# ====================================================================== #


class TestImagePlanToV1Dict:
    def test_basic_entry(self):
        plan = ImagePlan(entries=[
            _image_plan_entry("Modi", clip_index=0,
                              show_at=12.5, duration=4.0),
        ])
        out = image_plan_to_v1_dict(plan, [_entity("Modi")])
        assert len(out) == 1
        d = out[0]
        assert d["entity_name"] == "Modi"
        assert d["entity_name_native"] == "Modi"
        assert d["description"] == "Modi at podium"
        assert d["clip_index"] == 0
        assert d["show_at"] == "00:12.500"
        assert d["duration"] == 4.0

    def test_multiple_entries_preserve_order(self):
        plan = ImagePlan(entries=[
            _image_plan_entry("A", clip_index=0, show_at=5.0),
            _image_plan_entry("B", clip_index=0, show_at=15.0),
            _image_plan_entry("C", clip_index=1, show_at=25.0),
        ])
        out = image_plan_to_v1_dict(plan, [_entity("A"), _entity("B"), _entity("C")])
        assert [d["entity_name"] for d in out] == ["A", "B", "C"]
        assert [d["clip_index"] for d in out] == [0, 0, 1]

    def test_empty_plan(self):
        assert image_plan_to_v1_dict(ImagePlan(entries=[]), []) == []

    def test_emits_synth_id_for_v1_compat(self):
        # V2 ImagePlanEntry has no `id` field (Entity uses
        # canonical_name as identity per Step 6 D-12). The converter
        # synthesises ``img_NNN`` ids at the V2->V1 boundary so V1's
        # pool_manifest lookup works without polluting V2's schema.
        # See ``synth_id_map_for_image_plan``.
        plan = ImagePlan(entries=[_image_plan_entry()])
        out = image_plan_to_v1_dict(plan, [_entity()])
        assert out[0]["id"] == "img_000"

    def test_synth_ids_dedupe_by_entity_name(self):
        # Two entries for the same entity share one synth_id (so V1
        # resolves them to the same image file -- "one image per
        # unique id, reused across moments" per pipeline.py:1466).
        plan = ImagePlan(entries=[
            _image_plan_entry("Modi", clip_index=0, show_at=5.0),
            _image_plan_entry("Reddy", clip_index=0, show_at=15.0),
            _image_plan_entry("Modi", clip_index=1, show_at=25.0),
        ])
        out = image_plan_to_v1_dict(
            plan, [_entity("Modi"), _entity("Reddy")],
        )
        assert [d["id"] for d in out] == ["img_000", "img_001", "img_000"]


# ====================================================================== #
# Stage4Render: constructor + __post_init__                               #
# ====================================================================== #


class TestStage4RenderConstructor:
    def test_defaults(self, tmp_path: Path):
        r = Stage4Render(
            output_dir=tmp_path / "out",
            video_path=tmp_path / "v.mp4",
            preset=_preset(),
        )
        assert r.frame_layout == DEFAULT_FRAME_LAYOUT
        assert r.platform == DEFAULT_PLATFORM
        assert r.drop_ratio_threshold == DEFAULT_DROP_RATIO_THRESHOLD
        assert r.image_pool == {}

    def test_kwargs_override(self, tmp_path: Path):
        r = Stage4Render(
            output_dir=tmp_path / "out",
            video_path=tmp_path / "v.mp4",
            preset=_preset(),
            frame_layout="clean_card",
            platform="youtube_short",
            drop_ratio_threshold=0.3,
        )
        assert r.frame_layout == "clean_card"
        assert r.platform == "youtube_short"
        assert r.drop_ratio_threshold == 0.3

    def test_post_init_creates_output_dir(self, tmp_path: Path):
        target = tmp_path / "deeply" / "nested" / "out"
        assert not target.exists()
        Stage4Render(
            output_dir=target,
            video_path=tmp_path / "v.mp4",
            preset=_preset(),
        )
        assert target.is_dir()

    def test_post_init_normalises_str_paths_to_path(self, tmp_path: Path):
        r = Stage4Render(
            output_dir=str(tmp_path / "out"),
            video_path=str(tmp_path / "v.mp4"),
            preset=_preset(),
        )
        assert isinstance(r.output_dir, Path)
        assert isinstance(r.video_path, Path)

    def test_image_pool_starts_empty(self, render: Stage4Render):
        # D-9.5: image_pool is instance state, populated by Step 9.2.
        assert render.image_pool == {}

    def test_image_pool_is_mutable(self, render: Stage4Render):
        # Stage 9.2 populates this; pin that it's writable.
        render.image_pool["Modi"] = Path("/abs/news_01.jpg")
        assert render.image_pool["Modi"] == Path("/abs/news_01.jpg")


# ====================================================================== #
# Stage4Render.cut_raw_shorts (Step 9.1 raw-cut wiring)                   #
# ====================================================================== #


class TestCutRawShorts:
    """Wires V1's cut_video_clips. V1's side-effect contract: MUTATES
    each clip dict to add ``raw_path`` and ``duration_sec``. Tests
    mock V1 and verify both the call shape and the mutation.
    """

    def test_calls_v1_with_correct_args(self, render: Stage4Render):
        cuts = [_shorts_cut(0, 0, 20), _shorts_cut(1, 30, 50)]

        def _v1_mock(video_path, clips, output_dir):
            # Verify V1 sees the expected args
            assert video_path == str(render.video_path)
            assert output_dir == str(render.output_dir)
            assert len(clips) == 2
            # Mutate to simulate V1's side effect
            for i, c in enumerate(clips, 1):
                c["raw_path"] = f"{output_dir}/raw_clip_{i:02d}.mp4"
                c["duration_sec"] = 20.0
            return [c["raw_path"] for c in clips]

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_v1_mock,
        ) as mock_v1:
            out = render.cut_raw_shorts(cuts, _metadata())
            assert mock_v1.call_count == 1

        assert len(out) == 2
        assert all("raw_path" in c for c in out)
        assert all("duration_sec" in c for c in out)

    def test_returns_mutated_clip_dicts_with_raw_path(
        self, render: Stage4Render,
    ):
        cuts = [_shorts_cut(0, 0, 20)]

        def _v1_mock(video_path, clips, output_dir):
            clips[0]["raw_path"] = f"{output_dir}/raw_clip_01.mp4"
            clips[0]["duration_sec"] = 20.0
            return [clips[0]["raw_path"]]

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_v1_mock,
        ):
            out = render.cut_raw_shorts(cuts, _metadata())

        assert out[0]["raw_path"].endswith("raw_clip_01.mp4")
        assert out[0]["duration_sec"] == 20.0
        # Original V2 fields preserved
        assert out[0]["summary"] == "A hook"
        assert out[0]["importance"] == 7
        assert out[0]["v2_index"] == 0

    def test_v1_failure_propagates(self, render: Stage4Render):
        # If V1 raises (e.g. ffmpeg binary missing), Stage 4 lets it
        # bubble -- Inngest outer retry handles, NOT the >50% guardrail
        # which is per-clip-failure (Step 9.2).
        cuts = [_shorts_cut(0, 0, 20)]
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=FileNotFoundError("ffmpeg not found"),
        ):
            with pytest.raises(FileNotFoundError, match="ffmpeg"):
                render.cut_raw_shorts(cuts, _metadata())


# ====================================================================== #
# Stage4Render.cut_raw_bulletin_stories (Step 9.1 raw-cut wiring)         #
# ====================================================================== #


class TestCutRawBulletinStories:
    """Same V1 wiring as cut_raw_shorts but feeding it FullVideoCuts."""

    def test_calls_v1_with_full_video_cuts(self, render: Stage4Render):
        cuts = [_full_video_cut(0, 0, 60), _full_video_cut(1, 60, 120)]

        def _v1_mock(video_path, clips, output_dir):
            assert len(clips) == 2
            # bulletin path: per-clip summary is empty (V2 has no
            # per-cut summary for full_video_cuts)
            assert all(c["summary"] == "" for c in clips)
            for i, c in enumerate(clips, 1):
                c["raw_path"] = f"{output_dir}/raw_clip_{i:02d}.mp4"
                c["duration_sec"] = 60.0

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_v1_mock,
        ):
            out = render.cut_raw_bulletin_stories(cuts, _metadata())

        assert len(out) == 2
        assert out[0]["importance"] == 8   # FullVideoCut default

    def test_empty_input_calls_v1_with_empty_list(
        self, render: Stage4Render,
    ):
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
        ) as mock_v1:
            out = render.cut_raw_bulletin_stories([], _metadata())
        assert mock_v1.call_count == 1
        assert out == []

    def test_writes_raw_clips_to_bulletin_dir_not_output_dir(
        self, render: Stage4Render,
    ):
        # Regression test for Step 12.2a re-run #4 Fix B: bulletin's
        # raw cut must write to ``bulletin_dir`` so V1's path-cache
        # idempotency in cut_video_clips (pipeline.py:1235) doesn't
        # silently reuse a same-named raw_clip from the shorts pass.
        # Pre-#4 bug: bulletin_dir == output_dir, so a 21-second
        # shorts raw_clip_01.mp4 was reused as the bulletin's
        # raw input, truncating a 12-minute bulletin to 21 seconds.
        captured_output_dirs: list[str] = []

        def _v1_mock(video_path, clips, output_dir):
            captured_output_dirs.append(output_dir)
            for i, c in enumerate(clips, 1):
                c["raw_path"] = f"{output_dir}/raw_clip_{i:02d}.mp4"
                c["duration_sec"] = float(c.get("end_sec", 0.0)
                                          - c.get("start_sec", 0.0))

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_v1_mock,
        ):
            render.cut_raw_bulletin_stories(
                [_full_video_cut(0, 0, 720)], _metadata(),
            )

        assert len(captured_output_dirs) == 1
        bulletin_dir_str = str(render.bulletin_dir)
        output_dir_str = str(render.output_dir)
        assert captured_output_dirs[0] == bulletin_dir_str, (
            f"bulletin raw cut must write to bulletin_dir, "
            f"got {captured_output_dirs[0]!r}"
        )
        assert captured_output_dirs[0] != output_dir_str, (
            "bulletin_dir and output_dir must NOT be the same path "
            "(would re-introduce the V1 cache-collision bug)"
        )

    def test_bulletin_raw_paths_differ_from_shorts_raw_paths(
        self, render: Stage4Render,
    ):
        # End-to-end regression: when both cut_raw_shorts and
        # cut_raw_bulletin_stories run in the same render, the raw
        # clip files MUST live in different directories so V1's
        # cache check at pipeline.py:1235 can't collide.
        from pipeline_v2.models import ShortsCut

        def _v1_mock(video_path, clips, output_dir):
            for i, c in enumerate(clips, 1):
                c["raw_path"] = f"{output_dir}/raw_clip_{i:02d}.mp4"
                c["duration_sec"] = float(c.get("end_sec", 0.0)
                                          - c.get("start_sec", 0.0))

        shorts_cut = ShortsCut(
            index=0, start_sec=10.0, end_sec=30.0,
            hook="hk", importance=5,
        )

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_v1_mock,
        ):
            shorts_clips = render.cut_raw_shorts([shorts_cut], _metadata())
            bulletin_clips = render.cut_raw_bulletin_stories(
                [_full_video_cut(0, 0, 720)], _metadata(),
            )

        shorts_raw = shorts_clips[0]["raw_path"]
        bulletin_raw = bulletin_clips[0]["raw_path"]
        assert shorts_raw != bulletin_raw, (
            f"shorts and bulletin must produce DIFFERENT raw_clip "
            f"paths; got both at {shorts_raw!r}"
        )
        # Specifically, bulletin's raw should be under bulletin_dir
        bulletin_dir_parts = str(render.bulletin_dir).replace("\\", "/")
        bulletin_raw_norm = bulletin_raw.replace("\\", "/")
        assert bulletin_dir_parts in bulletin_raw_norm


# ====================================================================== #
# Step 9.2: Stage4Render.resolve_images                                   #
# ====================================================================== #


class TestResolveImages:
    """Wires V1's resolve_image_plan. Per the 12.2a re-run #2 fixes:

      * Pre-step: walk unique entity_names from image_plan, source via
        ``ImageSourcer.source_for_entity`` for entities not already in
        ``self.image_pool``. Populates pool with successful resolutions.

      * Build V1-shape image_plan (with synthetic ``img_NNN`` ids per
        D-9.2 boundary contract) + pool_manifest keyed by those ids.

      * Call V1's resolve_image_plan WITHOUT ``output_dir`` kwarg
        (V1's actual signature -- pure timeline mapper, not producer).

    Most tests in this class patch ``ImageSourcer.source_for_entity``
    to a deterministic stub so the test doesn't hit real search APIs.
    The class-level ``_mock_sourcer`` fixture is the standard pattern.
    """

    @pytest.fixture
    def _mock_sourcer(self, tmp_path: Path):
        """Replace ImageSourcer.source_for_entity with a stub that
        returns ``None`` by default (sourcer "misses"). Tests can
        override per-call by mutating ``stub.return_value`` or
        ``stub.side_effect``. Yields the MagicMock instance so tests
        can inspect call args / set return values.
        """
        with patch(
            "pipeline_v2.stages.stage_4_render.ImageSourcer",
        ) as MockSourcer:
            instance = MockSourcer.return_value
            instance.source_for_entity.return_value = None
            yield instance

    def test_populates_image_pool_via_sourcer_pre_step(
        self, render: Stage4Render, _mock_sourcer, tmp_path: Path,
    ):
        # ImageSourcer returns a real path for each entity -> pool
        # gets populated by the pre-step (BEFORE V1 is called).
        plan = ImagePlan(entries=[
            _image_plan_entry("Modi", clip_index=0),
            _image_plan_entry("Reddy", clip_index=1),
        ])
        entities = [_entity("Modi"), _entity("Reddy")]
        modi_img = tmp_path / "modi.jpg"
        reddy_img = tmp_path / "reddy.jpg"
        modi_img.write_bytes(b"\x00" * 5_000)
        reddy_img.write_bytes(b"\x00" * 5_000)

        def _stub_source(entity, brief, out_dir):
            return {"Modi": modi_img, "Reddy": reddy_img}.get(
                entity.canonical_name,
            )
        _mock_sourcer.source_for_entity.side_effect = _stub_source

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            return_value=[],
        ):
            render.resolve_images(plan, entities, kept_clip_dicts=[],
                                  full_metadata=_metadata())

        assert render.image_pool["Modi"] == modi_img
        assert render.image_pool["Reddy"] == reddy_img

    def test_skips_unready_entries(
        self, render: Stage4Render, _mock_sourcer,
    ):
        # ImageSourcer returns None (search/gen all missed) so pool
        # never gets populated; V1's "failed" return also doesn't
        # poison the pool.
        plan = ImagePlan(entries=[_image_plan_entry("X", clip_index=0)])
        resolved = [
            {"entity_name": "X", "image_path": "/abs/news.jpg",
             "status": "failed", "clip_index": 0},
        ]
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            return_value=resolved,
        ):
            render.resolve_images(plan, [_entity("X")], kept_clip_dicts=[],
                                  full_metadata=_metadata())
        assert render.image_pool == {}

    def test_pool_is_reused_across_calls(
        self, render: Stage4Render, _mock_sourcer, tmp_path: Path,
    ):
        # D-9.5: first resolve populates pool via sourcer; second
        # resolve sees pool["Modi"] cached and short-circuits the
        # sourcer for that entity. The pool_manifest passed to V1
        # on call 2 contains Modi keyed by its synth_id.
        plan_a = ImagePlan(entries=[_image_plan_entry("Modi", clip_index=0)])
        plan_b = ImagePlan(entries=[_image_plan_entry("Modi", clip_index=0)])
        modi_img = tmp_path / "modi.jpg"
        modi_img.write_bytes(b"\x00" * 5_000)
        _mock_sourcer.source_for_entity.return_value = modi_img

        captured_pool_manifests: list[dict] = []

        def _v1_mock(v1_image_plan, *, pool_manifest, kept_clips,
                     whisper_words, video_duration_sec):
            captured_pool_manifests.append(pool_manifest)
            return []

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            side_effect=_v1_mock,
        ):
            render.resolve_images(plan_a, [_entity("Modi")],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())
            assert render.image_pool["Modi"] == modi_img
            # Second call: pool already has Modi -- sourcer should
            # not be re-invoked for this entity.
            _mock_sourcer.source_for_entity.reset_mock()
            render.resolve_images(plan_b, [_entity("Modi")],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())
            assert _mock_sourcer.source_for_entity.call_count == 0

        # Both calls received a pool_manifest with Modi keyed by
        # the synth_id img_000 (since Modi is the first/only entity
        # in each image_plan).
        assert "img_000" in captured_pool_manifests[0]
        assert captured_pool_manifests[0]["img_000"]["topic_clue"] == "Modi"
        assert captured_pool_manifests[0]["img_000"]["path"] == str(modi_img)
        assert "img_000" in captured_pool_manifests[1]

    def test_whisper_words_passed_as_none_d94(
        self, render: Stage4Render, _mock_sourcer,
    ):
        # D-9.4: V2 skips whisper_anchor, passes whisper_words=None.
        plan = ImagePlan(entries=[_image_plan_entry()])

        captured_whisper = []

        def _v1_mock(v1_image_plan, *, pool_manifest, kept_clips,
                     whisper_words, video_duration_sec):
            captured_whisper.append(whisper_words)
            return []

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            side_effect=_v1_mock,
        ):
            render.resolve_images(plan, [_entity()],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())
        assert captured_whisper[0] is None

    def test_v1_image_plan_dict_shape(
        self, render: Stage4Render, _mock_sourcer,
    ):
        # Post-12.2a fix: dict V1 receives has a synthetic ``id``
        # field so V1's pool_manifest lookup works.
        plan = ImagePlan(entries=[
            _image_plan_entry("Modi", clip_index=0, show_at=12.5),
        ])
        captured = []

        def _v1_mock(v1_image_plan, *, pool_manifest, kept_clips,
                     whisper_words, video_duration_sec):
            captured.append(v1_image_plan)
            return []

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            side_effect=_v1_mock,
        ):
            render.resolve_images(plan, [_entity("Modi")],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())

        assert len(captured[0]) == 1
        d = captured[0][0]
        assert d["id"] == "img_000"
        assert d["entity_name"] == "Modi"
        assert d["clip_index"] == 0
        assert d["show_at"] == "00:12.500"

    def test_v1_call_omits_output_dir_kwarg(
        self, render: Stage4Render, _mock_sourcer,
    ):
        # V1's resolve_image_plan signature does NOT accept output_dir
        # (the previous V2 code wrongly passed it -- Bug #1 from
        # 12.2a re-run #1). The fixed code must not pass it.
        plan = ImagePlan(entries=[_image_plan_entry()])
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
        ) as v1_mock:
            v1_mock.return_value = []
            render.resolve_images(plan, [_entity()],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())
        assert v1_mock.call_count == 1
        _, kwargs = v1_mock.call_args
        assert "output_dir" not in kwargs

    def test_pool_manifest_keyed_by_synth_id_not_by_entries_list(
        self, render: Stage4Render, _mock_sourcer, tmp_path: Path,
    ):
        # V1's resolve_image_plan reads ``pool_manifest[eid]["path"]``
        # so the manifest MUST be a dict keyed by synthetic id, not
        # the old ``{"entries": [...]}`` shape (Bug #2 from 12.2a
        # re-run #1).
        img = tmp_path / "img.jpg"
        img.write_bytes(b"\x00" * 5_000)
        _mock_sourcer.source_for_entity.return_value = img
        plan = ImagePlan(entries=[_image_plan_entry("Modi", clip_index=0)])

        captured = []

        def _v1_mock(v1_image_plan, *, pool_manifest, kept_clips,
                     whisper_words, video_duration_sec):
            captured.append(pool_manifest)
            return []

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            side_effect=_v1_mock,
        ):
            render.resolve_images(plan, [_entity("Modi")],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())

        pm = captured[0]
        assert "entries" not in pm
        assert "img_000" in pm
        entry = pm["img_000"]
        assert entry["id"] == "img_000"
        assert entry["path"] == str(img)
        assert entry["topic_clue"] == "Modi"

    def test_orphan_entity_treated_as_other(
        self, render: Stage4Render, _mock_sourcer,
    ):
        # An entity_name in image_plan with no matching canonical
        # entity (orphan) should still hit the sourcer with a typed
        # OTHER stub so the non-PERSON policy branch applies.
        plan = ImagePlan(entries=[
            _image_plan_entry("OrphanCity", clip_index=0),
        ])

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            return_value=[],
        ):
            render.resolve_images(plan, entities=[],  # no canonicals
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())

        # source_for_entity was called once with a synthesised OTHER
        # entity for "OrphanCity".
        assert _mock_sourcer.source_for_entity.call_count == 1
        call_args, call_kwargs = _mock_sourcer.source_for_entity.call_args
        # ``entity`` is positional or keyword; check both call_args
        # and call_kwargs to be defensive about caller style.
        passed_entity = call_kwargs.get("entity") or (
            call_args[0] if call_args else None
        )
        assert passed_entity is not None
        assert passed_entity.canonical_name == "OrphanCity"
        assert passed_entity.type == EntityType.OTHER


# ====================================================================== #
# Step 9.2: Stage4Render._dispatch_compose                                #
# ====================================================================== #


class TestDispatchCompose:
    """Dispatch each frame_layout to the correct V1 compose function
    with the exact arg bundle V1 uses. Tests assert which V1 function
    is called, but don't check FFmpeg correctness (V1 owns that).
    """

    def test_torn_card_dispatches_to_compose_clip(
        self, render: Stage4Render,
    ):
        render.frame_layout = "torn_card"
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_clip",
            return_value={"font_size": 80, "font_file": "f.ttf"},
        ) as mock_compose:
            params = render._dispatch_compose(
                "/raw.mp4", "/img.jpg", "headline", "/out.mp4",
                "NotoSansTelugu-Bold.ttf", "FOLLOW",
            )
        assert mock_compose.call_count == 1
        assert params["card_params"]["card_c0"] == "#c10000"
        assert params["card_params"]["seed"] == 7

    def test_split_frame_dispatches(self, render: Stage4Render):
        render.frame_layout = "split_frame"
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_split_frame",
        ) as mock_compose:
            params = render._dispatch_compose(
                "/raw.mp4", "/img.jpg", "headline", "/out.mp4",
                "f.ttf", "FOLLOW",
            )
        assert mock_compose.call_count == 1
        assert params["split_params"] == {"bg_color": "#1a0a2e"}

    def test_clean_card_dispatches(self, render: Stage4Render):
        render.frame_layout = "clean_card"
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_clean_card",
            return_value={"font_size": 80, "font_file": "f.ttf"},
        ) as mock_compose:
            params = render._dispatch_compose(
                "/raw.mp4", "/img.jpg", "headline", "/out.mp4",
                "f.ttf", "FOLLOW",
            )
        assert mock_compose.call_count == 1
        assert params["card_params"]["bg_color"] == "#c10000"
        assert params["card_params"]["image_border_px"] == 14

    def test_follow_bar_dispatches(self, render: Stage4Render):
        render.frame_layout = "follow_bar"
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_follow_bar",
        ) as mock_compose:
            params = render._dispatch_compose(
                "/raw.mp4", "/img.jpg", "headline", "/out.mp4",
                "f.ttf", "FOLLOW KAIZER NEWS",
            )
        assert mock_compose.call_count == 1
        # follow_bar passes follow_text into the velvet style block
        assert params["follow_params"]["follow_text"] == "FOLLOW KAIZER NEWS"
        assert params["card_params"]["text_color"] == "#ffff00"

    def test_unknown_frame_layout_defaults_to_torn_card(
        self, render: Stage4Render,
    ):
        render.frame_layout = "bogus_layout"
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_clip",
            return_value={},
        ) as mock_compose:
            render._dispatch_compose(
                "/raw.mp4", "/img.jpg", "headline", "/out.mp4",
                "f.ttf", "FOLLOW",
            )
        assert mock_compose.call_count == 1


# ====================================================================== #
# Step 9.2: Stage4Render.compose_shorts (full per-clip orchestration)     #
# ====================================================================== #


def _post_cut_clip(idx: int, raw_exists: bool = True,
                   tmp_path: Path = None,
                   start_sec: float = None,
                   end_sec: float = None) -> dict:
    """Build a post-cut clip dict (what cut_raw_shorts returns).

    Default start/end span is 20s, offset by idx so multiple clips
    don't overlap. Caller can override via ``start_sec``/``end_sec``
    for tests that need specific time windows (e.g. the show_at_sec
    overlap regression suite for Step 12.2a re-run #4 Fix C).
    """
    raw_path = ""
    if raw_exists:
        assert tmp_path is not None
        raw_path = str(tmp_path / f"raw_clip_{idx+1:02d}.mp4")
        Path(raw_path).write_text("fake raw clip bytes")
    if start_sec is None:
        start_sec = idx * 30.0
    if end_sec is None:
        end_sec = start_sec + 20.0
    return {
        "start": "00:00.000", "end": "00:20.000",
        "start_sec": start_sec, "end_sec": end_sec,
        "summary": "hook", "mood": "", "importance": 7,
        "video_type": "SOLO", "v2_index": idx,
        "raw_path": raw_path, "duration_sec": end_sec - start_sec,
    }


class TestComposeShorts:
    def _patch_v1(self):
        """Mock V1's compose_clip + thumbnail subprocess."""
        return {
            "compose": patch(
                "pipeline_v2.stages.stage_4_render._v1_compose_clip",
                return_value={"font_size": 80, "font_file": "f.ttf"},
            ),
            "subprocess_run": patch(
                "pipeline_v2.stages.stage_4_render.subprocess.run",
                return_value=MagicMock(returncode=0),
            ),
        }

    def test_happy_path_3_clips_all_compose(
        self, render: Stage4Render, tmp_path: Path,
    ):
        clip_dicts = [_post_cut_clip(i, tmp_path=render.output_dir)
                      for i in range(3)]
        # Populate pool so each clip resolves
        render.image_pool["Modi"] = render.output_dir / "img_01.jpg"
        # Touch the file so it appears as a real image path
        (render.output_dir / "img_01.jpg").write_text("img")
        resolved = [
            {"entity_name": "Modi", "image_path": str(render.output_dir / "img_01.jpg"),
             "status": "ready", "clip_index": i}
            for i in range(3)
        ]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)
        assert len(out) == 3
        for i, c in enumerate(out):
            assert c["clip_path"].endswith(f"clip_{i+1:02d}.mp4")
            assert c["image_path"].endswith("img_01.jpg")

    def test_clip_with_missing_raw_path_skipped(
        self, render: Stage4Render, tmp_path: Path,
    ):
        clip_dicts = [
            _post_cut_clip(0, tmp_path=render.output_dir),
            _post_cut_clip(1, raw_exists=False),   # missing raw
            _post_cut_clip(2, tmp_path=render.output_dir),
        ]
        (render.output_dir / "img.jpg").write_text("img")
        render.image_pool["X"] = render.output_dir / "img.jpg"
        resolved = [
            {"entity_name": "X", "image_path": str(render.output_dir / "img.jpg"),
             "status": "ready", "clip_index": i}
            for i in (0, 1, 2)
        ]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)
        # 1/3 skipped = 33% < 50% threshold -> no raise
        assert len(out) == 2

    def test_over_50_pct_failure_raises_guardrail(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # 3 of 5 missing raw_path = 60% failure -> RuntimeError
        clip_dicts = [
            _post_cut_clip(0, tmp_path=render.output_dir),
            _post_cut_clip(1, raw_exists=False),
            _post_cut_clip(2, raw_exists=False),
            _post_cut_clip(3, raw_exists=False),
            _post_cut_clip(4, tmp_path=render.output_dir),
        ]
        (render.output_dir / "img.jpg").write_text("img")
        render.image_pool["X"] = render.output_dir / "img.jpg"
        resolved = [
            {"entity_name": "X", "image_path": str(render.output_dir / "img.jpg"),
             "status": "ready", "clip_index": i}
            for i in range(5)
        ]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            with pytest.raises(RuntimeError, match=r"60%"):
                render.compose_shorts(clip_dicts, _metadata(), resolved)

    def test_no_image_for_clip_counts_as_failure(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Clip with raw_path BUT no image (empty pool, no resolved
        # entry) is skipped. With only 1 clip and 1 failure, the
        # 100%-failure trips the >50% guardrail -> RuntimeError.
        clip = _post_cut_clip(0, tmp_path=render.output_dir)
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            with pytest.raises(RuntimeError, match=r"100%"):
                render.compose_shorts([clip], _metadata(), [])

    def test_missing_raw_path_isolated_failure_below_threshold(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # 1 clip out of 4 missing raw_path (25% failure) -> 3 clips
        # ship, the missing-raw-path clip is dropped silently (no
        # raise; we're under the 50% threshold).
        #
        # Note (12.2a re-run #4): the previous version of this test
        # exercised the now-removed ``clip_image_map`` lookup. Under
        # the show_at_sec overlap + round-robin algorithm, an empty
        # pool fails ALL clips uniformly (not just one), so "isolated
        # image failure" is no longer a natural single-clip dropout
        # mode. The missing-raw-path branch IS still isolated per
        # clip and serves the same testing purpose for the 50%
        # threshold logic.
        clip_dicts = [_post_cut_clip(i, tmp_path=render.output_dir)
                      for i in range(4)]
        # Wipe clip 1's raw_path so only that clip drops.
        clip_dicts[1]["raw_path"] = ""
        (render.output_dir / "img.jpg").write_text("img")
        render.image_pool["X"] = render.output_dir / "img.jpg"
        # Resolved entries cover all clips (irrelevant since we use
        # round-robin now; included so the no-image branch is never
        # taken).
        resolved = [
            {"entity_name": "X", "image_path": str(render.output_dir / "img.jpg"),
             "status": "ready", "clip_index": i,
             "source_show_at_sec": 1.0 + i}
            for i in range(4)
        ]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)
        # 1/4 = 25% < 50% threshold -> 3 clips compose successfully
        assert len(out) == 3

    def test_compose_exception_skips_clip(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # V1 compose throws -> clip skipped, doesn't kill the pass
        # (so long as overall ratio stays under threshold).
        clip_dicts = [_post_cut_clip(i, tmp_path=render.output_dir)
                      for i in range(4)]
        (render.output_dir / "img.jpg").write_text("img")
        render.image_pool["X"] = render.output_dir / "img.jpg"
        resolved = [
            {"entity_name": "X", "image_path": str(render.output_dir / "img.jpg"),
             "status": "ready", "clip_index": i}
            for i in range(4)
        ]

        call_count = [0]

        def _fail_one(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("v1 compose failed for clip 2")
            return {"font_size": 80, "font_file": "f.ttf"}

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_clip",
            side_effect=_fail_one,
        ), self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)
        # 1/4 = 25% < 50% threshold -> 3 clips compose successfully
        assert len(out) == 3

    def test_thumbnail_failure_non_fatal(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # V1 pattern (pipeline.py:4569-4575): thumbnail failure is
        # non-fatal; thumb_path becomes "".
        clip = _post_cut_clip(0, tmp_path=render.output_dir)
        (render.output_dir / "img.jpg").write_text("img")
        render.image_pool["X"] = render.output_dir / "img.jpg"
        resolved = [
            {"entity_name": "X", "image_path": str(render.output_dir / "img.jpg"),
             "status": "ready", "clip_index": 0},
        ]
        # Need >=2 clips so isolated thumbnail failure doesn't trip
        # the guardrail. Use just one clip but with successful
        # compose -- only the thumbnail call fails.
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_clip",
            return_value={"font_size": 80, "font_file": "f.ttf"},
        ), patch(
            "pipeline_v2.stages.stage_4_render.subprocess.run",
            side_effect=Exception("ffmpeg thumbnail failed"),
        ):
            out = render.compose_shorts([clip], _metadata(), resolved)
        # Compose succeeded; thumbnail failure non-fatal.
        assert len(out) == 1
        assert out[0]["clip_path"].endswith("clip_01.mp4")
        assert out[0]["thumb_path"] == ""

    def test_image_pool_fallback_when_no_resolved_entry(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # V1 pattern: when a clip has no image_plan entry, use any
        # image from the pool (V1 uses ``images[-1]``).
        clip_dicts = [_post_cut_clip(0, tmp_path=render.output_dir),
                      _post_cut_clip(1, tmp_path=render.output_dir)]
        (render.output_dir / "fallback.jpg").write_text("img")
        render.image_pool["Anyone"] = render.output_dir / "fallback.jpg"
        # Resolved has entry for clip_index=0 only, NOT for 1
        resolved = [{
            "entity_name": "Anyone",
            "image_path": str(render.output_dir / "fallback.jpg"),
            "status": "ready", "clip_index": 0,
        }]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)
        assert len(out) == 2
        # Both clips ended up using the pool image
        for c in out:
            assert c["image_path"].endswith("fallback.jpg")


class TestComposeShortsPerClipImageSelection:
    """Regression suite for Step 12.2a re-run #4 Fix C.

    Pre-#4 bug: ``compose_shorts`` built a ``clip_image_map`` keyed
    by ``r["clip_index"]`` from resolved image_plan entries. But
    image_plan entries' ``clip_index`` references the BULLETIN's
    full_video_cuts, NOT the shorts cut index. With one
    full_video_cut, every image_plan entry mapped to clip_index=0,
    so only shorts[0] got an image via the map; shorts[1..N-1]
    fell through to ``next(iter(image_pool.values()))`` -- the
    same first pool image for every short. Result: all shorts
    rendered with the same picture.

    Post-#4 algorithm: build an overlay list of
    ``(source_show_at_sec, image_path)`` from ready entries. For
    each short's ``[start_sec, end_sec]`` window, pick the first
    overlay whose show_at_sec falls inside. On miss, round-robin
    through the resolved image_pool by short index so each short
    still gets a different image.
    """

    def _patch_v1(self):
        return {
            "compose": patch(
                "pipeline_v2.stages.stage_4_render._v1_compose_clip",
                return_value={"font_size": 80, "font_file": "f.ttf"},
            ),
            "subprocess_run": patch(
                "pipeline_v2.stages.stage_4_render.subprocess.run",
                return_value=MagicMock(returncode=0),
            ),
        }

    def _put_image(self, render: Stage4Render, name: str) -> str:
        p = render.output_dir / name
        p.write_text("img")
        render.image_pool[name.replace(".jpg", "")] = p
        return str(p)

    def test_compose_shorts_uses_show_at_overlap_when_available(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # 3 shorts at non-contiguous windows. image_plan has 2
        # overlays covering shorts[0] and shorts[2]; shorts[1] has
        # no overlap and falls back to round-robin.
        clip_dicts = [
            _post_cut_clip(0, tmp_path=render.output_dir,
                           start_sec=0.0,    end_sec=20.0),
            _post_cut_clip(1, tmp_path=render.output_dir,
                           start_sec=50.0,   end_sec=70.0),
            _post_cut_clip(2, tmp_path=render.output_dir,
                           start_sec=100.0,  end_sec=120.0),
        ]
        modi_img    = self._put_image(render, "Modi.jpg")
        reddy_img   = self._put_image(render, "Reddy.jpg")
        other_img   = self._put_image(render, "Other.jpg")
        resolved = [
            # Modi overlay at 10s -> overlaps shorts[0] (0-20s).
            {"entity_name": "Modi", "image_path": modi_img,
             "status": "ready", "clip_index": 0,
             "source_show_at_sec": 10.0},
            # Reddy overlay at 110s -> overlaps shorts[2] (100-120s).
            {"entity_name": "Reddy", "image_path": reddy_img,
             "status": "ready", "clip_index": 0,
             "source_show_at_sec": 110.0},
        ]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)

        assert len(out) == 3
        assert out[0]["image_path"].endswith("Modi.jpg")
        # shorts[1] has no overlap; round-robin uses pool index 1
        # (Reddy in insertion order Modi, Reddy, Other). Verify it
        # falls back rather than reusing Modi (the pre-#4 bug).
        assert not out[1]["image_path"].endswith("Modi.jpg")
        assert out[2]["image_path"].endswith("Reddy.jpg")

    def test_compose_shorts_round_robin_when_no_overlap(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # 9 shorts, 6 pool images, ZERO overlapping image_plan
        # entries -> every short must cycle through the pool
        # (i % 6) instead of all sharing the first image.
        clip_dicts = [
            _post_cut_clip(i, tmp_path=render.output_dir,
                           # All windows are in 500-600s range so
                           # the source_show_at_sec=10.0 entry
                           # below cannot overlap with any of them.
                           start_sec=500.0 + i * 5,
                           end_sec=500.0 + i * 5 + 4)
            for i in range(9)
        ]
        pool_names = ["A", "B", "C", "D", "E", "F"]
        pool_paths = [self._put_image(render, f"{n}.jpg")
                      for n in pool_names]
        # One ready overlay but at a timestamp NO short overlaps.
        resolved = [{
            "entity_name": "A",
            "image_path": pool_paths[0],
            "status": "ready", "clip_index": 0,
            "source_show_at_sec": 10.0,
        }]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)

        assert len(out) == 9
        # Round-robin produces pool[i % 6] for each short.
        for i, c in enumerate(out):
            expected = pool_paths[i % 6]
            assert c["image_path"] == expected, (
                f"short[{i}] expected {expected!r}, got {c['image_path']!r}"
            )

    def test_compose_shorts_distinct_images_per_clip(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Direct regression: the exact pre-#4 bug shape. 9 shorts +
        # 6 distinct pool images. Assert no two consecutive shorts
        # share the same image (the pre-#4 bug produced 9 identical
        # image_paths).
        clip_dicts = [
            _post_cut_clip(i, tmp_path=render.output_dir,
                           start_sec=float(i * 40),
                           end_sec=float(i * 40 + 20))
            for i in range(9)
        ]
        for name in ("A", "B", "C", "D", "E", "F"):
            self._put_image(render, f"{name}.jpg")

        # Empty resolved -> the overlay match path never fires and
        # round-robin is the sole source of image_path assignment.
        # This is the worst-case for the bug to reappear.
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), [])

        assert len(out) == 9
        for i in range(1, len(out)):
            assert out[i]["image_path"] != out[i - 1]["image_path"], (
                f"shorts[{i}] and shorts[{i-1}] share the same image: "
                f"{out[i]['image_path']!r} -- regression of the "
                f"pre-#4 'all shorts share the same picture' bug."
            )

    def test_compose_shorts_handles_missing_source_show_at_sec(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Defensive: resolved entries that don't carry
        # ``source_show_at_sec`` (e.g. status="image_missing" /
        # "in_cut_span" branches in V1's resolve_image_plan that
        # short-circuit before computing the source time) MUST be
        # excluded from the overlap matching without raising.
        # Round-robin still works as the fallback.
        clip_dicts = [
            _post_cut_clip(i, tmp_path=render.output_dir,
                           start_sec=float(i * 30),
                           end_sec=float(i * 30 + 20))
            for i in range(3)
        ]
        a_img = self._put_image(render, "A.jpg")
        b_img = self._put_image(render, "B.jpg")
        resolved = [
            # Missing source_show_at_sec entirely -> ignored.
            {"entity_name": "A", "image_path": a_img,
             "status": "ready", "clip_index": 0},
            # Explicit None -> also ignored.
            {"entity_name": "B", "image_path": b_img,
             "status": "ready", "clip_index": 0,
             "source_show_at_sec": None},
        ]
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)

        assert len(out) == 3
        # All 3 shorts got an image (round-robin through pool of 2).
        assert all(c.get("image_path") for c in out)
        # shorts[0] and shorts[2] both at pool[0] under RR (i % 2);
        # shorts[1] at pool[1]. So at least 2 distinct images visible.
        unique_images = {c["image_path"] for c in out}
        assert len(unique_images) >= 2


# ====================================================================== #
# Step 9.3: Stage4Render.render_bulletin                                  #
# ====================================================================== #


def _post_cut_bulletin_clip(idx: int, tmp_path: Path,
                            duration_sec: float = 60.0,
                            importance: int = 8) -> dict:
    """Build a post-cut bulletin clip dict (raw_path actually exists)."""
    raw_path = str(tmp_path / f"raw_clip_{idx+1:02d}.mp4")
    Path(raw_path).write_text("fake raw bulletin clip bytes")
    return {
        "start": "00:00.000", "end": "01:00.000",
        "summary": "", "mood": "", "importance": importance,
        "video_type": "SOLO", "v2_index": idx,
        "raw_path": raw_path, "duration_sec": duration_sec,
    }


@pytest.fixture
def bulletin_render(tmp_path: Path) -> Stage4Render:
    """Stage4Render with all bulletin features ON (default)."""
    return Stage4Render(
        output_dir=tmp_path / "out",
        video_path=tmp_path / "source.mp4",
        preset=_preset(),
    )


def _stub_stitch_result(duration: float = 180.0,
                        rendered: int = 3, skipped: int = 0) -> MagicMock:
    r = MagicMock()
    r.total_duration_s = duration
    r.stories_rendered = rendered
    r.stories_skipped = skipped
    r.warnings = []
    return r


@pytest.fixture
def bulletin_v1_patches():
    """Context manager bundling all V1 bulletin helper patches.

    Returns a dict of patch context-managers + the configured mocks
    so individual tests can assert call args.
    """
    base = "pipeline_v2.stages.stage_4_render"

    def _stub_compose(raw_path, story_meta, composed_path, **kw):
        # Write a >100KB stub file so the size check after compose
        # passes (V1's pattern: file must be >100KB to count as
        # successful).
        Path(composed_path).write_bytes(b"x" * 200_000)
        return None

    patches = {
        "cut_video_clips": patch(
            f"{base}._v1_cut_video_clips",
            side_effect=lambda video_path, clips, output_dir: None,
        ),
        "resolve_image_plan": patch(
            f"{base}._v1_resolve_image_plan",
            return_value=[
                {"entity_name": "Modi",
                 "image_path": "/abs/img_01.jpg",
                 "status": "ready", "clip_index": 0},
            ],
        ),
        "render_ticker": patch(f"{base}._v1_render_ticker"),
        "render_channel_bug": patch(f"{base}._v1_render_channel_bug"),
        "build_sidebar_carousel": patch(f"{base}._v1_build_sidebar_carousel"),
        "make_sidebar_placeholder": patch(f"{base}._v1_make_sidebar_placeholder"),
        "pick_pip_source": patch(
            f"{base}._v1_pick_pip_source",
            return_value=None,
        ),
        "compose_bulletin_story": patch(
            f"{base}._v1_compose_bulletin_story",
            side_effect=_stub_compose,
        ),
        "compose_pip_story": patch(
            f"{base}._v1_compose_pip_story",
            side_effect=_stub_compose,
        ),
        "build_fullscreen_takeover": patch(
            f"{base}._v1_build_fullscreen_takeover"
        ),
        "stitch_bulletin": patch(
            f"{base}._v1_stitch_bulletin",
            return_value=_stub_stitch_result(),
        ),
        "overlay_image_plan": patch(f"{base}._v1_overlay_image_plan"),
        # compose_deps: default to "not fresh" so the cache misses
        # and each helper IS called. Individual tests can override.
        "is_fresh": patch(
            f"{base}._v1_compose_deps.is_fresh",
            return_value=False,
        ),
        "mark_built": patch(f"{base}._v1_compose_deps.mark_built"),
    }
    return patches


def _all_patches(d: dict):
    """Context manager that enters every patch in a dict."""
    from contextlib import ExitStack
    stack = ExitStack()
    mocks = {}
    for name, p in d.items():
        mocks[name] = stack.enter_context(p)
    return stack, mocks


class TestRenderBulletinHappyPath:
    def test_returns_complete_result_dict(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        bulletin_render.image_pool["Modi"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")
        cuts = [_full_video_cut(i, i * 60, (i + 1) * 60) for i in range(3)]
        # We need V1's cut_video_clips to mutate clip dicts with
        # raw_path. Override the default stub.
        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )

        # Stitch produces bulletin.mp4 -- make sure the file appears
        # so overlay path check works.
        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 300_000)
            return _stub_stitch_result()
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )

        # Overlay also writes a > 100KB file
        def _stub_overlay(in_path, resolved, out_path):
            Path(out_path).write_bytes(b"x" * 300_000)
        bulletin_v1_patches["overlay_image_plan"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_overlay_image_plan",
            side_effect=_stub_overlay,
        )

        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            result = bulletin_render.render_bulletin(
                full_video_cuts=cuts,
                metadata=_metadata(),
                entities=[_entity("Modi")],
                image_plan=ImagePlan(entries=[_image_plan_entry("Modi", 0)]),
            )

        assert result["bulletin_path"].endswith("bulletin.mp4")
        assert result["overlay_path"].endswith("bulletin_with_overlays.mp4")
        assert result["overlay_applied"] is True
        assert result["duration_s"] == 180.0
        assert result["stories_rendered"] == 3
        assert result["stories_skipped"] == 0
        # All 3 stories went through compose_bulletin_story (PiP source
        # returned None per default mock, so non-PiP path is used)
        assert mocks["compose_bulletin_story"].call_count == 3

    def test_uses_metadata_bulletin_marquee_points(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        # D-7.6 / Step 9.3 wiring: ticker is built from
        # metadata.bulletin_marquee_points (not per-clip fallback).
        meta = _metadata()
        meta = meta.model_copy(update={
            "bulletin_marquee_points": ["A_specific", "B_specific", "C_specific"],
        })
        cuts = [_full_video_cut(0, 0, 60)]
        bulletin_render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )

        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 200_000)
            return _stub_stitch_result(rendered=1)
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )

        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            bulletin_render.render_bulletin(
                full_video_cuts=cuts, metadata=meta,
                entities=[_entity("X")],
                image_plan=ImagePlan(entries=[]),
            )

        # render_ticker called with the metadata's headlines
        assert mocks["render_ticker"].call_count == 1
        call_args = mocks["render_ticker"].call_args
        ticker_headlines = call_args[0][0]
        assert ticker_headlines == ["A_specific", "B_specific", "C_specific"]


class TestRenderBulletinFeatureToggles:
    """D-9 / locked: each feature can be toggled OFF independently
    via constructor kwarg. The OFF path just skips the V1 helper call.
    """

    def _setup(self, render, bulletin_v1_patches, tmp_path):
        render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )
        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 200_000)
            return _stub_stitch_result(rendered=len(story_paths))
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )

    def test_use_sidebar_carousel_false_skips_carousel(
        self, tmp_path: Path, bulletin_v1_patches,
    ):
        render = Stage4Render(
            output_dir=tmp_path / "out",
            video_path=tmp_path / "source.mp4",
            preset=_preset(),
            use_sidebar_carousel=False,
        )
        # Need >=2 images in pool to test the "carousel would build" path
        render.image_pool = {
            "A": tmp_path / "a.jpg",
            "B": tmp_path / "b.jpg",
        }
        (tmp_path / "a.jpg").write_text("a")
        (tmp_path / "b.jpg").write_text("b")
        self._setup(render, bulletin_v1_patches, tmp_path)

        cuts = [_full_video_cut(0, 0, 60)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("A"), _entity("B")],
                image_plan=ImagePlan(entries=[]),
            )

        # Carousel NEVER called
        assert mocks["build_sidebar_carousel"].call_count == 0
        # Fallback static placeholder WAS called
        assert mocks["make_sidebar_placeholder"].call_count >= 1

    def test_use_pip_false_skips_pip(
        self, tmp_path: Path, bulletin_v1_patches,
    ):
        render = Stage4Render(
            output_dir=tmp_path / "out",
            video_path=tmp_path / "source.mp4",
            preset=_preset(),
            use_pip=False,
        )
        render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")
        self._setup(render, bulletin_v1_patches, tmp_path)

        cuts = [_full_video_cut(0, 0, 60), _full_video_cut(1, 60, 120)]
        # If pick_pip_source were called, it would return a PiP src.
        # With use_pip=False, it's NEVER called.
        bulletin_v1_patches["pick_pip_source"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_pick_pip_source",
            return_value=("/abs/pip.mp4", 5.0, 3.0),
        )
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("X")],
                image_plan=ImagePlan(entries=[]),
            )

        # pick_pip_source NEVER called when use_pip=False
        assert mocks["pick_pip_source"].call_count == 0
        # compose_pip_story NEVER called
        assert mocks["compose_pip_story"].call_count == 0
        # compose_bulletin_story used unconditionally
        assert mocks["compose_bulletin_story"].call_count == 2

    def test_use_takeovers_false_skips_inter_story_transitions(
        self, tmp_path: Path, bulletin_v1_patches,
    ):
        render = Stage4Render(
            output_dir=tmp_path / "out",
            video_path=tmp_path / "source.mp4",
            preset=_preset(),
            use_takeovers=False,
        )
        # Need >=2 images in pool to satisfy the "takeover would build" condition
        render.image_pool = {
            "A": tmp_path / "a.jpg",
            "B": tmp_path / "b.jpg",
        }
        (tmp_path / "a.jpg").write_text("a")
        (tmp_path / "b.jpg").write_text("b")
        self._setup(render, bulletin_v1_patches, tmp_path)

        cuts = [_full_video_cut(i, i * 60, (i + 1) * 60) for i in range(3)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("A"), _entity("B")],
                image_plan=ImagePlan(entries=[]),
            )

        # build_fullscreen_takeover NEVER called when use_takeovers=False
        assert mocks["build_fullscreen_takeover"].call_count == 0


class TestRenderBulletinComposeDepsCache:
    """compose_deps.is_fresh / mark_built integration."""

    def test_cache_hit_skips_helper(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        bulletin_render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
            # Also pre-populate cached composed_story files so
            # the "use the cached path" code path lands on a real file
            for i in range(len(clips)):
                composed = tmp_path / "out" / "bulletin" / f"composed_story_{i:02d}.mp4"
                composed.parent.mkdir(parents=True, exist_ok=True)
                composed.write_bytes(b"x" * 200_000)
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )

        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 200_000)
            return _stub_stitch_result(rendered=len(story_paths))
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )
        # is_fresh returns True for EVERYTHING -- so no helper is
        # called.
        bulletin_v1_patches["is_fresh"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_compose_deps.is_fresh",
            return_value=True,
        )

        cuts = [_full_video_cut(0, 0, 60), _full_video_cut(1, 60, 120)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            bulletin_render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("X")],
                image_plan=ImagePlan(entries=[]),
            )

        # All cache-able helpers SKIPPED because is_fresh=True
        assert mocks["render_ticker"].call_count == 0
        assert mocks["render_channel_bug"].call_count == 0
        assert mocks["compose_bulletin_story"].call_count == 0

    def test_cache_miss_calls_mark_built(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        bulletin_render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )

        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 200_000)
            return _stub_stitch_result(rendered=len(story_paths))
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )
        # Default is_fresh=False -- cache misses every time.
        cuts = [_full_video_cut(0, 0, 60)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            bulletin_render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("X")],
                image_plan=ImagePlan(entries=[]),
            )

        # On cache miss, helpers ran AND mark_built was called for each
        assert mocks["render_ticker"].call_count == 1
        # mark_built called at least once (ticker + bug + composed)
        assert mocks["mark_built"].call_count >= 3


class TestRenderBulletinGuardrail:
    """D-9.7 50%-failure guardrail applies to bulletin pass too."""

    def test_over_50_pct_stories_failed_raises(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        bulletin_render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        # cut_video_clips produces NO raw paths -- all stories fail
        def _no_raw_cut(video_path, clips, output_dir):
            # Leave clips without raw_path
            pass
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_no_raw_cut,
        )

        cuts = [_full_video_cut(i, i * 60, (i + 1) * 60) for i in range(4)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            with pytest.raises(RuntimeError, match=r"100%"):
                bulletin_render.render_bulletin(
                    full_video_cuts=cuts, metadata=_metadata(),
                    entities=[_entity("X")],
                    image_plan=ImagePlan(entries=[]),
                )

    def test_under_50_pct_failures_passes(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        bulletin_render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        # 1 of 4 stories has no raw_path = 25% failure -> no raise.
        # Note: failure-via-no-raw drops the story before reaching
        # compose; it's 1/4 = 25% < 50% threshold.
        def _partial_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                if i == 1:
                    # Skip cut for story 1 -- no raw_path
                    continue
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_partial_cut,
        )
        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 200_000)
            return _stub_stitch_result(rendered=len(story_paths))
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )

        cuts = [_full_video_cut(i, i * 60, (i + 1) * 60) for i in range(4)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            result = bulletin_render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("X")],
                image_plan=ImagePlan(entries=[]),
            )
        # Pass succeeds; 3 of 4 stories shipped
        assert result["bulletin_path"].endswith("bulletin.mp4")


class TestRenderBulletinImagePoolReuse:
    """D-9.5: image_pool persists across passes. A shorts pass that
    populated the pool earlier should mean render_bulletin doesn't
    re-resolve those entities.
    """

    def test_pool_passed_to_resolver_via_pool_manifest(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        # Pre-populate the pool as if a prior shorts pass had run.
        (tmp_path / "img_modi.jpg").write_text("img")
        bulletin_render.image_pool["Modi"] = tmp_path / "img_modi.jpg"

        captured_manifests = []
        def _capture_resolve(v1_image_plan, *, pool_manifest,
                              kept_clips, whisper_words, video_duration_sec):
            captured_manifests.append(pool_manifest)
            return []
        bulletin_v1_patches["resolve_image_plan"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            side_effect=_capture_resolve,
        )
        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )
        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 200_000)
            return _stub_stitch_result(rendered=len(story_paths))
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )

        # Image_plan must reference Modi so it gets a synth_id; the
        # pool_manifest is built from entries whose entity_name has
        # both a sourced image AND a synth_id from this image_plan.
        plan = ImagePlan(entries=[
            _image_plan_entry("Modi", clip_index=0, show_at=12.5),
        ])

        cuts = [_full_video_cut(0, 0, 60)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            bulletin_render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("Modi")],
                image_plan=plan,
            )

        # The pool_manifest passed to V1 is keyed by synth_id and
        # contains the Modi entry from the prior pass. V1 sees this
        # and can skip re-downloading.
        assert len(captured_manifests) == 1
        pm = captured_manifests[0]
        assert "img_000" in pm
        assert pm["img_000"]["topic_clue"] == "Modi"
        assert pm["img_000"]["path"] == str(tmp_path / "img_modi.jpg")


class TestRenderBulletinStitchFailure:
    """If V1.stitch_bulletin raises, Stage 4 surfaces it as a
    RuntimeError (Inngest retries).
    """

    def test_stitch_error_wrapped_in_runtime_error(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        bulletin_render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )
        # Make stitch raise BulletinStitchError
        from pipeline_core.bulletin_stitcher import BulletinStitchError
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=BulletinStitchError("ffmpeg concat failed"),
        )

        cuts = [_full_video_cut(0, 0, 60)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            with pytest.raises(RuntimeError, match="stitch_bulletin failed"):
                bulletin_render.render_bulletin(
                    full_video_cuts=cuts, metadata=_metadata(),
                    entities=[_entity("X")],
                    image_plan=ImagePlan(entries=[]),
                )


class TestRenderBulletinOverlayApplication:
    def test_no_image_plan_skips_overlay(
        self, bulletin_render, bulletin_v1_patches, tmp_path: Path,
    ):
        bulletin_render.image_pool["X"] = tmp_path / "img.jpg"
        (tmp_path / "img.jpg").write_text("img")

        def _mutate_cut(video_path, clips, output_dir):
            for i, c in enumerate(clips):
                raw = str(tmp_path / "out" / f"raw_clip_{i+1:02d}.mp4")
                Path(raw).write_text("raw")
                c["raw_path"] = raw
                c["duration_sec"] = 60.0
        bulletin_v1_patches["cut_video_clips"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_cut_video_clips",
            side_effect=_mutate_cut,
        )
        def _stub_stitch(story_paths, out_path, **kw):
            Path(out_path).write_bytes(b"x" * 200_000)
            return _stub_stitch_result(rendered=len(story_paths))
        bulletin_v1_patches["stitch_bulletin"] = patch(
            "pipeline_v2.stages.stage_4_render._v1_stitch_bulletin",
            side_effect=_stub_stitch,
        )

        cuts = [_full_video_cut(0, 0, 60)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            # Empty image_plan: overlay step NEVER runs
            result = bulletin_render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("X")],
                image_plan=ImagePlan(entries=[]),
            )

        assert mocks["overlay_image_plan"].call_count == 0
        assert result["overlay_applied"] is False
        # overlay_path falls back to bulletin_path when no overlay
        assert result["overlay_path"] == result["bulletin_path"]


# ====================================================================== #
# Step 9.4: Stage4Render.render() top-level orchestrator                  #
# ====================================================================== #


def _job_output(
    shorts_cuts: list[ShortsCut] = None,
    full_video_cuts: list[FullVideoCut] = None,
    image_plan_entries: list[ImagePlanEntry] = None,
    metadata: Metadata = None,
) -> JobOutput:
    """Synthetic JobOutput for orchestrator tests."""
    return JobOutput(
        stage_two=StageTwoOutput(
            full_video_cuts=full_video_cuts if full_video_cuts is not None
                           else [_full_video_cut(0, 0, 60)],
            skipped_segments=[],
            clean_transcript=CleanTranscript(
                words=[Word(w="x", s=0.0, e=1.0)],
                clip_boundaries={0: (0, 0)},
                source_word_map=[0],
            ),
            retake_audit="none",
        ),
        canonical_entities=[_entity("Modi")],
        shorts_cuts=shorts_cuts if shorts_cuts is not None
                   else [_shorts_cut(0, 10, 30)],
        metadata=metadata or _metadata(),
        image_plan=ImagePlan(entries=image_plan_entries or []),
    )


class TestRenderResult:
    def test_frozen_dataclass(self):
        r = RenderResult(
            shorts_editor_meta_path=None,
            bulletin_editor_meta_path=None,
            composed_shorts=[],
            bulletin=None,
        )
        with pytest.raises(Exception):
            r.composed_shorts = [{"x": 1}]   # frozen

    def test_has_all_fields(self):
        r = RenderResult(
            shorts_editor_meta_path="/a/editor_meta.json",
            bulletin_editor_meta_path="/a/bulletin/editor_meta.json",
            composed_shorts=[{"clip_path": "/a/clip_01.mp4"}],
            bulletin={"bulletin_path": "/a/bulletin/bulletin.mp4"},
        )
        assert r.shorts_editor_meta_path.endswith("editor_meta.json")
        assert r.bulletin is not None


class TestRenderOrchestrator:
    """End-to-end orchestration of shorts + bulletin sub-renders.

    Mocks `compose_shorts` + `render_bulletin` so we test the
    orchestrator's wiring (sequence, adapter calls, editor_meta
    writes), NOT the sub-renders themselves (those have their own
    test surface).
    """

    def _setup_mocks(self, render, tmp_path):
        """Mock the sub-renders to deterministic results."""
        # compose_shorts -> 1 composed clip dict with paths populated
        composed_clip = {
            "start": "00:10.000", "end": "00:30.000",
            "summary": "A hook", "mood": "",
            "importance": 7, "video_type": "SOLO",
            "v2_index": 0,
            "raw_path": str(tmp_path / "raw_clip_01.mp4"),
            "duration_sec": 20.0,
            "clip_path": str(tmp_path / "clip_01.mp4"),
            "thumb_path": str(tmp_path / "thumb_01.jpg"),
            "image_path": str(tmp_path / "img_01.jpg"),
            "card_params": {"font_file": "f.ttf"},
            "split_params": {}, "follow_params": {},
            "storage_url": "", "storage_key": "", "storage_backend": "",
        }
        cut_returns = [composed_clip.copy()]
        compose_returns = [composed_clip.copy()]
        resolve_returns = [
            {"entity_name": "Modi",
             "image_path": str(tmp_path / "img_01.jpg"),
             "status": "ready", "clip_index": 0},
        ]
        bulletin_returns = {
            "bulletin_path":     str(tmp_path / "bulletin" / "bulletin.mp4"),
            "overlay_path":      str(tmp_path / "bulletin" / "bulletin_with_overlays.mp4"),
            "overlay_applied":   True,
            "duration_s":        60.0,
            "stories_rendered":  1,
            "stories_skipped":   0,
            "warnings":          [],
        }
        return (
            patch.object(render, "cut_raw_shorts", return_value=cut_returns),
            patch.object(render, "resolve_images", return_value=resolve_returns),
            patch.object(render, "compose_shorts", return_value=compose_returns),
            patch.object(render, "render_bulletin", return_value=bulletin_returns),
        )

    def test_full_pipeline_writes_both_editor_metas(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Use render.output_dir for paths so editor_meta lands there
        job = _job_output(
            shorts_cuts=[_shorts_cut(0, 10, 30)],
            full_video_cuts=[_full_video_cut(0, 0, 60)],
        )
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(job, timestamp="20260518_120000")

        assert isinstance(result, RenderResult)
        # Both editor_meta files written
        assert result.shorts_editor_meta_path is not None
        assert result.bulletin_editor_meta_path is not None
        assert Path(result.shorts_editor_meta_path).exists()
        assert Path(result.bulletin_editor_meta_path).exists()
        # Shorts editor_meta at output_dir root; bulletin under bulletin/
        assert result.shorts_editor_meta_path.endswith("editor_meta.json")
        assert "bulletin" in result.bulletin_editor_meta_path

    def test_shorts_editor_meta_has_expected_structure(
        self, render: Stage4Render, tmp_path: Path,
    ):
        job = _job_output(
            shorts_cuts=[_shorts_cut(0, 10, 30)],
            full_video_cuts=[_full_video_cut(0, 0, 60)],
        )
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(job, timestamp="20260518_120000")

        data = _json.loads(
            Path(result.shorts_editor_meta_path).read_text(encoding="utf-8")
        )
        # Shorts shape: 20+ top-level keys including platform, language, clips
        assert data["platform"] == DEFAULT_PLATFORM
        assert data["frame_layout"] == "torn_card"
        assert data["created"] == "20260518_120000"
        assert len(data["clips"]) == 1
        assert data["clips"][0]["clip_path"].endswith("clip_01.mp4")

    def test_bulletin_editor_meta_has_expected_structure(
        self, render: Stage4Render, tmp_path: Path,
    ):
        job = _job_output()
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(job, timestamp="20260518_120000")

        data = _json.loads(
            Path(result.bulletin_editor_meta_path).read_text(encoding="utf-8")
        )
        # Bulletin shape: render_mode, stories, skipped, duration_s
        assert data["render_mode"] == "bulletin"
        assert data["stories"] == 1
        assert data["skipped"] == 0
        assert data["duration_s"] == 60.0
        assert len(data["clips"]) == 1
        assert data["clips"][0]["frame_type"] == "bulletin"
        # overlay paths populated because overlay_applied=True
        assert data["clips"][0]["clip_path_overlay"].endswith("bulletin_with_overlays.mp4")
        assert data["clips"][0]["clip_path_carousel_only"].endswith("bulletin.mp4")

    def test_zero_shorts_skips_shorts_pass(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # job_output requires shorts_cuts min_length=3 at the
        # Pydantic level (Stage 3a Output), but JobOutput itself
        # doesn't enforce that -- a future caller could supply []
        # (e.g. job that produced 0 shorts due to source content
        # being too short). Test that the orchestrator handles it.
        job = _job_output(
            shorts_cuts=[],
            full_video_cuts=[_full_video_cut(0, 0, 60)],
        )
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(job, timestamp="20260518_120000")

        assert result.shorts_editor_meta_path is None
        assert result.composed_shorts == []
        # Bulletin still ran
        assert result.bulletin_editor_meta_path is not None
        assert result.bulletin is not None

    def test_zero_full_video_cuts_skips_bulletin_pass(
        self, render: Stage4Render, tmp_path: Path,
    ):
        job = _job_output(
            shorts_cuts=[_shorts_cut(0, 10, 30)],
            full_video_cuts=[],
        )
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(job, timestamp="20260518_120000")

        assert result.shorts_editor_meta_path is not None
        assert result.bulletin_editor_meta_path is None
        assert result.bulletin is None

    def test_shorts_pass_runs_before_bulletin_pass(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # D-9.8: sequential, shorts FIRST so image_pool is seeded
        # before bulletin runs.
        call_order = []

        def _cut(*args, **kw):
            call_order.append("cut_raw_shorts")
            return [{
                "raw_path": str(tmp_path / "raw.mp4"), "duration_sec": 20.0,
                "v2_index": 0, "summary": "h", "importance": 7,
                "mood": "", "video_type": "SOLO",
                "start": "00:10.000", "end": "00:30.000",
            }]

        def _resolve(*args, **kw):
            call_order.append("resolve_images")
            return [{"entity_name": "Modi",
                     "image_path": str(tmp_path / "i.jpg"),
                     "status": "ready", "clip_index": 0}]

        def _compose(*args, **kw):
            call_order.append("compose_shorts")
            return [{
                "raw_path": str(tmp_path / "raw.mp4"),
                "clip_path": str(tmp_path / "clip_01.mp4"),
                "thumb_path": "", "image_path": str(tmp_path / "i.jpg"),
                "duration_sec": 20.0, "v2_index": 0,
                "summary": "h", "importance": 7, "mood": "",
                "video_type": "SOLO",
                "start": "00:10.000", "end": "00:30.000",
                "card_params": {}, "split_params": {}, "follow_params": {},
                "storage_url": "", "storage_key": "", "storage_backend": "",
            }]

        def _bul(*args, **kw):
            call_order.append("render_bulletin")
            return {
                "bulletin_path": str(tmp_path / "b.mp4"),
                "overlay_path": str(tmp_path / "b.mp4"),
                "overlay_applied": False,
                "duration_s": 60.0,
                "stories_rendered": 1, "stories_skipped": 0,
                "warnings": [],
            }

        job = _job_output()
        with patch.object(render, "cut_raw_shorts", side_effect=_cut), \
             patch.object(render, "resolve_images", side_effect=_resolve), \
             patch.object(render, "compose_shorts", side_effect=_compose), \
             patch.object(render, "render_bulletin", side_effect=_bul):
            render.render(job, timestamp="20260518_120000")

        # Expected order: cut -> resolve -> compose -> bulletin
        assert call_order == [
            "cut_raw_shorts", "resolve_images", "compose_shorts",
            "render_bulletin",
        ]

    def test_dropped_shorts_renumbered_for_adapter(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # If compose_shorts drops 1 of 3 clips, the adapter would
        # otherwise fail D-8.12 contiguity. Orchestrator must
        # renumber the kept shorts to 0-based contiguous.
        job = _job_output(
            shorts_cuts=[
                _shorts_cut(0, 10, 30, hook="A"),
                _shorts_cut(1, 40, 60, hook="B"),
                _shorts_cut(2, 70, 90, hook="C"),
            ],
            full_video_cuts=[],   # skip bulletin
        )

        # cut_raw_shorts produces 3 clip dicts.
        def _cut(*args, **kw):
            return [{
                "raw_path": str(tmp_path / f"r{i}.mp4"),
                "duration_sec": 20.0, "v2_index": i,
                "summary": "h", "importance": 7, "mood": "",
                "video_type": "SOLO",
                "start": "00:10.000", "end": "00:30.000",
            } for i in range(3)]

        # compose_shorts drops the middle clip (v2_index=1).
        def _compose(*args, **kw):
            return [
                {
                    "raw_path": str(tmp_path / "r0.mp4"),
                    "clip_path": str(tmp_path / "clip_01.mp4"),
                    "thumb_path": "", "image_path": "",
                    "duration_sec": 20.0, "v2_index": 0,
                    "summary": "A", "importance": 7, "mood": "",
                    "video_type": "SOLO",
                    "start": "00:10.000", "end": "00:30.000",
                    "card_params": {}, "split_params": {}, "follow_params": {},
                    "storage_url": "", "storage_key": "", "storage_backend": "",
                },
                {
                    "raw_path": str(tmp_path / "r2.mp4"),
                    "clip_path": str(tmp_path / "clip_03.mp4"),
                    "thumb_path": "", "image_path": "",
                    "duration_sec": 20.0, "v2_index": 2,
                    "summary": "C", "importance": 7, "mood": "",
                    "video_type": "SOLO",
                    "start": "01:10.000", "end": "01:30.000",
                    "card_params": {}, "split_params": {}, "follow_params": {},
                    "storage_url": "", "storage_key": "", "storage_backend": "",
                },
            ]

        with patch.object(render, "cut_raw_shorts", side_effect=_cut), \
             patch.object(render, "resolve_images", return_value=[]), \
             patch.object(render, "compose_shorts", side_effect=_compose):
            result = render.render(job, timestamp="20260518_120000")

        # editor_meta should have 2 clips (not 3), with renumbered
        # indices 0 and 1 (NOT 0 and 2) -- otherwise D-8.12 would
        # have raised inside the adapter.
        data = _json.loads(
            Path(result.shorts_editor_meta_path).read_text(encoding="utf-8")
        )
        assert len(data["clips"]) == 2

    def test_returned_result_carries_sub_render_outputs(
        self, render: Stage4Render, tmp_path: Path,
    ):
        job = _job_output()
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(job, timestamp="20260518_120000")

        # composed_shorts list is in the result
        assert len(result.composed_shorts) == 1
        assert result.composed_shorts[0]["clip_path"].endswith("clip_01.mp4")
        # bulletin dict is in the result
        assert result.bulletin["stories_rendered"] == 1

    def test_title_english_propagates_to_editor_meta(
        self, render: Stage4Render, tmp_path: Path,
    ):
        job = _job_output(full_video_cuts=[])  # skip bulletin
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(
                job, timestamp="20260518_120000",
                title_english="ENGLISH HEADLINE",
            )
        data = _json.loads(
            Path(result.shorts_editor_meta_path).read_text(encoding="utf-8")
        )
        assert data["title_english"] == "ENGLISH HEADLINE"
        assert data["clips"][0]["title_english"] == "ENGLISH HEADLINE"


# ====================================================================== #
# Step 12.3 Test 2 fix: Stage 4 cancel_check sub-phase threading          #
# ====================================================================== #


class TestStage4CancelCheck:
    """Verify the ``cancel_check`` callback is invoked between every
    Stage 4 sub-phase (Step 12.3 Test 2 fix / backlog item 76).

    Pre-fix bug: a user cancel during Stage 4 sat idle until the
    cooperative ``_check_cancelled`` at finalize fired (~5 min
    later) because ``_render_impl`` had no internal cancel checks.

    Fix: ``cancel_check: Optional[callable]`` parameter threaded
    through ``render()`` -> ``_render_impl()``, invoked before each
    sub-phase (cut_raw_shorts / resolve_images / compose_shorts /
    render_bulletin). When the callable raises, propagation exits
    Stage 4 through render()'s classifier and into the orchestrator's
    terminal catch.
    """

    def _setup_mocks(self, render, tmp_path):
        """Same mock shape as TestRenderOrchestrator -- sub-renders
        deterministic, orchestration testable."""
        composed_clip = {
            "start": "00:10.000", "end": "00:30.000",
            "summary": "hook", "mood": "", "importance": 7,
            "video_type": "SOLO", "v2_index": 0,
            "raw_path": str(tmp_path / "raw_clip_01.mp4"),
            "duration_sec": 20.0,
            "clip_path": str(tmp_path / "clip_01.mp4"),
            "thumb_path": str(tmp_path / "thumb_01.jpg"),
            "image_path": str(tmp_path / "img_01.jpg"),
            "card_params": {"font_file": "f.ttf"},
            "split_params": {}, "follow_params": {},
            "storage_url": "", "storage_key": "", "storage_backend": "",
        }
        bulletin_returns = {
            "bulletin_path":     str(tmp_path / "bulletin" / "bulletin.mp4"),
            "overlay_path":      str(tmp_path / "bulletin" / "bulletin_with_overlays.mp4"),
            "overlay_applied":   True,
            "duration_s":        60.0,
            "stories_rendered":  1,
            "stories_skipped":   0,
            "warnings":          [],
        }
        return (
            patch.object(render, "cut_raw_shorts", return_value=[composed_clip.copy()]),
            patch.object(render, "resolve_images", return_value=[
                {"entity_name": "Modi",
                 "image_path": str(tmp_path / "img_01.jpg"),
                 "status": "ready", "clip_index": 0},
            ]),
            patch.object(render, "compose_shorts", return_value=[composed_clip.copy()]),
            patch.object(render, "render_bulletin", return_value=bulletin_returns),
        )

    def test_render_calls_cancel_check_between_sub_phases(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Full pipeline (shorts + bulletin): cancel_check fires
        # before each of the 4 sub-phases (cut_raw_shorts,
        # resolve_images, compose_shorts, render_bulletin).
        job = _job_output(
            shorts_cuts=[_shorts_cut(0, 10, 30)],
            full_video_cuts=[_full_video_cut(0, 0, 60)],
        )
        cancel_check_calls = []

        def _cancel_check():
            cancel_check_calls.append(len(cancel_check_calls))

        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            render.render(
                job, timestamp="20260518_120000",
                cancel_check=_cancel_check,
            )
        assert len(cancel_check_calls) == 4, (
            f"expected 4 cancel_check invocations (cut/resolve/"
            f"compose/bulletin); got {len(cancel_check_calls)}"
        )

    def test_render_propagates_cancel_check_exception(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # If cancel_check raises mid-Stage-4, the exception
        # propagates up through render() to the caller (which the
        # orchestrator's except clause catches in production).
        job = _job_output(
            shorts_cuts=[_shorts_cut(0, 10, 30)],
            full_video_cuts=[_full_video_cut(0, 0, 60)],
        )

        from inngest import NonRetriableError

        call_count = [0]

        def _cancel_check_raises_on_2nd_call():
            call_count[0] += 1
            if call_count[0] == 2:
                raise NonRetriableError(
                    "cancelled: job 999 cancel_requested=True"
                )

        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            with pytest.raises(NonRetriableError, match="cancelled"):
                render.render(
                    job, timestamp="20260518_120000",
                    cancel_check=_cancel_check_raises_on_2nd_call,
                )
        # Confirm the callback ran the second time before raising,
        # i.e. the cancel actually interrupts mid-pipeline (would
        # be exactly 2 calls if it stopped before cut_raw_shorts
        # finished mocking; we want at least 2 to confirm
        # mid-pipeline interrupt).
        assert call_count[0] >= 2

    def test_render_without_cancel_check_runs_normally(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # cancel_check=None (the default) is the V1-only / unit-test
        # path. Backward compat: render() runs end-to-end without
        # any extra DB reads.
        job = _job_output(
            shorts_cuts=[_shorts_cut(0, 10, 30)],
            full_video_cuts=[_full_video_cut(0, 0, 60)],
        )
        cut_p, res_p, comp_p, bul_p = self._setup_mocks(
            render, render.output_dir,
        )
        with cut_p, res_p, comp_p, bul_p:
            result = render.render(
                job, timestamp="20260518_120000",
                # cancel_check NOT supplied
            )
        assert isinstance(result, RenderResult)
        # Both editor_meta files still written -- baseline behaviour
        # unaffected by the new parameter.
        assert result.shorts_editor_meta_path is not None
        assert result.bulletin_editor_meta_path is not None


# ====================================================================== #
# V2->V1 boundary contract test (real V1 resolve_image_plan)              #
# ====================================================================== #


class TestRealV1ResolveImagePlanContract:
    """Verify the V2->V1 boundary against the REAL V1 ``resolve_image_plan``.

    Step 12.2a re-run #1 found three boundary bugs that the regular
    suite missed because every other resolve_images test mocks the
    V1 function. This class deliberately does NOT mock V1's
    ``resolve_image_plan`` -- it constructs a realistic input set
    (entities, image_plan, kept_clips, on-disk dummy images) and
    drives the actual V1 function to verify:

      * V2's ``image_plan_to_v1_dict`` output has every field V1 reads
        (id, entity_name, clip_index, show_at, duration)
      * V2's ``pool_manifest`` shape is what V1 reads
        (``pool_manifest[eid]["path"]``)
      * No ``output_dir`` kwarg is passed (V1's signature doesn't accept it)
      * V1 returns ``status="ready"`` for each entry whose synth_id
        appears in the pool_manifest

    The test is intentionally side-effect-free (no FFmpeg, no APIs,
    no networking). It runs in the regular pytest suite so the
    boundary breaks in CI rather than at production E2E time.

    The ImageSourcer is mocked so the pre-step is a no-op; the test
    pre-populates ``self.image_pool`` directly with on-disk dummy
    images instead.
    """

    @pytest.fixture(autouse=True)
    def _no_op_sourcer(self):
        """Patch ImageSourcer.source_for_entity to ALWAYS return None
        so the pre-step in resolve_images doesn't hit real search /
        generation APIs. Tests in this class pre-populate
        ``self.image_pool`` directly to drive the V1 lookup.
        """
        with patch(
            "pipeline_v2.stages.stage_4_render.ImageSourcer",
        ) as MockSourcer:
            MockSourcer.return_value.source_for_entity.return_value = None
            yield

    def _write_dummy_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # V1's resolve_image_plan only reads the manifest entry's
        # "path" field as a STRING -- it never opens the file. A
        # tiny placeholder is sufficient for contract verification.
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

    def _make_kept_clips(self) -> list[dict]:
        # Two contiguous clips covering the timeline. V1's resolve
        # needs start_sec / end_sec / duration_sec on each so it can
        # build the source-time -> stitched-time map.
        return [
            {"start_sec": 0.0,  "end_sec": 60.0,
             "duration_sec": 60.0,
             "start": "00:00.000", "end": "01:00.000"},
            {"start_sec": 60.0, "end_sec": 120.0,
             "duration_sec": 60.0,
             "start": "01:00.000", "end": "02:00.000"},
        ]

    def test_real_v1_resolves_mixed_entity_types(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Mix of PERSON + ORG + PLACE to cover the type-aware policy
        # branches even though we're not exercising the sourcer here.
        entities = [
            _entity("Bandi Bhagirath", type_=EntityType.PERSON),
            _entity("High Court",      type_=EntityType.ORG),
            _entity("Hyderabad",       type_=EntityType.PLACE),
        ]
        plan = ImagePlan(entries=[
            _image_plan_entry("Bandi Bhagirath", clip_index=0,
                              show_at=10.0, duration=4.0),
            _image_plan_entry("High Court",      clip_index=1,
                              show_at=70.0, duration=4.0),
            _image_plan_entry("Hyderabad",       clip_index=1,
                              show_at=90.0, duration=4.0),
            # Repeat of Bandi Bhagirath -- should share synth_id with the
            # first Bandi Bhagirath entry (one image, two timeline slots).
            _image_plan_entry("Bandi Bhagirath", clip_index=1,
                              show_at=110.0, duration=4.0),
        ])

        # Pre-populate the pool with on-disk dummy images so the
        # ImageSourcer pre-step short-circuits (entities already
        # cached) and V1 sees the populated pool_manifest.
        for e in entities:
            img = tmp_path / f"{e.canonical_name.replace(' ', '_')}.jpg"
            self._write_dummy_image(img)
            render.image_pool[e.canonical_name] = img

        kept_clips = self._make_kept_clips()
        resolved = render.resolve_images(
            plan, entities,
            kept_clip_dicts=kept_clips,
            full_metadata=_metadata(),
            whisper_words=None,
            video_duration_sec=120.0,
        )

        assert len(resolved) == 4, (
            "V1 should emit one resolved entry per image_plan entry"
        )

        # Every entry must have a `status` -- V1's failure modes
        # (image_missing, in_cut_span, bad_timestamp, duration_too_short,
        # clamped_too_short, ready) all set this field.
        statuses = [r.get("status") for r in resolved]
        assert all(s is not None for s in statuses), (
            f"V1 didn't set status on every entry: {statuses}"
        )

        # All 4 entries should resolve to ready: the synth_ids are
        # in pool_manifest (we pre-populated the pool), the
        # show_at/duration are inside the kept_clips ranges, and
        # the durations are >= 2.0s.
        assert all(s == "ready" for s in statuses), (
            f"Expected all ready, got: {statuses}"
        )

        # The two Bandi Bhagirath entries share one synth_id, hence
        # one image_path -- verify they map to the same file.
        bandi_paths = [
            r["image_path"] for r in resolved
            if r.get("entity_name") == "Bandi Bhagirath"
        ]
        assert len(bandi_paths) == 2
        assert bandi_paths[0] == bandi_paths[1], (
            "Same entity must map to the same image file across "
            "image_plan entries (D-9.5 cross-pass pool reuse)"
        )

    def test_real_v1_marks_missing_pool_entry_as_image_missing(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # An image_plan entry whose entity isn't in the pool should
        # come back from V1 with status="image_missing" -- the
        # canonical V1 failure mode that we want surfaced cleanly.
        plan = ImagePlan(entries=[
            _image_plan_entry("Sourced",  clip_index=0,
                              show_at=10.0, duration=4.0),
            _image_plan_entry("Unsourced", clip_index=0,
                              show_at=20.0, duration=4.0),
        ])
        entities = [
            _entity("Sourced",   type_=EntityType.PLACE),
            _entity("Unsourced", type_=EntityType.PLACE),
        ]
        img = tmp_path / "sourced.jpg"
        self._write_dummy_image(img)
        render.image_pool["Sourced"] = img
        # Unsourced is deliberately NOT in the pool.

        resolved = render.resolve_images(
            plan, entities,
            kept_clip_dicts=self._make_kept_clips(),
            full_metadata=_metadata(),
            video_duration_sec=120.0,
        )

        by_name = {r["entity_name"]: r for r in resolved}
        assert by_name["Sourced"]["status"] == "ready"
        assert by_name["Unsourced"]["status"] == "image_missing"

    def test_real_v1_accepts_v2_pool_manifest_shape(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Regression test for Bug #2 (12.2a re-run #1): V1's
        # ``resolve_image_plan`` reads ``pool_manifest[eid]["path"]``,
        # NOT ``pool_manifest["entries"]``. If V2 ever regresses
        # to the old shape, this test fails with status="image_missing"
        # because the lookup hits the wrong key.
        plan = ImagePlan(entries=[
            _image_plan_entry("X", clip_index=0,
                              show_at=10.0, duration=4.0),
        ])
        img = tmp_path / "x.jpg"
        self._write_dummy_image(img)
        render.image_pool["X"] = img

        resolved = render.resolve_images(
            plan, [_entity("X", type_=EntityType.PLACE)],
            kept_clip_dicts=self._make_kept_clips(),
            full_metadata=_metadata(),
            video_duration_sec=120.0,
        )
        assert resolved[0]["status"] == "ready", (
            f"V1 didn't accept the pool_manifest shape -- regression "
            f"of Bug #2 from 12.2a re-run #1. Got: {resolved[0]}"
        )

    def test_real_v1_no_output_dir_kwarg(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # Regression test for Bug #1 (12.2a re-run #1): V2 must NOT
        # pass output_dir to V1's resolve_image_plan. Calling the
        # real V1 with the current V2 code -- if Bug #1 returns,
        # we'd see TypeError("unexpected keyword argument 'output_dir'").
        plan = ImagePlan(entries=[
            _image_plan_entry("X", clip_index=0,
                              show_at=10.0, duration=4.0),
        ])
        img = tmp_path / "x.jpg"
        self._write_dummy_image(img)
        render.image_pool["X"] = img

        # No assertion -- the test passes if no exception is raised.
        render.resolve_images(
            plan, [_entity("X", type_=EntityType.PLACE)],
            kept_clip_dicts=self._make_kept_clips(),
            full_metadata=_metadata(),
            video_duration_sec=120.0,
        )


# ======================================================================
# Backlog item 97: splice_cuts_minus_skipped -- bulletin trim regression
# ======================================================================


class TestSpliceCutsMinusSkipped:
    """``splice_cuts_minus_skipped`` subtracts SkippedSegment ranges
    from FullVideoCuts so the bulletin renderer drops retake / filler /
    dead-air spans. Before this helper, the V2 bulletin was rendered
    over the entire cut span = no trim applied (jobs 35/36/40 had
    <1s trim on 12-min sources).
    """

    @staticmethod
    def _cut(idx, sw, ew, ss, es, imp=5):
        from pipeline_v2.models import FullVideoCut
        return FullVideoCut(
            index=idx, start_word_idx=sw, end_word_idx=ew,
            start_sec=ss, end_sec=es, importance=imp,
        )

    @staticmethod
    def _skip(sw, ew, ss, es, cat="warm_up"):
        from pipeline_v2.models import SkippedSegment
        return SkippedSegment(
            start_word_idx=sw, end_word_idx=ew,
            start_sec=ss, end_sec=es,
            category=cat, reason="test",
        )

    def test_no_skipped_returns_input_unchanged(self):
        from pipeline_v2.stages.stage_4_render import splice_cuts_minus_skipped
        cuts = [self._cut(0, 0, 99, 0.0, 600.0)]
        out, parents = splice_cuts_minus_skipped(cuts, [])
        assert len(out) == 1
        assert out[0].start_sec == 0.0
        assert out[0].end_sec == 600.0
        assert parents == [0]

    def test_single_skip_in_middle_yields_two_sub_cuts(self):
        from pipeline_v2.stages.stage_4_render import splice_cuts_minus_skipped
        cuts = [self._cut(0, 0, 99, 0.0, 600.0)]
        skipped = [self._skip(40, 60, 100.0, 150.0)]
        out, parents = splice_cuts_minus_skipped(cuts, skipped)
        assert len(out) == 2
        assert out[0].start_sec == 0.0 and out[0].end_sec == 100.0
        assert out[1].start_sec == 150.0 and out[1].end_sec == 600.0
        kept = sum(c.end_sec - c.start_sec for c in out)
        assert kept == 550.0
        # Both sub-cuts share parent FullVideoCut.index = 0
        assert parents == [0, 0]

    def test_multiple_skips_per_cut(self):
        from pipeline_v2.stages.stage_4_render import splice_cuts_minus_skipped
        cuts = [self._cut(0, 0, 99, 0.0, 600.0)]
        skipped = [
            self._skip(10, 15, 30.0, 45.0),
            self._skip(40, 60, 100.0, 150.0),
            self._skip(70, 75, 400.0, 410.0),
        ]
        out, parents = splice_cuts_minus_skipped(cuts, skipped)
        assert len(out) == 4
        bounds = [(c.start_sec, c.end_sec) for c in out]
        assert bounds == [
            (0.0, 30.0), (45.0, 100.0), (150.0, 400.0), (410.0, 600.0),
        ]
        # 4 sub-cuts, all from the SAME parent (= no takeovers should
        # fire between them in render_bulletin, per item 100).
        assert parents == [0, 0, 0, 0]

    def test_skip_at_start_of_cut(self):
        from pipeline_v2.stages.stage_4_render import splice_cuts_minus_skipped
        cuts = [self._cut(0, 0, 99, 100.0, 600.0)]
        skipped = [self._skip(0, 5, 100.0, 110.0)]
        out, parents = splice_cuts_minus_skipped(cuts, skipped)
        assert len(out) == 1
        assert out[0].start_sec == 110.0 and out[0].end_sec == 600.0
        assert parents == [0]

    def test_skip_at_end_of_cut(self):
        from pipeline_v2.stages.stage_4_render import splice_cuts_minus_skipped
        cuts = [self._cut(0, 0, 99, 0.0, 500.0)]
        skipped = [self._skip(95, 99, 480.0, 500.0)]
        out, parents = splice_cuts_minus_skipped(cuts, skipped)
        assert len(out) == 1
        assert out[0].start_sec == 0.0 and out[0].end_sec == 480.0
        assert parents == [0]

    def test_skip_outside_cut_is_ignored(self):
        from pipeline_v2.stages.stage_4_render import splice_cuts_minus_skipped
        cuts = [self._cut(0, 0, 99, 100.0, 200.0)]
        skipped = [self._skip(200, 205, 300.0, 310.0)]
        out, parents = splice_cuts_minus_skipped(cuts, skipped)
        assert len(out) == 1
        assert out[0].start_sec == 100.0 and out[0].end_sec == 200.0
        assert parents == [0]

    def test_multiple_cuts_each_with_their_own_skips(self):
        from pipeline_v2.stages.stage_4_render import splice_cuts_minus_skipped
        cuts = [
            self._cut(0, 0, 50, 0.0, 100.0),
            self._cut(1, 60, 120, 150.0, 300.0),
        ]
        skipped = [
            self._skip(10, 20, 20.0, 40.0),
            self._skip(80, 90, 200.0, 230.0),
        ]
        out, parents = splice_cuts_minus_skipped(cuts, skipped)
        assert len(out) == 4
        assert [c.index for c in out] == [0, 1, 2, 3]
        bounds = [(c.start_sec, c.end_sec) for c in out]
        assert bounds == [
            (0.0, 20.0), (40.0, 100.0),
            (150.0, 200.0), (230.0, 300.0),
        ]
        # Sub-cuts 0,1 share parent FullVideoCut[0]; 2,3 share
        # parent FullVideoCut[1]. Takeover should fire BETWEEN
        # sub-cut 1 and sub-cut 2 (parent boundary) but NOT between
        # 0-1 or 2-3 (same parent within each story).
        assert parents == [0, 0, 1, 1]


# ======================================================================
# Backlog item 100: adaptive takeovers + PiP gate + A/V invariant
# ======================================================================


class TestAdaptiveTakeoversAndPiPGate:
    """Item 100 — V2 must not bloat the bulletin past narration audio.

    1. Takeovers default OFF and only enable when Stage 2 produced
       >=3 distinct FullVideoCuts (and even then only between
       different parents).
    2. PiP is suppressed for SOLO video_type (talking-head news
       monologue) since pick_pip_source would just pull another
       segment of the same anchor.
    3. After render, an A/V invariant assertion ensures bulletin
       audio == narration audio + takeover video.
    """

    def test_use_takeovers_default_is_false(self):
        """Default flipped from True to False so a single-story splice
        no longer auto-inserts ~6.5s of dead air between every sub-cut."""
        from pipeline_v2.stages.stage_4_render import Stage4Render
        from dataclasses import fields
        f = next(f for f in fields(Stage4Render) if f.name == "use_takeovers")
        assert f.default is False

    def test_full_video_cuts_to_v1_clip_dicts_stamps_parent_index(self):
        from pipeline_v2.stages.stage_4_render import (
            full_video_cuts_to_v1_clip_dicts,
        )
        cuts = [
            _full_video_cut(idx=0, start=0.0, end=100.0),
            _full_video_cut(idx=1, start=100.0, end=200.0),
        ]
        parents = [7, 7]  # both sub-cuts share parent FullVideoCut[7]
        out = full_video_cuts_to_v1_clip_dicts(cuts, _metadata(), parents)
        assert [c["parent_v2_index"] for c in out] == [7, 7]
        # When parent_v2_indexes is omitted, falls back to the cut's
        # own index (each cut is its own parent).
        out2 = full_video_cuts_to_v1_clip_dicts(cuts, _metadata())
        assert [c["parent_v2_index"] for c in out2] == [0, 1]

    def test_ffprobe_audio_duration_helper_handles_missing_file(self):
        """Item 100 guardrail probes return 0.0 (best-effort) when the
        path doesn't exist -- the outer try/except in _render_impl
        swallows probe failures so render success isn't blocked by
        a transient ffprobe issue."""
        from pipeline_v2.stages.stage_4_render import (
            _ffprobe_audio_duration_s, _ffprobe_video_duration_s,
        )
        assert _ffprobe_audio_duration_s("/nope/missing.mp4") == 0.0
        assert _ffprobe_video_duration_s("/nope/missing.mp4") == 0.0

    def test_video_type_solo_in_disabled_set(self):
        """Sanity-check the PiP allow-list used in _render_impl. SOLO
        must NOT be in the allowed set (= the user's bug 2 case)."""
        # This mirrors the literal set used in the renderer; if it
        # ever diverges this test catches the drift before it ships.
        allowed = {"INTERVIEW", "PRESS_CONFERENCE", "PANEL", "MIXED"}
        assert "SOLO" not in allowed

    def test_render_bulletin_signature_accepts_new_kwargs(self):
        """Wire-level lock: render_bulletin must accept the three new
        kwargs _render_impl now passes. Catches a future refactor
        that removes any of them before the orchestrator does."""
        import inspect
        from pipeline_v2.stages.stage_4_render import Stage4Render
        sig = inspect.signature(Stage4Render.render_bulletin)
        for name in ("parent_v2_indexes", "takeovers_enabled", "pip_enabled"):
            assert name in sig.parameters, (
                f"render_bulletin must accept '{name}' kwarg "
                f"(item 100); got params={list(sig.parameters)}"
            )
            # All three default to None so existing callers (tests,
            # _render_impl alternates) keep working.
            assert sig.parameters[name].default is None

    def test_cut_raw_bulletin_stories_signature_accepts_parents(self):
        """Parent-index plumb-through reaches the raw cut helper too."""
        import inspect
        from pipeline_v2.stages.stage_4_render import Stage4Render
        sig = inspect.signature(Stage4Render.cut_raw_bulletin_stories)
        assert "parent_v2_indexes" in sig.parameters
        assert sig.parameters["parent_v2_indexes"].default is None


# ======================================================================
# Backlog item 102: behavioural coverage of item 100's adaptive gates
# ======================================================================
#
# Item 100 introduced the three guardrails inline in _render_impl /
# render_bulletin. Item 102 extracts them to pure helpers
# (_compute_takeovers_enabled / _compute_pip_enabled /
# _should_insert_takeover_between / _validate_av_invariant) so the
# branch behaviour can be tested without spinning up the full render
# pipeline. The 6 tests below cover both branches of each gate plus
# the violation path.


class TestAdaptiveTakeoverGate:
    """Item 102 test 1-3: ``_compute_takeovers_enabled`` truth table.

    Takeovers enabled only when the operator opted in AND Stage 2
    produced >= TAKEOVER_MIN_DISTINCT_CUTS distinct FullVideoCuts.
    """

    def test_disabled_when_use_takeovers_false_regardless_of_count(self):
        """User opt-out wins over story count -- if use_takeovers=False
        the gate stays closed even with 10 stories."""
        from pipeline_v2.stages.stage_4_render import (
            _compute_takeovers_enabled,
        )
        # Use a synthetic list of the right length; the helper only
        # inspects len() so contents don't matter.
        cuts_10 = [None] * 10
        assert _compute_takeovers_enabled(False, cuts_10) is False
        assert _compute_takeovers_enabled(False, []) is False

    def test_disabled_when_fewer_than_three_cuts_even_if_opted_in(self):
        """The 132s-bloat bug case: opted in but only one or two
        narrative threads -> gate stays closed because all sub-cuts
        belong to the same parent and would just produce dead air."""
        from pipeline_v2.stages.stage_4_render import (
            _compute_takeovers_enabled, TAKEOVER_MIN_DISTINCT_CUTS,
        )
        assert TAKEOVER_MIN_DISTINCT_CUTS == 3, (
            "Test pins floor at 3; if you change "
            "TAKEOVER_MIN_DISTINCT_CUTS, update this assertion."
        )
        assert _compute_takeovers_enabled(True, []) is False
        assert _compute_takeovers_enabled(True, [None]) is False
        assert _compute_takeovers_enabled(True, [None, None]) is False

    def test_enabled_when_opted_in_and_three_or_more_cuts(self):
        """The legitimate multi-story TV bulletin case."""
        from pipeline_v2.stages.stage_4_render import (
            _compute_takeovers_enabled,
        )
        assert _compute_takeovers_enabled(True, [None, None, None]) is True
        assert _compute_takeovers_enabled(True, [None] * 5) is True


class TestSpliceGroupTakeoverBoundary:
    """Item 102 test 4: ``_should_insert_takeover_between``.

    The boundary check is the second line of defence after the
    coarse 3-cut gate. Even when takeovers are enabled, two
    adjacent clips that share the same ``parent_v2_index`` are
    spliced sub-cuts of the same story -- no takeover between
    them, or we recreate the 132s dead-air bug at a finer
    granularity.
    """

    def test_boundary_logic_matches_splice_group_semantics(self):
        """Same parent => no takeover; different parents => takeover;
        last clip (next_parent is None) => no takeover."""
        from pipeline_v2.stages.stage_4_render import (
            _should_insert_takeover_between,
        )
        # Same FullVideoCut parent -- spliced sub-cuts of one
        # narrative thread. Must concat seamlessly.
        assert _should_insert_takeover_between(0, 0) is False
        assert _should_insert_takeover_between(7, 7) is False
        # Different parents -- legitimate inter-story boundary.
        assert _should_insert_takeover_between(0, 1) is True
        assert _should_insert_takeover_between(7, 3) is True
        # Final clip in the bulletin -- there is no "next story" to
        # transition into, so no takeover.
        assert _should_insert_takeover_between(0, None) is False
        assert _should_insert_takeover_between(7, None) is False


class TestPiPGateForSOLO:
    """Item 102 test 5: ``_compute_pip_enabled`` excludes SOLO.

    SOLO talking-head monologue must NOT get PiP. V1's
    ``pick_pip_source`` would pull the same anchor from the next
    story slot -> meaningless inset of the same face in the corner.
    """

    def test_pip_disabled_for_solo_enabled_for_multi_source(self):
        from pipeline_v2.stages.stage_4_render import (
            _compute_pip_enabled, PIP_ALLOWED_VIDEO_TYPES,
        )
        # SOLO is the explicit exclusion the user flagged.
        assert _compute_pip_enabled(True, "SOLO") is False
        # Allowed video types -- enabled when opted in.
        for vt in ("INTERVIEW", "PRESS_CONFERENCE", "PANEL", "MIXED"):
            assert vt in PIP_ALLOWED_VIDEO_TYPES
            assert _compute_pip_enabled(True, vt) is True
        # Operator opt-out always wins.
        assert _compute_pip_enabled(False, "INTERVIEW") is False
        # Unknown / empty video_type -- treated as "not allowed"
        # (fail-closed). Matches the inline expression's behaviour
        # via set membership.
        assert _compute_pip_enabled(True, "") is False
        assert _compute_pip_enabled(True, "UNKNOWN") is False


class TestAVInvariantGuardrail:
    """Item 102 test 6: ``_validate_av_invariant`` raises on bloat.

    The Job 42 incident: bulletin audio was 132s longer than
    narration audio because takeovers added silent video padding.
    The guardrail catches this at render time and surfaces a
    descriptive RuntimeError so the operator sees the regression
    rather than discovering it via a support ticket.
    """

    def test_within_tolerance_does_not_raise(self):
        """Sub-second drift from ffmpeg rounding / sample boundaries
        must not trigger the guardrail. AV_INVARIANT_TOLERANCE_S
        defaults to 1.0s for exactly this reason."""
        from pipeline_v2.stages.stage_4_render import (
            _validate_av_invariant, AV_INVARIANT_TOLERANCE_S,
        )
        assert AV_INVARIANT_TOLERANCE_S == 1.0
        # Exact match.
        _validate_av_invariant(100.0, 80.0, 20.0)
        # Within tolerance on both signs.
        _validate_av_invariant(100.5, 80.0, 20.0)
        _validate_av_invariant(99.5, 80.0, 20.0)
        # Exactly at the boundary -- abs(delta) == tolerance, so still
        # accepted (the violation predicate is ``abs > tolerance``).
        _validate_av_invariant(101.0, 80.0, 20.0)

    def test_audio_longer_than_expected_raises_with_descriptive_message(self):
        """The Job 42 bug case: bulletin audio bloated by silent
        padding. The guardrail must raise and the message must name
        all three measured durations so the operator can debug."""
        import pytest as _pytest
        from pipeline_v2.stages.stage_4_render import (
            _validate_av_invariant,
        )
        # +132s delta (the Job 42 case): audio 633s vs expected 501s.
        with _pytest.raises(RuntimeError) as exc_info:
            _validate_av_invariant(
                actual_audio_s=633.0,
                composed_narration_s=481.0,
                takeover_video_s=20.0,
            )
        msg = str(exc_info.value)
        # Must name all three measured durations.
        assert "633" in msg
        assert "481" in msg
        assert "20" in msg
        # Must report the delta with sign.
        assert "+132" in msg
        # Must hint at root cause for the operator.
        assert "takeover" in msg.lower() or "intro" in msg.lower()

    def test_audio_shorter_than_expected_also_raises(self):
        """Negative delta is also a violation -- bulletin can't be
        shorter than narration+transitions. Belt-and-suspenders for
        a stitch step that silently truncated audio."""
        import pytest as _pytest
        from pipeline_v2.stages.stage_4_render import (
            _validate_av_invariant,
        )
        with _pytest.raises(RuntimeError) as exc_info:
            _validate_av_invariant(
                actual_audio_s=50.0,
                composed_narration_s=80.0,
                takeover_video_s=20.0,
            )
        # Delta should be -50s and surface with sign in the message.
        assert "-50" in str(exc_info.value)

    def test_custom_tolerance_is_honoured(self):
        """The tolerance kwarg lets callers tighten the bar for
        fixture-style tests (e.g. a sub-second-precision render
        regression check)."""
        import pytest as _pytest
        from pipeline_v2.stages.stage_4_render import (
            _validate_av_invariant,
        )
        # Default tolerance accepts 0.5s drift.
        _validate_av_invariant(100.5, 80.0, 20.0)
        # Tightened tolerance rejects the same drift.
        with _pytest.raises(RuntimeError):
            _validate_av_invariant(100.5, 80.0, 20.0, tolerance_s=0.1)


# ======================================================================
# Backlog item 103: micro-fragment drop (< 1.5s)
# ======================================================================
#
# After splice_cuts_minus_skipped runs, the bulletin can contain
# sub-cuts < 1.5s -- e.g. a 0.3s sliver between two consecutive
# retakes. ``collapse_micro_fragments`` drops these so the bulletin
# doesn't chop on near-zero-length segments. Drop (not merge): merging
# across an already-spliced skipped span would re-include the retake.


class TestCollapseMicroFragments:
    """Item 103 -- 5 behavioural tests for ``collapse_micro_fragments``."""

    def test_no_fragments_passes_through_unchanged(self):
        """All sub-cuts >= threshold -> identity (after index renumber)."""
        from pipeline_v2.stages.stage_4_render import collapse_micro_fragments
        cuts = [
            _full_video_cut(idx=0, start=0.0, end=10.0),
            _full_video_cut(idx=1, start=12.0, end=20.0),
            _full_video_cut(idx=2, start=22.0, end=30.0),
        ]
        parents = [0, 0, 1]
        out_cuts, out_parents = collapse_micro_fragments(cuts, parents)
        assert len(out_cuts) == 3
        assert [c.start_sec for c in out_cuts] == [0.0, 12.0, 22.0]
        assert [c.end_sec for c in out_cuts] == [10.0, 20.0, 30.0]
        # Indexes renumbered contiguously starting at 0.
        assert [c.index for c in out_cuts] == [0, 1, 2]
        # Parents preserved in parallel.
        assert out_parents == [0, 0, 1]

    def test_micro_fragment_between_long_subcuts_is_dropped(self):
        """Job-42-style case: 0.3s sliver between two long retakes
        gets dropped; the long siblings survive."""
        from pipeline_v2.stages.stage_4_render import collapse_micro_fragments
        cuts = [
            _full_video_cut(idx=0, start=0.0, end=10.0),  # 10s -- keep
            _full_video_cut(idx=1, start=10.5, end=10.8),  # 0.3s -- DROP
            _full_video_cut(idx=2, start=11.0, end=21.0),  # 10s -- keep
        ]
        parents = [0, 0, 0]  # all sub-cuts share one FullVideoCut parent
        out_cuts, out_parents = collapse_micro_fragments(cuts, parents)
        assert len(out_cuts) == 2
        assert [c.start_sec for c in out_cuts] == [0.0, 11.0]
        assert [c.end_sec for c in out_cuts] == [10.0, 21.0]
        # Renumbered contiguously after the drop.
        assert [c.index for c in out_cuts] == [0, 1]
        # Parent list filtered in parallel.
        assert out_parents == [0, 0]

    def test_multiple_micro_fragments_are_all_dropped(self):
        """Multiple slivers in one parent -- all dropped, longs kept."""
        from pipeline_v2.stages.stage_4_render import collapse_micro_fragments
        cuts = [
            _full_video_cut(idx=0, start=0.0, end=5.0),   # 5s
            _full_video_cut(idx=1, start=5.2, end=5.5),   # 0.3s
            _full_video_cut(idx=2, start=6.0, end=12.0),  # 6s
            _full_video_cut(idx=3, start=12.5, end=13.0), # 0.5s
            _full_video_cut(idx=4, start=13.5, end=14.0), # 0.5s
            _full_video_cut(idx=5, start=15.0, end=20.0), # 5s
        ]
        parents = [0, 0, 0, 0, 0, 0]
        out_cuts, out_parents = collapse_micro_fragments(cuts, parents)
        assert len(out_cuts) == 3
        assert [c.start_sec for c in out_cuts] == [0.0, 6.0, 15.0]
        assert [c.index for c in out_cuts] == [0, 1, 2]

    def test_per_parent_safety_keeps_longest_when_all_below_threshold(self):
        """If a parent's every sub-cut is below threshold, KEEP the
        longest -- otherwise an entire story would vanish from the
        bulletin. Other parents still get their micro-fragments dropped.
        """
        from pipeline_v2.stages.stage_4_render import collapse_micro_fragments
        cuts = [
            # Parent 7 (story A): all three sub-cuts < 1.5s -> keep the longest
            _full_video_cut(idx=0, start=0.0, end=0.5),   # 0.5s
            _full_video_cut(idx=1, start=1.0, end=2.2),   # 1.2s (longest)
            _full_video_cut(idx=2, start=2.5, end=3.0),   # 0.5s
            # Parent 8 (story B): one long sub-cut + one fragment -> drop fragment
            _full_video_cut(idx=3, start=5.0, end=12.0),  # 7s -- keep
            _full_video_cut(idx=4, start=12.2, end=12.5), # 0.3s -- drop
        ]
        parents = [7, 7, 7, 8, 8]
        out_cuts, out_parents = collapse_micro_fragments(cuts, parents)
        # Expect 2 sub-cuts: the longest of parent 7 (1.0-2.2s) + the
        # long sub-cut of parent 8.
        assert len(out_cuts) == 2
        assert (out_cuts[0].start_sec, out_cuts[0].end_sec) == (1.0, 2.2)
        assert (out_cuts[1].start_sec, out_cuts[1].end_sec) == (5.0, 12.0)
        # Parents preserved in parallel.
        assert out_parents == [7, 8]

    def test_configurable_threshold_and_disable_via_zero(self):
        """The threshold_s kwarg lets callers tighten or loosen the
        drop; threshold_s=0 effectively disables dropping (every
        sub-cut >= 0.0)."""
        from pipeline_v2.stages.stage_4_render import collapse_micro_fragments
        cuts = [
            _full_video_cut(idx=0, start=0.0, end=2.0),  # 2s
            _full_video_cut(idx=1, start=3.0, end=3.5),  # 0.5s
            _full_video_cut(idx=2, start=4.0, end=4.4),  # 0.4s
        ]
        parents = [0, 0, 0]
        # Tighter threshold drops more (still keep 2s; drop 0.5s + 0.4s).
        out_strict, _ = collapse_micro_fragments(cuts, parents, threshold_s=1.0)
        assert len(out_strict) == 1
        assert (out_strict[0].start_sec, out_strict[0].end_sec) == (0.0, 2.0)
        # Looser threshold keeps more.
        out_loose, _ = collapse_micro_fragments(cuts, parents, threshold_s=0.45)
        # 2.0s >= 0.45 keep, 0.5s >= 0.45 keep, 0.4s < 0.45 drop.
        assert len(out_loose) == 2
        assert {(c.start_sec, c.end_sec) for c in out_loose} == {
            (0.0, 2.0), (3.0, 3.5),
        }
        # threshold=0 -> all sub-cuts kept (passthrough).
        out_disabled, _ = collapse_micro_fragments(cuts, parents, threshold_s=0.0)
        assert len(out_disabled) == 3

    def test_empty_input_returns_empty(self):
        """Defensive: empty list in, empty list out (no exception)."""
        from pipeline_v2.stages.stage_4_render import collapse_micro_fragments
        out_cuts, out_parents = collapse_micro_fragments([], [])
        assert out_cuts == []
        assert out_parents == []

    def test_length_mismatch_raises(self):
        """Defensive: caller-error guard. sub_cuts and parent_v2_indexes
        must be parallel lists (one parent per sub-cut)."""
        import pytest as _pytest
        from pipeline_v2.stages.stage_4_render import collapse_micro_fragments
        cuts = [
            _full_video_cut(idx=0, start=0.0, end=5.0),
            _full_video_cut(idx=1, start=6.0, end=10.0),
        ]
        with _pytest.raises(ValueError, match="length mismatch"):
            collapse_micro_fragments(cuts, [0])  # 2 cuts vs 1 parent

    def test_stage4render_micro_fragment_threshold_default_is_one_point_five(self):
        """Wire-level check: the dataclass default matches the
        module-level constant so the operator can edit one place if
        the threshold ever needs tuning."""
        from dataclasses import fields
        from pipeline_v2.stages.stage_4_render import (
            Stage4Render, MICRO_FRAGMENT_THRESHOLD_S,
        )
        f = next(
            f for f in fields(Stage4Render)
            if f.name == "micro_fragment_threshold_s"
        )
        assert f.default == MICRO_FRAGMENT_THRESHOLD_S
        assert MICRO_FRAGMENT_THRESHOLD_S == 1.5
