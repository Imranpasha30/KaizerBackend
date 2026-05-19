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

    def test_no_id_field_emitted(self):
        # Per the post_v2_backlog "?" id bug: V2 ImagePlanEntry has no
        # `id` field. The converter must NOT emit one.
        plan = ImagePlan(entries=[_image_plan_entry()])
        out = image_plan_to_v1_dict(plan, [_entity()])
        assert "id" not in out[0]


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


# ====================================================================== #
# Step 9.2: Stage4Render.resolve_images                                   #
# ====================================================================== #


class TestResolveImages:
    """Wires V1's resolve_image_plan. Per D-9.5, populates the
    instance image_pool with entity_name -> image_path mappings so
    subsequent passes can reuse.
    """

    def _stub_resolved(self, entries: list[dict]) -> list[dict]:
        return entries

    def test_populates_image_pool_from_resolved_manifest(
        self, render: Stage4Render,
    ):
        plan = ImagePlan(entries=[
            _image_plan_entry("Modi", clip_index=0),
            _image_plan_entry("Reddy", clip_index=1),
        ])
        entities = [_entity("Modi"), _entity("Reddy")]

        resolved = [
            {"entity_name": "Modi",  "image_path": "/abs/news_01.jpg",
             "status": "ready", "clip_index": 0},
            {"entity_name": "Reddy", "image_path": "/abs/news_02.jpg",
             "status": "ready", "clip_index": 1},
        ]
        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            return_value=resolved,
        ):
            render.resolve_images(plan, entities, kept_clip_dicts=[],
                                  full_metadata=_metadata())

        assert render.image_pool["Modi"] == Path("/abs/news_01.jpg")
        assert render.image_pool["Reddy"] == Path("/abs/news_02.jpg")

    def test_skips_unready_entries(self, render: Stage4Render):
        # Entries with status != "ready" should NOT poison the pool.
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

    def test_pool_is_reused_across_calls(self, render: Stage4Render):
        # D-9.5 test: a second resolve call sees the pool from the
        # first call's resolution in its pool_manifest arg.
        plan_a = ImagePlan(entries=[_image_plan_entry("Modi", clip_index=0)])
        plan_b = ImagePlan(entries=[_image_plan_entry("Modi", clip_index=0)])
        resolved_first = [
            {"entity_name": "Modi", "image_path": "/abs/news_01.jpg",
             "status": "ready", "clip_index": 0},
        ]

        captured_pool_manifests: list[dict] = []

        def _v1_mock(v1_image_plan, *, output_dir, pool_manifest,
                     kept_clips, whisper_words, video_duration_sec):
            captured_pool_manifests.append(pool_manifest)
            return resolved_first

        with patch(
            "pipeline_v2.stages.stage_4_render._v1_resolve_image_plan",
            side_effect=_v1_mock,
        ):
            render.resolve_images(plan_a, [_entity("Modi")],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())
            # Second pass: image_pool already has Modi cached
            assert "Modi" in render.image_pool
            render.resolve_images(plan_b, [_entity("Modi")],
                                  kept_clip_dicts=[],
                                  full_metadata=_metadata())

        # First call: empty pool_manifest. Second call: pool_manifest
        # contains the Modi entry from the first resolution.
        assert captured_pool_manifests[0]["entries"] == []
        first_pool_names = {
            e["entity_name"] for e in captured_pool_manifests[1]["entries"]
        }
        assert "Modi" in first_pool_names

    def test_whisper_words_passed_as_none_d94(self, render: Stage4Render):
        # D-9.4: V2 skips whisper_anchor, passes whisper_words=None.
        plan = ImagePlan(entries=[_image_plan_entry()])

        captured_whisper = []

        def _v1_mock(v1_image_plan, *, output_dir, pool_manifest,
                     kept_clips, whisper_words, video_duration_sec):
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

    def test_v1_image_plan_dict_shape(self, render: Stage4Render):
        # The dict V1 receives is the one converted by
        # image_plan_to_v1_dict: 6 fields, no `id`.
        plan = ImagePlan(entries=[
            _image_plan_entry("Modi", clip_index=0, show_at=12.5),
        ])
        captured = []

        def _v1_mock(v1_image_plan, *, output_dir, pool_manifest,
                     kept_clips, whisper_words, video_duration_sec):
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
        assert d["entity_name"] == "Modi"
        assert d["clip_index"] == 0
        assert d["show_at"] == "00:12.500"
        assert "id" not in d


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
                   tmp_path: Path = None) -> dict:
    """Build a post-cut clip dict (what cut_raw_shorts returns)."""
    raw_path = ""
    if raw_exists:
        assert tmp_path is not None
        raw_path = str(tmp_path / f"raw_clip_{idx+1:02d}.mp4")
        Path(raw_path).write_text("fake raw clip bytes")
    return {
        "start": "00:00.000", "end": "00:20.000",
        "summary": "hook", "mood": "", "importance": 7,
        "video_type": "SOLO", "v2_index": idx,
        "raw_path": raw_path, "duration_sec": 20.0,
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

    def test_no_image_isolated_failure_below_threshold(
        self, render: Stage4Render, tmp_path: Path,
    ):
        # 1 image-less clip out of 4 (25% failure) -> 3 clips ship,
        # the missing-image clip is dropped silently (no raise).
        clip_dicts = [_post_cut_clip(i, tmp_path=render.output_dir)
                      for i in range(4)]
        (render.output_dir / "img.jpg").write_text("img")
        render.image_pool["X"] = render.output_dir / "img.jpg"
        # Resolved entries for clips 0, 2, 3 only -- clip 1 has none.
        resolved = [
            {"entity_name": "X", "image_path": str(render.output_dir / "img.jpg"),
             "status": "ready", "clip_index": i}
            for i in (0, 2, 3)
        ]
        # And remove the pool fallback so clip 1 truly has no image.
        render.image_pool.clear()
        with self._patch_v1()["compose"], self._patch_v1()["subprocess_run"]:
            # Even resolved entries can't help clip 1 because the
            # fallback path checks the resolved map by v2_index --
            # clip 1's index has no resolved entry.
            # But since pool is empty, fallback also fails.
            # Result: 1/4 dropped = 25%, no raise.
            out = render.compose_shorts(clip_dicts, _metadata(), resolved)
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
        def _capture_resolve(v1_image_plan, *, output_dir, pool_manifest,
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

        cuts = [_full_video_cut(0, 0, 60)]
        stack, mocks = _all_patches(bulletin_v1_patches)
        with stack:
            bulletin_render.render_bulletin(
                full_video_cuts=cuts, metadata=_metadata(),
                entities=[_entity("Modi")],
                image_plan=ImagePlan(entries=[]),
            )

        # The pool_manifest passed to V1 contains the Modi entry
        # that was already in the pool. V1 sees this and can skip
        # re-downloading.
        assert len(captured_manifests) == 1
        manifest_names = {
            e["entity_name"] for e in captured_manifests[0]["entries"]
        }
        assert "Modi" in manifest_names


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
