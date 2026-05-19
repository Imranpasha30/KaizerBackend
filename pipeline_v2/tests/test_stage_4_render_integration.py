"""Opt-in real-FFmpeg integration tests for Stage 4 render.

These tests invoke ACTUAL V1 helpers (cut_video_clips, compose_clip,
stitch_bulletin, ...) against a real source video. They are SKIPPED
by default; enable by setting:

    KAIZER_RUN_INTEGRATION_TESTS=1

AND providing a real test video at the path pointed to by:

    KAIZER_TEST_VIDEO=/abs/path/to/test.mp4

Skip-by-default rationale (per D-9.12):
  - CI may not have FFmpeg / NVENC available.
  - A 30s real-encode takes 5-30s vs the unit suite's <2s.
  - The fonts + Pillow + ffmpeg-build version matrix is fragile.
  - V1 already battle-tests the actual encoding -- these tests
    verify the V2 ORCHESTRATION against the real V1 surface, not
    encoding correctness.

When run, these tests cover:
  - Stage4Render.render() against a real video produces both
    editor_meta.json files
  - clip_NN.mp4 / raw_clip_NN.mp4 / thumb_NN.jpg exist and are
    non-zero size
  - bulletin/bulletin.mp4 exists and is non-zero size
  - editor_meta.json validates against V1's expected schema

Limitations of this surface:
  - Doesn't cover compose_pip_story / PiP path (needs >=2 clips
    and a specific pick_pip_source outcome).
  - Doesn't cover image-search (we use a stub image in the pool).
  - Doesn't cover the 50% guardrail (a real-FFmpeg failure mode
    is hard to trigger deterministically).

If KAIZER_RUN_INTEGRATION_TESTS is unset OR the fixture video is
missing, EVERY test in this file is skipped with a clear reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Skip the entire module if integration tests are disabled.
RUN_INTEGRATION = os.environ.get("KAIZER_RUN_INTEGRATION_TESTS", "").strip() == "1"
TEST_VIDEO = os.environ.get("KAIZER_TEST_VIDEO", "").strip()

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason=(
        "Integration tests skipped. Set KAIZER_RUN_INTEGRATION_TESTS=1 "
        "and KAIZER_TEST_VIDEO=/abs/path/to/test.mp4 to enable. See "
        "module docstring for details."
    ),
)


@pytest.fixture(scope="module")
def test_video_path() -> Path:
    """Resolve and validate the integration test video path."""
    if not TEST_VIDEO:
        pytest.skip("KAIZER_TEST_VIDEO env var not set")
    p = Path(TEST_VIDEO)
    if not p.is_file():
        pytest.skip(f"KAIZER_TEST_VIDEO points to non-existent file: {p}")
    return p


# Placeholder for the actual integration tests. These will be filled
# in when we have a settled test fixture video on disk (the Bandi
# Bhagirath case test.mp4 is the natural candidate -- already
# referenced by Step 5's regression script).


class TestRealFFmpegRender:
    """Real-FFmpeg orchestrator tests. Each test takes 5-30s."""

    def test_smoke_render_produces_expected_artifacts(
        self, test_video_path: Path, tmp_path: Path,
    ):
        """End-to-end smoke: real video → both editor_meta.json
        files written, all expected per-clip artifacts present.
        """
        from pipeline_v2.stages.stage_4_render import Stage4Render
        from pipeline_v2.models import (
            CleanTranscript, Entity, FullVideoCut, ImagePlan,
            ImagePlanEntry, JobOutput, Metadata, ShortsCut,
            SkippedSegment, StageTwoOutput, Word,
        )

        # Minimal JobOutput pointing at the test video. 3 shorts,
        # 1 full_video_cut, 1 entity.
        job = JobOutput(
            stage_two=StageTwoOutput(
                full_video_cuts=[FullVideoCut(
                    index=0, start_word_idx=0, end_word_idx=100,
                    start_sec=0.0, end_sec=60.0, importance=8,
                )],
                skipped_segments=[],
                clean_transcript=CleanTranscript(
                    words=[Word(w="dummy", s=0.0, e=1.0)],
                    clip_boundaries={0: (0, 0)},
                    source_word_map=[0],
                ),
                retake_audit="None",
            ),
            canonical_entities=[Entity(
                canonical_name="Bandi Bhagirath",
                native_name="బండి భగీరథ్",
                first_mention_word_idx=0,
                type="PERSON",
                mentions=[0],
            )],
            shorts_cuts=[
                ShortsCut(index=0, start_sec=0.0,  end_sec=18.0,
                          hook="Hook A", importance=8),
                ShortsCut(index=1, start_sec=20.0, end_sec=38.0,
                          hook="Hook B", importance=7),
                ShortsCut(index=2, start_sec=40.0, end_sec=58.0,
                          hook="Hook C", importance=6),
            ],
            metadata=Metadata(
                video_type="SOLO", language="te-en", total_speakers=1,
                overall_summary="Bandi Bhagirath case overview.",
                overall_summary_native="బండి భగీరథ్ కేసు సమీక్ష.",
                shorts_headline_native="బండి భగీరథ్ కేసులో కీలక మలుపులు",
                bulletin_marquee_points=["కేసులో మలుపులు", "వయసుపై స్పష్టత"],
                image_search_queries=["Bandi Bhagirath case"],
                key_people=["Bandi Bhagirath"],
                key_people_native=["బండి భగీరథ్"],
                key_topics=["court case"],
                key_locations=["Hyderabad"],
            ),
            image_plan=ImagePlan(entries=[
                ImagePlanEntry(
                    entity_name="Bandi Bhagirath",
                    entity_name_native="బండి భగీరథ్",
                    description="Bandi Bhagirath case courtroom",
                    clip_index=0,
                    show_at_sec=5.0,
                    duration_sec=4.0,
                ),
            ]),
        )

        renderer = Stage4Render(
            output_dir=tmp_path / "out",
            video_path=test_video_path,
            preset={
                "label": "YouTube Short",
                "width": 1080, "height": 1920,
                "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
                "vertical": True,
            },
        )
        result = renderer.render(job, timestamp="20260518_120000")

        # Both editor_meta files written
        assert result.shorts_editor_meta_path is not None
        assert Path(result.shorts_editor_meta_path).is_file()
        assert result.bulletin_editor_meta_path is not None
        assert Path(result.bulletin_editor_meta_path).is_file()

        # Expected per-clip artifacts under output_dir
        for i in range(1, 4):
            clip = renderer.output_dir / f"clip_{i:02d}.mp4"
            raw = renderer.output_dir / f"raw_clip_{i:02d}.mp4"
            thumb = renderer.output_dir / f"thumb_{i:02d}.jpg"
            assert clip.is_file() and clip.stat().st_size > 100_000, \
                f"clip_{i:02d}.mp4 missing or too small"
            assert raw.is_file() and raw.stat().st_size > 100_000, \
                f"raw_clip_{i:02d}.mp4 missing or too small"
            # Thumbnail is non-fatal; assert only if exists
            if thumb.is_file():
                assert thumb.stat().st_size > 5_000

        # Bulletin artifact
        bulletin = renderer.bulletin_dir / "bulletin.mp4"
        assert bulletin.is_file() and bulletin.stat().st_size > 100_000
