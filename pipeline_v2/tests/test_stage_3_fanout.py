"""Integration tests for Stage 3 fan-out -- Step 7.5.

The Stage3FanOut wires together 3a / 3b / 3c via asyncio.gather.
These tests:

  1. Confirm all three sub-stages are called concurrently with the
     correct inputs.
  2. Confirm the merged Stage3Output carries through each sub-stage's
     validated output without double-validation.
  3. Confirm exception in any one sub-stage propagates (Inngest's
     outer retry handles).
  4. Confirm the constructor accepts dependency-injected sub-stage
     instances (for production wiring + this test surface).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    FullVideoCut,
    ImagePlan,
    ImagePlanEntry,
    Metadata,
    ShortsCut,
    Stage3aOutput,
    Word,
)
from pipeline_v2.stages.stage_3_fanout import (
    Stage3FanOut,
    Stage3Output,
)


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _clean(n: int = 30) -> CleanTranscript:
    words = [Word(w=f"w{i}", s=i * 1.0, e=i * 1.0 + 0.8) for i in range(n)]
    return CleanTranscript(
        words=words,
        clip_boundaries={0: (0, n - 1)},
        source_word_map=list(range(n)),
    )


def _cut(index: int = 0, start: float = 0.0, end: float = 30.0) -> FullVideoCut:
    return FullVideoCut(
        index=index,
        start_word_idx=int(start), end_word_idx=int(end),
        start_sec=start, end_sec=end, importance=7,
    )


def _entity(name: str = "Modi") -> Entity:
    return Entity(
        canonical_name=name, native_name=name,
        first_mention_word_idx=0, type="PERSON", mentions=[0],
    )


def _shorts_output() -> Stage3aOutput:
    return Stage3aOutput(shorts_cuts=[
        ShortsCut(index=0, start_sec=0.0, end_sec=20.0,
                  hook="A hook", importance=8),
        ShortsCut(index=1, start_sec=25.0, end_sec=55.0,
                  hook="B hook", importance=7),
        ShortsCut(index=2, start_sec=60.0, end_sec=80.0,
                  hook="C hook", importance=6),
    ])


def _metadata() -> Metadata:
    return Metadata(
        video_type="SOLO",
        language="te-en",
        total_speakers=1,
        overall_summary="Summary",
        overall_summary_native="సారాంశం",
        shorts_headline_native="హెడ్‌లైన్",
        bulletin_marquee_points=["A", "B", "C"],
        image_search_queries=["q1", "q2"],
        key_people=["Modi"],
        key_people_native=["మోదీ"],
        key_topics=["topic1"],
        key_locations=["Hyderabad"],
    )


def _image_plan() -> ImagePlan:
    return ImagePlan(entries=[
        ImagePlanEntry(
            entity_name="Modi", entity_name_native="మోదీ",
            description="PM at podium",
            clip_index=0, show_at_sec=5.0, duration_sec=4.0,
        ),
    ])


def _make_fanout(*, shorts=None, metadata=None, image_plan=None) -> Stage3FanOut:
    """Construct Stage3FanOut with mocked sub-stages."""
    shorts_gen = MagicMock()
    shorts_gen.generate = AsyncMock(return_value=shorts or _shorts_output())

    metadata_ext = MagicMock()
    metadata_ext.extract = AsyncMock(return_value=metadata or _metadata())

    image_planner = MagicMock()
    image_planner.plan = AsyncMock(return_value=image_plan or _image_plan())

    return Stage3FanOut(
        shorts_generator=shorts_gen,
        metadata_extractor=metadata_ext,
        image_planner=image_planner,
    )


# ====================================================================== #
# Happy path: all three sub-stages succeed                                #
# ====================================================================== #


class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_returns_merged_stage3output(self):
        fanout = _make_fanout()
        result = await fanout.run(_clean(30), [_cut()], [_entity()])
        assert isinstance(result, Stage3Output)
        assert len(result.shorts_cuts) == 3
        assert result.metadata.video_type == "SOLO"
        assert len(result.image_plan.entries) == 1

    @pytest.mark.asyncio
    async def test_each_sub_stage_called_once_with_correct_inputs(self):
        fanout = _make_fanout()
        clean = _clean(20)
        cuts = [_cut(0, 0.0, 20.0), _cut(1, 20.0, 40.0)]
        entities = [_entity("Modi"), _entity("Reddy")]

        await fanout.run(clean, cuts, entities)

        # Stage 3a: generate(clean, entities)
        fanout.shorts_generator.generate.assert_awaited_once_with(
            clean, entities,
        )
        # Stage 3b: extract(clean, entities)
        fanout.metadata_extractor.extract.assert_awaited_once_with(
            clean, entities,
        )
        # Stage 3c: plan(clean, full_video_cuts, entities)
        fanout.image_planner.plan.assert_awaited_once_with(
            clean, cuts, entities,
        )

    @pytest.mark.asyncio
    async def test_sub_stage_outputs_carry_through_unchanged(self):
        # No double-validation -- the carrier preserves identity.
        custom_shorts = _shorts_output()
        custom_meta = _metadata()
        custom_plan = _image_plan()

        fanout = _make_fanout(
            shorts=custom_shorts,
            metadata=custom_meta,
            image_plan=custom_plan,
        )
        result = await fanout.run(_clean(10), [_cut()], [_entity()])

        # shorts_cuts: same list (carrier unpacks .shorts_cuts attribute)
        assert result.shorts_cuts == custom_shorts.shorts_cuts
        # metadata: same object
        assert result.metadata is custom_meta
        # image_plan: same object
        assert result.image_plan is custom_plan


# ====================================================================== #
# Exception propagation                                                   #
# ====================================================================== #


class TestExceptionPropagation:
    @pytest.mark.asyncio
    async def test_stage_3a_failure_raises(self):
        fanout = _make_fanout()
        fanout.shorts_generator.generate = AsyncMock(
            side_effect=RuntimeError("3a corrective retry failed"),
        )
        with pytest.raises(RuntimeError, match="3a corrective"):
            await fanout.run(_clean(10), [_cut()], [_entity()])

    @pytest.mark.asyncio
    async def test_stage_3b_failure_raises(self):
        fanout = _make_fanout()
        fanout.metadata_extractor.extract = AsyncMock(
            side_effect=RuntimeError("3b corrective retry failed"),
        )
        with pytest.raises(RuntimeError, match="3b corrective"):
            await fanout.run(_clean(10), [_cut()], [_entity()])

    @pytest.mark.asyncio
    async def test_stage_3c_failure_raises(self):
        fanout = _make_fanout()
        fanout.image_planner.plan = AsyncMock(
            side_effect=RuntimeError("3c >50% dropped"),
        )
        with pytest.raises(RuntimeError, match="3c >50"):
            await fanout.run(_clean(10), [_cut()], [_entity()])

    @pytest.mark.asyncio
    async def test_non_runtime_exception_propagates(self):
        # ConnectionError (no corrective retry) propagates as-is.
        fanout = _make_fanout()
        fanout.shorts_generator.generate = AsyncMock(
            side_effect=ConnectionError("net down"),
        )
        with pytest.raises(ConnectionError, match="net down"):
            await fanout.run(_clean(10), [_cut()], [_entity()])


# ====================================================================== #
# Constructor                                                             #
# ====================================================================== #


class TestConstructor:
    def test_default_constructor_creates_real_sub_stages(self, monkeypatch):
        # Fake API key so construction doesn't blow up at lazy
        # client init time (it won't actually be called in this test).
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        fanout = Stage3FanOut()
        # All three sub-stages instantiated with defaults
        assert fanout.shorts_generator.__class__.__name__ == "Stage3aShortsGenerator"
        assert fanout.metadata_extractor.__class__.__name__ == "Stage3bMetadataExtractor"
        assert fanout.image_planner.__class__.__name__ == "Stage3cImagePlanner"

    def test_dependency_injection_kwargs(self):
        # All three sub-stages can be injected
        s = MagicMock(); m = MagicMock(); i = MagicMock()
        fanout = Stage3FanOut(
            shorts_generator=s,
            metadata_extractor=m,
            image_planner=i,
        )
        assert fanout.shorts_generator is s
        assert fanout.metadata_extractor is m
        assert fanout.image_planner is i


# ====================================================================== #
# Stage3Output dataclass                                                  #
# ====================================================================== #


class TestStage3OutputDataclass:
    def test_frozen(self):
        # @dataclass(frozen=True) -- can't mutate after construction
        out = Stage3Output(
            shorts_cuts=_shorts_output().shorts_cuts,
            metadata=_metadata(),
            image_plan=_image_plan(),
        )
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            out.metadata = _metadata()

    def test_holds_three_fields(self):
        out = Stage3Output(
            shorts_cuts=_shorts_output().shorts_cuts,
            metadata=_metadata(),
            image_plan=_image_plan(),
        )
        assert hasattr(out, "shorts_cuts")
        assert hasattr(out, "metadata")
        assert hasattr(out, "image_plan")
