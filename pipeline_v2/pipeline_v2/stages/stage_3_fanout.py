"""Stage 3 fan-out helper -- runs 3a / 3b / 3c in parallel.

Per D-7.4: ONE Inngest step calls ``Stage3FanOut.run(...)`` which
fires three concurrent Gemini Flash calls via ``asyncio.gather``.
Wasted-retry cost on any single sub-stage failure is acceptable
because:

  - Each sub-stage is short (~2-5s wall on Flash).
  - Single-stage retries (one of 3a/3b/3c stochastically flaking)
    are already handled by each class's own corrective-retry layer.
  - All-three-retry triggers ONLY on a systemic failure, where
    Inngest's exponential backoff is the right outer layer.

The return type is a plain ``Stage3Output`` dataclass -- not a
Pydantic model. All three sub-fields are ALREADY validated by their
respective stage classes; another layer of validation would be
ceremony.

Python 3.10 caveat: ``asyncio.gather`` with ``return_exceptions=False``
raises on first exception but does NOT auto-cancel siblings. The
non-failing stages continue running in the background, wasting some
API budget. In Python 3.11+ we'd use ``asyncio.TaskGroup`` for
clean cancellation; for now (3.10) we accept the bounded waste.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    FullVideoCut,
    ImagePlan,
    Metadata,
    ShortsCut,
)
from pipeline_v2.stages.stage_3a_shorts import Stage3aShortsGenerator
from pipeline_v2.stages.stage_3b_metadata import Stage3bMetadataExtractor
from pipeline_v2.stages.stage_3c_image_plan import Stage3cImagePlanner

logger = logging.getLogger("pipeline_v2.stage_3_fanout")


@dataclass(frozen=True)
class Stage3Output:
    """Carrier for the three Stage 3 sub-stage results.

    Each field is already validated by its producing class:
      - ``shorts_cuts`` from Stage3aShortsGenerator (incl. 15-60s
        duration validator, importance 1-10, 3-10 count band)
      - ``metadata`` from Stage3bMetadataExtractor (incl. video_type
        Literal, all 12 required fields)
      - ``image_plan`` from Stage3cImagePlanner (incl. the three
        post-validate invariants + the 50%-drop guardrail)

    No further validation in this carrier -- it's just a tuple-like
    grouping for the orchestrator to unpack into JobOutput.
    """
    shorts_cuts: list[ShortsCut]
    metadata: Metadata
    image_plan: ImagePlan


class Stage3FanOut:
    """Fan-out helper for Stage 3 (3a + 3b + 3c).

    Usage::

        fanout = Stage3FanOut()
        result = await fanout.run(clean, full_video_cuts, entities)
        # result.shorts_cuts, result.metadata, result.image_plan

    Constructor takes optional pre-built sub-stage instances so tests
    can inject mocks. Default constructor uses fresh instances of
    each sub-stage class with their respective defaults.
    """

    def __init__(
        self,
        *,
        shorts_generator: Optional[Stage3aShortsGenerator] = None,
        metadata_extractor: Optional[Stage3bMetadataExtractor] = None,
        image_planner: Optional[Stage3cImagePlanner] = None,
    ):
        self.shorts_generator = shorts_generator or Stage3aShortsGenerator()
        self.metadata_extractor = metadata_extractor or Stage3bMetadataExtractor()
        self.image_planner = image_planner or Stage3cImagePlanner()

    async def run(
        self,
        clean: CleanTranscript,
        full_video_cuts: list[FullVideoCut],
        entities: list[Entity],
    ) -> Stage3Output:
        """Execute 3a / 3b / 3c in parallel via ``asyncio.gather``.

        On any sub-stage failure, the exception propagates and the
        outer Inngest step retries. The two non-failing stages may
        complete in the background (Python 3.10 limitation) -- the
        Inngest step's subsequent retry will re-run all three.
        """
        logger.info(
            "stage_3_fanout: starting parallel fan-out (3a shorts + "
            "3b metadata + 3c image_plan) -- %d entities, %d clips, "
            "%d clean words",
            len(entities), len(full_video_cuts), len(clean.words),
        )

        shorts_task = self.shorts_generator.generate(clean, entities)
        metadata_task = self.metadata_extractor.extract(clean, entities)
        image_plan_task = self.image_planner.plan(
            clean, full_video_cuts, entities,
        )

        stage3a_out, metadata, image_plan = await asyncio.gather(
            shorts_task, metadata_task, image_plan_task,
        )

        logger.info(
            "stage_3_fanout: completed -- %d shorts, video_type=%s, "
            "%d image overlays",
            len(stage3a_out.shorts_cuts),
            metadata.video_type,
            len(image_plan.entries),
        )

        return Stage3Output(
            shorts_cuts=stage3a_out.shorts_cuts,
            metadata=metadata,
            image_plan=image_plan,
        )
