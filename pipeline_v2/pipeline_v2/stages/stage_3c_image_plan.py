"""Stage 3c -- Image Plan Generator (Gemini 2.5 Flash, temperature 0.2).

Takes the clean transcript + full_video_cuts + canonical entities
and emits a list of image overlays scheduled inside bulletin clips.
Each ``ImagePlanEntry`` references an entity by name (must match a
canonical_entity) and a clip by index (must be a valid full_video_cut
index), with a ``show_at_sec`` / ``duration_sec`` window that MUST
fall inside the referenced clip's time range.

LLM contract: ``ImagePlan`` (passed as ``response_schema=`` directly).

SDK pattern: manual ``model_validate`` path, 1 corrective retry on
JSONDecodeError / ValidationError, RuntimeError on second failure.

Temperature is 0.2 (per D-7.5) -- this is purely structural work:
every output field is bound to existing input (an entity name, a
clip index, a time range). Creativity here = hallucinations.

Post-validate filter (the heart of Stage 3c per D-7.10):

  1. Build a set of valid entity names (from canonical_entities).
  2. Build a dict clip_index -> (start_sec, end_sec).
  3. For each ImagePlanEntry, check three invariants:
     a. ``entity_name`` is in the valid-names set
     b. ``clip_index`` is in the valid-indices dict
     c. ``[show_at_sec, show_at_sec + duration_sec]`` is entirely
        inside ``[clip_start, clip_end]``
  4. Drop violating entries with a structured logger.warning() (per
     entry, naming the violation + the offending fields).
  5. **D-7.10 GUARDRAIL**: if (dropped / total) > 0.5, raise
     RuntimeError. That ratio indicates systemic prompt/model
     failure, not occasional outliers -- let Inngest retry.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    FullVideoCut,
    ImagePlan,
    ImagePlanEntry,
)

logger = logging.getLogger("pipeline_v2.stage_3c")


# --- Defaults (locked per Step 7 decisions) --------------------------

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2                # D-7.5: structural
DEFAULT_THINKING_BUDGET = 0              # D-7.6: mechanical task
DEFAULT_MAX_OUTPUT_TOKENS = 4096         # D-7.7
DEFAULT_DROP_RATIO_THRESHOLD = 0.5       # D-7.10 hard-fail guardrail
DEFAULT_PROMPT_PATH = Path(__file__).parent / "stage_3c_prompt.md"


# --- Helpers ---------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _serialize_clean_transcript_for_prompt(clean: CleanTranscript) -> str:
    """Compact word array for Stage 3c. Indices matter because the
    image plan's `show_at_sec` must align with clip word boundaries.
    """
    arr = [
        {
            "idx": i,
            "w": w.w,
            "s": round(float(w.s), 3),
            "e": round(float(w.e), 3),
        }
        for i, w in enumerate(clean.words)
    ]
    return json.dumps(arr, ensure_ascii=False)


def _serialize_full_video_cuts_for_prompt(cuts: list[FullVideoCut]) -> str:
    arr = [
        {
            "index": c.index,
            "start_sec": round(float(c.start_sec), 3),
            "end_sec": round(float(c.end_sec), 3),
            "importance": c.importance,
        }
        for c in cuts
    ]
    return json.dumps(arr, ensure_ascii=False)


def _serialize_entities_for_prompt(entities: list[Entity]) -> str:
    """Entity list serialized for Stage 3c. Reduced to ONLY the fields
    Stage 3c needs (canonical_name, native_name, type). Mention
    indices are not relevant here -- the boundary check is on clip
    range, not mention range.
    """
    arr = [
        {
            "canonical_name": e.canonical_name,
            "native_name": e.native_name,
            "type": e.type.value,
        }
        for e in entities
    ]
    return json.dumps(arr, ensure_ascii=False)


# --- Post-validate filter (D-7.10 -- the heart of Stage 3c) ----------


def _validate_and_filter(
    plan: ImagePlan,
    full_video_cuts: list[FullVideoCut],
    entities: list[Entity],
    drop_ratio_threshold: float = DEFAULT_DROP_RATIO_THRESHOLD,
) -> ImagePlan:
    """Apply the three invariants. Drop violations with warnings.

    Hard-fail guardrail per D-7.10: if more than
    ``drop_ratio_threshold`` (default 50%) of entries are dropped,
    raise RuntimeError. That ratio indicates systemic prompt/model
    failure, not occasional outliers.
    """
    valid_entity_names = {e.canonical_name for e in entities}
    cut_ranges: dict[int, tuple[float, float]] = {
        c.index: (c.start_sec, c.end_sec) for c in full_video_cuts
    }

    total = len(plan.entries)
    kept: list[ImagePlanEntry] = []
    dropped_reasons: list[dict] = []

    for entry in plan.entries:
        # Invariant 1: entity_name must match a canonical entity
        if entry.entity_name not in valid_entity_names:
            dropped_reasons.append({
                "entity_name": entry.entity_name,
                "clip_index": entry.clip_index,
                "reason": "orphan_entity_name",
            })
            logger.warning(
                "stage_3c: dropping ImagePlanEntry -- entity_name "
                "'%s' not in canonical_entities (clip_index=%d, "
                "show_at_sec=%.2f). Valid names: %s",
                entry.entity_name, entry.clip_index, entry.show_at_sec,
                sorted(valid_entity_names),
            )
            continue

        # Invariant 2: clip_index must reference a real cut
        if entry.clip_index not in cut_ranges:
            dropped_reasons.append({
                "entity_name": entry.entity_name,
                "clip_index": entry.clip_index,
                "reason": "invalid_clip_index",
            })
            logger.warning(
                "stage_3c: dropping ImagePlanEntry -- clip_index=%d "
                "not in full_video_cuts (valid indices: %s). "
                "entity_name='%s', show_at_sec=%.2f",
                entry.clip_index, sorted(cut_ranges.keys()),
                entry.entity_name, entry.show_at_sec,
            )
            continue

        # Invariant 3: [show_at, show_at + duration] inside [clip_start, clip_end]
        clip_start, clip_end = cut_ranges[entry.clip_index]
        overlay_end = entry.show_at_sec + entry.duration_sec
        if entry.show_at_sec < clip_start or overlay_end > clip_end:
            dropped_reasons.append({
                "entity_name": entry.entity_name,
                "clip_index": entry.clip_index,
                "show_at_sec": entry.show_at_sec,
                "overlay_end": overlay_end,
                "clip_range": (clip_start, clip_end),
                "reason": "boundary_violation",
            })
            logger.warning(
                "stage_3c: dropping ImagePlanEntry -- boundary "
                "violation. entity_name='%s', clip_index=%d, "
                "overlay [%.2f, %.2f] not inside clip [%.2f, %.2f].",
                entry.entity_name, entry.clip_index,
                entry.show_at_sec, overlay_end, clip_start, clip_end,
            )
            continue

        kept.append(entry)

    dropped_count = total - len(kept)
    if total > 0 and (dropped_count / total) > drop_ratio_threshold:
        raise RuntimeError(
            f"Stage 3c dropped {dropped_count}/{total} entries "
            f"({dropped_count/total:.0%} > "
            f"{drop_ratio_threshold:.0%} threshold). This indicates "
            f"systemic prompt/model failure, not occasional "
            f"outliers. Dropped reasons: {dropped_reasons!r}. "
            f"Inngest will retry the step."
        )

    if dropped_count:
        logger.info(
            "stage_3c: kept %d/%d entries after post-validate "
            "(dropped %d below the %d%% threshold).",
            len(kept), total, dropped_count,
            int(drop_ratio_threshold * 100),
        )

    return ImagePlan(entries=kept)


# --- Stage entry -----------------------------------------------------


class Stage3cImagePlanner:
    """Gemini 2.5 Flash image-plan generator.

    Usage::

        planner = Stage3cImagePlanner()
        plan = await planner.plan(clean, full_video_cuts, entities)
        # plan: ImagePlan with all entries post-validated
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        drop_ratio_threshold: float = DEFAULT_DROP_RATIO_THRESHOLD,
        prompt_path: Optional[Path] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.drop_ratio_threshold = drop_ratio_threshold
        self.prompt_path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        self._client: Optional[genai.Client] = None
        self._prompt_template: Optional[str] = None

    # ---- lazy resources ---------------------------------------------

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY env var is empty/unset. Add it to "
                "KaizerBackend/.env."
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _load_prompt(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        if not self.prompt_path.is_file():
            raise FileNotFoundError(
                f"Stage 3c prompt template not found at {self.prompt_path}. "
                f"Step 7.4 fills this in. For testing earlier, pass "
                f"prompt_path= explicitly to the constructor."
            )
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    # ---- prompt construction ----------------------------------------

    def _build_prompt(
        self,
        clean: CleanTranscript,
        full_video_cuts: list[FullVideoCut],
        entities: list[Entity],
        correction_note: str = "",
    ) -> str:
        template = self._load_prompt()
        words_json = _serialize_clean_transcript_for_prompt(clean)
        cuts_json = _serialize_full_video_cuts_for_prompt(full_video_cuts)
        entities_json = _serialize_entities_for_prompt(entities)

        prompt = (
            f"{template}\n\n"
            f"## Inputs\n"
            f"### Canonical entities (entity_name MUST match one of "
            f"these canonical_name values)\n"
            f"{entities_json}\n"
            f"\n"
            f"### Full video cuts (clip_index MUST be one of these "
            f"indices; show_at + duration MUST fall inside the "
            f"clip's [start_sec, end_sec])\n"
            f"{cuts_json}\n"
            f"\n"
            f"### Clean transcript word array\n"
            f"{words_json}\n"
        )
        if correction_note:
            prompt += f"\n## Correction note\n{correction_note}\n"
        return prompt

    # ---- Gemini call + parse ----------------------------------------

    async def _call_gemini(self, prompt: str):
        client = self._get_client()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ImagePlan,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            ),
        )
        return await client.aio.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=config,
        )

    def _parse_response(self, response) -> ImagePlan:
        raw_text = getattr(response, "text", "") or ""
        if not raw_text.strip():
            raise json.JSONDecodeError(
                "response.text is empty (Gemini returned no JSON body; "
                "check finish_reason for MAX_TOKENS truncation or "
                "thinking-budget exhaustion)",
                "",
                0,
            )
        cleaned = _strip_markdown_fences(raw_text)
        data = json.loads(cleaned)
        return ImagePlan.model_validate(data)

    # ---- Public entry -----------------------------------------------

    async def plan(
        self,
        clean: CleanTranscript,
        full_video_cuts: list[FullVideoCut],
        entities: list[Entity],
    ) -> ImagePlan:
        """Generate post-validated image plan.

        Same corrective-retry pattern + the D-7.10 drop-ratio
        guardrail (raises RuntimeError if >50% of entries violate
        invariants).
        """
        base_prompt = self._build_prompt(clean, full_video_cuts, entities)

        first_error: Optional[Exception] = None
        try:
            response = await self._call_gemini(base_prompt)
            raw_plan = self._parse_response(response)
            return _validate_and_filter(
                raw_plan, full_video_cuts, entities,
                drop_ratio_threshold=self.drop_ratio_threshold,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            first_error = exc
            logger.warning(
                "stage_3c: first attempt failed validation: %s. "
                "Issuing 1 corrective retry with the error appended.",
                exc,
            )

        correction_note = (
            "Your previous response failed validation with this error:\n"
            f"  {first_error}\n\n"
            "Regenerate a JSON response that conforms EXACTLY to the "
            "ImagePlan schema. Do not wrap in markdown code fences. "
            "Emit only the JSON object. Each entry's `entity_name` "
            "MUST match a canonical_entity's canonical_name. Each "
            "entry's `clip_index` MUST be one of the listed "
            "full_video_cuts. `show_at_sec + duration_sec` MUST fall "
            "ENTIRELY inside the referenced clip's [start_sec, "
            "end_sec]. `duration_sec` MUST be >= 2.0."
        )
        retry_prompt = self._build_prompt(
            clean, full_video_cuts, entities,
            correction_note=correction_note,
        )
        try:
            response = await self._call_gemini(retry_prompt)
            raw_plan = self._parse_response(response)
            return _validate_and_filter(
                raw_plan, full_video_cuts, entities,
                drop_ratio_threshold=self.drop_ratio_threshold,
            )
        except (json.JSONDecodeError, ValidationError) as second_exc:
            raise RuntimeError(
                f"Stage 3c failed after corrective retry. "
                f"First error: {first_error!r}. "
                f"Second error: {second_exc!r}. "
                f"Inngest's outer retry will handle if applicable."
            ) from second_exc
