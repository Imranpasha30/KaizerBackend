"""Stage 3a -- Shorts Generator (Gemini 2.5 Flash, temperature 0.7).

Takes the clean transcript + canonical entities and emits 3-10
``ShortsCut`` entries: 15-60s short-form video segments suitable for
1080x1920 vertical render. The 15-60s duration is enforced at three
levels:

  1. Prompt body -- explicit "DURATION CONSTRAINT" block at the top
     of stage_3a_prompt.md (added in Step 12.2a re-run #3 after
     Gemini was observed non-deterministically emitting <15s shorts).
  2. ``ShortsCut.@field_validator("end_sec")`` -- raises
     ValidationError per offending entry.
  3. Lenient per-entry parse in ``_parse_response`` -- invalid
     entries are DROPPED with a structured ``stage_3a_dropped_invalid_short``
     log warning rather than sinking the entire response. The 3-tier
     outcome logic in ``generate`` decides whether to accept, retry,
     or raise based on the count of surviving valid entries.

SDK pattern mirrors Stage 2 / Stage 2.5: manual ``model_validate``
path (NEVER ``response.parsed``). Defense changes for 12.2a re-run #3:
  * Per-entry validation in ``_parse_response`` so a single duration
    violator doesn't sink the whole batch (Option E from the
    operator's locked spec).
  * 3-tier outcome on attempt 1: >=5 valid -> success; 3-4 valid ->
    retry; <3 valid -> retry. On attempt 2: >=3 accept (degraded
    output beats total failure for Beta); <3 raise.

Temperature is 0.7 (per D-7.5) -- this is creative work: picking
"interesting moments" benefits from variety. The Pydantic schema
constraints + 15-60s validator do the safety work.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    ShortsCut,
    Stage3aOutput,
)

logger = logging.getLogger("pipeline_v2.stage_3a")


# --- Defaults (locked per Step 7 decisions) --------------------------

DEFAULT_MODEL = "gemini-2.5-flash"       # D-7.5 / Step 6 D7
DEFAULT_TEMPERATURE = 0.7                # D-7.5: creative
DEFAULT_THINKING_BUDGET = 512            # D-7.6 pushback: conservative
DEFAULT_MAX_OUTPUT_TOKENS = 4096         # D-7.7
DEFAULT_PROMPT_PATH = Path(__file__).parent / "stage_3a_prompt.md"


# --- Outcome thresholds (Option E, locked 12.2a re-run #3) ------------

# Stage3aOutput.shorts_cuts has min_length=3, max_length=10 (models.py).
# Mirror those caps here so the lenient parser respects the same band
# without importing private Pydantic state.
_STAGE3A_MIN_SHORTS = 3      # min valid shorts to accept any output
_STAGE3A_MAX_SHORTS = 10     # cap (Stage3aOutput.max_length)

# Healthy band: attempt-1 outputs at or above this count return
# immediately without a retry. Below the band but >= min triggers a
# corrective retry (the bulletin needs variety; 3-4 shorts is
# degraded). Locked at 5 per the operator spec.
_STAGE3A_HEALTHY_MIN_SHORTS = 5


# --- Lenient-parse result + structured drop log ----------------------


@dataclass(frozen=True)
class _ParseResult:
    """Outcome of one Gemini-response parse attempt."""

    output: Optional[Stage3aOutput]
    """Constructed Stage3aOutput when valid_count >= _STAGE3A_MIN_SHORTS;
    otherwise None. Caller's 3-tier outcome logic decides whether to
    return it, retry, or raise."""

    valid_count: int
    """Number of shorts_cuts entries that passed individual validation
    (after capping at _STAGE3A_MAX_SHORTS)."""

    dropped_count: int
    """Number of shorts_cuts entries that failed individual validation
    and were dropped with structured log warnings."""


def _log_dropped_short(
    attempt: int,
    index_in_response: int,
    entry: dict,
    exc: ValidationError,
) -> None:
    """Structured-log a dropped shorts_cut entry.

    The ``extra`` payload is the telemetry signal that drives future
    prompt refinement: a high rate of ``below_15s`` violations on
    attempt 1 across many jobs is a sign Gemini Flash isn't getting
    the duration rule from the prompt and we should strengthen the
    constraint block (or switch to Stage3aOutput.with_response_schema).
    """
    start = entry.get("start_sec") if isinstance(entry, dict) else None
    end = entry.get("end_sec") if isinstance(entry, dict) else None
    duration: Optional[float] = None
    violation = "other"
    try:
        if start is not None and end is not None:
            duration = float(end) - float(start)
            if duration < 15.0:
                violation = "below_15s"
            elif duration > 60.0:
                violation = "above_60s"
    except (TypeError, ValueError):
        pass

    hook = entry.get("hook") if isinstance(entry, dict) else None
    hook_preview = (str(hook)[:60]) if hook else None

    logger.warning(
        "stage_3a_dropped_invalid_short",
        extra={
            "event":               "stage_3a_dropped_invalid_short",
            "attempt":             attempt,
            "index_in_response":   index_in_response,
            "start_sec":           start,
            "end_sec":             end,
            "duration_sec":        duration,
            "duration_violation":  violation,
            "hook_preview":        hook_preview,
            "validation_error":    str(exc),
        },
    )


# --- Helpers ---------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Same defensive strip as Stage 2 / 2.5. Duplicated per-stage so
    a future maintainer can delete one stage without breaking the
    other two.
    """
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
    """Render clean words as compact JSON. Each entry includes the
    clean-array index so shorts_cut start_sec/end_sec can be tied to
    word boundaries (the LLM is told to align cut edges to word
    boundaries in the prompt).
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


def _serialize_entities_for_prompt(entities: list[Entity]) -> str:
    """Compact entity list for context. Stage 3a doesn't reference
    entities in its OUTPUT, but seeing them in the input prompt helps
    the model rank which moments matter (entity-dense moments are
    more share-worthy).
    """
    arr = [
        {
            "canonical_name": e.canonical_name,
            "type": e.type.value,
            "mention_count": len(e.mentions),
        }
        for e in entities
    ]
    return json.dumps(arr, ensure_ascii=False)


# --- Stage entry -----------------------------------------------------


class Stage3aShortsGenerator:
    """Gemini 2.5 Flash shorts generator.

    Usage::

        gen = Stage3aShortsGenerator()
        out = await gen.generate(clean_transcript, entities)
        # out: Stage3aOutput with 3-10 ShortsCut entries (each 15-60s)

    Constructor overrides are kwargs-only.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        prompt_path: Optional[Path] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
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
                f"Stage 3a prompt template not found at {self.prompt_path}. "
                f"Step 7.2 fills this in. For testing earlier, pass "
                f"prompt_path= explicitly to the constructor."
            )
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    # ---- prompt construction ----------------------------------------

    def _build_prompt(
        self,
        clean: CleanTranscript,
        entities: list[Entity],
        correction_note: str = "",
    ) -> str:
        template = self._load_prompt()
        words_json = _serialize_clean_transcript_for_prompt(clean)
        entities_json = _serialize_entities_for_prompt(entities)
        word_count = len(clean.words)
        duration_sec = clean.words[-1].e if clean.words else 0.0

        prompt = (
            f"{template}\n\n"
            f"## Clean transcript metadata\n"
            f"- Word count:        {word_count}\n"
            f"- Total duration:    {duration_sec:.1f}s\n"
            f"- Canonical entities (context only -- do NOT reference "
            f"in your output): {entities_json}\n"
            f"\n"
            f"## Clean transcript (idx = clean-array index)\n"
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
            response_schema=Stage3aOutput,
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

    def _parse_response(
        self, response, attempt: int,
    ) -> "_ParseResult":
        """Lenient per-entry parse (Option E, locked 12.2a re-run #3).

        Behaviour:
          1. Parse JSON to dict; raise ``json.JSONDecodeError`` if the
             text is empty or non-JSON.
          2. Iterate ``data.get("shorts_cuts", [])`` and validate each
             entry individually via ``ShortsCut.model_validate``.
          3. Invalid entries are DROPPED with a structured
             ``stage_3a_dropped_invalid_short`` log warning carrying
             enough telemetry to drive future prompt refinement.
          4. Cap valid entries at the Stage3aOutput max_length=10.
          5. If >= 3 valid entries: construct and return Stage3aOutput.
             If < 3 valid entries: return _ParseResult with output=None;
             caller decides retry vs raise.

        ``attempt`` (1 or 2) is recorded in the dropped-entry log so
        post-launch telemetry can distinguish first-attempt drops from
        retry drops.
        """
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

        raw_cuts = data.get("shorts_cuts", []) or []

        valid_cuts: list[ShortsCut] = []
        dropped_count = 0
        for idx, entry in enumerate(raw_cuts):
            try:
                valid_cuts.append(ShortsCut.model_validate(entry))
            except ValidationError as exc:
                dropped_count += 1
                _log_dropped_short(attempt, idx, entry, exc)

        # Cap at Stage3aOutput.max_length so the final
        # Stage3aOutput construction never trips on count.
        capped_cuts = valid_cuts[: _STAGE3A_MAX_SHORTS]

        # Renumber surviving shorts to 0-based contiguous indices.
        # Stage 3a's prompt asks Gemini to emit 0..N-1 sequentially.
        # Option E's lenient-drop step breaks that invariant when any
        # entry fails per-entry validation. Downstream contracts
        # (notably build_v1_shorts_editor_meta's D-8.12 contiguity
        # guardrail) require contiguous indices. Re-establish here.
        renumbered_cuts = [
            sc.model_copy(update={"index": i})
            for i, sc in enumerate(capped_cuts)
        ]

        # Defensive: if model_copy fails to set index (Pydantic edge
        # case), we want to fail loudly at Stage 3a's boundary, not
        # cascade into Stage 4's adapter where the error message is
        # less actionable.
        for i, sc in enumerate(renumbered_cuts):
            assert sc.index == i, (
                f"renumber post-condition failed at position {i}: "
                f"expected index={i}, got index={sc.index}"
            )

        if len(renumbered_cuts) >= _STAGE3A_MIN_SHORTS:
            output = Stage3aOutput(shorts_cuts=renumbered_cuts)
        else:
            output = None

        return _ParseResult(
            output=output,
            valid_count=len(renumbered_cuts),
            dropped_count=dropped_count,
        )

    # ---- Public entry -----------------------------------------------

    async def generate(
        self,
        clean: CleanTranscript,
        entities: list[Entity],
    ) -> Stage3aOutput:
        """Generate 3-10 short-form clip selections.

        Locked 3-tier outcome (Option E, 12.2a re-run #3):

          Attempt 1:
            * >= 5 valid shorts -> return immediately (healthy).
            * 3-4 valid shorts -> trigger corrective retry (degraded;
              we want >=5 for variety on the shorts pass).
            * < 3 valid shorts -> trigger corrective retry (insufficient).
            * JSON parse failure -> trigger corrective retry.

          Attempt 2:
            * >= 3 valid shorts -> accept (Beta degraded-output
              acceptance: a working bulletin with 3 shorts beats
              total failure).
            * < 3 valid shorts or JSON parse failure -> raise
              RuntimeError. Inngest's outer retry handles if applicable.
        """
        base_prompt = self._build_prompt(clean, entities)

        first_summary: str = ""
        first_result: Optional[_ParseResult] = None
        try:
            response = await self._call_gemini(base_prompt)
            first_result = self._parse_response(response, attempt=1)
        except json.JSONDecodeError as exc:
            first_summary = f"JSONDecodeError: {exc!r}"
            logger.warning(
                "stage_3a: attempt 1 yielded unparseable JSON: %s. "
                "Issuing corrective retry.", exc,
            )
        else:
            if first_result.valid_count >= _STAGE3A_HEALTHY_MIN_SHORTS:
                return first_result.output
            first_summary = (
                f"only {first_result.valid_count} valid shorts "
                f"({first_result.dropped_count} dropped); need >="
                f"{_STAGE3A_HEALTHY_MIN_SHORTS} for the no-retry path"
            )
            logger.warning(
                "stage_3a: attempt 1 produced %d valid / %d dropped "
                "shorts. Issuing corrective retry.",
                first_result.valid_count, first_result.dropped_count,
            )

        correction_note = (
            "Your previous response had issues:\n"
            f"  {first_summary}\n\n"
            "Regenerate a JSON response that conforms EXACTLY to the "
            "Stage3aOutput schema. Do not wrap in markdown code "
            "fences. Emit only the JSON object. Every shorts_cut MUST "
            "have (end_sec - start_sec) BETWEEN 15.0 AND 60.0 seconds "
            "inclusive. Emit 5-7 shorts (3 minimum). importance is "
            "1-10 inclusive. The hook is a short attention-grabbing "
            "phrase, not a full sentence. Before emitting each entry, "
            "verify duration is in the 15.0-60.0 second range."
        )
        retry_prompt = self._build_prompt(
            clean, entities, correction_note=correction_note,
        )
        try:
            response = await self._call_gemini(retry_prompt)
            retry_result = self._parse_response(response, attempt=2)
        except json.JSONDecodeError as second_exc:
            raise RuntimeError(
                f"Stage 3a corrective retry produced unparseable JSON: "
                f"{second_exc!r}. Attempt 1: {first_summary}. "
                f"Inngest's outer retry will handle if applicable."
            ) from second_exc

        if retry_result.valid_count >= _STAGE3A_MIN_SHORTS:
            if retry_result.valid_count < _STAGE3A_HEALTHY_MIN_SHORTS:
                logger.warning(
                    "stage_3a: accepting degraded retry output: %d "
                    "valid / %d dropped shorts. Beta acceptance: a "
                    "working bulletin with fewer shorts beats total "
                    "failure.",
                    retry_result.valid_count, retry_result.dropped_count,
                )
            return retry_result.output

        raise RuntimeError(
            f"Stage 3a failed after corrective retry. "
            f"Attempt 1: {first_summary}. "
            f"Attempt 2: {retry_result.valid_count} valid / "
            f"{retry_result.dropped_count} dropped (need >="
            f"{_STAGE3A_MIN_SHORTS}). "
            f"Inngest's outer retry will handle if applicable."
        )
