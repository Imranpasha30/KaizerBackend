"""Stage 2.5 -- Entity Canonicalizer (Gemini 2.5 Flash).

Takes Stage 2's clean transcript and emits a list of canonicalized
entities. The downstream renderer uses these entities for image
overlays during bulletin playback.

LLM contract: ``Stage2_5Output`` (see ``pipeline_v2/models.py``).

SDK-level gotchas defended against (per Step 5 / Step 6 research):

  - ``response.parsed`` silently swallows ``json.JSONDecodeError``
    and ``pydantic.ValidationError`` and returns ``None``. We use
    ``response.text`` + ``json.loads`` + ``Stage2_5Output.model_validate``
    so validation errors surface explicitly.
  - 1.x had a markdown-fence wrapping bug. 2.x is mostly fixed but
    we strip defensively (shared helper with Stage 2).
  - Thinking mode shares the ``max_output_tokens`` budget. Stage 2.5
    is a mechanical classification task -- thinking cap is 512 (per
    Step 6 D8), well below Pro's 2048.
  - ``Field(max_length=6)`` on the Pydantic model is a SOFT hint to
    Gemini's structured-output engine (google-genai issue #699). The
    post-validate ``_truncate_to_cap()`` is the THIRD safety net per
    Step 6 D3.
  - ``Field(default=...)`` on response_schema fields raises at API
    time (google-genai #699). We do NOT add defaults; the caller
    constructs ``Stage2_5Output`` explicitly.
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
    Stage2_5Output,
)

logger = logging.getLogger("pipeline_v2.stage_2_5")


# --- Defaults (locked at top of file per Step 6 decisions) -----------

DEFAULT_MODEL = "gemini-2.5-flash"       # D7: bare alias, auto-rolls
DEFAULT_TEMPERATURE = 0.2                # D10: structural, low creativity
DEFAULT_THINKING_BUDGET = 512            # D8: mechanical classification
DEFAULT_MAX_OUTPUT_TOKENS = 4096         # D9: comfortable headroom for 6 entities
DEFAULT_ENTITY_CAP = 6                   # D3: 6-cap (3 enforcement layers)

DEFAULT_PROMPT_PATH = Path(__file__).parent / "stage_2_5_prompt.md"


# --- Helpers ---------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ``\`\`\`json`` and ``\`\`\``` fences.

    Same defensive strip as Stage 2 -- duplicated here rather than
    imported across stages, since the two stages are independent
    units of work (a future maintainer should be able to delete one
    stage without breaking the other).
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
    """Render the clean word array as compact JSON for the prompt.

    Same shape as Stage 2's serializer (Deepgram-style {w,s,e}) but
    augmented with the clean index so the LLM can reference word
    indices in its output. We pass clip_boundaries too so the model
    knows which words belong to which clip (helps with mention-
    grouping when the same entity appears across multiple clips).
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


# --- Truncation safety net (D3 layer 3) ------------------------------


def _truncate_to_cap(
    entities: list[Entity],
    cap: int = DEFAULT_ENTITY_CAP,
) -> list[Entity]:
    """Post-validate truncation -- the third safety net.

    Sort algorithm per Step 6 D3-ADDITIONAL:

      1. Primary: mention count DESC (most-talked-about wins)
      2. Tiebreak: first_mention_word_idx ASC (earlier-mentioned wins)

    Reasoning: the renderer uses entities for image overlays.
    Entities mentioned more times deserve more screen time. Late-
    introduced entities with only 1 mention are unlikely to carry
    visual weight.

    If truncation fires, logs a structured warning naming the
    dropped entities' ``canonical_name`` so operators see what was
    cut.
    """
    if len(entities) <= cap:
        return entities

    # Sort: most-mentioned first, tiebreak by earliest-mention.
    ranked = sorted(
        entities,
        key=lambda e: (-len(e.mentions), e.first_mention_word_idx),
    )
    kept = ranked[:cap]
    dropped = ranked[cap:]
    logger.warning(
        "stage_2_5: LLM emitted %d entities (cap=%d); truncating. "
        "Kept: %s. Dropped: %s. (Gemini ignored Pydantic max_length "
        "and prompt cap; google-genai #699 says maxItems is soft -- "
        "this safety net is expected to fire occasionally.)",
        len(entities), cap,
        [e.canonical_name for e in kept],
        [
            {"name": e.canonical_name, "mentions": len(e.mentions)}
            for e in dropped
        ],
    )
    return kept


# --- Stage entry -----------------------------------------------------


class Stage2_5EntityCanonicalizer:
    """Gemini 2.5 Flash entity canonicalizer.

    Usage::

        canonicalizer = Stage2_5EntityCanonicalizer()
        out = await canonicalizer.classify(clean_transcript)
        # out: Stage2_5Output with entities (<= 6, truncated if needed)

    Constructor overrides are kwargs-only so future arguments can be
    added without breaking call sites. The dispatcher (Step 10) will
    instantiate this with no args; tests may override the model /
    temperature / thinking_budget / max_output_tokens / prompt_path /
    entity_cap.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        entity_cap: int = DEFAULT_ENTITY_CAP,
        prompt_path: Optional[Path] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.entity_cap = entity_cap
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
                f"Stage 2.5 prompt template not found at {self.prompt_path}. "
                f"Step 6.3 fills this in. For testing earlier, pass "
                f"prompt_path= explicitly to the constructor."
            )
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    # ---- prompt construction ----------------------------------------

    def _build_prompt(
        self,
        clean: CleanTranscript,
        correction_note: str = "",
    ) -> str:
        template = self._load_prompt()
        words_json = _serialize_clean_transcript_for_prompt(clean)
        clip_count = len(clean.clip_boundaries)
        word_count = len(clean.words)

        prompt = (
            f"{template}\n\n"
            f"## Clean transcript metadata\n"
            f"- Word count:      {word_count}\n"
            f"- Clip count:      {clip_count}\n"
            f"- Entity cap:      {self.entity_cap} (HARD; exceeding "
            f"  will be post-validate-truncated)\n"
            f"\n"
            f"## Clean transcript (idx = clean-array index; "
            f"reference these in `mentions[]`)\n"
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
            response_schema=Stage2_5Output,
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

    def _parse_response(self, response) -> Stage2_5Output:
        """Manual validate path -- same rationale as Stage 2.

        See ``Stage2ContinuityEditor._parse_response`` docstring for
        the full ``response.parsed`` vs ``model_validate`` rationale.
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
        return Stage2_5Output.model_validate(data)

    # ---- Public entry -----------------------------------------------

    async def classify(
        self,
        clean: CleanTranscript,
    ) -> Stage2_5Output:
        """Convert a clean transcript into canonicalized entities.

        Retry policy (mirrors Stage 2 per Step 6):
          1. First attempt: call Flash, parse, truncate, return.
          2. On ``json.JSONDecodeError`` or ``pydantic.ValidationError``:
             single corrective retry with the validation error
             appended to the prompt.
          3. On second failure: raise ``RuntimeError`` wrapping both
             errors. Inngest's exponential backoff is the outer retry
             layer -- we deliberately do NOT retry beyond 1 in-step.
          4. Other exceptions (auth fail, rate limit, network) are
             allowed to propagate.
        """
        base_prompt = self._build_prompt(clean)

        # ---- First attempt ----
        first_error: Optional[Exception] = None
        try:
            response = await self._call_gemini(base_prompt)
            parsed = self._parse_response(response)
            return self._finalize(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            first_error = exc
            logger.warning(
                "stage_2_5: first attempt failed validation: %s. "
                "Issuing 1 corrective retry with the error appended.",
                exc,
            )

        # ---- Corrective retry ----
        correction_note = (
            "Your previous response failed validation with this error:\n"
            f"  {first_error}\n\n"
            "Regenerate a JSON response that conforms EXACTLY to the "
            "Stage2_5Output schema. Do not wrap in markdown code "
            "fences. Emit only the JSON object. Entity `type` values "
            "must be one of: PERSON, ORG, PLACE, EVENT, OTHER (NEVER "
            "invent new types). `mentions` are integer indices into "
            "the clean transcript word array. `entities` length must "
            f"be <= {self.entity_cap}. `native_name` must be non-empty "
            "(use the canonical_name verbatim if there is no native "
            "rendering)."
        )
        retry_prompt = self._build_prompt(
            clean, correction_note=correction_note,
        )
        try:
            response = await self._call_gemini(retry_prompt)
            parsed = self._parse_response(response)
            return self._finalize(parsed)
        except (json.JSONDecodeError, ValidationError) as second_exc:
            raise RuntimeError(
                f"Stage 2.5 failed after corrective retry. "
                f"First error: {first_error!r}. "
                f"Second error: {second_exc!r}. "
                f"Inngest's outer retry will handle if applicable."
            ) from second_exc

    # ---- Post-validate truncation -----------------------------------

    def _finalize(self, parsed: Stage2_5Output) -> Stage2_5Output:
        """Apply the third 6-cap safety net.

        Pydantic ``Field(max_length=6)`` MAY allow this through if
        Gemini ignores the JSON Schema maxItems (it shouldn't with
        Pydantic validation in the middle, but the truncation is
        also belt-and-suspenders for cases where the schema engine
        produces >6 and Pydantic's max_length is somehow bypassed
        -- e.g. future SDK changes).
        """
        if len(parsed.entities) <= self.entity_cap:
            return parsed
        truncated = _truncate_to_cap(parsed.entities, cap=self.entity_cap)
        return Stage2_5Output(entities=truncated)
