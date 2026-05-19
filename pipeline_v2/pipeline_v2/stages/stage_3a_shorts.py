"""Stage 3a -- Shorts Generator (Gemini 2.5 Flash, temperature 0.7).

Takes the clean transcript + canonical entities and emits 3-10
``ShortsCut`` entries: 15-60s short-form video segments suitable for
1080x1920 vertical render. The 15-60s duration is enforced at three
levels (per Step 7 D-7.11):

  1. Prompt body (HARD RULES section)
  2. ``ShortsCut.@field_validator("end_sec")`` (raises ValidationError
     -> corrective-retry catches)
  3. (No third layer in 3a -- the validator-as-ValidationError +
     corrective-retry pattern is the entire defense. Unlike Stage
     2.5 we don't post-validate-truncate because duration is a
     hard constraint, not a "too many" constraint.)

SDK pattern mirrors Stage 2 / Stage 2.5: manual ``model_validate``
path (NEVER ``response.parsed``), 1 corrective retry on
ValidationError / JSONDecodeError, RuntimeError on second failure.

Temperature is 0.7 (per D-7.5) -- this is creative work: picking
"interesting moments" benefits from variety. The Pydantic schema
constraints + 15-60s validator do the safety work.
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
    Stage3aOutput,
)

logger = logging.getLogger("pipeline_v2.stage_3a")


# --- Defaults (locked per Step 7 decisions) --------------------------

DEFAULT_MODEL = "gemini-2.5-flash"       # D-7.5 / Step 6 D7
DEFAULT_TEMPERATURE = 0.7                # D-7.5: creative
DEFAULT_THINKING_BUDGET = 512            # D-7.6 pushback: conservative
DEFAULT_MAX_OUTPUT_TOKENS = 4096         # D-7.7
DEFAULT_PROMPT_PATH = Path(__file__).parent / "stage_3a_prompt.md"


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

    def _parse_response(self, response) -> Stage3aOutput:
        """Manual validate path. See Stage 2 / 2.5 for rationale."""
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
        return Stage3aOutput.model_validate(data)

    # ---- Public entry -----------------------------------------------

    async def generate(
        self,
        clean: CleanTranscript,
        entities: list[Entity],
    ) -> Stage3aOutput:
        """Generate 3-10 short-form clip selections.

        Same retry policy as Stage 2 / 2.5:
          1. First attempt: call Flash, parse, return.
          2. On JSONDecodeError / ValidationError (incl. the 15-60s
             duration validator): single corrective retry with error
             appended.
          3. On second failure: RuntimeError; Inngest outer retry.
        """
        base_prompt = self._build_prompt(clean, entities)

        first_error: Optional[Exception] = None
        try:
            response = await self._call_gemini(base_prompt)
            return self._parse_response(response)
        except (json.JSONDecodeError, ValidationError) as exc:
            first_error = exc
            logger.warning(
                "stage_3a: first attempt failed validation: %s. "
                "Issuing 1 corrective retry with the error appended.",
                exc,
            )

        correction_note = (
            "Your previous response failed validation with this error:\n"
            f"  {first_error}\n\n"
            "Regenerate a JSON response that conforms EXACTLY to the "
            "Stage3aOutput schema. Do not wrap in markdown code "
            "fences. Emit only the JSON object. Every shorts_cut MUST "
            "have (end_sec - start_sec) BETWEEN 15.0 AND 60.0 seconds "
            "inclusive. Emit 3-10 shorts (5-7 ideal). importance is "
            "1-10 inclusive. The hook is a short attention-grabbing "
            "phrase, not a full sentence."
        )
        retry_prompt = self._build_prompt(
            clean, entities, correction_note=correction_note,
        )
        try:
            response = await self._call_gemini(retry_prompt)
            return self._parse_response(response)
        except (json.JSONDecodeError, ValidationError) as second_exc:
            raise RuntimeError(
                f"Stage 3a failed after corrective retry. "
                f"First error: {first_error!r}. "
                f"Second error: {second_exc!r}. "
                f"Inngest's outer retry will handle if applicable."
            ) from second_exc
