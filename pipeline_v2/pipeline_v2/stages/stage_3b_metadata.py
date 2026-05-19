"""Stage 3b -- Metadata Extractor (Gemini 2.5 Flash, temperature 0.7).

Takes the clean transcript + canonical entities and emits a single
``Metadata`` object describing the bulletin: video_type, language,
total_speakers, overall_summary (English + native), the
shorts_headline_native (one Telugu/native headline for all shorts),
bulletin_marquee_points (ticker text), image_search_queries, plus
4 entity-extracted lists (key_people, key_people_native, key_topics,
key_locations).

LLM contract: ``Metadata`` (passed as ``response_schema=`` directly --
no wrapper class needed; Metadata is already an object, not a list).

SDK pattern mirrors Stage 2 / 2.5 / 3a: manual ``model_validate``
path, 1 corrective retry on JSONDecodeError / ValidationError,
RuntimeError on second failure.

Temperature is 0.7 (per D-7.5) -- this is creative work: writing
summaries and headlines benefits from variety. The Pydantic schema
constraints handle the safety work (video_type Literal locks the
5 allowed values; everything else is free-form text).
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
    Metadata,
)

logger = logging.getLogger("pipeline_v2.stage_3b")


# --- Defaults (locked per Step 7 decisions) --------------------------

DEFAULT_MODEL = "gemini-2.5-flash"       # D-7.5 / Step 6 D7
DEFAULT_TEMPERATURE = 0.7                # D-7.5: creative
DEFAULT_THINKING_BUDGET = 512            # D-7.6
DEFAULT_MAX_OUTPUT_TOKENS = 2048         # D-7.7: 13 fields, mostly short
DEFAULT_PROMPT_PATH = Path(__file__).parent / "stage_3b_prompt.md"


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
    """For Stage 3b, drop word-level timestamps (the model doesn't
    need them for metadata extraction). Emit a single concatenated
    prose string -- much cheaper input tokens, same info content for
    summary-writing.
    """
    return " ".join(w.w for w in clean.words)


def _serialize_entities_for_prompt(entities: list[Entity]) -> str:
    """Compact entity list. Stage 3b uses these to populate
    key_people / key_topics / key_locations lists in its output.
    """
    arr = [
        {
            "canonical_name": e.canonical_name,
            "native_name": e.native_name,
            "type": e.type.value,
            "mention_count": len(e.mentions),
        }
        for e in entities
    ]
    return json.dumps(arr, ensure_ascii=False)


# --- Stage entry -----------------------------------------------------


class Stage3bMetadataExtractor:
    """Gemini 2.5 Flash metadata extractor.

    Usage::

        ext = Stage3bMetadataExtractor()
        meta = await ext.extract(clean_transcript, entities)
        # meta: Metadata with 12 fields populated
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
                f"Stage 3b prompt template not found at {self.prompt_path}. "
                f"Step 7.3 fills this in. For testing earlier, pass "
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
        transcript_prose = _serialize_clean_transcript_for_prompt(clean)
        entities_json = _serialize_entities_for_prompt(entities)
        word_count = len(clean.words)
        duration_sec = clean.words[-1].e if clean.words else 0.0

        prompt = (
            f"{template}\n\n"
            f"## Clean transcript metadata\n"
            f"- Word count:        {word_count}\n"
            f"- Total duration:    {duration_sec:.1f}s\n"
            f"- Canonical entities: {entities_json}\n"
            f"\n"
            f"## Clean transcript (concatenated prose)\n"
            f"{transcript_prose}\n"
        )
        if correction_note:
            prompt += f"\n## Correction note\n{correction_note}\n"
        return prompt

    # ---- Gemini call + parse ----------------------------------------

    async def _call_gemini(self, prompt: str):
        client = self._get_client()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Metadata,
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

    def _parse_response(self, response) -> Metadata:
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
        return Metadata.model_validate(data)

    # ---- Public entry -----------------------------------------------

    async def extract(
        self,
        clean: CleanTranscript,
        entities: list[Entity],
    ) -> Metadata:
        """Extract bulletin metadata.

        Same retry policy as Stage 2 / 2.5 / 3a.
        """
        base_prompt = self._build_prompt(clean, entities)

        first_error: Optional[Exception] = None
        try:
            response = await self._call_gemini(base_prompt)
            return self._parse_response(response)
        except (json.JSONDecodeError, ValidationError) as exc:
            first_error = exc
            logger.warning(
                "stage_3b: first attempt failed validation: %s. "
                "Issuing 1 corrective retry with the error appended.",
                exc,
            )

        correction_note = (
            "Your previous response failed validation with this error:\n"
            f"  {first_error}\n\n"
            "Regenerate a JSON response that conforms EXACTLY to the "
            "Metadata schema. Do not wrap in markdown code fences. "
            "Emit only the JSON object. `video_type` MUST be one of: "
            "SOLO, INTERVIEW, PRESS_CONFERENCE, PANEL, MIXED. "
            "Every native-script field (`*_native`) MUST be in the "
            "target script (Telugu / Hindi / etc.), NEVER Latin "
            "transliteration."
        )
        retry_prompt = self._build_prompt(
            clean, entities, correction_note=correction_note,
        )
        try:
            response = await self._call_gemini(retry_prompt)
            return self._parse_response(response)
        except (json.JSONDecodeError, ValidationError) as second_exc:
            raise RuntimeError(
                f"Stage 3b failed after corrective retry. "
                f"First error: {first_error!r}. "
                f"Second error: {second_exc!r}. "
                f"Inngest's outer retry will handle if applicable."
            ) from second_exc
