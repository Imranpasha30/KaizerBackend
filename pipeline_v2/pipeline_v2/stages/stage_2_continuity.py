"""Stage 2 -- Continuity Editor (Gemini 2.5 Pro).

Takes Stage 1's word-level transcript and emits cut decisions: which
ranges become bulletin clips, which ranges are skipped (retakes /
hesitations / asides / etc.), and a one-sentence ``retake_audit``
prose summary.

LLM contract: ``Stage2Output`` (see ``pipeline_v2/models.py``). The
LLM does NOT emit ``clean_transcript`` -- that's reconstructed
deterministically client-side in Step 5.4 from the original word
array minus skipped spans.

SDK-level gotchas defended against (per Step 5 research):

  - ``response.parsed`` silently swallows ``json.JSONDecodeError`` and
    ``pydantic.ValidationError`` and returns ``None``. We use
    ``response.text`` + ``json.loads`` + ``Stage2Output.model_validate``
    so validation errors surface explicitly.
  - 1.x had a markdown-fence wrapping bug (``\`\`\`json ... \`\`\```).
    2.x is mostly fixed but we strip defensively.
  - Thinking mode shares the ``max_output_tokens`` budget. We cap
    thinking at 2048 (per Step 5 decision) so it doesn't silently
    eat output tokens and truncate the JSON.
  - No SDK-level retry. We do 1 corrective retry with the validation
    error appended to the prompt; second failure raises so Inngest's
    exponential backoff is the outer retry layer.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

# Eager imports. If the SDK isn't installed, the module fails to
# import -- there's no graceful degradation here (Stage 2 IS the
# Gemini step).
from google import genai
from google.genai import types
from pydantic import ValidationError

from pipeline_v2.models import (
    CleanTranscript,
    FullVideoCut,
    SkippedSegment,
    Stage1Output,
    Stage2Output,
    StageTwoOutput,
    Word,
)

logger = logging.getLogger("pipeline_v2.stage_2")


# --- Defaults (locked at top of file so a future reader sees them) ----

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_THINKING_BUDGET = 2048      # cap thinking; large output is the priority
DEFAULT_MAX_OUTPUT_TOKENS = 16384   # well under 65k Gemini cap

DEFAULT_PROMPT_PATH = Path(__file__).parent / "stage_2_prompt.md"


# --- Helpers ---------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ``\`\`\`json`` and ``\`\`\``` fences.

    Older google-genai versions had a bug where Gemini wrapped JSON
    output in markdown fences despite ``response_mime_type=
    "application/json"``. The 2.4.0 SDK mostly fixed this but the
    behavior is empirically still inconsistent on Gemini 2.5 Pro for
    long structured outputs. Stripping defensively costs nothing.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    # Drop the opening fence (```json or ```)
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    # Drop the closing fence if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _serialize_words_for_prompt(stage1_output: Stage1Output) -> str:
    """Render Stage 1's word array as compact JSON for the prompt.

    Deepgram-style shape: ``[{"w":"hello","s":0.10,"e":0.45}, ...]``.
    We round timestamps to 3 decimals to keep the prompt tokens-tight
    (3-decimal precision = 1ms, plenty for cut decisions). Speaker
    and confidence are omitted -- Stage 2 doesn't use them at the
    prompt level (continuity is mostly textual signal).
    """
    arr = [
        {
            "w": w.w,
            "s": round(float(w.s), 3),
            "e": round(float(w.e), 3),
        }
        for w in stage1_output.transcript.words
    ]
    return json.dumps(arr, ensure_ascii=False)


# --- Stage entry -----------------------------------------------------


class Stage2ContinuityEditor:
    """Gemini 2.5 Pro continuity editor.

    Usage::

        editor = Stage2ContinuityEditor()
        decisions = await editor.transcribe_to_decisions(stage1_output)
        # decisions: Stage2Output with full_video_cuts / skipped_segments
        # / retake_audit. CleanTranscript is reconstructed elsewhere
        # (Step 5.4).

    Constructor overrides are kwargs-only so future arguments can be
    added without breaking call sites. The dispatcher (Step 10) will
    instantiate this with no args; tests may override the model /
    temperature / thinking_budget / max_output_tokens / prompt_path.
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
        """Construct the Gemini client on first call.

        API key is fetched lazily so a missing key doesn't kill
        construction. Tests can monkey-patch ``GEMINI_API_KEY`` to a
        fake value -- the SDK only complains when an actual API call
        is made (which tests mock).
        """
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
                f"Stage 2 prompt template not found at {self.prompt_path}. "
                f"Step 5.3 fills this in. For testing earlier, pass "
                f"prompt_path= explicitly to the constructor."
            )
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    # ---- prompt construction ----------------------------------------

    def _build_prompt(
        self,
        stage1_output: Stage1Output,
        correction_note: str = "",
    ) -> str:
        template = self._load_prompt()
        words_json = _serialize_words_for_prompt(stage1_output)
        # Stage 1 metadata that may be useful in the prompt:
        lang = stage1_output.stt_language_detected or "unknown"
        provider = stage1_output.stt_provider
        duration = stage1_output.stt_audio_duration_sec

        prompt = (
            f"{template}\n\n"
            f"## Audio metadata\n"
            f"- Detected language: {lang}\n"
            f"- STT provider:      {provider}\n"
            f"- Duration:          {duration:.1f}s\n"
            f"- Word count:        {len(stage1_output.transcript.words)}\n"
            f"\n"
            f"## Word array (Deepgram-style timestamps, one entry per word)\n"
            f"{words_json}\n"
        )
        if correction_note:
            prompt += f"\n## Correction note\n{correction_note}\n"
        return prompt

    # ---- Gemini call + parse ---------------------------------------

    async def _call_gemini(self, prompt: str):
        client = self._get_client()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Stage2Output,
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

    def _parse_response(self, response) -> Stage2Output:
        """Manual validate path -- do NOT trust ``response.parsed``.

        Per the google-genai 2.4.0 research (issue #289):
          - ``response.parsed`` silently catches ``json.JSONDecodeError``
            and ``pydantic.ValidationError``, returning ``None`` instead
            of raising
          - ``response.parsed`` does NOT run our ``@field_validator``
            decorators (e.g. ShortsCut's 15-60s duration check)

        We use ``response.text`` + ``json.loads`` + ``Stage2Output.
        model_validate(...)`` so every validation step happens
        explicitly and surfaces errors for the corrective-retry layer.
        """
        raw_text = getattr(response, "text", "") or ""
        if not raw_text.strip():
            # Empty response.text: treat as JSON decode error so the
            # corrective-retry catches it like any other parse failure.
            raise json.JSONDecodeError(
                "response.text is empty (Gemini returned no JSON body; "
                "check finish_reason for MAX_TOKENS truncation or "
                "thinking-budget exhaustion)",
                "",
                0,
            )
        cleaned = _strip_markdown_fences(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            raise
        return Stage2Output.model_validate(data)

    # ---- Public entry -----------------------------------------------

    async def transcribe_to_decisions(
        self,
        stage1_output: Stage1Output,
    ) -> Stage2Output:
        """Convert a Stage 1 word transcript into Stage 2 cut decisions.

        Retry policy (locked by Step 5 design):
          1. First attempt: call Gemini, parse, return.
          2. On ``json.JSONDecodeError`` or ``pydantic.ValidationError``:
             single corrective retry with the validation error appended
             to the prompt.
          3. On second failure: raise ``RuntimeError`` wrapping both
             errors. Inngest's exponential backoff is the outer retry
             layer -- we deliberately do NOT retry beyond 1 in-step.
          4. Other exceptions (auth fail, rate limit, network) are
             allowed to propagate immediately -- the dispatcher / Inngest
             handles those.
        """
        base_prompt = self._build_prompt(stage1_output)

        # ---- First attempt ----
        first_error: Optional[Exception] = None
        try:
            response = await self._call_gemini(base_prompt)
            return self._parse_response(response)
        except (json.JSONDecodeError, ValidationError) as exc:
            first_error = exc
            logger.warning(
                "stage_2: first attempt failed validation: %s. "
                "Issuing 1 corrective retry with the error appended.",
                exc,
            )

        # ---- Corrective retry ----
        correction_note = (
            "Your previous response failed validation with this error:\n"
            f"  {first_error}\n\n"
            "Regenerate a JSON response that conforms EXACTLY to the "
            "Stage2Output schema. Do not wrap in markdown code fences. "
            "Emit only the JSON object. All skipped_segments[*].category "
            "values must be one of: warm_up, retake, crew_talk, "
            "hesitation, aside, self_correction (NEVER invent new "
            "categories)."
        )
        retry_prompt = self._build_prompt(
            stage1_output, correction_note=correction_note,
        )
        try:
            response = await self._call_gemini(retry_prompt)
            return self._parse_response(response)
        except (json.JSONDecodeError, ValidationError) as second_exc:
            raise RuntimeError(
                f"Stage 2 failed after corrective retry. "
                f"First error: {first_error!r}. "
                f"Second error: {second_exc!r}. "
                f"Inngest's outer retry will handle if applicable."
            ) from second_exc


# --- Step 5.4: clean_transcript reconstruction --------------------------
#
# The Stage 2 LLM emits ONLY decisions (full_video_cuts + skipped_segments
# + retake_audit) -- it does NOT emit the clean word array. Reconstructing
# the clean transcript in code (rather than asking Gemini to echo every
# kept word back) saves output tokens AND eliminates an entire class of
# hallucination risk (the LLM cannot accidentally rewrite a kept word).
#
# These are pure functions: no I/O, no SDK calls, fully unit-testable
# without mocks. The orchestrator (Step 10) composes them like:
#     decisions = await editor.transcribe_to_decisions(stage1)
#     stage_two = assemble_stage_two_output(stage1, decisions)


def build_clean_transcript(
    original_words: list[Word],
    skipped_segments: list[SkippedSegment],
    full_video_cuts: list[FullVideoCut],
) -> CleanTranscript:
    """Reconstruct the clean word array deterministically.

    Walks the ORIGINAL Stage 1 word array, drops words covered by any
    ``SkippedSegment``, and builds the three ``CleanTranscript`` fields:

      - ``words``: surviving Word objects (preserved verbatim --
        same w/s/e/speaker/confidence as the source).
      - ``source_word_map``: ``clean_idx -> original_idx``. Invariant:
        ``original_words[source_word_map[i]]`` is the source of
        ``clean_words[i]`` with zero coordinate drift.
      - ``clip_boundaries``: ``cut.index -> (first_clean_idx,
        last_clean_idx)``. Translates ``FullVideoCut`` ranges (which
        use ORIGINAL indices) into clean-array indices, snapping the
        boundaries inward if they land on a skipped word.

    Validation (raises ``ValueError`` on failure -- the corrective
    retry layer above does NOT catch these; an invalid Stage 2 result
    that passed Pydantic but fails semantic checks is an Inngest-level
    failure):

      - Every ``skipped_segment`` index must be in bounds.
      - ``start_word_idx <= end_word_idx`` per segment.
      - No two ``skipped_segments`` may cover the same original index
        (HARD RULE #5 in the prompt; enforced here too as a safety
        net against prompt drift).
      - Every ``full_video_cut`` index must be in bounds.

    Edge case -- a ``FullVideoCut`` whose entire range is skipped
    (every word inside ``[start_word_idx, end_word_idx]`` appears in
    some ``SkippedSegment``) is dropped from ``clip_boundaries`` with
    a structured warning. The cut still appears in
    ``StageTwoOutput.full_video_cuts`` for telemetry; downstream
    renderers skip cuts with no boundary entry.
    """
    n = len(original_words)

    # ---- Validate + collect skipped original indices ----
    skipped_indices: set[int] = set()
    for seg in skipped_segments:
        if not (0 <= seg.start_word_idx <= seg.end_word_idx < n):
            raise ValueError(
                f"SkippedSegment out of bounds (word array len={n}): "
                f"start={seg.start_word_idx} end={seg.end_word_idx}"
            )
        for idx in range(seg.start_word_idx, seg.end_word_idx + 1):
            if idx in skipped_indices:
                raise ValueError(
                    f"Overlapping SkippedSegments at original word idx "
                    f"{idx}. HARD RULE #5 forbids this -- the LLM "
                    f"violated the prompt; treat as a Stage 2 failure "
                    f"and let Inngest retry."
                )
            skipped_indices.add(idx)

    # ---- Build clean words + source_word_map ----
    clean_words: list[Word] = []
    source_word_map: list[int] = []
    orig_to_clean: dict[int, int] = {}  # original_idx -> clean_idx
    for orig_idx, w in enumerate(original_words):
        if orig_idx in skipped_indices:
            continue
        orig_to_clean[orig_idx] = len(clean_words)
        clean_words.append(w)
        source_word_map.append(orig_idx)

    # ---- Build clip_boundaries ----
    clip_boundaries: dict[int, tuple[int, int]] = {}
    for cut in full_video_cuts:
        if not (0 <= cut.start_word_idx <= cut.end_word_idx < n):
            raise ValueError(
                f"FullVideoCut out of bounds (word array len={n}): "
                f"index={cut.index} start={cut.start_word_idx} "
                f"end={cut.end_word_idx}"
            )
        # Snap inward: first non-skipped at or after start_word_idx.
        first_clean: Optional[int] = None
        for i in range(cut.start_word_idx, cut.end_word_idx + 1):
            if i in orig_to_clean:
                first_clean = orig_to_clean[i]
                break
        # Snap inward: last non-skipped at or before end_word_idx.
        last_clean: Optional[int] = None
        for i in range(cut.end_word_idx, cut.start_word_idx - 1, -1):
            if i in orig_to_clean:
                last_clean = orig_to_clean[i]
                break
        if first_clean is None or last_clean is None:
            logger.warning(
                "stage_2: FullVideoCut index=%d (original word range "
                "%d-%d) is entirely covered by skipped_segments. "
                "Dropping from clip_boundaries; renderer will skip it.",
                cut.index, cut.start_word_idx, cut.end_word_idx,
            )
            continue
        clip_boundaries[cut.index] = (first_clean, last_clean)

    # 100%-skipped recordings are rare but legitimate (e.g. Stage 2
    # concluded the entire raw take is unusable). Surface via logger
    # so operators see the signal without forcing the orchestrator
    # to special-case empty CleanTranscript.
    if not clean_words:
        logger.warning(
            "stage_2: build_clean_transcript produced an empty word "
            "array -- every word in the %d-word input was covered by "
            "skipped_segments. Downstream stages will receive an empty "
            "transcript; renderer will produce no clips.",
            n,
        )

    return CleanTranscript(
        words=clean_words,
        clip_boundaries=clip_boundaries,
        source_word_map=source_word_map,
    )


def assemble_stage_two_output(
    stage1_output: Stage1Output,
    stage2_decisions: Stage2Output,
) -> StageTwoOutput:
    """Compose Stage 2's full return value from LLM decisions + reconstruction.

    The Step 10 orchestrator calls this immediately after
    ``Stage2ContinuityEditor.transcribe_to_decisions(...)`` returns.

    ``StageTwoOutput.retake_audit`` is mandatory non-None at the
    Pydantic level. The LLM-side ``Stage2Output.retake_audit`` is
    optional only as a safety net (see ``Stage2Output`` docstring).
    If we reach this assembler with ``retake_audit`` empty/missing,
    the corrective-retry already failed -- raise a clear ValueError
    rather than fabricating one (no silent defaulting).
    """
    clean = build_clean_transcript(
        original_words=stage1_output.transcript.words,
        skipped_segments=stage2_decisions.skipped_segments,
        full_video_cuts=stage2_decisions.full_video_cuts,
    )
    retake_audit = stage2_decisions.retake_audit
    if not retake_audit or not retake_audit.strip():
        raise ValueError(
            "Stage 2 produced empty/missing retake_audit. The Step 5.2 "
            "corrective retry should have caught this; reaching the "
            "assembler with no audit means the retry also failed. "
            "Treat as Stage 2 failure -- Inngest will retry the step."
        )
    return StageTwoOutput(
        full_video_cuts=stage2_decisions.full_video_cuts,
        skipped_segments=stage2_decisions.skipped_segments,
        clean_transcript=clean,
        retake_audit=retake_audit,
    )
