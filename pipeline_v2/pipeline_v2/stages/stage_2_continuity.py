"""Stage 2 -- Continuity Editor (provider-agnostic dispatcher).

Takes Stage 1's word-level transcript and emits cut decisions: which
ranges become bulletin clips, which ranges are skipped (retakes /
hesitations / asides / etc.), and a one-sentence ``retake_audit``
prose summary.

LLM contract: ``Stage2Output`` (see ``pipeline_v2/models.py``). The
LLM does NOT emit ``clean_transcript`` -- that's reconstructed
deterministically client-side in Step 5.4 from the original word
array minus skipped spans.

Item 114: the actual SDK call lives in ``stage_2_providers.py``
behind a ``Stage2Provider`` abstraction. ``Stage2ContinuityEditor``
keeps the corrective-retry policy and is the only thing the
orchestrator calls. Provider selection is per-job via the
``provider_name`` constructor arg (defaults to "gemini").

Retry policy (unchanged from the Gemini-only era):

  - ``response.parsed`` silently swallows ``json.JSONDecodeError`` and
    ``pydantic.ValidationError`` and returns ``None``. The Gemini
    provider uses ``response.text`` + ``json.loads`` +
    ``Stage2Output.model_validate`` so validation errors surface.
  - Markdown-fence wrapping bug (``\`\`\`json ... \`\`\```): both
    providers strip defensively.
  - Thinking mode shares the ``max_output_tokens`` budget. The Gemini
    provider caps thinking at 2048; the Claude provider disables
    thinking entirely for first ship.
  - No SDK-level retry. We do 1 corrective retry with the validation
    error appended to the prompt; second failure raises so Inngest's
    exponential backoff is the outer retry layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

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
from pipeline_v2.stages.stage_2_providers import (
    DEFAULT_PROVIDER,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_CLAUDE_MODEL,
    PROVIDER_CLAUDE,
    PROVIDER_GEMINI,
    Stage2Provider,
    create_provider,
    # Backward-compat aliases for older callers / tests that imported
    # these helpers from stage_2_continuity directly. They moved to
    # stage_2_providers in item 114; re-export so import sites
    # don't have to change in lockstep.
    _serialize_words_for_prompt,   # noqa: F401
    _strip_markdown_fences,        # noqa: F401
)

logger = logging.getLogger("pipeline_v2.stage_2")


# --- Defaults (locked at top of file so a future reader sees them) ----

# Backward-compat aliases used by older tests + the orchestrator.
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_THINKING_BUDGET = 2048      # Gemini-only
DEFAULT_MAX_OUTPUT_TOKENS = 16384   # well under 65k Gemini cap

DEFAULT_PROMPT_PATH = Path(__file__).parent / "stage_2_prompt.md"


# --- Stage entry -----------------------------------------------------


class Stage2ContinuityEditor:
    """Provider-agnostic continuity editor.

    Usage::

        editor = Stage2ContinuityEditor()                           # default: gemini
        editor = Stage2ContinuityEditor(provider_name="claude")     # explicit Claude
        decisions = await editor.transcribe_to_decisions(stage1_output)

    The dispatcher (Step 10) instantiates this with ``provider_name``
    taken from the job's Inngest envelope (which the runner stamps
    from ``Job.stage_2_provider``). Tests may override the model /
    temperature / thinking_budget / max_output_tokens / prompt_path.

    Provider exposure: after a successful call, ``self.provider_name``
    and ``self.last_cost_usd`` reflect what was actually used. The
    orchestrator reads both for the cost ledger.
    """

    def __init__(
        self,
        *,
        provider_name: str = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        prompt_path: Optional[Path] = None,
    ):
        self.provider_name = provider_name
        # Resolve default model from provider when caller didn't pin one.
        # Backward-compat: legacy callers / tests that read editor.model
        # before any call expect a non-None default; the previous
        # constructor pre-item-114 had ``model: str = DEFAULT_MODEL``.
        if model is None:
            if provider_name == PROVIDER_CLAUDE:
                model = DEFAULT_CLAUDE_MODEL
            else:
                model = DEFAULT_GEMINI_MODEL
        self.model = model
        # Same pattern for temperature: legacy default was 0.2 (Gemini).
        # Claude defaults to 0.0 in its provider. Surface a sensible
        # default per provider so callers can inspect ``editor.temperature``.
        if temperature is None:
            if provider_name == PROVIDER_CLAUDE:
                temperature = 0.0
            else:
                temperature = DEFAULT_TEMPERATURE
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.prompt_path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        # Provider instance -- built lazily on first call. Built once
        # per Stage2ContinuityEditor so the prompt template is loaded
        # only once and (for Claude) the prompt cache breakpoint
        # stays warm across the corrective retry.
        self._provider: Optional[Stage2Provider] = None

    def _get_provider(self) -> Stage2Provider:
        if self._provider is not None:
            return self._provider
        kwargs: dict = {
            "max_output_tokens": self.max_output_tokens,
        }
        # Forward only the params that match. create_provider's per-
        # provider sig_keys filter prevents irrelevant kwargs from
        # crashing the constructor.
        if self.model is not None:
            kwargs["model"] = self.model
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.thinking_budget is not None:
            kwargs["thinking_budget"] = self.thinking_budget
        self._provider = create_provider(
            self.provider_name,
            prompt_path=self.prompt_path,
            **kwargs,
        )
        return self._provider

    @property
    def last_cost_usd(self) -> float:
        """Last call's USD cost, or 0 if no call has been made yet."""
        return self._provider.last_cost_usd if self._provider else 0.0

    @property
    def last_usage(self) -> dict:
        """Last call's token usage breakdown."""
        return self._provider.last_usage if self._provider else {}

    # ---- Backward-compat delegates ----------------------------------
    #
    # Pre-item-114 callers (mostly the existing test suite) probed
    # _load_prompt and read self._prompt_template directly on the
    # editor. Forward those to the underlying provider so the same
    # access patterns keep working.

    def _load_prompt(self) -> str:
        """Delegate -- loads the prompt template via the provider."""
        return self._get_provider()._load_prompt()

    @property
    def _prompt_template(self) -> Optional[str]:
        """Reflect the provider's cached template (None if not yet
        loaded). Read-only; tests that want to invalidate the cache
        should construct a fresh editor."""
        return self._get_provider()._prompt_template

    @_prompt_template.setter
    def _prompt_template(self, value: Optional[str]) -> None:
        """Tests sometimes pre-populate the cache to skip the file
        read. Honour that by writing through to the provider."""
        self._get_provider()._prompt_template = value

    # ---- Public entry -----------------------------------------------

    async def transcribe_to_decisions(
        self,
        stage1_output: Stage1Output,
    ) -> Stage2Output:
        """Convert a Stage 1 word transcript into Stage 2 cut decisions.

        Retry policy (unchanged from Gemini-only era):
          1. First attempt: call provider.decide(), return.
          2. On ``json.JSONDecodeError`` or ``pydantic.ValidationError``:
             single corrective retry with the validation error
             appended to the prompt.
          3. On second failure: raise ``RuntimeError`` wrapping both
             errors. Inngest's exponential backoff is the outer retry
             layer -- we deliberately do NOT retry beyond 1 in-step.
          4. Other exceptions (auth fail, rate limit, network) are
             allowed to propagate immediately.
        """
        provider = self._get_provider()

        # ---- First attempt ----
        first_error: Optional[Exception] = None
        try:
            return await provider.decide(stage1_output)
        except (json.JSONDecodeError, ValidationError) as exc:
            first_error = exc
            logger.warning(
                "stage_2 [%s]: first attempt failed validation: %s. "
                "Issuing 1 corrective retry with the error appended.",
                self.provider_name, exc,
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
        try:
            return await provider.decide(
                stage1_output, correction_note=correction_note,
            )
        except (json.JSONDecodeError, ValidationError) as second_exc:
            raise RuntimeError(
                f"Stage 2 failed after corrective retry "
                f"(provider={self.provider_name!r}). "
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
