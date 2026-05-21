"""Item 114 -- Stage 2 provider abstraction.

The original ``Stage2ContinuityEditor`` hard-coded Gemini 2.5 Pro
(``google-genai`` SDK + ``response_schema=`` structured output).
Item 114 introduces a provider-agnostic interface so we can swap in
alternative LLMs (Claude Sonnet 4.6 today; OpenAI / Mistral / etc.
in the future) without re-plumbing the orchestrator or any
downstream code.

The boundary is intentionally narrow:

  - ``Stage2Provider`` exposes ONE async method: ``decide(stage1,
    correction_note="")`` which takes the Stage 1 transcript and
    returns a validated ``Stage2Output`` (the LLM's editorial
    decisions: full_video_cuts + skipped_segments + retake_audit).
  - Each provider class owns its SDK client, prompt assembly,
    response parsing, and cost calculation.
  - The retry policy (one corrective retry on validation failure,
    then raise) lives in ``Stage2ContinuityEditor`` -- providers
    are stateless single-call objects.

Both providers must:
  - Accept the same prompt template (``stage_2_prompt.md``)
  - Return identical ``Stage2Output`` Pydantic instances
  - Populate ``self.cost_usd`` after each call so the orchestrator
    can ledger provider-specific spend

API key resolution is lazy (first call) so a missing key doesn't
crash construction. Tests can monkey-patch the env var to a fake
value and mock the SDK at the call site.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from pipeline_v2.models import Stage1Output, Stage2Output

# Module-level genai import so test mocking can target
# ``pipeline_v2.stages.stage_2_providers.genai.Client`` directly. The
# import is eager (the SDK is a hard dep of the Gemini provider) but
# the SDK doesn't actually contact Google until ``Client(...)`` runs.
from google import genai

logger = logging.getLogger("pipeline_v2.stage_2_providers")


# --- Pricing (per 1M tokens, USD) --------------------------------------
# Update these alongside the model strings if pricing changes.

GEMINI_2_5_PRO_INPUT_PER_M_USD: float = 1.25
GEMINI_2_5_PRO_OUTPUT_PER_M_USD: float = 10.00

# Claude Sonnet 4.6 pricing (per Anthropic's published rates).
# Prompt caching: writes ~1.25x base rate, reads ~0.1x base rate.
CLAUDE_SONNET_4_6_INPUT_PER_M_USD: float = 3.00
CLAUDE_SONNET_4_6_OUTPUT_PER_M_USD: float = 15.00
CLAUDE_SONNET_4_6_CACHE_WRITE_PER_M_USD: float = 3.75   # = 1.25x input
CLAUDE_SONNET_4_6_CACHE_READ_PER_M_USD: float = 0.30    # = 0.10x input


PROVIDER_GEMINI: str = "gemini"
PROVIDER_CLAUDE: str = "claude"

VALID_PROVIDERS: frozenset[str] = frozenset({PROVIDER_GEMINI, PROVIDER_CLAUDE})

DEFAULT_PROVIDER: str = PROVIDER_GEMINI

# Locked at module top so a reader sees the catalog at a glance.
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-pro"
DEFAULT_CLAUDE_MODEL: str = "claude-sonnet-4-6"


# --- Helpers (shared between providers) -------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ```json / ``` fences.

    Both Gemini (older SDK builds) and Claude (rarely, but possible)
    can wrap structured JSON output in markdown fences. Strip
    defensively so ``json.loads`` doesn't reject the response.
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


def _serialize_words_for_prompt(stage1_output: Stage1Output) -> str:
    """Render Stage 1's word array as compact JSON for the prompt.

    Deepgram-style shape: ``[{"w":"hello","s":0.10,"e":0.45}, ...]``.
    Speaker and confidence are omitted -- Stage 2 doesn't use them
    at the prompt level. Timestamps rounded to 3 decimals (1ms).
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


def _build_user_payload(
    stage1_output: Stage1Output,
    correction_note: str = "",
) -> str:
    """The dynamic per-job content that follows the (cacheable) system
    prompt: audio metadata, word array, and an optional correction
    note appended by the retry layer."""
    words_json = _serialize_words_for_prompt(stage1_output)
    lang = stage1_output.stt_language_detected or "unknown"
    provider = stage1_output.stt_provider
    duration = stage1_output.stt_audio_duration_sec
    payload = (
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
        payload += f"\n## Correction note\n{correction_note}\n"
    return payload


# --- Abstract base ----------------------------------------------------


class Stage2Provider(ABC):
    """Provider-agnostic Stage 2 interface.

    Subclasses implement ``decide`` for their SDK. The retry layer
    (``Stage2ContinuityEditor``) treats every provider identically:
    one corrective retry on ``json.JSONDecodeError`` /
    ``pydantic.ValidationError``, then propagate.
    """

    name: str = "abstract"

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: int = 16384,
        prompt_path: Optional[Path] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_path = prompt_path
        # Per-call cost, set after every successful decide().
        self.last_cost_usd: float = 0.0
        # Per-call token usage for the audit ledger.
        self.last_usage: dict = {}
        self._prompt_template: Optional[str] = None

    def _load_prompt(self) -> str:
        """Lazily load + cache the prompt template from disk."""
        if self._prompt_template is not None:
            return self._prompt_template
        if self.prompt_path is None or not self.prompt_path.is_file():
            raise FileNotFoundError(
                f"Stage 2 prompt template not found at {self.prompt_path}. "
                f"Step 5.3 fills this in. For testing earlier, pass "
                f"prompt_path= explicitly to the constructor."
            )
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    @abstractmethod
    async def decide(
        self,
        stage1: Stage1Output,
        correction_note: str = "",
    ) -> Stage2Output:
        """Make a single LLM call, parse + validate the response.

        Raises ``json.JSONDecodeError`` or ``pydantic.ValidationError``
        on parse / validation failure (so the orchestrator's
        corrective-retry layer catches them). All other exceptions
        (auth fail, rate limit, network) propagate immediately.
        """


# --- Gemini provider (existing behaviour) ------------------------------


class GeminiStage2Provider(Stage2Provider):
    """Gemini 2.5 Pro continuity editor. Lift-and-shift of the original
    ``Stage2ContinuityEditor._call_gemini`` + ``_parse_response`` so the
    provider interface stays a thin wrapper around the existing path.
    """

    name = PROVIDER_GEMINI

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.2,
        thinking_budget: int = 2048,
        max_output_tokens: int = 16384,
        prompt_path: Optional[Path] = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            prompt_path=prompt_path,
        )
        self.thinking_budget = thinking_budget
        self._client = None   # lazy

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY env var is empty/unset. Add it to "
                "KaizerBackend/.env."
            )
        # Uses module-level ``genai`` so test mocking via
        # ``patch("pipeline_v2.stages.stage_2_providers.genai.Client")``
        # reaches this call site.
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _build_prompt(
        self,
        stage1_output: Stage1Output,
        correction_note: str = "",
    ) -> str:
        """Gemini takes a single-string ``contents`` -- concatenate
        the cacheable prompt template and the per-job payload."""
        return self._load_prompt() + "\n\n" + _build_user_payload(
            stage1_output, correction_note=correction_note,
        )

    async def decide(
        self,
        stage1: Stage1Output,
        correction_note: str = "",
    ) -> Stage2Output:
        from google.genai import types

        prompt = self._build_prompt(stage1, correction_note=correction_note)
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
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=config,
        )

        # Cost ledger from response.usage_metadata.
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            self.last_cost_usd = (
                in_tok * GEMINI_2_5_PRO_INPUT_PER_M_USD / 1_000_000
                + out_tok * GEMINI_2_5_PRO_OUTPUT_PER_M_USD / 1_000_000
            )
            self.last_usage = {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            }

        # Parse: NEVER trust ``response.parsed`` (it silently swallows
        # JSONDecodeError + ValidationError). Use response.text +
        # json.loads + Pydantic.
        raw = getattr(response, "text", "") or ""
        if not raw.strip():
            raise json.JSONDecodeError(
                "Gemini response.text is empty (check finish_reason "
                "for MAX_TOKENS / thinking-budget exhaustion)",
                "",
                0,
            )
        cleaned = _strip_markdown_fences(raw)
        data = json.loads(cleaned)
        return Stage2Output.model_validate(data)


# --- Claude provider (item 114) ---------------------------------------


class ClaudeStage2Provider(Stage2Provider):
    """Claude Sonnet 4.6 continuity editor.

    Uses ``client.messages.parse(output_format=Stage2Output)`` for
    native Pydantic-validated structured output.

    Prompt caching: the stage_2_prompt.md template (~8K tokens) is
    sent as the ``system`` block with ``cache_control: {"type":
    "ephemeral"}``. First request writes the cache (~1.25x input
    price for the template); subsequent requests read at ~0.1x.
    Across many jobs this is a substantial saving on the static
    portion of the prompt.

    Determinism: temperature defaults to 0 -- Sonnet 4.6 still
    accepts sampling params (unlike Opus 4.7 which removed them).
    Note that temperature=0 does NOT guarantee identical outputs.

    Thinking: ``thinking: {"type": "disabled"}`` for first ship.
    Stage 2 is a classification task; extended reasoning didn't
    help in the diagnostic. Can revisit with adaptive thinking +
    low effort if catch-rate plateaus.
    """

    name = PROVIDER_CLAUDE

    def __init__(
        self,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = 16384,
        prompt_path: Optional[Path] = None,
        enable_prompt_cache: bool = True,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            prompt_path=prompt_path,
        )
        self.enable_prompt_cache = enable_prompt_cache
        self._client = None   # lazy AsyncAnthropic

    def _get_client(self):
        if self._client is not None:
            return self._client
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY env var is empty/unset. Add it to "
                "KaizerBackend/.env."
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    def _build_system_blocks(self) -> list[dict]:
        """The (cacheable) system prompt. The stage_2_prompt.md
        template is invariant per-job -- mark it cacheable so
        repeat jobs read at 0.1x input price."""
        template = self._load_prompt()
        block = {"type": "text", "text": template}
        if self.enable_prompt_cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _build_user_messages(
        self,
        stage1_output: Stage1Output,
        correction_note: str = "",
    ) -> list[dict]:
        """The per-job payload (audio metadata + word array)."""
        payload = _build_user_payload(stage1_output, correction_note=correction_note)
        return [{"role": "user", "content": payload}]

    async def decide(
        self,
        stage1: Stage1Output,
        correction_note: str = "",
    ) -> Stage2Output:
        import anthropic   # noqa: F401 -- imported for exception types

        client = self._get_client()

        system_blocks = self._build_system_blocks()
        messages = self._build_user_messages(
            stage1, correction_note=correction_note,
        )

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system_blocks,
            "messages": messages,
            "temperature": self.temperature,
            "thinking": {"type": "disabled"},
        }

        # Use client.messages.parse() with output_format= for
        # native Pydantic validation. If parse() raises a Pydantic
        # ValidationError, the corrective-retry layer catches it.
        response = await client.messages.parse(
            output_format=Stage2Output,
            **kwargs,
        )

        # Cost ledger.
        usage = response.usage
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.last_cost_usd = (
            in_tok * CLAUDE_SONNET_4_6_INPUT_PER_M_USD / 1_000_000
            + out_tok * CLAUDE_SONNET_4_6_OUTPUT_PER_M_USD / 1_000_000
            + cache_write * CLAUDE_SONNET_4_6_CACHE_WRITE_PER_M_USD / 1_000_000
            + cache_read * CLAUDE_SONNET_4_6_CACHE_READ_PER_M_USD / 1_000_000
        )
        self.last_usage = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
        }

        # client.messages.parse() returns a ParsedMessage whose
        # ParsedTextBlock entries each carry a ``parsed_output``
        # field already validated against ``Stage2Output``. (The
        # SDK invokes ``parse_response`` internally; see
        # anthropic.lib._parse._response.) The top-level response
        # has NO ``parsed_output`` attribute -- look inside the
        # content blocks instead.
        parsed: Optional[Stage2Output] = None
        text_block = None
        for block in response.content:
            if getattr(block, "type", "") != "text":
                continue
            text_block = block
            candidate = getattr(block, "parsed_output", None)
            if candidate is not None:
                parsed = candidate
                break
        if parsed is not None:
            return parsed

        # Fallback: parse the raw text ourselves if the SDK didn't
        # populate parsed_output (e.g. refusal, partial response).
        # We still raise JSONDecodeError / ValidationError so the
        # corrective-retry layer catches them.
        raw = getattr(text_block, "text", "") if text_block is not None else ""
        if not raw.strip():
            raise json.JSONDecodeError(
                "Claude response had no parsed_output and no text block "
                "(check stop_reason for max_tokens / refusal)",
                "",
                0,
            )
        cleaned = _strip_markdown_fences(raw)
        data = json.loads(cleaned)
        return Stage2Output.model_validate(data)


# --- Factory ----------------------------------------------------------


def is_valid_provider(name: Optional[str]) -> bool:
    """Strict catalog membership check. Used by the create-job
    endpoint to validate the operator's selection."""
    return name in VALID_PROVIDERS


def create_provider(
    name: Optional[str],
    *,
    prompt_path: Optional[Path] = None,
    **kwargs,
) -> Stage2Provider:
    """Resolve a provider name to an instantiated provider.

    Blank or unknown name -> the default provider (Gemini). Caller-
    supplied kwargs are forwarded to the provider's constructor;
    use this to override ``model``, ``temperature``, etc. per call.

    The actual constructor accepts a subset of kwargs per provider --
    we silently ignore keys the target class doesn't know about so
    a generic dispatch can pass through whatever the orchestrator
    threads in.
    """
    if not name or name not in VALID_PROVIDERS:
        name = DEFAULT_PROVIDER

    if name == PROVIDER_GEMINI:
        sig_keys = {
            "model", "temperature", "thinking_budget",
            "max_output_tokens", "prompt_path",
        }
        filtered = {k: v for k, v in kwargs.items() if k in sig_keys}
        return GeminiStage2Provider(prompt_path=prompt_path, **filtered)
    if name == PROVIDER_CLAUDE:
        sig_keys = {
            "model", "temperature", "max_output_tokens",
            "prompt_path", "enable_prompt_cache",
        }
        filtered = {k: v for k, v in kwargs.items() if k in sig_keys}
        return ClaudeStage2Provider(prompt_path=prompt_path, **filtered)
    # Unreachable -- VALID_PROVIDERS membership was checked above.
    raise RuntimeError(f"Unhandled provider: {name!r}")
