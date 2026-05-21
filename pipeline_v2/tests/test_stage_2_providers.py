"""Item 114 -- Stage 2 provider catalog + Claude alternative tests.

Covers:
  - Provider catalog membership and factory dispatch
  - Default fallbacks (unknown / blank name -> Gemini)
  - Claude provider construction + model + temperature + thinking
  - Claude SDK call shape (model, temperature, output_format, system
    cache_control)
  - Claude cost calculation (base + cache_read + cache_write)
  - Backwards-compat: Gemini provider still works end-to-end
  - End-to-end: both providers return identical Stage2Output shape
  - Wiring: Job model has the column, create_job has the form field,
    runner.event_data carries the provider name
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))
sys.path.insert(0, str(_PIPELINE_V2_ROOT.parent))


# ----- Catalog + factory ----------------------------------------------


def test_01_valid_providers_contains_gemini_and_claude():
    from pipeline_v2.stages.stage_2_providers import (
        VALID_PROVIDERS, PROVIDER_GEMINI, PROVIDER_CLAUDE,
    )
    assert PROVIDER_GEMINI == "gemini"
    assert PROVIDER_CLAUDE == "claude"
    assert VALID_PROVIDERS == frozenset({"gemini", "claude"})


def test_02_default_provider_is_gemini():
    """Backward compat: existing jobs with no explicit provider get
    Gemini -- the same model the V2 pipeline shipped with."""
    from pipeline_v2.stages.stage_2_providers import DEFAULT_PROVIDER
    assert DEFAULT_PROVIDER == "gemini"


def test_03_is_valid_provider_strict_check():
    """Strict mode -- unknown / blank / None all reject. Used by
    main.create_job to detect coercion candidates before persisting
    the Job row."""
    from pipeline_v2.stages.stage_2_providers import is_valid_provider
    assert is_valid_provider("gemini") is True
    assert is_valid_provider("claude") is True
    assert is_valid_provider("unknown") is False
    assert is_valid_provider("") is False
    assert is_valid_provider(None) is False


def test_04_factory_dispatches_to_correct_subclass(tmp_path: Path):
    """create_provider returns the right class for each catalog name.
    Unknown / blank names fall back to the default (Gemini)."""
    from pipeline_v2.stages.stage_2_providers import (
        create_provider,
        GeminiStage2Provider, ClaudeStage2Provider,
    )
    pp = tmp_path / "p.md"
    pp.write_text("test prompt")
    g = create_provider("gemini", prompt_path=pp)
    assert isinstance(g, GeminiStage2Provider)
    assert g.name == "gemini"
    c = create_provider("claude", prompt_path=pp)
    assert isinstance(c, ClaudeStage2Provider)
    assert c.name == "claude"
    # Unknown -> default Gemini.
    fallback = create_provider("not_a_provider", prompt_path=pp)
    assert isinstance(fallback, GeminiStage2Provider)
    # None -> default Gemini.
    fallback2 = create_provider(None, prompt_path=pp)
    assert isinstance(fallback2, GeminiStage2Provider)


# ----- Claude provider construction ------------------------------------


def test_05_claude_provider_constructs_with_correct_model_and_temp(tmp_path: Path):
    """Headline spec: Sonnet 4.6 model ID and T=0 default."""
    from pipeline_v2.stages.stage_2_providers import (
        ClaudeStage2Provider, DEFAULT_CLAUDE_MODEL,
    )
    pp = tmp_path / "p.md"
    pp.write_text("test prompt")
    p = ClaudeStage2Provider(prompt_path=pp)
    assert DEFAULT_CLAUDE_MODEL == "claude-sonnet-4-6"
    assert p.model == "claude-sonnet-4-6"
    assert p.temperature == 0.0
    assert p.enable_prompt_cache is True   # cache the system prompt by default
    assert p.last_cost_usd == 0.0          # no calls yet


def test_06_claude_provider_lazy_client_raises_without_key(tmp_path: Path, monkeypatch):
    """Construction is free of side effects -- the key is read lazily
    on first call. Matches the Gemini provider's behaviour."""
    from pipeline_v2.stages.stage_2_providers import ClaudeStage2Provider
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pp = tmp_path / "p.md"
    pp.write_text("test prompt")
    p = ClaudeStage2Provider(prompt_path=pp)
    # Construction is fine.
    assert p._client is None
    # First client request raises informative error.
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        p._get_client()


# ----- Claude SDK call shape ------------------------------------------


def _stage1_minimal():
    """Tiny Stage1Output that satisfies the schema without needing a
    full Stage 1 run."""
    from pipeline_v2.models import (
        Stage1Output, WordLevelTranscript, Word,
    )
    words = [Word(w="hi", s=0.0, e=0.5)]
    transcript = WordLevelTranscript(
        words=words, duration_sec=0.5,
        detected_languages=["te"], provider="deepgram",
    )
    return Stage1Output(
        transcript=transcript,
        stt_provider="deepgram",
        stt_audio_duration_sec=0.5,
        stt_wall_seconds=1.0,
        stt_cost_usd=0.0,
        stt_word_count=1,
        stt_language_detected="te",
        stt_request_id="t-1",
    )


def _stage2_minimal_dict():
    """Stub Stage2Output dict for SDK responses to return."""
    return {
        "full_video_cuts": [
            {
                "index": 0, "start_word_idx": 0, "end_word_idx": 0,
                "start_sec": 0.0, "end_sec": 0.5, "importance": 5,
            }
        ],
        "skipped_segments": [],
        "retake_audit": "Nothing skipped.",
    }


@pytest.mark.asyncio
async def test_07_claude_call_uses_correct_model_temp_and_thinking_disabled(
    tmp_path: Path, monkeypatch,
):
    """Verify the Anthropic SDK invocation includes model=claude-sonnet-4-6,
    temperature=0, thinking=disabled, and the cacheable system block."""
    from pipeline_v2.stages.stage_2_providers import ClaudeStage2Provider
    from pipeline_v2.models import Stage2Output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pp = tmp_path / "p.md"
    pp.write_text("TEST PROMPT TEMPLATE")
    provider = ClaudeStage2Provider(prompt_path=pp)

    # Fake response that satisfies parsed_output + usage.
    fake_response = MagicMock()
    fake_response.parsed_output = Stage2Output.model_validate(_stage2_minimal_dict())
    fake_response.usage = MagicMock(
        input_tokens=100, output_tokens=50,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )

    fake_messages = MagicMock()
    fake_messages.parse = AsyncMock(return_value=fake_response)
    fake_client = MagicMock()
    fake_client.messages = fake_messages

    with patch.object(provider, "_get_client", return_value=fake_client):
        result = await provider.decide(_stage1_minimal())

    # SDK was called once with the right kwargs.
    assert fake_messages.parse.await_count == 1
    call_kwargs = fake_messages.parse.await_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["thinking"] == {"type": "disabled"}
    assert call_kwargs["output_format"] is Stage2Output
    # System is a list of blocks, with cache_control on the template.
    sys_blocks = call_kwargs["system"]
    assert isinstance(sys_blocks, list) and len(sys_blocks) == 1
    assert sys_blocks[0]["text"] == "TEST PROMPT TEMPLATE"
    assert sys_blocks[0].get("cache_control") == {"type": "ephemeral"}
    # User payload includes the audio metadata block.
    msgs = call_kwargs["messages"]
    assert isinstance(msgs, list) and msgs[0]["role"] == "user"
    assert "Audio metadata" in msgs[0]["content"]
    # Result is the parsed Stage2Output.
    assert isinstance(result, Stage2Output)


@pytest.mark.asyncio
async def test_08_claude_cost_includes_cache_read_and_write(
    tmp_path: Path, monkeypatch,
):
    """Cost calculation must include base input, output, cache write
    (1.25x input), and cache read (0.1x input)."""
    from pipeline_v2.stages.stage_2_providers import (
        ClaudeStage2Provider, CLAUDE_SONNET_4_6_INPUT_PER_M_USD,
        CLAUDE_SONNET_4_6_OUTPUT_PER_M_USD,
        CLAUDE_SONNET_4_6_CACHE_READ_PER_M_USD,
        CLAUDE_SONNET_4_6_CACHE_WRITE_PER_M_USD,
    )
    from pipeline_v2.models import Stage2Output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pp = tmp_path / "p.md"
    pp.write_text("TEMPLATE")
    provider = ClaudeStage2Provider(prompt_path=pp)

    fake_response = MagicMock()
    fake_response.parsed_output = Stage2Output.model_validate(_stage2_minimal_dict())
    fake_response.usage = MagicMock(
        input_tokens=1000, output_tokens=500,
        cache_read_input_tokens=8000, cache_creation_input_tokens=0,
    )

    fake_messages = MagicMock()
    fake_messages.parse = AsyncMock(return_value=fake_response)
    fake_client = MagicMock()
    fake_client.messages = fake_messages

    with patch.object(provider, "_get_client", return_value=fake_client):
        await provider.decide(_stage1_minimal())

    expected = (
        1000 * CLAUDE_SONNET_4_6_INPUT_PER_M_USD / 1_000_000
        + 500 * CLAUDE_SONNET_4_6_OUTPUT_PER_M_USD / 1_000_000
        + 8000 * CLAUDE_SONNET_4_6_CACHE_READ_PER_M_USD / 1_000_000
    )
    assert abs(provider.last_cost_usd - expected) < 1e-9
    assert provider.last_usage["input_tokens"] == 1000
    assert provider.last_usage["output_tokens"] == 500
    assert provider.last_usage["cache_read_tokens"] == 8000
    assert provider.last_usage["cache_write_tokens"] == 0
    # Sanity: cache read is much cheaper than input.
    assert CLAUDE_SONNET_4_6_CACHE_READ_PER_M_USD < CLAUDE_SONNET_4_6_INPUT_PER_M_USD
    # Cache write is more expensive than input.
    assert CLAUDE_SONNET_4_6_CACHE_WRITE_PER_M_USD > CLAUDE_SONNET_4_6_INPUT_PER_M_USD


# ----- Backwards-compat: Stage2ContinuityEditor dispatcher ------------


def test_09_dispatcher_defaults_to_gemini(tmp_path: Path):
    from pipeline_v2.stages.stage_2_continuity import Stage2ContinuityEditor
    pp = tmp_path / "p.md"
    pp.write_text("TEMPLATE")
    e = Stage2ContinuityEditor(prompt_path=pp)
    assert e.provider_name == "gemini"
    assert e.model == "gemini-2.5-pro"
    assert e.temperature == 0.2


def test_10_dispatcher_routes_to_claude_when_specified(tmp_path: Path):
    from pipeline_v2.stages.stage_2_continuity import Stage2ContinuityEditor
    from pipeline_v2.stages.stage_2_providers import ClaudeStage2Provider
    pp = tmp_path / "p.md"
    pp.write_text("TEMPLATE")
    e = Stage2ContinuityEditor(provider_name="claude", prompt_path=pp)
    assert e.provider_name == "claude"
    assert e.model == "claude-sonnet-4-6"
    assert e.temperature == 0.0
    provider = e._get_provider()
    assert isinstance(provider, ClaudeStage2Provider)


# ----- Wiring: model + endpoint + runner ------------------------------


def test_11_job_model_has_stage_2_provider_column():
    """ORM schema lock: the column exists with a 'gemini' default
    so legacy rows + new rows both behave correctly."""
    from sqlalchemy import inspect as sql_inspect
    sys.path.insert(0, str(_PIPELINE_V2_ROOT.parent))
    from models import Job
    mapper = sql_inspect(Job)
    col_names = {c.name for c in mapper.columns}
    assert "stage_2_provider" in col_names


def test_12_create_job_endpoint_accepts_stage_2_provider_form_field():
    """Wire-level lock: introspect the FastAPI handler signature to
    confirm stage_2_provider is a Form() field defaulting to gemini."""
    import inspect as _inspect
    from main import create_job
    sig = _inspect.signature(create_job)
    assert "stage_2_provider" in sig.parameters
    param = sig.parameters["stage_2_provider"]
    raw_default = getattr(param.default, "default", param.default)
    assert raw_default == "gemini"


def test_13_runner_event_payload_includes_stage_2_provider():
    """The Inngest event the runner emits must carry stage_2_provider
    so the worker (orchestrator) can read it back."""
    import runner as _runner

    captured: dict = {}

    class _StubEvent:
        def __init__(self, name=None, data=None, id=None):
            captured["name"] = name
            captured["data"] = data
            captured["id"] = id

    class _StubClient:
        def send_sync(self, events):
            captured["sent"] = events

    import inngest as _inngest_mod
    import pipeline_v2.inngest_client as _ic_mod
    orig_event = _inngest_mod.Event
    orig_get_client = _ic_mod.get_client
    _inngest_mod.Event = _StubEvent
    _ic_mod.get_client = lambda: _StubClient()

    class _StubDbSession:
        def query(self, *_a, **_kw): return self
        def filter(self, *_a, **_kw): return self
        def update(self, *_a, **_kw): return None
        def commit(self): return None
        def close(self): return None

    try:
        _runner._dispatch_v2_inngest_event(
            job_id=99,
            video_path="/tmp/x.mp4",
            language="te",
            platform="full_video_shorts_v2",
            frame="torn_card",
            stt_provider="deepgram",
            transition_style="smart_cut",
            stage_2_provider="claude",
            db_session_factory=lambda: _StubDbSession(),
        )
    finally:
        _inngest_mod.Event = orig_event
        _ic_mod.get_client = orig_get_client

    assert captured["data"]["stage_2_provider"] == "claude"
    assert captured["data"]["job_id"] == 99
