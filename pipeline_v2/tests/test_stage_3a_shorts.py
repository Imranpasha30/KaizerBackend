"""Unit tests for Stage 3a (Shorts Generator) -- Step 7.2.

Same testing pattern as Stage 2 / 2.5:

  - All Gemini calls mocked.
  - Manual model_validate path (NEVER response.parsed).
  - Corrective retry on JSONDecodeError / ValidationError; second
    failure raises RuntimeError.
  - The 15-60s duration validator on ShortsCut IS part of the
    Pydantic validation -- corrective retry catches violations.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    Stage3aOutput,
    Word,
)
from pipeline_v2.stages.stage_3a_shorts import (
    Stage3aShortsGenerator,
    _strip_markdown_fences,
)


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _word(text: str, s: float, e: float) -> Word:
    return Word(w=text, s=s, e=e)


def _clean(n: int = 60) -> CleanTranscript:
    """Synthetic clean transcript -- n sequential 0.5s words spanning
    n*0.55 seconds. Long enough that 15-60s shorts can fit.
    """
    words = [_word(f"w{i}", i * 0.55, i * 0.55 + 0.5) for i in range(n)]
    return CleanTranscript(
        words=words,
        clip_boundaries={0: (0, n - 1)},
        source_word_map=list(range(n)),
    )


def _entity(name: str = "X", mentions=None, type_: str = "PERSON") -> Entity:
    return Entity(
        canonical_name=name,
        native_name=name,
        first_mention_word_idx=min(mentions) if mentions else 0,
        type=type_,
        mentions=mentions or [0],
    )


def _shorts_cut(idx: int = 0, start: float = 10.0, end: float = 28.0,
                hook: str = "Test hook", importance: int = 7) -> dict:
    return {
        "index": idx,
        "start_sec": start,
        "end_sec": end,
        "hook": hook,
        "importance": importance,
    }


def _good_response_body(n: int = 3) -> dict:
    return {
        "shorts_cuts": [
            _shorts_cut(idx=i, start=10.0 + i * 30, end=30.0 + i * 30,
                        hook=f"Hook {i}",
                        # importance is 1-10 inclusive (ShortsCut.Field
                        # ge=1, le=10) -- modulate so n up to 10 stays
                        # within range.
                        importance=((i % 10) + 1))
            for i in range(n)
        ],
    }


def _mock_response(json_text: str) -> MagicMock:
    r = MagicMock()
    r.text = json_text
    return r


def _writable_prompt(tmp_path: Path, body: str = "STAGE 3a STUB PROMPT") -> Path:
    p = tmp_path / "stage_3a_prompt.md"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def generator(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    prompt_path = _writable_prompt(tmp_path, "TEMPLATE")
    return Stage3aShortsGenerator(prompt_path=prompt_path)


# ====================================================================== #
# _strip_markdown_fences                                                  #
# ====================================================================== #


class TestStripMarkdownFences:
    def test_passthrough_clean_json(self):
        assert _strip_markdown_fences('{"x": 1}') == '{"x": 1}'

    def test_strip_json_fence(self):
        assert _strip_markdown_fences('```json\n{"x": 1}\n```') == '{"x": 1}'

    def test_strip_plain_fence(self):
        assert _strip_markdown_fences('```\n{"x": 1}\n```') == '{"x": 1}'


# ====================================================================== #
# Constructor + lazy resources                                            #
# ====================================================================== #


class TestConstructor:
    def test_defaults(self):
        g = Stage3aShortsGenerator()
        assert g.model == "gemini-2.5-flash"
        assert g.temperature == 0.7
        assert g.thinking_budget == 512
        assert g.max_output_tokens == 4096

    def test_kwargs_override(self):
        g = Stage3aShortsGenerator(
            model="gemini-2.5-flash-lite",
            temperature=0.5,
            thinking_budget=1024,
            max_output_tokens=2048,
        )
        assert g.model == "gemini-2.5-flash-lite"
        assert g.temperature == 0.5
        assert g.thinking_budget == 1024
        assert g.max_output_tokens == 2048

    def test_init_is_side_effect_free(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        g = Stage3aShortsGenerator(prompt_path=Path("/nope/missing.md"))
        assert g._client is None
        assert g._prompt_template is None

    def test_missing_api_key_raises_on_first_use(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        g = Stage3aShortsGenerator()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            g._get_client()

    def test_missing_prompt_raises_on_first_use(self):
        g = Stage3aShortsGenerator(prompt_path=Path("/nope/missing.md"))
        with pytest.raises(FileNotFoundError, match="Stage 3a prompt"):
            g._load_prompt()


# ====================================================================== #
# Prompt construction                                                     #
# ====================================================================== #


class TestPromptConstruction:
    def test_prompt_contains_metadata_and_entities(self, generator):
        clean = _clean(40)
        entities = [_entity("Revanth Reddy", [5, 10, 20], "PERSON")]
        prompt = generator._build_prompt(clean, entities)
        assert "TEMPLATE" in prompt
        assert "Word count:        40" in prompt
        # Entities appear in context section (NOT in output)
        assert "Revanth Reddy" in prompt
        # Word array surfaces
        assert '"idx": 0' in prompt
        assert '"w": "w0"' in prompt

    def test_empty_entities_works(self, generator):
        # 0 entities is legitimate -- the bulletin might have no
        # recognized named entities. Stage 3a should still produce
        # shorts.
        clean = _clean(20)
        prompt = generator._build_prompt(clean, [])
        assert "[]" in prompt  # entities serializes to []

    def test_correction_note_appended(self, generator):
        clean = _clean(20)
        prompt = generator._build_prompt(clean, [], correction_note="FIX X")
        assert "## Correction note" in prompt
        assert "FIX X" in prompt


# ====================================================================== #
# _parse_response: manual validate path                                   #
# ====================================================================== #


class TestParseResponse:
    """Lenient per-entry validation (Option E, 12.2a re-run #3).

    Contract change vs. pre-Option-E:
      * ``_parse_response`` no longer raises ValidationError on a
        single per-entry duration violation. It DROPS the offending
        entry with a structured log and returns a ``_ParseResult``.
      * It still raises ``json.JSONDecodeError`` for total JSON
        parse failures (the only "structural" failure left in
        ``_parse_response``).
    """

    def test_happy_path(self, generator):
        body = _good_response_body(n=4)
        resp = _mock_response(json.dumps(body))
        result = generator._parse_response(resp, attempt=1)
        assert result.output is not None
        assert isinstance(result.output, Stage3aOutput)
        assert len(result.output.shorts_cuts) == 4
        assert result.valid_count == 4
        assert result.dropped_count == 0

    def test_strips_fences(self, generator):
        body = _good_response_body(n=3)
        resp = _mock_response(f"```json\n{json.dumps(body)}\n```")
        result = generator._parse_response(resp, attempt=1)
        assert result.valid_count == 3

    def test_empty_text_raises_json_decode_error(self, generator):
        resp = _mock_response("")
        with pytest.raises(json.JSONDecodeError):
            generator._parse_response(resp, attempt=1)

    def test_invalid_json_raises_json_decode_error(self, generator):
        resp = _mock_response("not json")
        with pytest.raises(json.JSONDecodeError):
            generator._parse_response(resp, attempt=1)

    def test_parse_drops_invalid_entries_keeps_valid(self, generator):
        # New contract: a duration violation no longer raises -- the
        # invalid entry is dropped and the valid ones survive.
        body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0,   20.0),                # 20s OK
                _shorts_cut(1, 30.0,  37.0, "too short"),   # 7s -- DROP
                _shorts_cut(2, 50.0,  72.0),                # 22s OK
                _shorts_cut(3, 80.0,  100.0),               # 20s OK
                _shorts_cut(4, 110.0, 117.0, "too short"),  # 7s -- DROP
                _shorts_cut(5, 130.0, 155.0),               # 25s OK
                _shorts_cut(6, 160.0, 178.0),               # 18s OK
                _shorts_cut(7, 190.0, 199.0, "too short"),  # 9s -- DROP
            ],
        }
        resp = _mock_response(json.dumps(body))
        result = generator._parse_response(resp, attempt=1)
        assert result.output is not None
        assert result.valid_count == 5      # kept the 5 valid
        assert result.dropped_count == 3
        # The kept entries are the valid ones, in order.
        kept_starts = [c.start_sec for c in result.output.shorts_cuts]
        assert kept_starts == [0.0, 50.0, 80.0, 130.0, 160.0]

    def test_parse_logs_structured_warning_on_drop(
        self, generator, caplog,
    ):
        body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 20.0),                  # OK
                _shorts_cut(1, 30.0, 38.0, "too short"),    # 8s -- DROP
                _shorts_cut(2, 50.0, 70.0),                 # OK
                _shorts_cut(3, 80.0, 100.0),                # OK
            ],
        }
        import logging as _logging
        with caplog.at_level(_logging.WARNING,
                             logger="pipeline_v2.stage_3a"):
            result = generator._parse_response(
                _mock_response(json.dumps(body)), attempt=1,
            )
        # One drop record with the right slug + telemetry.
        drop_records = [
            rec for rec in caplog.records
            if getattr(rec, "event", None) == "stage_3a_dropped_invalid_short"
        ]
        assert len(drop_records) == 1
        rec = drop_records[0]
        assert rec.attempt == 1
        assert rec.index_in_response == 1
        assert rec.start_sec == 30.0
        assert rec.end_sec == 38.0
        assert rec.duration_sec == 8.0
        assert rec.duration_violation == "below_15s"
        # Hook preview present (entry had "too short" hook).
        assert rec.hook_preview == "too short"
        assert result.valid_count == 3
        assert result.dropped_count == 1

    def test_parse_drops_above_60s_as_separate_violation_slug(
        self, generator, caplog,
    ):
        body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 20.0),                   # OK
                _shorts_cut(1, 30.0, 105.0, "too long"),     # 75s -- DROP
                _shorts_cut(2, 110.0, 130.0),                # OK
                _shorts_cut(3, 140.0, 158.0),                # OK
            ],
        }
        import logging as _logging
        with caplog.at_level(_logging.WARNING,
                             logger="pipeline_v2.stage_3a"):
            generator._parse_response(
                _mock_response(json.dumps(body)), attempt=1,
            )
        drop_records = [
            rec for rec in caplog.records
            if getattr(rec, "event", None) == "stage_3a_dropped_invalid_short"
        ]
        assert len(drop_records) == 1
        assert drop_records[0].duration_violation == "above_60s"

    def test_parse_under_3_valid_returns_none_output(self, generator):
        # 2 valid + 2 dropped = below the _STAGE3A_MIN_SHORTS=3 cutoff.
        # The lenient parser returns _ParseResult with output=None;
        # the caller's 3-tier logic decides retry vs raise.
        body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 5.0),                  # 5s -- DROP
                _shorts_cut(1, 10.0, 30.0),                # OK
                _shorts_cut(2, 35.0, 42.0),                # 7s -- DROP
                _shorts_cut(3, 50.0, 70.0),                # OK
            ],
        }
        resp = _mock_response(json.dumps(body))
        result = generator._parse_response(resp, attempt=1)
        assert result.output is None
        assert result.valid_count == 2
        assert result.dropped_count == 2

    def test_parse_caps_above_10_at_max_length(self, generator):
        # 12 entries, all valid -> we cap at the Stage3aOutput
        # max_length=10. No drop count for the excess (it's a cap,
        # not a validation failure).
        body = {
            "shorts_cuts": [
                _shorts_cut(i, i * 30.0, i * 30.0 + 20.0)
                for i in range(12)
            ],
        }
        resp = _mock_response(json.dumps(body))
        result = generator._parse_response(resp, attempt=1)
        assert result.output is not None
        assert result.valid_count == 10
        assert len(result.output.shorts_cuts) == 10
        # dropped_count counts validation drops only, NOT cap-trims.
        assert result.dropped_count == 0

    def test_parse_attempt_2_logged_in_drop_extras(self, generator, caplog):
        # The attempt number in the drop log is what telemetry uses to
        # distinguish "first attempt drops" (prompt-quality signal)
        # from "retry drops" (Gemini persistent misbehaviour signal).
        body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 7.0, "drop"),            # 7s -- DROP
                _shorts_cut(1, 10.0, 30.0),
                _shorts_cut(2, 40.0, 60.0),
                _shorts_cut(3, 70.0, 90.0),
            ],
        }
        import logging as _logging
        with caplog.at_level(_logging.WARNING,
                             logger="pipeline_v2.stage_3a"):
            generator._parse_response(
                _mock_response(json.dumps(body)), attempt=2,
            )
        drop_records = [
            rec for rec in caplog.records
            if getattr(rec, "event", None) == "stage_3a_dropped_invalid_short"
        ]
        assert len(drop_records) == 1
        assert drop_records[0].attempt == 2

    # ---- 12.2a re-run #5 fix: contiguous index renumbering ---------

    def test_parse_renumbers_surviving_shorts_to_contiguous_indices(
        self, generator, caplog,
    ):
        # Regression for the 12.2a re-run #4 cascade: Option E's
        # lenient drop kept original Gemini-emitted indices, which
        # left gaps when ANY entry was dropped. Downstream
        # (build_v1_shorts_editor_meta D-8.12) requires 0-based
        # contiguous indices. The renumber step in _parse_response
        # re-establishes that invariant at the Stage 3a boundary.
        #
        # Input: 7 entries at original indices 0..6; entries at
        # positions 1, 4, 6 violate the 15-60s duration constraint.
        # Expected: 4 survivors renumbered to [0, 1, 2, 3].
        body = {
            "shorts_cuts": [
                _shorts_cut(0,  0.0,  20.0,  "kept"),       # OK
                _shorts_cut(1,  30.0, 38.0,  "drop"),       # 8s DROP
                _shorts_cut(2,  45.0, 65.0,  "kept"),       # OK
                _shorts_cut(3,  70.0, 90.0,  "kept"),       # OK
                _shorts_cut(4,  100.0, 107.0, "drop"),      # 7s DROP
                _shorts_cut(5,  115.0, 135.0, "kept"),      # OK
                _shorts_cut(6,  140.0, 148.0, "drop"),      # 8s DROP
            ],
        }
        import logging as _logging
        with caplog.at_level(_logging.WARNING,
                             logger="pipeline_v2.stage_3a"):
            result = generator._parse_response(
                _mock_response(json.dumps(body)), attempt=1,
            )

        assert result.output is not None
        assert result.valid_count == 4
        assert result.dropped_count == 3
        # Indices must be contiguous 0..N-1.
        assert [c.index for c in result.output.shorts_cuts] == [0, 1, 2, 3]
        # At least one drop log emitted for telemetry.
        drop_records = [
            rec for rec in caplog.records
            if getattr(rec, "event", None) == "stage_3a_dropped_invalid_short"
        ]
        assert len(drop_records) >= 1

    def test_parse_renumbers_when_first_short_dropped(self, generator):
        # Edge case: position 0 is the dropped entry. The new index 0
        # must carry the content from original-position-1, NOT the
        # dropped content.
        body = {
            "shorts_cuts": [
                _shorts_cut(0,  0.0,  5.0,  "drop"),    # 5s DROP
                _shorts_cut(1,  10.0, 30.0, "first surviving"),
                _shorts_cut(2,  40.0, 60.0, "second"),
                _shorts_cut(3,  70.0, 90.0, "third"),
                _shorts_cut(4,  100.0, 120.0, "fourth"),
            ],
        }
        result = generator._parse_response(
            _mock_response(json.dumps(body)), attempt=1,
        )
        assert result.output is not None
        assert result.valid_count == 4
        assert [c.index for c in result.output.shorts_cuts] == [0, 1, 2, 3]
        # New index 0 carries the first-surviving hook (original pos 1).
        assert result.output.shorts_cuts[0].hook == "first surviving"

    def test_parse_renumbers_when_consecutive_drops(self, generator):
        # Edge case: positions 2 and 3 (consecutive) are dropped. The
        # output must still be contiguous 0..N-1 with no gap.
        body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0,   20.0,  "first"),
                _shorts_cut(1, 30.0,  50.0,  "second"),
                _shorts_cut(2, 55.0,  62.0,  "drop"),    # 7s DROP
                _shorts_cut(3, 65.0,  72.0,  "drop"),    # 7s DROP
                _shorts_cut(4, 80.0,  100.0, "third"),
                _shorts_cut(5, 110.0, 130.0, "fourth"),
            ],
        }
        result = generator._parse_response(
            _mock_response(json.dumps(body)), attempt=1,
        )
        assert result.output is not None
        assert result.valid_count == 4
        assert [c.index for c in result.output.shorts_cuts] == [0, 1, 2, 3]
        # New index 2 carries the content from original position 4
        # (the entry that filled the slot after the two consecutive drops).
        assert result.output.shorts_cuts[2].hook == "third"


# ====================================================================== #
# async generate: end-to-end with mocked Gemini                           #
# ====================================================================== #


class TestGenerateHappyPath:
    @pytest.mark.asyncio
    async def test_first_attempt_with_5_valid_succeeds_no_retry(
        self, generator,
    ):
        # New contract: attempt 1 with >= 5 valid shorts returns
        # immediately. (4 valid would trigger a retry under the new
        # 3-tier logic.)
        clean = _clean(60)
        body = _good_response_body(n=5)
        mock_call = AsyncMock(return_value=_mock_response(json.dumps(body)))
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert len(out.shorts_cuts) == 5
        assert mock_call.await_count == 1

    @pytest.mark.asyncio
    async def test_first_attempt_with_8_valid_succeeds_no_retry(
        self, generator,
    ):
        # 8 valid shorts (well above the healthy threshold) -- single
        # Gemini call.
        clean = _clean(80)
        body = _good_response_body(n=8)
        mock_call = AsyncMock(return_value=_mock_response(json.dumps(body)))
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert len(out.shorts_cuts) == 8
        assert mock_call.await_count == 1


class TestGenerateCorrectiveRetry:
    """3-tier outcome (Option E):

      Attempt 1 valid >= 5: return immediately.
      Attempt 1 valid 3-4: retry (degraded -- want more variety).
      Attempt 1 valid <  3: retry (insufficient).
      Attempt 2 valid >= 3: accept (degraded acceptance beats failure).
      Attempt 2 valid <  3: raise RuntimeError.
    """

    @pytest.mark.asyncio
    async def test_attempt1_with_4_valid_triggers_retry(self, generator):
        # 4 valid shorts is in the 3-4 degraded band -- retry.
        clean = _clean(80)
        first_body = _good_response_body(n=4)
        retry_body = _good_response_body(n=5)
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(first_body)),
            _mock_response(json.dumps(retry_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert mock_call.await_count == 2
        assert len(out.shorts_cuts) == 5

    @pytest.mark.asyncio
    async def test_attempt1_with_3_valid_triggers_retry(self, generator):
        # Exactly at the min cutoff -- still below healthy (5), retry.
        clean = _clean(80)
        first_body = _good_response_body(n=3)
        retry_body = _good_response_body(n=6)
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(first_body)),
            _mock_response(json.dumps(retry_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert mock_call.await_count == 2
        assert len(out.shorts_cuts) == 6

    @pytest.mark.asyncio
    async def test_attempt1_duration_violation_triggers_retry(
        self, generator,
    ):
        # Attempt 1 has 2 valid + 1 dropped (12s < 15s) = 2 valid -
        # below the >=3 min, so retry.
        clean = _clean(80)
        bad_body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 12.0, "TOO SHORT"),   # 12s -- DROP
                _shorts_cut(1, 30.0, 50.0),               # OK
                _shorts_cut(2, 60.0, 80.0),               # OK
            ],
        }
        good_body = _good_response_body(n=5)
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(bad_body)),
            _mock_response(json.dumps(good_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert mock_call.await_count == 2
        assert len(out.shorts_cuts) == 5

    @pytest.mark.asyncio
    async def test_json_decode_triggers_retry(self, generator):
        clean = _clean(80)
        good_body = _good_response_body(n=5)
        mock_call = AsyncMock(side_effect=[
            _mock_response("not json"),
            _mock_response(json.dumps(good_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_attempt2_with_3_valid_accepts_degraded(self, generator):
        # Attempt 1 yields 4 valid (retry triggered). Attempt 2
        # yields 3 valid (above min, below healthy). The new
        # contract ACCEPTS this (degraded but functional).
        clean = _clean(80)
        first_body = _good_response_body(n=4)
        retry_body = _good_response_body(n=3)
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(first_body)),
            _mock_response(json.dumps(retry_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            out = await generator.generate(clean, [])
        assert mock_call.await_count == 2
        # Degraded acceptance: 3 shorts is enough to render a bulletin.
        assert len(out.shorts_cuts) == 3

    @pytest.mark.asyncio
    async def test_attempt2_under_3_valid_raises(self, generator):
        # Both attempts yield <3 valid -- raise RuntimeError so the
        # outer Inngest retry layer can handle the structural failure.
        clean = _clean(80)
        # Attempt 1: 1 valid + 4 short violations = 1 valid (<3)
        attempt1_body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 5.0, "drop"),       # 5s DROP
                _shorts_cut(1, 10.0, 14.0, "drop"),     # 4s DROP
                _shorts_cut(2, 20.0, 40.0),             # OK
                _shorts_cut(3, 50.0, 58.0, "drop"),     # 8s DROP
                _shorts_cut(4, 60.0, 67.0, "drop"),     # 7s DROP
            ],
        }
        # Attempt 2: still only 2 valid
        attempt2_body = {
            "shorts_cuts": [
                _shorts_cut(0, 0.0, 20.0),
                _shorts_cut(1, 30.0, 50.0),
                _shorts_cut(2, 60.0, 67.0, "drop"),     # 7s DROP
            ],
        }
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(attempt1_body)),
            _mock_response(json.dumps(attempt2_body)),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await generator.generate(clean, [])
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_double_json_failure_raises_runtime_error(self, generator):
        clean = _clean(80)
        mock_call = AsyncMock(side_effect=[
            _mock_response("not json"),
            _mock_response("still not json"),
        ])
        with patch.object(generator, "_call_gemini", new=mock_call):
            with pytest.raises(RuntimeError, match="unparseable JSON"):
                await generator.generate(clean, [])
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_other_exceptions_propagate(self, generator):
        clean = _clean(80)
        mock_call = AsyncMock(side_effect=ConnectionError("network down"))
        with patch.object(generator, "_call_gemini", new=mock_call):
            with pytest.raises(ConnectionError, match="network down"):
                await generator.generate(clean, [])
        assert mock_call.await_count == 1
