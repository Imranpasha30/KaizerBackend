"""Unit tests for Stage 3b (Metadata Extractor) -- Step 7.3.

Same testing pattern as Stage 2 / 2.5 / 3a. All Gemini calls mocked.
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
    Metadata,
    Word,
)
from pipeline_v2.stages.stage_3b_metadata import (
    Stage3bMetadataExtractor,
    _strip_markdown_fences,
)


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _clean(n: int = 30) -> CleanTranscript:
    words = [Word(w=f"w{i}", s=i * 0.5, e=i * 0.5 + 0.3) for i in range(n)]
    return CleanTranscript(
        words=words,
        clip_boundaries={0: (0, n - 1)},
        source_word_map=list(range(n)),
    )


def _entity(name: str, type_: str = "PERSON") -> Entity:
    return Entity(
        canonical_name=name,
        native_name=name,
        first_mention_word_idx=0,
        type=type_,
        mentions=[0],
    )


def _good_metadata_body() -> dict:
    return {
        "video_type": "SOLO",
        "language": "te-en",
        "total_speakers": 1,
        "overall_summary": "Anchor delivers a single-topic bulletin on the day's news.",
        "overall_summary_native": "ఈరోజు ఒక topic పైన anchor monologue.",
        "shorts_headline_native": "ఈరోజు ప్రధాన వార్త — షాక్",
        "bulletin_marquee_points": [
            "ఈరోజు ప్రధాన వార్త 1",
            "ఈరోజు ప్రధాన వార్త 2",
            "ఈరోజు ప్రధాన వార్త 3",
        ],
        "image_search_queries": ["news studio", "anchor microphone"],
        "key_people": [],
        "key_people_native": [],
        "key_topics": ["news"],
        "key_locations": [],
    }


def _mock_response(json_text: str) -> MagicMock:
    r = MagicMock()
    r.text = json_text
    return r


def _writable_prompt(tmp_path: Path, body: str = "STAGE 3b STUB") -> Path:
    p = tmp_path / "stage_3b_prompt.md"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    prompt_path = _writable_prompt(tmp_path, "TEMPLATE")
    return Stage3bMetadataExtractor(prompt_path=prompt_path)


# ====================================================================== #
# Constructor + lazy resources                                            #
# ====================================================================== #


class TestConstructor:
    def test_defaults(self):
        e = Stage3bMetadataExtractor()
        assert e.model == "gemini-2.5-flash"
        assert e.temperature == 0.7
        assert e.thinking_budget == 512
        assert e.max_output_tokens == 2048

    def test_init_is_side_effect_free(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        e = Stage3bMetadataExtractor(prompt_path=Path("/nope.md"))
        assert e._client is None
        assert e._prompt_template is None

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        e = Stage3bMetadataExtractor()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            e._get_client()

    def test_missing_prompt_raises(self):
        e = Stage3bMetadataExtractor(prompt_path=Path("/nope.md"))
        with pytest.raises(FileNotFoundError, match="Stage 3b prompt"):
            e._load_prompt()


# ====================================================================== #
# Prompt construction                                                     #
# ====================================================================== #


class TestPromptConstruction:
    def test_contains_prose_not_word_array(self, extractor):
        # Stage 3b drops word-level timestamps -- it uses prose.
        clean = _clean(10)
        prompt = extractor._build_prompt(clean, [])
        # Prose serialisation: words joined with spaces, no "idx" field
        assert "w0 w1 w2" in prompt
        assert '"idx":' not in prompt

    def test_includes_entities(self, extractor):
        clean = _clean(10)
        prompt = extractor._build_prompt(
            clean, [_entity("Revanth Reddy", "PERSON")],
        )
        assert "Revanth Reddy" in prompt
        assert "PERSON" in prompt

    def test_correction_note_appended(self, extractor):
        prompt = extractor._build_prompt(_clean(5), [], correction_note="FIX")
        assert "## Correction note" in prompt
        assert "FIX" in prompt


# ====================================================================== #
# _parse_response                                                         #
# ====================================================================== #


class TestParseResponse:
    def test_happy_path(self, extractor):
        resp = _mock_response(json.dumps(_good_metadata_body()))
        out = extractor._parse_response(resp)
        assert isinstance(out, Metadata)
        assert out.video_type == "SOLO"
        assert out.language == "te-en"

    def test_strips_fences(self, extractor):
        body = _good_metadata_body()
        resp = _mock_response(f"```json\n{json.dumps(body)}\n```")
        out = extractor._parse_response(resp)
        assert out.total_speakers == 1

    def test_empty_text_raises(self, extractor):
        with pytest.raises(json.JSONDecodeError):
            extractor._parse_response(_mock_response(""))

    def test_invented_video_type_caught(self, extractor):
        # video_type is Literal["SOLO","INTERVIEW","PRESS_CONFERENCE","PANEL","MIXED"]
        # Anything else -> ValidationError -> corrective retry.
        body = _good_metadata_body()
        body["video_type"] = "STUDIO"   # invented
        with pytest.raises(ValidationError):
            extractor._parse_response(_mock_response(json.dumps(body)))

    def test_missing_required_field_caught(self, extractor):
        # All 12 fields required (no defaults per D-7.8).
        body = _good_metadata_body()
        del body["shorts_headline_native"]
        with pytest.raises(ValidationError):
            extractor._parse_response(_mock_response(json.dumps(body)))


# ====================================================================== #
# async extract: corrective retry                                         #
# ====================================================================== #


class TestExtractRetry:
    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(self, extractor):
        mock_call = AsyncMock(
            return_value=_mock_response(json.dumps(_good_metadata_body()))
        )
        with patch.object(extractor, "_call_gemini", new=mock_call):
            out = await extractor.extract(_clean(10), [])
        assert out.video_type == "SOLO"
        assert mock_call.await_count == 1

    @pytest.mark.asyncio
    async def test_video_type_violation_triggers_retry(self, extractor):
        bad = _good_metadata_body()
        bad["video_type"] = "STUDIO"
        good = _good_metadata_body()
        mock_call = AsyncMock(side_effect=[
            _mock_response(json.dumps(bad)),
            _mock_response(json.dumps(good)),
        ])
        with patch.object(extractor, "_call_gemini", new=mock_call):
            out = await extractor.extract(_clean(10), [])
        assert mock_call.await_count == 2
        assert out.video_type == "SOLO"

    @pytest.mark.asyncio
    async def test_double_failure_raises_runtime_error(self, extractor):
        mock_call = AsyncMock(side_effect=[
            _mock_response("not json"),
            _mock_response("still not json"),
        ])
        with patch.object(extractor, "_call_gemini", new=mock_call):
            with pytest.raises(RuntimeError, match="corrective retry"):
                await extractor.extract(_clean(10), [])
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_other_exceptions_propagate(self, extractor):
        mock_call = AsyncMock(side_effect=ConnectionError("net down"))
        with patch.object(extractor, "_call_gemini", new=mock_call):
            with pytest.raises(ConnectionError, match="net down"):
                await extractor.extract(_clean(10), [])
        assert mock_call.await_count == 1
