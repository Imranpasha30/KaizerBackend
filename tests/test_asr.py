"""
tests/test_asr.py
==================
Pytest coverage for pipeline_core.asr.

Speed contract
--------------
- Fast tests (< 1 s): mock Whisper pipeline + helpers via pytest-mock.
- Slow tests (@pytest.mark.slow): call transcribe() for real — skipped by default.

Run fast tests only:
    pytest tests/test_asr.py -m "not slow"
"""
from __future__ import annotations

import os
import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from pipeline_core.asr import (
    Word,
    Sentence,
    TranscriptionResult,
    _group_words_into_sentences,
    _resolve_model,
    _TELUGU_MODEL_PATH,
    _SMALL_MODEL_PATH,
    _HF_FALLBACK,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def _fake_pipeline_output(words: list[Word]) -> dict:
    """Build a fake HF pipeline output dict from a list of Word objects."""
    return {
        "text": " ".join(w.text for w in words),
        "chunks": [
            {"text": w.text, "timestamp": (w.start, w.end)} for w in words
        ],
    }


# ── Dataclass shape ────────────────────────────────────────────────────────────

def test_transcription_result_dataclass_shape():
    """TranscriptionResult must have all documented fields with correct types."""
    field_names = {f.name for f in dataclasses.fields(TranscriptionResult)}
    for expected in ("sentences", "language", "model_used", "full_text", "warnings"):
        assert expected in field_names, f"Missing field {expected!r} in TranscriptionResult"

    # Defaults
    result = TranscriptionResult(
        sentences=[], language="en", model_used="openai/whisper-small", full_text=""
    )
    assert isinstance(result.warnings, list), "warnings must default to list"
    assert result.warnings == [], f"warnings default must be empty, got {result.warnings}"

    # Word and Sentence shape
    w_fields = {f.name for f in dataclasses.fields(Word)}
    assert {"text", "start", "end"} <= w_fields, f"Word missing fields: {w_fields}"

    s_fields = {f.name for f in dataclasses.fields(Sentence)}
    assert {"text", "start", "end", "words"} <= s_fields, f"Sentence missing fields: {s_fields}"


# ── Model routing (fast, no real Whisper) ─────────────────────────────────────

def test_language_routing_telugu_picks_telugu_model(mocker):
    """language='te' + local telugu model dir exists → returns telugu model path."""
    mocker.patch("os.path.isdir", side_effect=lambda p: (
        True if p == _TELUGU_MODEL_PATH else False
    ))
    model_path, warnings = _resolve_model(language="te", model_hint=None)
    assert model_path == _TELUGU_MODEL_PATH, (
        f"Expected Telugu model path, got {model_path!r}"
    )
    assert warnings == [], f"Expected no warnings for Telugu routing, got {warnings}"


def test_language_routing_default_picks_whisper_small(mocker):
    """language=None + local whisper-small exists → returns whisper-small path."""
    mocker.patch("os.path.isdir", side_effect=lambda p: (
        True if p == _SMALL_MODEL_PATH else False
    ))
    model_path, warnings = _resolve_model(language=None, model_hint=None)
    assert model_path == _SMALL_MODEL_PATH, (
        f"Expected small model path, got {model_path!r}"
    )
    assert warnings == [], f"Expected no warnings, got {warnings}"


def test_model_hint_overrides_language_routing(mocker):
    """model_hint must be returned as-is, bypassing all directory checks."""
    # Even if isdir returns True for known paths, model_hint wins
    mocker.patch("os.path.isdir", return_value=True)
    hint = "/custom/my-fine-tuned-whisper"
    model_path, warnings = _resolve_model(language="te", model_hint=hint)
    assert model_path == hint, (
        f"Expected model_hint {hint!r} to be returned, got {model_path!r}"
    )
    assert warnings == [], f"Expected no warnings when hint used, got {warnings}"


def test_language_routing_falls_back_to_hf_hub_when_no_local(mocker):
    """No local models present → returns HF Hub fallback with a warning."""
    mocker.patch("os.path.isdir", return_value=False)
    model_path, warnings = _resolve_model(language=None, model_hint=None)
    assert model_path == _HF_FALLBACK, (
        f"Expected HF Hub fallback {_HF_FALLBACK!r}, got {model_path!r}"
    )
    assert len(warnings) >= 1, f"Expected a fallback warning, got {warnings}"


# ── Sentence grouping (pure Python, no I/O) ───────────────────────────────────

def test_sentence_grouping_on_period():
    """Words ending with '.' trigger a sentence boundary."""
    words = [
        _w("Hello", 0.0, 0.5),
        _w("world.", 0.6, 1.0),
        _w("Next", 1.1, 1.4),
        _w("sentence.", 1.5, 2.0),
    ]
    sentences = _group_words_into_sentences(words)
    assert len(sentences) == 2, (
        f"Expected 2 sentences from period split, got {len(sentences)}: {sentences}"
    )
    assert sentences[0].text == "Hello world.", f"First sentence wrong: {sentences[0].text!r}"
    assert sentences[1].text == "Next sentence.", f"Second sentence wrong: {sentences[1].text!r}"


def test_sentence_grouping_on_long_pause():
    """A pause > 0.7 s between words triggers a sentence boundary."""
    words = [
        _w("First", 0.0, 0.3),
        _w("clause", 0.4, 0.7),
        # gap: 0.7+1.0=1.7 → next.start - this.end = 1.0 > 0.7
        _w("second", 1.7, 2.0),
        _w("clause", 2.1, 2.4),
    ]
    sentences = _group_words_into_sentences(words)
    assert len(sentences) == 2, (
        f"Expected 2 sentences after long-pause split, got {len(sentences)}: "
        f"{[s.text for s in sentences]}"
    )


def test_sentence_grouping_on_devanagari_purna_viram():
    """The Devanagari danda '।' is treated as a sentence terminal."""
    words = [
        _w("यह", 0.0, 0.3),
        _w("वाक्य।", 0.4, 0.9),    # ends with '।'
        _w("अगला", 1.0, 1.3),
        _w("वाक्य।", 1.4, 1.8),
    ]
    sentences = _group_words_into_sentences(words)
    assert len(sentences) == 2, (
        f"Expected 2 sentences for Devanagari danda, got {len(sentences)}: "
        f"{[s.text for s in sentences]}"
    )


def test_sentence_grouping_empty_words():
    """Empty word list returns empty sentence list — no crash."""
    result = _group_words_into_sentences([])
    assert result == [], f"Expected [] for empty words, got {result}"


# ── Video input → audio extraction (subprocess mock) ──────────────────────────

def test_video_input_extracts_audio(mocker, tmp_path):
    """For a video file input, transcribe() must invoke _extract_audio, which
    in turn calls ffmpeg with -vn and -ar flags.

    Strategy
    --------
    We patch pipeline_core.asr._extract_audio directly so we can:
      1. Confirm it was called for a .mp4 input (video path triggers extraction).
      2. Inspect the ffmpeg command it would have built without any subprocess I/O.

    This avoids the cascading-mock problem where patching subprocess.run prevents
    Windows platform.machine() and shutil.which from working during module reload.
    """
    import inspect
    import pipeline_core.asr as asr_mod

    # Save a reference to the REAL _extract_audio before it gets patched.
    # inspect.getsource must be called on this real function, not the mock.
    real_extract_audio = asr_mod._extract_audio

    # Create a fake .mp4 on disk — transcribe() checks os.path.isfile()
    fake_video = str(tmp_path / "fake_video.mp4")
    with open(fake_video, "wb") as fh:
        fh.write(b"\x00" * 64)

    # Capture the call to _extract_audio; return a dummy wav path
    fake_wav = str(tmp_path / "audio_16k.wav")
    with open(fake_wav, "wb") as fh:
        fh.write(b"\x00" * 16)

    extract_mock = mocker.patch(
        "pipeline_core.asr._extract_audio",
        return_value=fake_wav,
    )

    # Stub out the HF pipeline so no real model loading happens.
    # hf_pipeline is imported as `from transformers import pipeline as hf_pipeline`
    # inside the function, so we patch it via sys.modules.
    fake_output = {
        "text": "test",
        "chunks": [{"text": "test", "timestamp": (0.0, 0.5)}],
    }
    mock_pipe_instance = MagicMock(return_value=fake_output)
    mock_hf_pipeline_factory = MagicMock(return_value=mock_pipe_instance)

    mock_auto_processor = MagicMock()
    mock_auto_processor.from_pretrained.return_value = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = MagicMock()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor = mock_auto_processor
    mock_transformers.AutoModelForSpeechSeq2Seq = mock_auto_model
    mock_transformers.pipeline = mock_hf_pipeline_factory

    mocker.patch.dict(
        "sys.modules",
        {"torch": mock_torch, "transformers": mock_transformers},
    )
    mocker.patch("os.path.isdir", return_value=False)

    # Run transcribe on the fake .mp4
    result = asr_mod.transcribe(fake_video, language="en")

    # _extract_audio must have been called with the video path
    assert extract_mock.called, (
        "Expected _extract_audio to be called for a .mp4 video input"
    )
    called_video_path = extract_mock.call_args[0][0]
    assert called_video_path == fake_video, (
        f"Expected _extract_audio called with {fake_video!r}, "
        f"got {called_video_path!r}"
    )

    # Verify the ffmpeg command constants in _extract_audio source.
    # Use the real function saved before patching — asr_mod._extract_audio
    # is now the mock, not the original implementation.
    src = inspect.getsource(real_extract_audio)
    assert '"-vn"' in src or "'-vn'" in src, (
        f"_extract_audio source must contain '-vn' (no-video) ffmpeg flag.\n"
        f"Source:\n{src}"
    )
    assert '"-ar"' in src or "'-ar'" in src, (
        f"_extract_audio source must contain '-ar' (audio-rate) ffmpeg flag.\n"
        f"Source:\n{src}"
    )


# ── Slow test — real transcription ────────────────────────────────────────────

@pytest.mark.slow
def test_real_transcription_on_short_audio_clip(valid_short_mp4):
    """Invoke transcribe() for real on a short video (requires Whisper on CPU).

    This test is marked slow and skipped in normal CI runs.
    The synthetic test video has a 440 Hz sine tone — Whisper will produce
    silence or noise transcription; that's fine.  We only assert the shape.
    """
    from pipeline_core.asr import transcribe
    result = transcribe(valid_short_mp4)
    assert isinstance(result, TranscriptionResult), (
        f"Expected TranscriptionResult, got {type(result)}"
    )
    assert isinstance(result.sentences, list), "sentences must be a list"
    assert isinstance(result.language, str), "language must be a string"
    assert isinstance(result.model_used, str), "model_used must be a string"
    assert isinstance(result.full_text, str), "full_text must be a string"
    assert isinstance(result.warnings, list), "warnings must be a list"
