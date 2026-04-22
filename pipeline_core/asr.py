"""
kaizer.pipeline.asr
===================
HuggingFace Whisper wrapper for automatic speech recognition with word-level
timestamps.  Supports Telugu, Hindi, English and any other Whisper-supported
language out of the box.

Usage
-----
    from pipeline_core.asr import transcribe, TranscriptionResult

    result = transcribe("/path/to/video.mp4", language="te")
    for sentence in result.sentences:
        print(f"[{sentence.start:.2f}s – {sentence.end:.2f}s]  {sentence.text}")

TranscriptionResult fields
--------------------------
  sentences  : list[Sentence]  — Grouped sentence objects with start/end times.
  language   : str             — ISO 639-1 code detected or provided (e.g. "te").
  model_used : str             — Absolute path or HF Hub id of the model used.
  full_text  : str             — Raw concatenation of all word texts.
  warnings   : list[str]       — Non-fatal issues collected during transcription.

Model selection
---------------
  Priority (highest to lowest):
    1. ``model_hint`` argument (overrides everything).
    2. Language == "te" (or auto-detected Telugu) + local
       ``models/whisper-telugu-medium/`` exists → use that model.
    3. Local ``models/whisper-small/`` exists → use it.
    4. HF Hub ``openai/whisper-small`` (may download; logged as a warning).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.asr")

# ── Constants ─────────────────────────────────────────────────────────────────

# Pause threshold (seconds) between consecutive words that triggers a new
# sentence even without terminal punctuation.
_PAUSE_THRESHOLD_S: float = 0.7

# Sentence-terminal punctuation characters (ASCII + Devanagari danda).
_TERMINAL_PUNCT: str = ".?!।"

# Video file extensions that require audio extraction first.
_VIDEO_EXTS: frozenset[str] = frozenset({
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".flv", ".wmv", ".m4v",
})

# Base directory used to resolve local model paths.
_BASE_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_MODELS_DIR: str = os.path.join(_BASE_DIR, "models")

_TELUGU_MODEL_PATH: str = os.path.join(_MODELS_DIR, "whisper-telugu-medium")
_SMALL_MODEL_PATH: str = os.path.join(_MODELS_DIR, "whisper-small")
_HF_FALLBACK: str = "openai/whisper-small"


# ── Public dataclasses ─────────────────────────────────────────────────────────

@dataclass
class Word:
    """A single recognised word with its time boundaries."""

    text: str
    start: float  # seconds from media start
    end: float    # seconds from media start


@dataclass
class Sentence:
    """A sentence assembled from one or more consecutive Words."""

    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Output of :func:`transcribe`."""

    sentences: list[Sentence]
    language: str          # ISO 639-1
    model_used: str        # absolute path or HF id
    full_text: str
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_model(language: Optional[str], model_hint: Optional[str]) -> tuple[str, list[str]]:
    """Return (model_path_or_id, warnings).

    Selection order:
      1. model_hint (if provided).
      2. Telugu language + local whisper-telugu-medium present.
      3. Local whisper-small present.
      4. HF Hub fallback (emits a warning).
    """
    warnings: list[str] = []

    if model_hint:
        logger.debug("ASR: using model_hint=%s", model_hint)
        return model_hint, warnings

    if language == "te":
        if os.path.isdir(_TELUGU_MODEL_PATH):
            logger.info("ASR: Telugu language detected — using local whisper-telugu-medium")
            return _TELUGU_MODEL_PATH, warnings
        else:
            warnings.append(
                f"Telugu model directory not found at {_TELUGU_MODEL_PATH!r}; "
                "falling back to whisper-small."
            )

    if os.path.isdir(_SMALL_MODEL_PATH):
        logger.info("ASR: using local whisper-small at %s", _SMALL_MODEL_PATH)
        return _SMALL_MODEL_PATH, warnings

    warnings.append(
        f"Local whisper-small not found at {_SMALL_MODEL_PATH!r}. "
        f"Falling back to HF Hub '{_HF_FALLBACK}' — this may trigger a download."
    )
    logger.warning("ASR: falling back to HF Hub model '%s'", _HF_FALLBACK)
    return _HF_FALLBACK, warnings


def _extract_audio(video_path: str, tmp_dir: str) -> str:
    """Extract 16 kHz mono PCM WAV from *video_path* into *tmp_dir*.

    Returns the path to the extracted WAV file.
    Raises RuntimeError if ffmpeg fails.
    """
    from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore

    out_wav = os.path.join(tmp_dir, "audio_16k.wav")
    cmd = [
        FFMPEG_BIN,
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        out_wav,
    ]
    logger.debug("ASR: extracting audio: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}"
        )
    return out_wav


def _load_audio_array(path: str) -> tuple[any, int]:  # type: ignore[type-arg]
    """Load audio using librosa, returning (array_float32, sample_rate)."""
    import librosa  # type: ignore
    array, sr = librosa.load(path, sr=16000, mono=True)
    return array, sr


def _words_from_pipeline_output(output: dict) -> list[Word]:
    """Parse HF pipeline output dict into a flat list of Words.

    The pipeline returns something like:
        {"text": "...", "chunks": [{"text": "word", "timestamp": (start, end)}, ...]}
    When return_timestamps="word" is used.
    """
    words: list[Word] = []
    chunks = output.get("chunks") or []
    for chunk in chunks:
        ts = chunk.get("timestamp") or (None, None)
        start = float(ts[0]) if ts[0] is not None else 0.0
        end = float(ts[1]) if ts[1] is not None else start
        text = (chunk.get("text") or "").strip()
        if text:
            words.append(Word(text=text, start=start, end=end))
    return words


def _group_words_into_sentences(words: list[Word]) -> list[Sentence]:
    """Group words into sentences using punctuation and pause heuristics.

    Splits on:
      - Terminal punctuation: . ? ! । (Devanagari full stop)
      - Inter-word pause > _PAUSE_THRESHOLD_S seconds
    """
    if not words:
        return []

    sentences: list[Sentence] = []
    current_words: list[Word] = []

    for i, word in enumerate(words):
        current_words.append(word)

        is_terminal = bool(re.search(r"[.?!।]$", word.text.strip()))

        # Check pause to next word
        long_pause = False
        if i < len(words) - 1:
            gap = words[i + 1].start - word.end
            if gap > _PAUSE_THRESHOLD_S:
                long_pause = True

        if is_terminal or long_pause or i == len(words) - 1:
            text = " ".join(w.text for w in current_words).strip()
            if text:
                sentences.append(
                    Sentence(
                        text=text,
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        words=list(current_words),
                    )
                )
            current_words = []

    return sentences


def _detect_language_from_output(output: dict, provided: Optional[str]) -> str:
    """Extract the language code from HF pipeline output, falling back to provided."""
    # Some pipeline outputs include a "language" key; others do not.
    lang = output.get("language")
    if lang and isinstance(lang, str) and len(lang) <= 10:
        return lang.lower()
    return provided or "unknown"


# ── Public API ─────────────────────────────────────────────────────────────────

def transcribe(
    audio_or_video_path: str,
    *,
    language: Optional[str] = None,
    model_hint: Optional[str] = None,
    chunk_length_s: float = 30.0,
) -> TranscriptionResult:
    """Transcribe an audio or video file with word-level timestamps.

    Parameters
    ----------
    audio_or_video_path : str
        Path to a video (.mp4, .mkv, …) or audio (.wav, .mp3, …) file.
    language : str | None
        ISO 639-1 language code (e.g. ``"te"``, ``"hi"``, ``"en"``).
        ``None`` triggers Whisper's built-in language auto-detection.
    model_hint : str | None
        Absolute path to a local HF model directory or an HF Hub model id.
        Overrides automatic model selection.
    chunk_length_s : float
        Length of each audio chunk passed to the pipeline (default: 30 s).

    Returns
    -------
    TranscriptionResult
        Never raises; all failures are captured in ``warnings``.
    """
    warnings: list[str] = []
    path = audio_or_video_path

    # ── Existence check ───────────────────────────────────────────────────────
    if not os.path.isfile(path):
        warnings.append(f"File not found: {path!r}. Returning empty transcription.")
        return TranscriptionResult(
            sentences=[], language=language or "unknown",
            model_used="", full_text="", warnings=warnings,
        )

    ext = Path(path).suffix.lower()
    tmp_dir: Optional[str] = None
    audio_path = path

    try:
        # ── Video → WAV extraction ────────────────────────────────────────────
        if ext in _VIDEO_EXTS:
            try:
                tmp_dir = tempfile.mkdtemp(prefix="kaizer_asr_")
                audio_path = _extract_audio(path, tmp_dir)
                logger.info("ASR: extracted audio from video to %s", audio_path)
            except Exception as exc:
                warnings.append(
                    f"FFmpeg audio extraction failed ({exc}); "
                    "attempting to feed video directly to Whisper."
                )
                audio_path = path

        # ── Model resolution ──────────────────────────────────────────────────
        model_path, model_warnings = _resolve_model(language, model_hint)
        warnings.extend(model_warnings)

        # ── Load model via transformers ────────────────────────────────────────
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoProcessor,
            AutoModelForSpeechSeq2Seq,
            pipeline as hf_pipeline,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("ASR: loading model %s on %s (%s)", model_path, device, torch_dtype)

        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

        # Build generate_kwargs for language forcing when provided
        generate_kwargs: dict = {}
        if language and language != "unknown":
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"

        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=chunk_length_s,
            stride_length_s=5.0,
            return_timestamps="word",
            # transformers 4.x _sanitize_parameters calls .pop() on generate_kwargs,
            # which raises TypeError: 'NoneType' object is not iterable when None
            # is passed. Always provide at least an empty dict.
            generate_kwargs=generate_kwargs or {},
        )

        logger.info("ASR: transcribing %s", audio_path)
        output = pipe(audio_path)

        # ── Parse output ──────────────────────────────────────────────────────
        words = _words_from_pipeline_output(output)
        logger.info("ASR: got %d word chunks", len(words))

        if not words:
            # Fallback: treat the whole text as a single sentence without timestamps
            raw_text = (output.get("text") or "").strip()
            warnings.append(
                "No word-level timestamp chunks returned; using full-text fallback."
            )
            if raw_text:
                words = [Word(text=raw_text, start=0.0, end=0.0)]

        sentences = _group_words_into_sentences(words)
        detected_lang = _detect_language_from_output(output, language)
        full_text = " ".join(w.text for w in words)

        logger.info(
            "ASR: transcription complete — %d sentences, language=%s, model=%s",
            len(sentences), detected_lang, model_path,
        )
        return TranscriptionResult(
            sentences=sentences,
            language=detected_lang,
            model_used=model_path,
            full_text=full_text,
            warnings=warnings,
        )

    except Exception as exc:
        logger.exception("ASR: transcription failed: %s", exc)
        warnings.append(f"Transcription failed: {exc}")
        return TranscriptionResult(
            sentences=[], language=language or "unknown",
            model_used=model_hint or "", full_text="", warnings=warnings,
        )
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
