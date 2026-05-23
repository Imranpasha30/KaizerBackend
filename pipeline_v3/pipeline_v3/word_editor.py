"""V3 word editor — Deepgram word-level STT + Claude analysis.

Architecture: V3 = V1 + Claude (instead of Gemini) for the analysis step.
This module produces V1's compound analysis JSON (clips, full_video_cuts,
shorts_cuts, image_plan, skipped_segments, etc.) so V1's pipeline.py can
consume it via the KAIZER_REUSE_ANALYSIS_FROM env var and run its full
render path (sidebar carousel, lower-third, image card, ticker, channel
bug).

This module does NOT render anything. The render path is V1's job.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pipeline_v3.word_editor")

PROMPT_PATH = Path(__file__).parent / "prompts" / "claude_v1_analysis.md"

LANGUAGE_NAME_MAP = {
    "te": ("Telugu", "Telugu"),
    "hi": ("Hindi", "Devanagari"),
    "en": ("English", "Latin"),
    "ta": ("Tamil", "Tamil"),
    "kn": ("Kannada", "Kannada"),
    "ml": ("Malayalam", "Malayalam"),
    "bn": ("Bengali", "Bengali"),
    "mr": ("Marathi", "Devanagari"),
    "gu": ("Gujarati", "Gujarati"),
}


@dataclass
class V3AnalysisResult:
    analysis: dict             # V1-compatible compound analysis dict
    n_words_in: int
    audio_duration_sec: float
    llm_cost_usd: float
    llm_wall_sec: float


def _deepgram_words_from_audio(audio_path: str, language: str = "multi") -> tuple[list[dict], float]:
    """Run Deepgram nova-3 on audio. Returns (word_array, duration_sec).

    SDK 7.x sync client. Multi-lingual handles Telugu/Hindi/English code-switch.
    """
    from deepgram import DeepgramClient
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set in env")
    dg = DeepgramClient(api_key=api_key)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    kwargs = {
        "request": audio_bytes,
        "model": "nova-3",
        "punctuate": True,
        "diarize": False,
        "smart_format": True,
    }
    if language:
        kwargs["language"] = language
    response = dg.listen.v1.media.transcribe_file(**kwargs)

    metadata = getattr(response, "metadata", None)
    audio_duration_sec = float(getattr(metadata, "duration", 0.0) or 0.0)
    results = getattr(response, "results", None)
    channels = getattr(results, "channels", None) or []
    if not channels:
        return [], audio_duration_sec
    alts = getattr(channels[0], "alternatives", None) or []
    if not alts:
        return [], audio_duration_sec
    raw_words = getattr(alts[0], "words", None) or []
    words: list[dict] = []
    for w in raw_words:
        text = (
            getattr(w, "punctuated_word", None)
            or getattr(w, "word", "")
            or ""
        ).strip()
        if not text:
            continue
        words.append({
            "i": len(words),
            "w": text,
            "s": float(getattr(w, "start", 0.0) or 0.0),
            "e": float(getattr(w, "end", 0.0) or 0.0),
        })
    return words, audio_duration_sec


def _call_claude_compound_analysis(
    words: list[dict],
    language: str,
    audio_duration_sec: float,
    preset: dict,
) -> tuple[dict, float, float]:
    """Send the word array to Claude and ask for V1's compound schema.

    Returns (parsed_json, cost_usd, wall_sec). Strips the ``<<END>>`` sentinel
    and any code fences from Claude's response before parsing.
    """
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    client = Anthropic(api_key=api_key)

    language_name, script_name = LANGUAGE_NAME_MAP.get(language, ("English", "Latin"))
    min_dur = int(preset.get("min_dur", 30))
    max_clips = int(preset.get("max_clips", 6))

    raw_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    # The prompt uses .format placeholders ({language_name}, {script_name},
    # {min_dur}, {max_clips}, {language_code}). Double-braces in JSON
    # examples survive the format() pass.
    system_prompt = raw_prompt.format(
        language_name=language_name,
        script_name=script_name,
        min_dur=min_dur,
        max_clips=max_clips,
        language_code=language,
    )

    payload = {
        "language": language,
        "script": script_name,
        "n_words": len(words),
        "audio_duration_sec": round(audio_duration_sec, 3),
        "words": words,
    }
    user_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    t0 = time.time()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        temperature=0.0,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_payload}],
    )
    wall = time.time() - t0

    # Concat all text blocks
    text = ""
    for block in resp.content:
        if hasattr(block, "text") and block.text:
            text += block.text
    text = text.strip()

    # Strip <<END>> sentinel
    if "<<END>>" in text:
        text = text.split("<<END>>")[0].strip()
    # Strip markdown fences if Claude added them despite instructions
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
    # Find outermost { ... } if there's still prose around it
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)

    parsed = json.loads(text)

    # Cost
    usage = resp.usage
    in_tok = getattr(usage, "input_tokens", 0)
    out_tok = getattr(usage, "output_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_write = getattr(usage, "cache_creation_input_tokens", 0)
    cost = (
        (in_tok - cache_read - cache_write) * 3.0 / 1_000_000
        + cache_read * 0.30 / 1_000_000
        + cache_write * 3.75 / 1_000_000
        + out_tok * 15.0 / 1_000_000
    )
    return parsed, cost, wall


def _call_gemini_compound_analysis(
    words: list[dict],
    language: str,
    audio_duration_sec: float,
    preset: dict,
) -> tuple[dict, float, float]:
    """Send the word array to Gemini 2.5 Pro and ask for V1's compound schema.

    Uses google-genai SDK (sync API). The prompt is the SAME claude_v1_analysis.md
    file (same rules + same output format). Schema validation happens via
    response_mime_type="application/json".

    Returns (parsed_json, cost_usd, wall_sec).
    """
    from google import genai
    from google.genai import types as genai_types

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in env")

    language_name, script_name = LANGUAGE_NAME_MAP.get(language, ("English", "Latin"))
    min_dur = int(preset.get("min_dur", 30))
    max_clips = int(preset.get("max_clips", 6))

    raw_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    system_prompt = raw_prompt.format(
        language_name=language_name,
        script_name=script_name,
        min_dur=min_dur,
        max_clips=max_clips,
        language_code=language,
    )

    payload = {
        "language": language,
        "script": script_name,
        "n_words": len(words),
        "audio_duration_sec": round(audio_duration_sec, 3),
        "words": words,
    }
    user_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,           # Gemini 2.5 Pro default; tighter than Claude's 0.0
                                    # because Gemini's T=0 still has determinism
                                    # issues and 0.2 was V1/V2's production setting.
        max_output_tokens=16000,
        system_instruction=system_prompt,
    )

    t0 = time.time()
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[user_payload],
        config=config,
    )
    wall = time.time() - t0

    raw_text = (getattr(resp, "text", "") or "").strip()
    if not raw_text:
        raise ValueError(
            "Gemini response.text is empty -- check finish_reason for MAX_TOKENS "
            "or safety blocks. Switch to Claude provider as a fallback."
        )
    # Strip <<END>> sentinel + markdown fences (defensive, Gemini usually
    # honours response_mime_type="application/json")
    if "<<END>>" in raw_text:
        raw_text = raw_text.split("<<END>>")[0].strip()
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        if lines[-1].startswith("```"):
            raw_text = "\n".join(lines[1:-1])
        else:
            raw_text = "\n".join(lines[1:])
    m = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if m:
        raw_text = m.group(0)
    parsed = json.loads(raw_text)

    # Cost ledger (Gemini 2.5 Pro pricing: $1.25/M input, $10/M output)
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        cost = in_tok * 1.25 / 1_000_000 + out_tok * 10.0 / 1_000_000
    else:
        cost = 0.0
    return parsed, cost, wall


def produce_v1_analysis(
    audio_mp3_path: str,
    language: str,
    preset: Optional[dict] = None,
    provider: str = "claude",
) -> V3AnalysisResult:
    """End-to-end: audio -> Deepgram -> {Claude|Gemini} -> V1-compatible analysis JSON.

    Args:
        audio_mp3_path: Path to audio file (mp3 or wav).
        language: ISO 639-1 (te / hi / en / etc).
        preset: Optional. Defaults to a sensible bulletin preset.
        provider: "claude" (default, Sonnet 4.6 T=0) or "gemini" (2.5 Pro T=0.2).
                  Same prompt + same output schema -- only the model differs.

    Returns:
        V3AnalysisResult containing the analysis dict + diagnostics.

    Raises:
        RuntimeError on missing API keys.
        ValueError on invalid LLM output.
    """
    if preset is None:
        preset = {"width": 1080, "height": 1920, "min_dur": 30, "max_clips": 6}

    provider_norm = (provider or "claude").strip().lower()
    if provider_norm not in ("claude", "gemini"):
        logger.warning("v3 word_editor: unknown provider %r, defaulting to claude", provider)
        provider_norm = "claude"

    logger.info("v3 word_editor: Deepgram nova-3 STT on %s", audio_mp3_path)
    words, audio_dur = _deepgram_words_from_audio(audio_mp3_path, language="multi")
    logger.info(
        "v3 word_editor: Deepgram returned n_words=%d duration=%.2fs",
        len(words), audio_dur,
    )
    if len(words) < 5:
        raise ValueError(f"v3 word_editor: too few words ({len(words)}) -- STT failed?")

    if provider_norm == "gemini":
        logger.info("v3 word_editor: Gemini 2.5 Pro producing V1 compound analysis")
        analysis, cost, wall = _call_gemini_compound_analysis(
            words=words,
            language=language,
            audio_duration_sec=audio_dur,
            preset=preset,
        )
    else:
        logger.info("v3 word_editor: Claude sonnet-4-6 producing V1 compound analysis")
        analysis, cost, wall = _call_claude_compound_analysis(
            words=words,
            language=language,
            audio_duration_sec=audio_dur,
            preset=preset,
        )
    # Sanity: V1 expects these keys at minimum
    missing = [k for k in ("clips", "full_video_cuts", "shorts_cuts", "image_plan", "skipped_segments", "retake_audit") if k not in analysis]
    if missing:
        raise ValueError(f"v3 word_editor: Claude analysis missing keys: {missing}")

    n_clips = len(analysis.get("clips", []))
    n_full = len(analysis.get("full_video_cuts", []))
    n_shorts = len(analysis.get("shorts_cuts", []))
    n_skip = len(analysis.get("skipped_segments", []))
    n_img = len(analysis.get("image_plan", []))
    logger.info(
        "v3 word_editor: analysis ready -- clips=%d full_video_cuts=%d shorts_cuts=%d image_plan=%d skipped_segments=%d cost=$%.4f wall=%.1fs",
        n_clips, n_full, n_shorts, n_img, n_skip, cost, wall,
    )

    return V3AnalysisResult(
        analysis=analysis,
        n_words_in=len(words),
        audio_duration_sec=audio_dur,
        llm_cost_usd=cost,
        llm_wall_sec=wall,
    )
