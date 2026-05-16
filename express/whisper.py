"""Whisper transcription via Groq (preferred) or OpenAI.

Direct port of the teammate's ``transcribeWithSegments`` — same wire
format, same prompt biasing, returns the same shape:

    { "text": str, "segments": [{start, end, text}],
      "language": str, "duration": float }

Why Groq is preferred (per teammate's README):
  - Whisper Large v3 is dramatically more accurate for Telugu/Hindi
    than the OpenAI-hosted whisper-1, which still uses Large v2.
  - Free tier covers comfortably for a single user's daily usage.
  - OpenAI-compatible endpoint, so the same code works for either.

Environment / config:
  - GROQ_API_KEY → uses Groq with whisper-large-v3
  - OPENAI_API_KEY → fallback, uses whisper-1
  - The user can override via the Express Mode UI which provider to
    use; the request includes the key explicitly so we don't pull
    from env on the request hot path.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx


GROQ_BASE = "https://api.groq.com/openai/v1"
OPENAI_BASE = "https://api.openai.com/v1"
# Groq's flagship as of late 2024. The non-turbo "whisper-large-v3"
# endpoint has been returning 500s on real audio (confirmed empirically
# 2026-05-14 against multiple file shapes); turbo is the recommended
# variant per Groq's own docs and the one their console picks by default.
GROQ_MODEL = "whisper-large-v3-turbo"
OPENAI_MODEL = "whisper-1"


class WhisperError(RuntimeError):
    """Whisper API call failed or response was unparseable."""


def extract_audio_mp3(video_path: str, out_path: str, *, timeout_s: int = 300) -> None:
    """Pull mono 16kHz mp3 audio out of the video for Whisper.

    Matches the teammate's ``extractAudioToFile`` ffmpeg invocation
    *exactly* — explicit ``libmp3lame`` codec is load-bearing on
    Windows ffmpeg builds where ``-f mp3`` alone can produce mp3s
    with non-standard frame headers that Groq's decoder rejects (the
    teammate hit + fixed this exact bug; same fix applies here).
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                       # drop video
        "-acodec", "libmp3lame",     # explicit codec — see docstring
        "-ab", "64k",                # 64 kbps audio bitrate
        "-ar", "16000",              # 16 kHz — Whisper sweet spot
        "-ac", "1",                  # mono — better for ASR
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        raise WhisperError(f"ffmpeg audio extract failed: {stderr}")


def transcribe(
    audio_path: str,
    *,
    api_key: str,
    provider: str = "groq",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    names_hint: Optional[str] = None,
    timeout_s: int = 300,
) -> dict:
    """POST audio to a Whisper endpoint, return parsed transcription.

    Parameters
    ----------
    audio_path : path to mp3/wav/mp4 audio (ffmpeg-readable).
    api_key    : Bearer token for the provider.
    provider   : "groq" | "openai" | "custom"
    base_url   : override for "custom" providers (e.g. self-hosted).
                 Ignored when provider is "groq" or "openai".
    model      : override the default model.
    language   : ISO code ("te", "hi", "en", …). Blank = auto-detect.
    names_hint : prompt-biasing for proper nouns. Capped at 800 chars
                 (the prompt param is limited to ~224 tokens by Whisper).
    """
    if provider == "groq":
        base_url = GROQ_BASE
        model    = model or GROQ_MODEL
    elif provider == "openai":
        base_url = OPENAI_BASE
        model    = model or OPENAI_MODEL
    elif provider == "custom":
        if not base_url:
            raise WhisperError("custom provider requires base_url")
        model = model or "whisper-large-v3"
    else:
        raise WhisperError(f"unknown provider {provider!r}")

    # Per-request key wins; fall back to env so server-side testing
    # works without a UI key paste each time.
    if not api_key:
        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise WhisperError(f"missing API key for provider {provider!r}")

    audio_p = Path(audio_path)
    if not audio_p.is_file():
        raise WhisperError(f"audio file not found: {audio_path}")

    # Surface size before the request so 500/413 root cause is obvious.
    size_bytes = audio_p.stat().st_size
    size_mb    = size_bytes / 1_000_000
    print(f"[whisper] {provider} {model} audio={audio_p.name} size={size_mb:.2f}MB")
    if provider == "groq" and size_mb > 24.5:
        # Groq's documented limit is 25 MB. Fail early with a clear
        # message instead of letting Groq 413/500 the request.
        raise WhisperError(
            f"audio is {size_mb:.1f}MB but Groq caps at 25MB — switch transcription provider "
            f"to OpenAI, or trim the source video first (source must be ≤50 min at our 64kbps mono)."
        )

    mime = "audio/mpeg" if audio_p.suffix.lower() == ".mp3" else "application/octet-stream"

    data = {
        "model":           model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }
    if language:
        data["language"] = language
    if names_hint:
        data["prompt"] = names_hint[:800]

    url = f"{base_url.rstrip('/')}/audio/transcriptions"

    # Retry on 5xx / 429 with exponential backoff. Groq returns 500
    # under transient load (rare but happens — esp on free tier) and
    # the request is fully idempotent on their side, so retry is safe.
    import time as _time
    last_err: str = ""
    backoffs = [0, 2, 5]   # 3 attempts total
    for attempt, wait_s in enumerate(backoffs):
        if wait_s:
            _time.sleep(wait_s)
        try:
            with open(audio_p, "rb") as fh:
                files = {"file": (audio_p.name, fh, mime)}
                with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=20)) as cli:
                    r = cli.post(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        data=data,
                        files=files,
                    )
        except httpx.HTTPError as exc:
            last_err = f"network error: {exc}"
            print(f"[whisper] attempt {attempt + 1}/{len(backoffs)} network: {exc}")
            continue

        if r.status_code < 400:
            break

        body = r.text[:600]
        last_err = f"{provider} HTTP {r.status_code}: {body}"
        # Only retry on transient 429/5xx. 4xx other than 429 = caller
        # error (bad key, bad audio, model rejected), no point retrying.
        if r.status_code not in (429, 500, 502, 503, 504):
            raise WhisperError(last_err)
        print(f"[whisper] attempt {attempt + 1}/{len(backoffs)} got {r.status_code}; will retry")
    else:
        raise WhisperError(f"{last_err} (after {len(backoffs)} attempts)")

    if r.status_code >= 400:
        raise WhisperError(last_err)

    try:
        payload = r.json()
    except ValueError as exc:
        raise WhisperError(f"{provider} returned non-JSON: {r.text[:300]}") from exc

    # Normalise the response to the same shape the teammate's code
    # expects downstream: {text, segments, language, duration}.
    return {
        "text":     payload.get("text", "") or "",
        "segments": payload.get("segments") or [],
        "language": payload.get("language") or language or "",
        "duration": float(payload.get("duration") or 0.0),
    }
