"""Two-tier transcript fetcher for HeyGen avatar generation.

Strategy
--------
Given a YouTube URL, return the full transcript as plain text.

  1. **YouTube caption track** (fast, free, ~5 s). yt-dlp pulls the
     channel-uploaded subtitles or YouTube's auto-generated captions.
     We prefer manually-uploaded subs when present (they're cleanly
     punctuated); fall back to auto-captions otherwise.
  2. **Audio + Whisper** (~10-30 s, costs Groq quota). When no captions
     exist (channel disabled them or the video is too new for auto-
     captions), download the lowest-bitrate audio, transcode to mp3
     via ``express.whisper.extract_audio_mp3``, then send to
     ``express.whisper.transcribe`` with Groq Whisper Large v3.

Returns the same shape regardless of source:

    {
      "transcript":  str,             # plain text, no timestamps
      "source":      "captions"|"whisper",
      "duration_s":  float,           # 0.0 when not known
      "language":    str,             # ISO code, blank when unknown
    }

Module is self-contained; reuses ``express.whisper`` for the audio
path (no duplication of the Groq retry / size-cap logic).
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional


class TranscriptError(RuntimeError):
    """Both caption + whisper paths failed."""


# ── Caption-track strategy ─────────────────────────────────────────

def _strip_vtt_cues(vtt_text: str) -> str:
    """Convert a .vtt subtitle file into plain prose.

    Removes WEBVTT header, timestamp lines (``00:00:01.000 --> ...``),
    cue identifiers, and inline tags like ``<c.colorE5E5E5>``. Joins
    surviving lines with spaces, de-duplicates consecutive identical
    lines (auto-captions repeat each phrase as it slides through the
    viewport — a kerning artifact, not new content)."""
    out: list[str] = []
    seen_last = ""
    for raw in vtt_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        # Cue identifier (numeric or hash)
        if re.fullmatch(r"[\w\-]+", line) and "-->" not in line and len(line) <= 12:
            continue
        # Strip inline tags
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if not clean:
            continue
        if clean == seen_last:
            continue
        seen_last = clean
        out.append(clean)
    return " ".join(out).strip()


def _fetch_captions(video_url: str, languages: list[str]) -> Optional[dict]:
    """Try to pull a subtitle track for ``video_url``. Returns the
    transcript dict on success, None when no track is available."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("[heygen/transcript] yt-dlp not installed — skipping caption path")
        return None

    with tempfile.TemporaryDirectory(prefix="heygen-cap-") as td:
        out_tmpl = str(Path(td) / "%(id)s.%(ext)s")
        opts = {
            "skip_download":     True,
            "writesubtitles":    True,
            "writeautomaticsub": True,
            "subtitleslangs":    languages + ["en"],   # always try English as a last resort
            "subtitlesformat":   "vtt",
            "outtmpl":           out_tmpl,
            "quiet":             True,
            "no_warnings":       True,
            "extract_flat":      False,
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
        except Exception as exc:
            print(f"[heygen/transcript] caption fetch failed: {exc}")
            return None

        # yt-dlp writes <id>.<lang>.vtt — pick the largest .vtt for
        # whichever language landed (manually-uploaded subs are
        # usually longer + cleaner than auto-captions).
        vtts = sorted(Path(td).glob("*.vtt"), key=lambda p: p.stat().st_size, reverse=True)
        if not vtts:
            return None
        text = _strip_vtt_cues(vtts[0].read_text(encoding="utf-8", errors="replace"))
        if not text or len(text) < 40:
            print(f"[heygen/transcript] caption track found but empty/too short ({len(text)} chars)")
            return None

        # Detect which language we ended up with from the filename
        # (yt-dlp embeds it as ``video_id.te.vtt`` etc.).
        m = re.search(r"\.([a-zA-Z\-]{2,6})\.vtt$", vtts[0].name)
        lang = (m.group(1).split("-")[0] if m else "").lower()

        duration = float(info.get("duration") or 0.0)
        print(f"[heygen/transcript] captions found: {len(text)} chars, lang={lang or '?'}, duration={duration:.0f}s")
        return {
            "transcript": text,
            "source":     "captions",
            "duration_s": duration,
            "language":   lang,
        }


# ── Whisper fallback ───────────────────────────────────────────────

def _download_audio(video_url: str, out_path: str) -> Optional[float]:
    """Pull the audio stream of ``video_url`` and save as mp3 at
    ``out_path``. Returns the duration in seconds (yt-dlp probes it)
    or None on failure."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return None

    # We write the source media to a temp file inside a temp dir, then
    # post-process to mp3 separately so we control the codec — yt-dlp
    # can do FFmpegExtractAudio inline but it embeds metadata we don't
    # want for Whisper.
    with tempfile.TemporaryDirectory(prefix="heygen-aud-") as td:
        src_tmpl = str(Path(td) / "%(id)s.%(ext)s")
        opts = {
            "format":      "bestaudio/best",
            "outtmpl":     src_tmpl,
            "quiet":       True,
            "no_warnings": True,
            "noplaylist":  True,
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
        except Exception as exc:
            print(f"[heygen/transcript] audio download failed: {exc}")
            return None

        # Pick whichever file yt-dlp produced.
        files = list(Path(td).iterdir())
        if not files:
            return None
        src = max(files, key=lambda p: p.stat().st_size)

        # Use Express Mode's existing helper to transcode to 16 kHz
        # mono mp3 — already calibrated for Whisper input shape.
        try:
            from express.whisper import extract_audio_mp3
            extract_audio_mp3(str(src), out_path, timeout_s=180)
        except Exception as exc:
            print(f"[heygen/transcript] mp3 extract failed: {exc}")
            return None

        return float(info.get("duration") or 0.0)


def _fetch_via_whisper(video_url: str, language_hint: Optional[str]) -> Optional[dict]:
    """Download audio + run Groq Whisper. Returns the transcript dict
    or None on hard failure."""
    fd, mp3_path = tempfile.mkstemp(suffix=".mp3", prefix="heygen-whisper-")
    os.close(fd)
    try:
        duration = _download_audio(video_url, mp3_path)
        if duration is None or not os.path.isfile(mp3_path):
            return None

        # Lazy import so the heygen module loads even when express
        # isn't fully importable for some reason (defensive).
        from express import whisper as express_whisper

        # Server-side env keys win when present; per-request keys
        # are reserved for the Express Mode UI flow.
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        provider = "groq" if api_key else "openai"
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("[heygen/transcript] no GROQ_API_KEY or OPENAI_API_KEY — cannot Whisper")
            return None

        try:
            tr = express_whisper.transcribe(
                mp3_path,
                api_key=api_key,
                provider=provider,
                language=language_hint or None,
                names_hint=None,
            )
        except express_whisper.WhisperError as exc:
            print(f"[heygen/transcript] Whisper failed: {exc}")
            return None

        text = (tr.get("text") or "").strip()
        if not text:
            return None
        print(f"[heygen/transcript] whisper transcript: {len(text)} chars, lang={tr.get('language','?')}")
        return {
            "transcript": text,
            "source":     "whisper",
            "duration_s": float(tr.get("duration") or duration),
            "language":   tr.get("language") or language_hint or "",
        }
    finally:
        try:
            Path(mp3_path).unlink(missing_ok=True)
        except OSError:
            pass


# ── Public entry point ─────────────────────────────────────────────

def fetch_transcript(
    video_url: str,
    *,
    language: Optional[str] = None,
    timeout_total_s: int = 5 * 60,
) -> dict:
    """Return the full transcript of ``video_url``.

    Parameters
    ----------
    video_url : YouTube URL. Anything yt-dlp can parse is fine
                (youtube.com/watch?v=…, youtu.be/…, /shorts/…).
    language  : ISO code hint (e.g. "te", "hi", "en"). Used both for
                caption-track preference order and as a Whisper hint
                when we fall through.

    Raises
    ------
    TranscriptError when neither path produces usable text.
    """
    if not video_url:
        raise TranscriptError("video_url is empty")

    lang_pref = []
    if language:
        lang_pref.append(language)
    # Channel-uploaded subs are usually in the channel's primary
    # language; for Telugu news that's "te" with English as a
    # frequent fallback.
    for fallback in ("te", "hi", "en"):
        if fallback not in lang_pref:
            lang_pref.append(fallback)

    # Path 1: captions.
    cap = _fetch_captions(video_url, lang_pref)
    if cap and cap.get("transcript"):
        return cap

    # Path 2: Whisper.
    print("[heygen/transcript] no captions — falling back to whisper")
    wh = _fetch_via_whisper(video_url, language)
    if wh and wh.get("transcript"):
        return wh

    raise TranscriptError(
        f"could not get transcript for {video_url} via captions or Whisper"
    )
