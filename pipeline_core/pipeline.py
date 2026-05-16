"""
KAIZER NEWS — API-Based Production Pipeline
=============================================
Architecture:
    Video → Gemini 2.0 Flash  (cut timestamps + summary + image keywords)
          → FFmpeg             (cut video at Gemini timestamps)
          → GPT-4o-mini        (Telugu headline / title from summary)
          → Pexels API         (free images from Gemini keywords)
          → PIL + FFmpeg       (compose broadcast layout)
          → Web Editor         (layout adjust, font, color, image swap)

Run:
    python scripts/11_api_pipeline.py
    python scripts/11_api_pipeline.py "path/to/video.mp4"

Requires:
    pip install google-generativeai openai requests Pillow
"""

import os, sys, json, time, re, subprocess, math, random, shutil
from datetime import datetime
from typing import Optional

# When this module is run as a standalone script (e.g. via
# `python pipeline_core/pipeline.py ...` from the upload worker
# subprocess, or directly), the `pipeline_core` package's parent
# directory (KaizerBackend/) is NOT automatically on sys.path — so
# the `from pipeline_core.xxx import ...` lines below fail with
# ModuleNotFoundError. Bootstrap it here so the file runs cleanly
# both as a module (`from pipeline_core.pipeline import ...`) AND as
# a script.
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)   # KaizerBackend/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


def _find_binary(name):
    """Find ffmpeg/ffprobe: checks PATH first, then common nix/Railway paths."""
    import shutil as _sh
    p = _sh.which(name)
    if p:
        return p
    for prefix in ["/usr/bin", "/usr/local/bin", "/nix/var/nix/profiles/default/bin",
                   "/run/current-system/sw/bin"]:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return name  # fallback — will fail with a clear error


FFMPEG_BIN  = _find_binary("ffmpeg")
FFPROBE_BIN = _find_binary("ffprobe")

# ═══════════════════════════════════════════════════════════
# STANDARD ENCODE ARGS
#
# Uses hardware-accelerated H.264 encoder (NVENC / QSV / AMF) when
# available, falls back to libx264 CPU. Switchover is driven by
# pipeline_core.hw_accel.ACTIVE_ENCODER — probed once at import.
# Set KAIZER_FORCE_CPU_ENCODE=1 in .env to disable GPU fallback for
# debugging.
#
# Dropping libx264 to NVENC on a Windows 32 GB / RTX 5060 box reduces
# per-render CPU to ~5 % and frees 500 MB – 1 GB of RAM per concurrent
# ffmpeg subprocess — key after a CPU-encoder-caused OOM shutdown.
#
# _BASE_ENCODE_ARGS: bitrate-capped + colour-normalised + AAC 48 kHz.
#   Used for INTERMEDIATE cuts (cut_video_clips) where loudnorm would
#   be applied twice if included — once here, once at final compose.
#
# ENCODE_ARGS_SHORT_FORM: _BASE + loudnorm I=-14:TP=-1.5:LRA=11.
#   Used at the FINAL compose step only, so audio hits the normaliser
#   exactly once.
# ═══════════════════════════════════════════════════════════
from pipeline_core.hw_accel import h264_args as _h264_args, ACTIVE_ENCODER as _ACTIVE_ENCODER

_VIDEO_ARGS = _h264_args(bitrate_kbps=8000, maxrate_kbps=10000, bufsize_kbps=16000)

_COLOR_TAGS = [
    "-color_range",      "tv",
    "-color_primaries",  "bt709",
    "-color_trc",        "bt709",
    "-colorspace",       "bt709",
]

_AUDIO_ARGS = [
    "-c:a",  "aac",
    "-b:a",  "192k",
    "-ar",   "48000",
]

_BASE_ENCODE_ARGS = _VIDEO_ARGS + _COLOR_TAGS + _AUDIO_ARGS

ENCODE_ARGS_INTERMEDIATE = list(_BASE_ENCODE_ARGS)

ENCODE_ARGS_SHORT_FORM = _BASE_ENCODE_ARGS + [
    # Three-layer audio defence so the QA -0.5 dBTP gate is never even
    # close to being touched on real-world inputs:
    #   1. volume=-3dB  — pre-attenuate; never trust upstream peak claims.
    #   2. loudnorm     — TP target -2 dBTP (was -1.5; widened margin).
    #   3. alimiter     — brick-wall at 0.708 lin ≈ -3 dBTP, with explicit
    #                     level_in/level_out so behaviour is reproducible
    #                     across ffmpeg builds (the "level=disabled" form
    #                     was being silently ignored on Railway's nixpacks
    #                     ffmpeg, letting peaks reach +1.5 dBTP in prod).
    # Net result: post-mix peaks land near -3 dBTP. -14 LUFS integrated
    # loudness is unchanged because loudnorm self-calibrates.
    "-af", (
        "volume=-3dB,"
        "loudnorm=I=-14:TP=-2:LRA=11,"
        "alimiter=level_in=1:level_out=1:limit=0.708:attack=5:release=50"
    ),
]

# ═══════════════════════════════════════════════════════════
# PIPELINE EXCEPTIONS
# ═══════════════════════════════════════════════════════════

class PipelineQAError(RuntimeError):
    """Raised when an output clip fails QA validation.

    Attributes
    ----------
    qa_errors : list[str]
        The hard-failure messages from QAResult.errors.
    qa_warnings : list[str]
        Any soft warnings from QAResult.warnings.
    """

    def __init__(self, qa_errors: list, qa_warnings: list | None = None):
        self.qa_errors = qa_errors
        self.qa_warnings = qa_warnings or []
        msg = "Pipeline QA failed: " + "; ".join(qa_errors)
        super().__init__(msg)

# Fix Windows console encoding for Telugu/Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging as _logging
_pipeline_logger = _logging.getLogger("kaizer.pipeline")


def _maybe_upload_final(
    local_path: str,
    key: str,
    *,
    content_type: str = "video/mp4",
) -> tuple:
    """Upload *local_path* to the configured storage backend if it is not 'local'.

    Returns a ``(url, key, backend)`` triple.  When the backend is local (or
    STORAGE_BACKEND is unset / 'local') all three values are empty strings so
    callers can detect "no upload happened" without extra logic.

    This helper is intentionally simple — it does NOT delete the local file.
    The caller (run_pipeline) owns cleanup.

    Exceptions from the storage provider are logged and re-raised so the
    pipeline fails loudly rather than silently producing a clip with no cloud
    copy.
    """
    try:
        from pipeline_core.storage import get_storage_provider
        # Honour STORAGE_BACKEND so local dev keeps files on disk and
        # prod (STORAGE_BACKEND=r2) ships them to R2. The local branch
        # below early-returns an empty triple — the file is already on
        # disk where the renderer wrote it, so there's nothing to "copy
        # to local storage" and no DB columns to populate. The worker's
        # _ensure_local_clip uses clip.file_path in that case.
        provider = get_storage_provider()
        if provider.name == "local":
            return ("", "", "")
        # Forward-slash keys on all platforms.
        safe_key = key.replace("\\", "/")
        stored = provider.upload(local_path, safe_key, content_type=content_type)
        _pipeline_logger.info(
            "_maybe_upload_final: uploaded %r → key=%r url=%r",
            local_path, safe_key, stored.url,
        )
        return (stored.url, safe_key, stored.backend)
    except Exception as exc:
        _pipeline_logger.error(
            "_maybe_upload_final: upload FAILED for %r key=%r: %s",
            local_path, key, exc,
        )
        raise


# pipeline_core/ lives inside kaizer/backend/ — BASE_DIR = kaizer/backend/
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

RESOURCES   = os.path.join(BASE_DIR, "resources")
FONTS_DIR   = os.path.join(RESOURCES, "fonts")
# Per-job logo: resolved by runner.py from the selected channel's logo_asset
# and passed through KAIZER_DEFAULT_LOGO env var.  Empty / missing = NO logo
# overlay at all (intentional — users must opt in by setting a channel logo).
_DEFAULT_LOGO_ENV = (os.environ.get("KAIZER_DEFAULT_LOGO", "") or "").strip()
DEFAULT_LOGO = _DEFAULT_LOGO_ENV if (_DEFAULT_LOGO_ENV and os.path.exists(_DEFAULT_LOGO_ENV)) else ""
OUTPUT_ROOT = os.environ.get("KAIZER_OUTPUT_ROOT",
              os.path.join(BASE_DIR, "output", "api_pipeline"))

# ═══════════════════════════════════════════════════════════
# API KEYS — load from .env or set here
# ═══════════════════════════════════════════════════════════
def _load_env():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

# ═══════════════════════════════════════════════════════════
# INSTALL / IMPORT DEPENDENCIES
# ═══════════════════════════════════════════════════════════

def _ensure_package(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        print(f"  Installing {pip_name or pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg])

_ensure_package("google.genai", "google-genai")
_ensure_package("openai", "openai")
_ensure_package("requests")
_ensure_package("PIL", "Pillow")

from google import genai
from google.genai import types as genai_types
import openai
import requests
from PIL import Image, ImageDraw, ImageFont

# Phase 12 — admin accounting of every Gemini call.  Import is best-effort so
# pipeline_core.pipeline stays usable as a standalone script (no Kaizer DB).
try:
    from learning.gemini_log import log_gemini_call as _log_gemini_call
except Exception:  # pragma: no cover
    from contextlib import contextmanager as _cm
    @_cm
    def _log_gemini_call(*_a, **_kw):
        class _NoOp:
            def record(self, *a, **k): return None
        yield _NoOp()


def _truetype(path, size):
    """Load font with raqm/HarfBuzz for proper Telugu conjunct shaping.
    Falls back to basic layout if libraqm is not installed."""
    try:
        return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        return ImageFont.truetype(path, size)


# ═══════════════════════════════════════════════════════════
# PLATFORM PRESETS
# ═══════════════════════════════════════════════════════════
PLATFORM_PRESETS = {
    "instagram_reel": {
        "label": "Instagram Reel", "width": 1080, "height": 1920,
        "min_dur": 15, "max_dur": 90, "ideal_dur": 30, "vertical": True,
    },
    "youtube_short": {
        "label": "YouTube Short", "width": 1080, "height": 1920,
        "min_dur": 15, "max_dur": 60, "ideal_dur": 45, "vertical": True,
    },
    "youtube_full": {
        "label": "YouTube Full", "width": 1920, "height": 1080,
        "min_dur": 60, "max_dur": None, "ideal_dur": None, "vertical": False,
    },
}

MAX_CLIPS = 5


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def ts_to_sec(ts: str, video_dur: float = None) -> float:
    """
    Parse timestamps from Gemini. Handles multiple formats:
      HH:MM:SS.ss  →  3600*H + 60*M + S
      MM:SS.ss     →  60*M + S
      MM:SS:mmm    →  60*M + S + ms/1000   (last part ≥3 digits)
      MM:SS:FF     →  60*M + S + FF/100    (Gemini centiseconds variant)

    If video_dur is given and the HH:MM:SS parse exceeds it, auto-reparse as MM:SS:cs.
    """
    ts = ts.strip()
    parts = ts.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            last = parts[2]
            last_int = last.split(".")[0]
            if len(last_int) >= 3:
                # MM:SS:mmm (milliseconds)
                return float(parts[0]) * 60 + float(parts[1]) + float(last) / 1000
            # Try HH:MM:SS first
            hms = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            # If result exceeds video duration, re-interpret as MM:SS:cs (centiseconds)
            if video_dur is not None and hms > video_dur:
                return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 100
            return hms
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except ValueError:
        return 0.0


def build_stitched_time_map(kept_clips: list) -> list:
    """Build a source→stitched timestamp map from validated kept clips.

    After cut_video_clips runs, the final bulletin is the concatenation
    of each kept clip end-to-end. So a source-video timestamp `src_t`
    inside the k-th kept clip translates to:

        stitched_t = sum(durations of clips 0..k-1) + (src_t - clip_k.start)

    A timestamp inside a CUT span (between two kept clips or before/after
    the entire kept set) has no stitched equivalent — caller drops the
    image_plan entry.
    """
    rows: list[dict] = []
    cum = 0.0
    for k in kept_clips:
        ss = float(k.get("start_sec") or 0.0)
        se = float(k.get("end_sec") or 0.0)
        if se <= ss:
            continue
        rows.append({
            "src_start":       round(ss, 3),
            "src_end":         round(se, 3),
            "stitched_offset": round(cum, 3),
            "stitched_end":    round(cum + (se - ss), 3),
            "clip_index":      k.get("index"),
        })
        cum += (se - ss)
    return rows


def to_stitched(src_t: float, time_map: list):
    """Convert a source-video timestamp to a stitched-video timestamp.

    Returns (stitched_t, clip_index, span) where ``span`` is the row from
    ``time_map`` containing ``src_t``. Returns ``None`` when ``src_t``
    falls inside a cut span or past the end of the last kept clip.
    """
    for row in time_map:
        if row["src_start"] <= src_t < row["src_end"]:
            stitched_t = row["stitched_offset"] + (src_t - row["src_start"])
            return (stitched_t, row["clip_index"], row)
    return None


def _word_text(w) -> str:
    if isinstance(w, dict):
        return str(w.get("text") or w.get("word") or "")
    return str(getattr(w, "text", "") or getattr(w, "word", ""))


def _word_start(w) -> float:
    if isinstance(w, dict):
        return float(w.get("start") or w.get("start_sec") or 0.0)
    return float(getattr(w, "start", 0.0) or getattr(w, "start_sec", 0.0))


def whisper_anchor(
    entity_name: str,
    source_time: float,
    whisper_words: list,
    window_sec: float = 3.0,
) -> tuple:
    """Snap an image_plan show_at to the spoken word that names the entity.

    Looks for ``entity_name``'s first token (case-insensitive, punctuation
    stripped) inside ``[source_time - window, source_time + window]`` of
    the Whisper word list. If found, returns ``(matched_word.start, info)``.
    If not, returns ``(source_time, miss_info)`` so the caller falls back
    to Gemini's original timing.

    Designed to accept either Word dataclass instances OR plain dicts.
    """
    name = (entity_name or "").strip()
    if not name or not whisper_words:
        return source_time, {"matched": False, "reason": "empty entity or empty transcript"}

    first_token = re.sub(r"[^\w]+", "", name.split()[0]).lower()
    if not first_token:
        return source_time, {"matched": False, "reason": "entity has no alphanumeric tokens"}

    lo = source_time - window_sec
    hi = source_time + window_sec
    best = None  # (abs_drift, word_start, word_text)

    for w in whisper_words:
        ws = _word_start(w)
        if ws < lo or ws > hi:
            continue
        wt = re.sub(r"[^\w]+", "", _word_text(w)).lower()
        if not wt:
            continue
        if first_token in wt or wt in first_token:
            drift = abs(ws - source_time)
            if best is None or drift < best[0]:
                best = (drift, ws, _word_text(w))

    if best is None:
        return source_time, {
            "matched": False,
            "reason": f"'{name}' not found in Whisper transcript within ±{window_sec}s of {source_time:.2f}s",
        }
    drift, word_start, matched_text = best
    return word_start, {
        "matched": True,
        "word": matched_text,
        "word_start_sec": round(word_start, 3),
        "drift_corrected_sec": round(word_start - source_time, 3),
    }


def sec_to_ts(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sc = s % 60
    return f"{h:02d}:{m:02d}:{sc:05.2f}"


def _e(n):
    """Round to nearest even integer (FFmpeg needs even dimensions)."""
    v = int(n)
    return v if v % 2 == 0 else v - 1


def _hex_to_rgb(hex_color):
    """Convert #rrggbb or #rgb hex string to (r, g, b) tuple."""
    h = (hex_color or '#000000').lstrip('#')
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    try:
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def _sc(idx, w, h):
    return (f"[{idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}")


def _ffp(path):
    return path.replace("\\", "/").replace(":", "\\:")


def _fte(text):
    return (str(text).replace("\\", "\\\\").replace("'", "\\'")
            .replace(":", "\\:").replace("%", "\\%").replace("\n", " "))


def _ascii_text(text, max_len=60):
    clean = "".join(c for c in str(text) if ord(c) < 128)
    return clean[:max_len].strip() or "KAIZER NEWS"


def get_video_info(path):
    cmd = [FFPROBE_BIN, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(r.stdout)
        dur = float(info["format"]["duration"])
        fps, w, h = 30.0, 1920, 1080
        for s in info.get("streams", []):
            if s["codec_type"] == "video":
                parts = s.get("r_frame_rate", "30/1").split("/")
                fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else 30.0
                w = int(s.get("width", 1920))
                h = int(s.get("height", 1080))
                break
        return {"duration": dur, "fps": fps, "width": w, "height": h}
    except Exception as e:
        print(f"  ffprobe error: {e}")
        return None


def run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("  FFmpeg error (last 15 lines):")
        for line in result.stderr.decode("utf-8", errors="replace").splitlines()[-15:]:
            print(f"    {line}")
        raise subprocess.CalledProcessError(result.returncode, cmd)


# ═══════════════════════════════════════════════════════════
# STEP 1: GEMINI — Video Analysis + Cuts + Summary + Keywords
# ═══════════════════════════════════════════════════════════

# Base schema block — the COMPLETE single-platform JSON schema. Used when
# the caller is NOT in compound mode (instagram_reel / youtube_short /
# youtube_full standalone jobs). The Plan A redesigned prompt expects the
# {compound_schema_block} placeholder to expand to the FULL schema text
# — single mode gets this base-only version; compound mode gets the
# extended version below that includes the additional fields.
#
# Both blocks use str.format placeholders for {language_name} + {script_name}
# and are .format()'d BEFORE being spliced into the parent prompt. JSON
# braces are doubled (`{{` / `}}`) so they survive the parent's .format().
_BASE_SCHEMA_BLOCK = r"""{{
  "video_type": "SOLO|INTERVIEW|PRESS_CONFERENCE|PANEL|MIXED",
  "language": "{language_name}|English|Mixed",
  "total_speakers": <number>,
  "clips": [
    {{
      "index": 1,
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "summary": "<2-3 sentence summary in English>",
      "summary_native": "<same summary in {language_name} using {script_name} script>",
      "mood": "serious|dramatic|emotional|calm|heated|funny",
      "speakers": <number of speakers in this clip>,
      "importance": <1-10 score>
    }}
  ],
  "overall_summary": "<5-6 sentence overall summary in English>",
  "overall_summary_native": "<same summary in {language_name} using {script_name} script>",
  "image_search_queries": [
    "<SPECIFIC Google search query to find a real photo of the main person/event — in English>",
    "<Second search query — different angle, e.g. location or incident photo>",
    "<Third search query — IN {language_name} ({script_name} script) for local news article images>",
    "<Fourth search query — another relevant person or topic>"
  ],
  "key_people": ["full name 1", "full name 2"],
  "key_people_native": ["name1 in {script_name}", "name2 in {script_name}"],
  "key_topics": ["topic 1", "topic 2", "topic 3"],
  "key_locations": ["location 1", "location 2"],
  "skipped_segments": [
    {{
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "reason": "<one-line explanation, e.g. 'abandoned take, restarted at 00:42.10' or 'mic check before delivery began'>",
      "category": "retake|warm_up|crew_talk|hesitation|aside|self_correction"
    }}
  ],
  "retake_audit": "<one sentence with numbers proving the self-check ran: 'Scanned 5 clips, found 3 retake repetitions and 1 warm-up; tightened clip 1 start by 4.2s and dropped 7 spans totalling 38s.' Never SKIPPED.>"
}}"""


# Compound schema block — the COMPLETE base+compound JSON schema. Used
# only when the caller asks for "Full Video + Shorts" via mode="compound".
# Returns TWO cut sets (full_video_cuts + shorts_cuts) plus image_plan
# that drives content-aligned bulletin overlays, plus the Telugu/Hindi
# headline for the shorts compose step, plus the marquee points for the
# bulletin ticker. NO ChatGPT round-trip needed after — everything
# text-decision lives here.
_COMPOUND_SCHEMA_BLOCK = r"""{{
  "video_type": "SOLO|INTERVIEW|PRESS_CONFERENCE|PANEL|MIXED",
  "language": "{language_name}|English|Mixed",
  "total_speakers": <number>,
  "clips": [
    {{
      "index": 1,
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "summary": "<2-3 sentence summary in English>",
      "summary_native": "<same summary in {language_name} using {script_name} script>",
      "mood": "serious|dramatic|emotional|calm|heated|funny",
      "speakers": <number of speakers in this clip>,
      "importance": <1-10 score>
    }}
  ],
  "overall_summary": "<5-6 sentence overall summary in English>",
  "overall_summary_native": "<same summary in {language_name} using {script_name} script>",
  "image_search_queries": [
    "<SPECIFIC Google search query to find a real photo of the main person/event — in English>",
    "<Second search query — different angle, e.g. location or incident photo>",
    "<Third search query — IN {language_name} ({script_name} script) for local news article images>",
    "<Fourth search query — another relevant person or topic>"
  ],
  "key_people": ["full name 1", "full name 2"],
  "key_people_native": ["name1 in {script_name}", "name2 in {script_name}"],
  "key_topics": ["topic 1", "topic 2", "topic 3"],
  "key_locations": ["location 1", "location 2"],
  "full_video_cuts": [
    {{
      "index": 1,
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "summary": "<2-3 sentence summary in English>",
      "summary_native": "<same summary in {language_name} using {script_name} script>",
      "importance": <1-10 score>
    }}
  ],
  "shorts_cuts": [
    {{
      "index": 1,
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "hook": "<one-line punchy English hook describing why this moment goes viral>",
      "importance": <1-10 score>
    }}
  ],
  "image_plan": [
    {{
      "id": "img_01",
      "topic_clue": "rahul_gandhi",
      "entity_name": "Rahul Gandhi",
      "description": "<one-sentence photo description fed to the image generator>",
      "search_query": "<English Google query to find a real news photo of this entity>",
      "search_query_native": "<same query in {script_name} script>",
      "clip_index": 2,
      "show_at": "MM:SS.ss",
      "duration": 5.0,
      "reason": "<why this image at this moment, e.g. 'speaker mentions Rahul Gandhi from 02:14 to 02:21'>"
    }}
  ],
  "shorts_headline_native": "<the Telugu / {language_name} headline burned into the shorts torn-card frame — 5-8 words MAX, {script_name} script only>",
  "bulletin_marquee_points": [
    "<short headline phrase 1 — {script_name} script — 4-7 words>",
    "<short headline phrase 2 — {script_name} script>",
    "<short headline phrase 3 — {script_name} script>"
  ],
  "skipped_segments": [
    {{
      "start": "MM:SS.ss",
      "end": "MM:SS.ss",
      "reason": "<one-line explanation; for retakes, name the repeated phrase>",
      "category": "retake|warm_up|crew_talk|hesitation|aside|self_correction"
    }}
  ],
  "retake_audit": "<one sentence with numbers proving the self-check ran: 'Scanned 5 clips, found 3 retake repetitions and 1 warm-up; tightened clip 1 start by 4.2s and dropped 7 spans totalling 38s.' Never SKIPPED.>"
}}"""


GEMINI_PROMPT = """You are an expert {language_name} news video editor. Watch this raw, unedited footage carefully and produce a single JSON plan for downstream video processing.

OUTPUT TOKEN BUDGET: ~3000 tokens total. Pace yourself: ~150 tokens per clip summary, ~80 tokens per image_plan entry. Compress prose. End your response with the closing }} of the JSON immediately followed by the sentinel <<END>> — nothing after.

OUTPUT FORMAT (strict):
- Pure JSON object only. NO markdown fences (no ```json). NO trailing commas. NO comments. NO prose before or after.
- Final characters of your response must be the JSON closing brace and the sentinel literal.

REASONING ORDER (mandatory, in this sequence):
1. SCRUB for skipped_segments FIRST. Identify warm_up, retake, crew_talk, hesitation, aside, and self_correction spans. This is your self-check — it must produce visible output, not silent thought.
2. PICK full_video_cuts inside the retake-free spans only. Aim for {min_dur}s minimum per clip, {max_clips} clips max. Cut at natural pauses, never mid-sentence.
3. PICK shorts_cuts: 15–60s each, 3–6 entries, can overlap with full_video_cuts. Choose for punch, not coverage.
4. CANONICALIZE entities the speaker mentions. Aggregate aliases into one canonical name (see Worked Example A). Hard cap: 6 unique IDs.
5. SCHEDULE image_plan entries within clip boundaries (see Worked Example B).
6. WRITE native-script metadata last.

SKIPPED_SEGMENTS CATEGORIES (use one of these exactly):
- warm_up: mic checks, "test test", flat affect, throat clearing before delivery voice begins
- retake: same sentence attempted ≥2 times — keep ONLY the last clean version, drop earlier attempts
- crew_talk: speaker addresses someone off-camera ("Ravi, did you start recording?")
- hesitation: filler "umm", "uhh" between abandoned and final take
- aside: speaker breaks performance briefly ("one minute", "where was I", looks off-camera)
- self_correction: factual error explicitly replaced ("Monday... sorry, on Tuesday")

EMPHASIS IS NOT A RETAKE:
- "chala chala bagundi" (intensifier) → KEEP all words
- "really really good" → KEEP
- "bahut bahut dhanyavaad" → KEEP
- Pattern: repetition without a pause is emphasis. Pause >500ms + restart from semantic anchor is a retake.

WORKED EXAMPLE A — ENTITY ID REUSE (drives 50% failure rate to ~0):
Speaker mentions Narendra Modi at 01:20 as "PM Modi" and again at 08:45 as "Modi gaaru".

WRONG (wastes the 6-ID budget):
  image_plan[0] = {{"id": "img_01", "entity_name": "PM Modi", "show_at": "01:20.00", ...}}
  image_plan[1] = {{"id": "img_02", "entity_name": "Modi gaaru", "show_at": "08:45.00", ...}}

RIGHT (canonicalize aliases, reuse id):
  image_plan[0] = {{"id": "img_modi", "entity_name": "Narendra Modi", "show_at": "01:20.00", ...}}
  image_plan[1] = {{"id": "img_modi", "entity_name": "Narendra Modi", "show_at": "08:45.00", ...}}

Aliases like "PM Modi", "Modi gaaru", "Bharath Pradhana Mantri", "Narendra Modi" all refer to ONE entity. Use ONE id. Same applies to all repeated entities.

WORKED EXAMPLE B — IMAGE_PLAN BOUNDARIES (drives 5% drop rate to 0):
Given: full_video_cuts[1] = {{"start": "01:00.00", "end": "02:00.00"}}

VALID:   {{"clip_index": 1, "show_at": "01:30.00", "duration": 5.0}}
  → 01:30 + 5s = 01:35 ∈ [01:00, 02:00] ✓

INVALID: {{"clip_index": 1, "show_at": "01:58.00", "duration": 5.0}}
  → 01:58 + 5s = 02:03 > 02:00 ✗ — software will silently drop this

INVALID: {{"clip_index": 1, "show_at": "03:15.00", "duration": 4.0}}
  → 03:15 outside clip 1 entirely ✗

Verify show_at + duration ≤ clip_end for every entry.

HARD RULES:
- Never cut mid-sentence (rule applies AFTER retake filtering).
- Never include warm-up or crew talk inside any kept clip.
- skipped_segments MUST be non-empty for SOLO videos. Every single-take recording has at least a warm-up. If you emit [], you did not look hard enough — default expectation is 3–10 entries.
- retake_audit MUST be a real sentence with numbers, never "SKIPPED".
- image_plan description must describe what's in the frame ("politician in white kurta at a press conference"), NEVER name real public figures. The entity_name field carries identity separately.
- duration ≥ 2.0 seconds for every image_plan entry.
- shorts_headline_native and bulletin_marquee_points in {script_name} script ONLY. No Latin transliteration.

LANGUAGE-SPECIFIC RETAKE PATTERNS:
- Telugu retake: "ee paddati lo... ee paddati lo manam chestamu" → drop first attempt
- Telugu emphasis: "chala chala bagundi" → keep
- Hindi retake: "yeh kahani... yeh kahani aaj ki" → drop first
- Hindi emphasis: "bahut bahut dhanyavaad" → keep
- English retake: "the bowler has done... let me say again..." → drop earlier
- English emphasis: "really really good" → keep
- Code-mixed: same rules apply across language boundaries

VIDEO TYPE AFFECTS RETAKE FREQUENCY:
- SOLO: aggressive (3–10 skipped_segments typical)
- INTERVIEW: moderate (1–4)
- PRESS_CONFERENCE: light (0–2)
- PANEL: light (0–2)

COMPOUND MODE FIELDS (this mode):
- full_video_cuts[]: 3–6 coherent {min_dur}s+ chunks covering overall scenario
- shorts_cuts[]: 3–6 punchy 15–60s clips — CAN overlap full_video_cuts, CAN be outside them
- image_plan[]: bulletin-only overlays, scheduled inside full_video_cuts clips
- shorts_headline_native: ONE {script_name}-script headline (5–8 words) burned on every short
- bulletin_marquee_points: 3–7 ticker phrases (4–7 words each) in {script_name} script

RESPOND IN EXACTLY THIS JSON SCHEMA:
{compound_schema_block}

Remember: end with }}<<END>>. No exceptions.
"""


def _fix_json_strings(text: str) -> str:
    """Replace literal control characters inside JSON string values with escape sequences.
    Handles: unescaped newlines, carriage returns, tabs, and stray backslashes.
    Uses a simple state machine so it correctly skips over \\" inside strings.
    """
    result = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_string:
            if c == '\\':
                # Pass through the escape sequence unchanged
                result.append(c)
                i += 1
                if i < len(text):
                    result.append(text[i])
                    i += 1
                continue
            elif c == '"':
                in_string = False
                result.append(c)
            elif c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            else:
                result.append(c)
        else:
            if c == '"':
                in_string = True
                result.append(c)
            else:
                result.append(c)
        i += 1
    return ''.join(result)


def _parse_gemini_json(raw_text: str) -> dict:
    """Robust multi-strategy parser for Gemini JSON responses.

    Gemini sometimes returns:
      - Markdown code fences (```json ... ```)
      - Trailing commas before } or ]
      - Literal newlines / tabs inside string values (invalid JSON)
      - Truncated responses

    Each strategy is tried in order; the first that succeeds is returned.
    As a last resort the clips array is extracted clip-by-clip so at least
    the cut decisions are preserved even if metadata fields are garbled.
    """
    text = raw_text.strip()

    def _try(t):
        return json.loads(t)

    # 1. Direct parse
    try:
        return _try(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    if "```" in text:
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            try:
                return _try(text)
            except json.JSONDecodeError:
                pass

    # 3. Find outermost { ... } block
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)

    # 4. Remove trailing commas before } or ]
    fixed = re.sub(r',(\s*[}\]])', r'\1', text)
    try:
        return _try(fixed)
    except json.JSONDecodeError:
        pass

    # 5. Fix literal control characters inside strings
    fixed2 = _fix_json_strings(fixed)
    try:
        return _try(fixed2)
    except json.JSONDecodeError:
        pass

    # 6. Also fix the original (unfixed-commas) text with string repair
    fixed3 = _fix_json_strings(text)
    fixed3 = re.sub(r',(\s*[}\]])', r'\1', fixed3)
    try:
        return _try(fixed3)
    except json.JSONDecodeError:
        pass

    # 7. Last resort — extract clip objects individually so we keep cut decisions
    clips = []
    for clip_m in re.finditer(
        r'\{\s*"index"\s*:.*?"importance"\s*:\s*\d+\s*\}',
        fixed3, re.DOTALL
    ):
        try:
            clips.append(json.loads(clip_m.group(0)))
        except json.JSONDecodeError:
            pass

    if clips:
        print(f"    [warn] Gemini JSON was malformed — recovered {len(clips)} clip(s) from partial parse")
        return {
            "video_type": "MIXED",
            "language": "Mixed",
            "total_speakers": 1,
            "clips": clips,
            "overall_summary": "",
            "overall_summary_native": "",
            "image_search_queries": ["breaking news today"],
            "key_people": [],
            "key_people_native": [],
            "key_topics": [],
            "key_locations": [],
        }

    print(f"    [error] Could not parse Gemini response. Raw (first 600 chars):\n{raw_text[:600]}")
    raise ValueError("Could not parse Gemini JSON response — all repair strategies failed")


def _gemini_client() -> "genai.Client":
    """Construct a google.genai client from the configured API key.

    Replaces the deprecated `genai.configure(api_key=...)` module-level
    state from google-generativeai. Each call returns a fresh Client —
    cheap (just stores the API key + a requests session under the hood).
    """
    return genai.Client(api_key=GEMINI_API_KEY)


def upload_video_to_gemini(video_path: str) -> object:
    """Upload video to Gemini File API and wait for processing.

    Migrated to google.genai. Same polling contract — `state.name`
    transitions from PROCESSING → ACTIVE / FAILED. The client.files
    namespace replaces the old top-level `genai.upload_file` /
    `genai.get_file` helpers.
    """
    client = _gemini_client()
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"    Uploading video ({size_mb:.1f} MB) to Gemini ...", end="", flush=True)
    vf = client.files.upload(
        file=video_path,
        config=genai_types.UploadFileConfig(mime_type="video/mp4"),
    )
    while getattr(vf.state, "name", str(vf.state)) == "PROCESSING":
        time.sleep(4)
        vf = client.files.get(name=vf.name)
        print(".", end="", flush=True)
    if getattr(vf.state, "name", str(vf.state)) == "FAILED":
        raise RuntimeError("Gemini upload failed")
    print(f" done")
    return vf


def _video_cache_key(video_path: str, preset: dict) -> str:
    """Content-hash of video (first 4MB + size) + preset params.

    First 4 MB + byte-length is enough to distinguish any two real
    videos; hashing 771 MB to look up a cache entry would defeat the
    purpose of having a cache.

    ``mtime`` is INTENTIONALLY excluded — re-uploading the same video
    creates a new file with a fresh mtime, which would bust the cache
    even though the bytes are identical. The whole point of this
    cache is to skip Gemini when the user re-runs the same source.
    """
    import hashlib
    h = hashlib.sha256()
    try:
        st = os.stat(video_path)
        h.update(f"{st.st_size}".encode())
        with open(video_path, "rb") as f:
            h.update(f.read(4 * 1024 * 1024))
    except Exception:
        h.update(video_path.encode())
    h.update(json.dumps({
        "min_dur":   preset.get("min_dur"),
        "max_dur":   preset.get("max_dur"),
        "ideal_dur": preset.get("ideal_dur"),
        "max_clips": MAX_CLIPS,
        # Prompt version: bump this string whenever GEMINI_PROMPT changes
        # in a way that should invalidate cached responses (e.g. new rules,
        # new fields, stricter retake detection). Old cache files become
        # unreachable; first re-run of any video pays for one fresh
        # Gemini call, then re-caches against the new key.
        "prompt_version": "2026.05.16-stricter-retake-audit",
    }, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _cache_dir() -> str:
    """Persistent cache directory — survives per-run OUTPUT_DIR timestamps."""
    d = os.path.join(OUTPUT_ROOT, "_gemini_cache")
    os.makedirs(d, exist_ok=True)
    return d


def analyze_video_with_gemini(
    video_path: str,
    preset: dict,
    language: str = "te",
    mode: str = "single",
) -> dict:
    """Send video to Gemini, get cut decisions + summary + keywords.

    Cached: identical video + preset + language → same Gemini response, read from disk.
    Override with KAIZER_CACHE_GEMINI=false to force re-run.

    Compound-job reuse: when ``KAIZER_REUSE_ANALYSIS_FROM`` env var
    points at a previously-written analysis JSON, we skip the Gemini
    call entirely and load that file. The compound runner uses this
    to do ONE Gemini call across both pipeline passes — pass 1
    produces the analysis, pass 2 reads it.

    `mode` controls the schema Gemini is asked to return:
      - "single"   → existing single-platform schema (clips[] only, no
                     image_plan, no shorts_cuts, no bulletin marquee).
                     Used by standalone youtube_short / youtube_full /
                     instagram_reel jobs. Fully backward-compatible.
      - "compound" → the new combined schema for "Full Video + Shorts".
                     Adds full_video_cuts[], shorts_cuts[], image_plan[],
                     shorts_headline_native, bulletin_marquee_points[].
                     ONE Gemini call decides both formats and the
                     bulletin imagery in a single round-trip.
    """
    # Lazy import so test harnesses that don't set sys.path can still load.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import languages as _langs
    lang_cfg = _langs.get(language)

    # ── Compound-job reuse hook (highest priority) ───────────────
    # The runner sets this env var on pass 2 of a compound job after
    # capturing pass 1's [kaizer:analysis] stdout marker. Skipping
    # Gemini saves ~$0.50 and ~30s per compound job.
    _reuse_path = os.environ.get("KAIZER_REUSE_ANALYSIS_FROM", "").strip()
    if _reuse_path and os.path.isfile(_reuse_path):
        try:
            with open(_reuse_path, "r", encoding="utf-8") as _f:
                _cached = json.load(_f)
            print(f"    ✓ Reusing Gemini analysis from prior pass ({os.path.basename(_reuse_path)}) — 0 quota burned")
            return _cached
        except Exception as _e:
            print(f"    [reuse] failed to load {_reuse_path}: {_e} — falling back to fresh Gemini call")

    # ── Cache lookup (key includes language so te/hi outputs don't collide) ──
    # Two-tier cache: Redis first (cross-pod, enterprise scale), then
    # the legacy on-disk JSON cache (per-pod warm cache for cold-start
    # cost). A Redis miss + disk hit lifts the entry into Redis on the
    # way through so future pods benefit. A Redis hit doesn't bother
    # touching disk.
    cache_enabled = os.getenv("KAIZER_CACHE_GEMINI", "true").lower() != "false"
    cache_preset = {**preset, "_lang": lang_cfg.code} if cache_enabled else preset
    cache_key = _video_cache_key(video_path, cache_preset) if cache_enabled else None
    cache_path = os.path.join(_cache_dir(), f"{cache_key}.json") if cache_key else None

    # Redis tier — content-hash via the shared gemini_cache module so
    # the key matches across all worker pods.
    redis_key: Optional[str] = None
    if cache_enabled and cache_key:
        try:
            import sys as _sys2
            _sys2.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            import gemini_cache as _gc
            redis_key = _gc.make_key(
                _gc.hash_file_prefix(video_path),
                cache_preset,
            )
            cached = _gc.cache_get("video", redis_key)
            if cached is not None:
                print(f"    ✓ Gemini analysis served from Redis cache ({redis_key[:12]}…) — 0 quota burned")
                return cached
        except Exception as exc:
            # Redis cache is best-effort; fall through to disk + live call.
            print(f"    [cache:redis] lookup failed ({exc}) — falling back to disk")
            redis_key = None

    if cache_enabled and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"    ✓ Gemini analysis served from disk cache ({cache_key}) — 0 quota burned")
            # Lift into Redis so the next pod gets a faster hit.
            if redis_key:
                try:
                    import gemini_cache as _gc
                    _gc.cache_set("video", redis_key, cached)
                except Exception:
                    pass
            return cached
        except Exception as e:
            print(f"    [cache] read failed ({e}) — re-calling Gemini")

    client = _gemini_client()

    # Upload
    video_file = upload_video_to_gemini(video_path)

    # Build prompt with platform-specific constraints + target language
    min_dur = preset.get("min_dur", 15)
    max_dur = preset.get("max_dur", 90) or 300
    ideal_dur = preset.get("ideal_dur", 30) or 120

    # Schema block selection: Plan A's redesigned prompt ends with
    # "RESPOND IN EXACTLY THIS JSON SCHEMA: {compound_schema_block}",
    # expecting the placeholder to carry the COMPLETE schema text for
    # whichever mode is active.
    #   - single mode  → _BASE_SCHEMA_BLOCK     (no compound fields)
    #   - compound mode → _COMPOUND_SCHEMA_BLOCK (base + compound fields)
    # Both blocks are themselves str.format()'d for {language_name} +
    # {script_name} BEFORE being spliced into the parent prompt template.
    if mode == "compound":
        compound_schema_block = _COMPOUND_SCHEMA_BLOCK.format(
            language_name=lang_cfg.name_english,
            script_name=lang_cfg.script,
        )
    else:
        compound_schema_block = _BASE_SCHEMA_BLOCK.format(
            language_name=lang_cfg.name_english,
            script_name=lang_cfg.script,
        )

    prompt = GEMINI_PROMPT.format(
        min_dur=min_dur, max_dur=max_dur,
        ideal_dur=ideal_dur, max_clips=MAX_CLIPS,
        language_name=lang_cfg.name_english,
        script_name=lang_cfg.script,
        compound_schema_block=compound_schema_block,
    )

    # Model fallback chain — first entry is primary, rest are fallbacks tried
    # on either quota (429) OR model-not-found (404). Accuracy-first ordering,
    # using only models currently live on v1beta. The 1.5-* family is retired.
    #   2.0-flash       → 2.0-flash-lite → 2.5-flash-lite → 2.5-flash
    # Override via KAIZER_GEMINI_VIDEO_MODELS="model_a,model_b,..." in .env.
    models_raw = os.getenv(
        "KAIZER_GEMINI_VIDEO_MODELS",
        "gemini-2.0-flash,gemini-2.0-flash-lite,gemini-2.5-flash-lite,gemini-2.5-flash",
    )
    # Back-compat: if someone set the old single-model var, put it first.
    legacy_single = os.getenv("KAIZER_GEMINI_VIDEO_MODEL", "").strip()
    candidates = [m.strip() for m in models_raw.split(",") if m.strip()]
    if legacy_single and legacy_single not in candidates:
        candidates.insert(0, legacy_single)

    response = None
    last_error = None
    for model_name in candidates:
        try:
            print(f"    Asking {model_name} to analyze and suggest cuts ...", end="", flush=True)
            # Only retry in-place on transient server errors; 429 and 404
            # bubble up to the outer loop so we try the next model instead.
            for attempt in range(3):
                try:
                    with _log_gemini_call(
                        db=None, model=model_name, purpose="video-cut",
                    ) as _gcall:
                        # google-genai: client.models.generate_content takes
                        # the model name as a kwarg + the multipart contents
                        # (video file ref + text prompt). Old SDK's
                        # request_options={"timeout": 180} has no direct
                        # equivalent — google-genai uses internal HTTP
                        # timeouts; we rely on those defaults.
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[video_file, prompt],
                        )
                        _gcall.record(response)
                    break
                except Exception as e:
                    msg = str(e)
                    is_quota = ("429" in msg) or ("ResourceExhausted" in msg) or ("quota" in msg.lower())
                    is_missing = ("404" in msg) or ("NotFound" in msg) or ("not found for API version" in msg) or ("is not supported for generateContent" in msg)
                    is_server = any(code in msg for code in ("500", "502", "503", "504"))
                    if is_quota or is_missing:
                        raise  # Try next model in chain
                    if attempt < 2 and is_server:
                        print(f"\n    Retry {attempt+1} (server error): {e}")
                        time.sleep(5)
                    else:
                        raise
            if response is not None:
                break  # Got a response — stop trying fallback models
        except Exception as e:
            msg = str(e)
            is_quota = ("429" in msg) or ("ResourceExhausted" in msg) or ("quota" in msg.lower())
            is_missing = ("404" in msg) or ("NotFound" in msg) or ("not found for API version" in msg) or ("is not supported for generateContent" in msg)
            skip = is_quota or is_missing
            if skip and model_name != candidates[-1]:
                reason = "quota exhausted" if is_quota else "model unavailable (404)"
                print(f"\n    [skip] {model_name}: {reason} — falling back to next model in chain")
                last_error = e
                continue
            # Non-recoverable error, or last model in chain → give up
            if is_quota:
                print("\n    [quota] All Gemini models exhausted for today. Options:")
                print("      • Enable billing on your Google Cloud project (recommended)")
                print("      • Wait for daily reset (midnight UTC)")
                print("      • Add more models to KAIZER_GEMINI_VIDEO_MODELS")
            elif is_missing:
                print(f"\n    [404] All configured models are unavailable. Update KAIZER_GEMINI_VIDEO_MODELS in .env.")
            raise

    if response is None:
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    raw_text = response.text.strip()
    print(" done")

    result = _parse_gemini_json(raw_text)

    # Persist to BOTH cache tiers so re-runs are free across all pods.
    # Disk write first (process-local fastest path), then Redis (shared).
    # Either side failing is non-fatal — the result is already in hand.
    if cache_enabled and cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"    ✓ Cached Gemini analysis at {cache_key}.json (disk)")
            # Machine-parseable marker — the runner watches stdout for
            # this line and uses the absolute path as
            # ``KAIZER_REUSE_ANALYSIS_FROM`` on subsequent compound-job
            # passes, so the second pass skips Gemini entirely.
            print(f"[kaizer:analysis] {os.path.abspath(cache_path)}")
        except Exception as e:
            print(f"    [cache:disk] write failed: {e}")
    if cache_enabled and redis_key:
        try:
            import gemini_cache as _gc
            _gc.cache_set("video", redis_key, result)
            print(f"    ✓ Cached Gemini analysis in Redis ({redis_key[:12]}…)")
        except Exception as e:
            print(f"    [cache:redis] write failed: {e}")

    # Clean up the file from Gemini servers
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════
# STEP 2: FFmpeg — Cut Video at Gemini Timestamps
# ═══════════════════════════════════════════════════════════

def cut_video_clips(video_path: str, clips: list, output_dir: str) -> list:
    """Cut the source video into individual clips based on Gemini timestamps."""
    vid_info = get_video_info(video_path)
    vid_dur = vid_info["duration"] if vid_info else None

    cut_paths = []
    for i, clip in enumerate(clips, 1):
        start = ts_to_sec(clip["start"], vid_dur)
        end   = ts_to_sec(clip["end"],   vid_dur)
        # Clamp to actual video length
        if vid_dur:
            start = min(start, vid_dur - 1)
            end   = min(end,   vid_dur)
        dur = end - start
        if dur <= 0:
            continue

        out_path = os.path.join(output_dir, f"raw_clip_{i:02d}.mp4")
        # Idempotency: if a previous run already produced this slice,
        # reuse it. Lets --resume-dir skip the cut step entirely on
        # retry. We require >100KB to filter out empty / corrupted
        # files left behind by a half-run that crashed.
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100_000:
            print(f"    Clip {i}: cached at {os.path.basename(out_path)} (skipping cut)")
            clip["raw_path"] = out_path
            clip["duration_sec"] = round(dur, 2)
            cut_paths.append(out_path)
            continue
        cmd = (
            [FFMPEG_BIN, "-y",
             "-ss", str(round(start, 3)), "-t", str(round(dur, 3)),
             "-i", video_path]
            + ENCODE_ARGS_INTERMEDIATE
            + [out_path]
        )
        print(f"    Cutting clip {i}: {sec_to_ts(start)} -> {sec_to_ts(end)} ({dur:.1f}s)")
        run_ffmpeg(cmd)
        clip["raw_path"] = out_path
        clip["duration_sec"] = round(dur, 2)
        cut_paths.append(out_path)

    return cut_paths


# ═══════════════════════════════════════════════════════════
# STEP 3: ChatGPT — Generate Title from Summary
# ═══════════════════════════════════════════════════════════

TITLE_PROMPT = """You are a {language_name} breaking news flash writer.
Write ONE ultra-short {language_name} flash headline — ticker style, 5-8 words max.

Rules:
1. {script_name} script ONLY for the native headline. No transliteration.
2. MAX 8 words. No full sentences — just the key fact.
3. Include the main person/place name if mentioned.
4. {style_hint}
5. Do NOT add explanation or context — just the punchy headline.

Video summary:
{summary}

Key people: {people}
Key topics: {topics}

Respond with ONLY the JSON (no markdown):
{{"title_native": "<{language_name} headline in {script_name} script>", "title_english": "<English translation>", "subtitle": "<1-line English subtitle for context>"}}
"""


def generate_title_chatgpt(summary: str, people: list, topics: list, language: str = "te") -> dict:
    """Call GPT-4o-mini to generate a native-language news headline.

    Returns dict with title_native + title_english + subtitle. Callers that
    still expect the old key 'title_telugu' get it back as an alias (legacy).
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import languages as _langs
    lang_cfg = _langs.get(language)

    if not OPENAI_API_KEY:
        print("    No OPENAI_API_KEY — truncating summary to flash headline")
        words = (summary or "KAIZER NEWS").split()
        short = " ".join(words[:7])
        return {
            "title_native":  short,
            "title_telugu":  short,  # legacy alias
            "title_english": short,
            "subtitle": "",
        }

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = TITLE_PROMPT.format(
        summary=summary,
        people=", ".join(people) if people else "unknown",
        topics=", ".join(topics) if topics else "general news",
        language_name=lang_cfg.name_english,
        script_name=lang_cfg.script,
        style_hint=lang_cfg.title_style_hint or "Breaking news style — punchy and direct.",
    )

    print(f"    Generating {lang_cfg.name_english} title with GPT-4o-mini ...", end="", flush=True)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    print(" done")

    json_text = raw
    if "```" in json_text:
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", json_text, re.DOTALL)
        if m:
            json_text = m.group(1).strip()
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError:
        result = {"title_native": raw[:80], "title_english": raw[:80], "subtitle": ""}

    # Legacy alias so downstream code that still references title_telugu keeps working
    result.setdefault("title_native", result.get("title_telugu", raw[:80]))
    result["title_telugu"] = result.get("title_native", "")
    return result


# ═══════════════════════════════════════════════════════════
# STEP 4: News Image Search (Google + DuckDuckGo + Pexels)
# ═══════════════════════════════════════════════════════════

GOOGLE_CSE_ID  = os.environ.get("GOOGLE_CSE_ID", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")  # for Custom Search


def _download_image(url: str, dest_path: str, timeout: int = 20) -> bool:
    """Download a single image URL to disk. Returns True on success."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code == 200 and len(resp.content) > 2000:
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


def _search_google_cse(query: str, count: int = 3) -> list:
    """Google Custom Search (100 free/day). Returns list of image URLs."""
    if not GOOGLE_CSE_ID or not GOOGLE_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID,
                "q": query, "searchType": "image",
                "num": min(count, 10), "imgSize": "large",
                "safe": "active"
            },
            timeout=15
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            return [it["link"] for it in items if "link" in it]
    except Exception:
        pass
    return []


_ddg_last_call = 0.0  # rate-limit tracker

def _search_duckduckgo(query: str, count: int = 5) -> list:
    """DuckDuckGo image search — free, no API key needed. Handles rate limits."""
    global _ddg_last_call

    # Throttle: at least 2s between DDG requests to avoid 403
    elapsed = time.time() - _ddg_last_call
    if elapsed < 2.0:
        time.sleep(2.0 - elapsed)

    try:
        try:
            from duckduckgo_search import DDGS
            _ddg_last_call = time.time()
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=count))
                return [r["image"] for r in results if "image" in r]
        except ImportError:
            pass
        except Exception:
            # Rate limit or other DDG error — skip gracefully
            pass

        # Fallback: DuckDuckGo lite scraping
        _ddg_last_call = time.time()
        resp = requests.get(
            "https://duckduckgo.com/",
            params={"q": query + " news", "iax": "images", "ia": "images"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15
        )
        vqd_match = re.search(r'vqd=([\d-]+)', resp.text)
        if not vqd_match:
            return []
        vqd = vqd_match.group(1)

        time.sleep(1)
        img_resp = requests.get(
            "https://duckduckgo.com/i.js",
            params={"l": "us-en", "o": "json", "q": query, "vqd": vqd, "f": ",,,,,", "p": "1"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15
        )
        if img_resp.status_code == 200:
            results = img_resp.json().get("results", [])
            return [r["image"] for r in results[:count] if "image" in r]
    except Exception:
        pass
    return []


def _search_pexels(query: str, count: int = 3) -> list:
    """Pexels fallback for generic background images."""
    if not PEXELS_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": count, "orientation": "landscape"},
            timeout=15
        )
        if resp.status_code == 200:
            photos = resp.json().get("photos", [])
            return [p["src"]["large2x"] or p["src"]["large"] for p in photos if p.get("src")]
    except Exception:
        pass
    return []



def generate_image_pool_from_plan(
    image_plan: list,
    job_dir: str,
    language: str = "en",
    max_unique: int = 6,
    dual_backend: bool = True,
) -> dict:
    """Dedup Gemini's image_plan by `id` and generate ONE image per unique id.

    The image_plan may contain many entries (same entity referenced at
    several moments) but only ONE image file per id is generated — the
    overlay step reuses the same file at every show_at moment.

    Returns a manifest keyed by id:
        {
          "img_01": {
            "id": "img_01",
            "topic_clue": "rahul_gandhi",
            "description": "Rahul Gandhi at rally…",
            "path":        "/abs/path/to/openai/img_01__rahul_gandhi.jpg",
            "openai_path": "/abs/path/to/openai/img_01__rahul_gandhi.jpg",
            "imagen_path": "/abs/path/to/imagen/img_01__rahul_gandhi.jpg",
            "cached":      False,
          },
          ...
        }

    `path` is what downstream code reads to lay the image on screen. We
    keep `path` pointing at the OpenAI render by default; the Imagen
    render is saved alongside for A/B comparison. Once the operator
    picks a winner they can flip `path` to point at the Imagen file.

    Files land under <job_dir>/pool/openai/ and <job_dir>/pool/imagen/.
    """
    from pipeline_core import openai_images as _oi
    if dual_backend:
        from pipeline_core import imagen as _im

    openai_dir = os.path.join(job_dir, "pool", "openai")
    imagen_dir = os.path.join(job_dir, "pool", "imagen")
    os.makedirs(openai_dir, exist_ok=True)
    if dual_backend:
        os.makedirs(imagen_dir, exist_ok=True)

    # Dedupe by id, preserving Gemini's emission order. Cap at max_unique.
    by_id: dict = {}
    for entry in image_plan or []:
        eid = (entry.get("id") or "").strip()
        if not eid or eid in by_id:
            continue
        by_id[eid] = entry
        if len(by_id) >= max_unique:
            break

    manifest: dict = {}
    for eid, entry in by_id.items():
        clue = re.sub(r"[^a-z0-9_]+", "_", (entry.get("topic_clue") or "").lower()).strip("_") or "image"
        fname = f"{eid}__{clue}.jpg"
        openai_path = os.path.join(openai_dir, fname)
        imagen_path = os.path.join(imagen_dir, fname)

        # Idempotency: skip OpenAI generation if a prior run produced it.
        openai_cached = os.path.exists(openai_path) and os.path.getsize(openai_path) > 10_000
        imagen_cached = os.path.exists(imagen_path) and os.path.getsize(imagen_path) > 10_000

        # Build subject text — Gemini's per-entry description wins; fall
        # back to entity_name / search_query.
        subject_desc = (entry.get("description") or "").strip()
        subject_q    = (entry.get("entity_name") or entry.get("search_query") or "").strip()
        entities     = [entry["entity_name"]] if entry.get("entity_name") else []

        if not openai_cached:
            try:
                _oi.generate_news_image(
                    query=subject_desc or subject_q,
                    entities=entities,
                    topics=[clue],
                    language=language,
                    out_path=openai_path,
                )
            except Exception as exc:
                print(f"    [image_pool] openai failed for {eid}: {exc}")
        if dual_backend and not imagen_cached:
            try:
                _im.generate_news_image(
                    query=subject_q,
                    description=subject_desc or None,
                    entities=entities,
                    topics=[clue],
                    language=language,
                    out_path=imagen_path,
                )
            except Exception as exc:
                print(f"    [image_pool] imagen failed for {eid}: {exc}")

        openai_ok = os.path.exists(openai_path) and os.path.getsize(openai_path) > 10_000
        imagen_ok = os.path.exists(imagen_path) and os.path.getsize(imagen_path) > 10_000

        if not openai_ok and not imagen_ok:
            print(f"    [image_pool] both backends failed for {eid} ({clue}) — entries with this id will be dropped")
            continue

        # Default render path: prefer OpenAI; fall back to Imagen if OpenAI failed.
        render_path = openai_path if openai_ok else imagen_path
        manifest[eid] = {
            "id":          eid,
            "topic_clue":  clue,
            "description": entry.get("description") or "",
            "path":        render_path,
            "openai_path": openai_path if openai_ok else None,
            "imagen_path": imagen_path if imagen_ok else None,
            "cached":      openai_cached or imagen_cached,
        }
        if openai_ok and imagen_ok:
            print(f"    [image_pool] {fname} ✓ openai+imagen")
        elif openai_ok:
            print(f"    [image_pool] {fname} ✓ openai (imagen skipped/failed)")
        else:
            print(f"    [image_pool] {fname} ✓ imagen only (openai failed)")

    return manifest


def resolve_image_plan(
    image_plan: list,
    pool_manifest: dict,
    kept_clips: list,
    whisper_words: list | None,
    video_duration_sec: float,
) -> list:
    """Project Gemini's image_plan onto the stitched bulletin timeline.

    For each entry, in order:
      1. Look up the image file by `id` in pool_manifest. If missing,
         drop with status=image_missing.
      2. Map `show_at` (source time) → stitched time. If show_at fell
         in a cut span, drop with status=in_cut_span.
      3. If a Whisper word list is supplied, snap `show_at` to the
         word the speaker said for ``entity_name`` (±3s window).
      4. Clamp `duration` so the overlay doesn't bleed past its clip's
         stitched end.
    """
    time_map = build_stitched_time_map(kept_clips)
    resolved: list = []

    for entry in image_plan or []:
        eid = entry.get("id") or ""
        out = {
            "id":          eid,
            "topic_clue":  entry.get("topic_clue") or "",
            "entity_name": entry.get("entity_name") or "",
            "description": entry.get("description") or "",
            "clip_index":  entry.get("clip_index"),
        }

        if eid not in pool_manifest:
            out["status"] = "image_missing"
            out["reason"] = f"id '{eid}' has no generated image (pool exceeded cap or generation failed)"
            resolved.append(out)
            continue
        out["image_path"] = pool_manifest[eid]["path"]

        try:
            src_show_at = ts_to_sec(entry.get("show_at", "00:00"), video_duration_sec)
            duration    = float(entry.get("duration", 0.0) or 0.0)
        except Exception as exc:
            out["status"] = "bad_timestamp"
            out["reason"] = f"parse failed: {exc}"
            resolved.append(out)
            continue

        if duration < 2.0:
            out["status"] = "duration_too_short"
            out["reason"] = f"duration {duration:.2f}s < 2.0s minimum"
            resolved.append(out)
            continue
        out["source_show_at_sec"] = round(src_show_at, 3)

        # Whisper anchor (optional)
        anchor_info: dict = {"matched": False, "reason": "no transcript supplied"}
        if whisper_words:
            anchored_t, anchor_info = whisper_anchor(
                entry.get("entity_name") or "", src_show_at, whisper_words, window_sec=3.0,
            )
            src_show_at = anchored_t
        out["whisper_anchor"] = anchor_info

        mapped = to_stitched(src_show_at, time_map)
        if mapped is None:
            out["status"] = "in_cut_span"
            out["reason"] = (
                f"source time {src_show_at:.2f}s fell inside a cut span "
                f"(image meant for a moment we removed)"
            )
            resolved.append(out)
            continue
        stitched_t, clip_index_resolved, span = mapped
        out["stitched_show_at_sec"] = round(stitched_t, 3)
        out["clip_index_resolved"]  = clip_index_resolved

        # Boundary clamp
        clip_stitched_end = span["stitched_end"]
        max_allowed_dur = max(0.0, clip_stitched_end - stitched_t)
        clamped_dur = min(duration, max_allowed_dur)
        if clamped_dur < duration:
            out["clamped_from_sec"] = round(duration, 3)
        out["duration_sec"] = round(clamped_dur, 3)

        if clamped_dur < 2.0:
            out["status"] = "clamped_too_short"
            out["reason"] = f"after clamp duration {clamped_dur:.2f}s < 2.0s — image would barely flash"
            resolved.append(out)
            continue

        out["status"] = "ready"
        resolved.append(out)

    return resolved


def overlay_image_plan(
    stitched_video_path: str,
    resolved_plan: list,
    out_path: str,
) -> str | None:
    """Composite scheduled overlays onto the already-stitched bulletin.

    Only entries with ``status == "ready"`` are rendered. Each unique
    image_path becomes one ffmpeg input; overlays chain linearly using
    ``enable='between(t,start,end)'`` so a single ffmpeg invocation
    produces the final file.

    Layout: news-broadcast inset — scaled to 30% of frame width, placed
    in the upper-right with a 30px margin. Square images keep their
    aspect ratio.

    Returns out_path on success, None when there's nothing to render
    (caller keeps the un-overlaid bulletin).
    """
    ready = [e for e in (resolved_plan or []) if e.get("status") == "ready" and e.get("image_path")]
    if not ready:
        print("  [overlay] no ready image_plan entries — keeping bulletin unchanged")
        return None

    # Dedupe inputs so the same image file is added once.
    path_to_input_idx: dict = {}
    inputs: list = ["-i", stitched_video_path]
    for entry in ready:
        p = entry["image_path"]
        if p not in path_to_input_idx:
            path_to_input_idx[p] = len(path_to_input_idx) + 1
            inputs += ["-i", p]

    fg_parts: list = []
    # Pre-scale each unique image so overlay positioning is predictable.
    for p, idx in path_to_input_idx.items():
        fg_parts.append(f"[{idx}:v]scale=576:-1,format=yuva420p[i{idx}]")

    prev = "[0:v]"
    for k, entry in enumerate(ready, 1):
        idx = path_to_input_idx[entry["image_path"]]
        start_t = float(entry["stitched_show_at_sec"])
        end_t   = start_t + float(entry["duration_sec"])
        out_pin = f"[v{k}]"
        fg_parts.append(
            f"{prev}[i{idx}]overlay=x=W-w-30:y=30:"
            f"enable='between(t,{start_t:.3f},{end_t:.3f})'{out_pin}"
        )
        prev = out_pin

    fg = ";".join(fg_parts)
    cmd = (
        [FFMPEG_BIN, "-y"]
        + inputs
        + ["-filter_complex", fg,
           "-map", prev,
           "-map", "0:a?"]
        + ENCODE_ARGS_INTERMEDIATE
        + [out_path]
    )
    print(f"  [overlay] applying {len(ready)} entries across {len(path_to_input_idx)} unique image(s)")
    run_ffmpeg(cmd)
    return out_path


def search_news_images(search_queries: list, people: list, topics: list,
                       output_dir: str, count: int = 5,
                       language: str = "en",
                       skip_openai: bool = False) -> list:
    """
    Multi-source news image search.
    Priority: OpenAI (gpt-image-1) → Google CSE → DuckDuckGo → Pexels → Generated cards.
    Uses specific search queries from Gemini (real people, events, locations).

    The ``language`` parameter is used by the OpenAI generator to pick a
    geographic location hint (Telugu → Andhra/Telangana, Hindi → North
    India, etc.) so the visual reads as local news. Falls back to
    "India" for unknown / missing languages.

    ``skip_openai=True`` jumps straight to the web-search chain (Google
    CSE → DuckDuckGo → Pexels). Use this when the subject is a real
    public figure or a specific named incident — OpenAI refuses to
    render them by name and even if it didn't, viewers expect the
    ACTUAL photo for those. Stories about generic themes should keep
    skip_openai=False (the default) so AI-generated B-roll fills the
    pool when Pexels has nothing topical.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    downloaded = []
    seen_urls = set()

    # Build search queries — specific first, generic last
    queries = []
    for q in (search_queries or []):
        if q and q.strip():
            queries.append(q.strip())
    # Add people-name searches
    for person in (people or []):
        if person and person.strip():
            queries.append(f"{person} news photo")
    # Add topic searches
    for topic in (topics or []):
        if topic and topic.strip():
            queries.append(topic)

    if not queries:
        queries = ["Telugu news today"]

    # ── OpenAI image generation pass (preferred when enabled) ────────
    # Tries gpt-image-1 first for each requested image. Each call that
    # succeeds fills one slot; failures fall through silently to the
    # web-search loop below for that slot. Per-image fallback (not
    # all-or-nothing) means refusals on real-public-figure prompts
    # don't sink the whole batch.
    #
    # ``skip_openai=True`` short-circuits this step — used by the
    # bulletin pool builder when the caller already knows the subject
    # is a real public figure / named incident and wants the ACTUAL
    # web photo, not an AI render.
    try:
        from pipeline_core.openai_images import generate_news_image, is_enabled
        if is_enabled() and not skip_openai:
            print(f"    Trying OpenAI gpt-image-1 for up to {count} images ...")
            for qi, query in enumerate(queries):
                if len(downloaded) >= count:
                    break
                idx = len(downloaded) + 1
                out_path = os.path.join(images_dir, f"news_{idx:02d}.jpg")
                generated = generate_news_image(
                    query=query,
                    entities=people,
                    topics=topics,
                    language=language,
                    out_path=out_path,
                )
                if generated:
                    downloaded.append(generated)
            if downloaded:
                print(f"    OpenAI filled {len(downloaded)}/{count} slots")
            if len(downloaded) >= count:
                print(f"    Total: {len(downloaded)} news images downloaded")
                return downloaded
    except Exception as _oai_exc:
        # Module-level failure (import error, unexpected exception):
        # log + fall through to the web-search loop. Pipeline never
        # blocks on this path.
        print(f"    OpenAI image generation skipped ({_oai_exc})")

    print(f"    Searching for real news images ({len(queries)} queries) ...")

    for qi, query in enumerate(queries):
        if len(downloaded) >= count:
            break

        print(f"      [{qi+1}] \"{query[:60]}\" ...", end="", flush=True)
        urls = []

        try:
            # Try Google CSE first (most reliable)
            if not urls:
                urls = _search_google_cse(query, count=3)
                if urls:
                    print(f" Google:{len(urls)}", end="")

            # Try DuckDuckGo (free, no setup)
            if not urls:
                urls = _search_duckduckgo(query, count=4)
                if urls:
                    print(f" DDG:{len(urls)}", end="")

            # Pexels as last resort for this query
            if not urls:
                urls = _search_pexels(query, count=2)
                if urls:
                    print(f" Pexels:{len(urls)}", end="")
        except Exception as _e:
            print(f" (search error: {_e})")
            continue

        if not urls:
            print(" (no results)")
            continue

        # Download images
        dl_count = 0
        for url in urls:
            if len(downloaded) >= count:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            idx = len(downloaded) + 1
            ext = ".jpg"
            if ".png" in url.lower():
                ext = ".png"
            img_path = os.path.join(images_dir, f"news_{idx:02d}{ext}")
            if _download_image(url, img_path):
                downloaded.append(img_path)
                dl_count += 1

        print(f" -> {dl_count} saved")

    print(f"    Total: {len(downloaded)} news images downloaded")
    return downloaded


# ═══════════════════════════════════════════════════════════
# NEWS CARD GENERATOR (fallback when no Pexels images)
# ═══════════════════════════════════════════════════════════

_NEWS_SCHEMES = [
    {"bg": (8, 20, 45),  "accent": (220, 35, 35),  "text": (255, 255, 255)},
    {"bg": (15, 15, 15), "accent": (255, 165, 0),   "text": (255, 255, 255)},
    {"bg": (28, 0, 0),   "accent": (200, 0, 0),     "text": (255, 240, 200)},
    {"bg": (5, 30, 60),  "accent": (0, 120, 220),   "text": (255, 255, 255)},
    {"bg": (20, 20, 20), "accent": (0, 180, 100),   "text": (255, 255, 255)},
]


def _best_font(size):
    for fn in ("Roboto-Bold.ttf", "Oswald-Bold.ttf", "NotoSansTelugu-Bold.ttf"):
        fp = os.path.join(FONTS_DIR, fn)
        if os.path.exists(fp):
            try:
                return _truetype(fp, size)
            except Exception:
                pass
    try:
        return _truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def generate_news_card(query: str, out_path: str, width: int, height: int):
    """Generate a fallback news card graphic when no image is found."""
    cs = _NEWS_SCHEMES[abs(hash(query)) % len(_NEWS_SCHEMES)]
    img = Image.new("RGB", (width, height), cs["bg"])
    draw = ImageDraw.Draw(img)

    # Gradient background
    for y in range(height):
        r = min(255, cs["bg"][0] + int(y / height * 35))
        g = min(255, cs["bg"][1] + int(y / height * 25))
        b = min(255, cs["bg"][2] + int(y / height * 20))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Top accent bar
    bar_h = max(6, height // 50)
    draw.rectangle([0, 0, width, bar_h], fill=cs["accent"])

    # Breaking badge
    bw, bh = max(110, width // 7), max(28, height // 14)
    bx, by = max(14, width // 25), bar_h + max(14, height // 25)
    draw.rectangle([bx, by, bx + bw, by + bh], fill=cs["accent"])
    f_badge = _best_font(max(13, bh // 2))
    draw.text((bx + bw // 2, by + bh // 2), "BREAKING",
              fill=(255, 255, 255), font=f_badge, anchor="mm")

    # Topic text
    words = query.upper().replace("-", " ").split()
    max_chars = max(10, width // max(14, width // 20))
    lines, line = [], []
    for word in words:
        if len(" ".join(line + [word])) > max_chars and line:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    lines = lines[:4]

    f_big = _best_font(max(28, width // 8))
    line_h = max(36, width // 7)
    total_h = len(lines) * line_h
    ty = (height - total_h) // 2
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=f_big)
        tw = bbox[2] - bbox[0]
        sx = (width - tw) // 2
        draw.text((sx + 3, ty + 3), ln, fill=(0, 0, 0), font=f_big)
        draw.text((sx, ty), ln, fill=cs["text"], font=f_big)
        ty += line_h

    # Bottom ticker
    tick_h = max(22, height // 22)
    draw.rectangle([0, height - tick_h, width, height], fill=cs["accent"])

    img.save(out_path, "JPEG", quality=95)
    return out_path


# ═══════════════════════════════════════════════════════════
# TORN-PAPER TEXT CARD GENERATOR (PIL)
# ═══════════════════════════════════════════════════════════

def generate_torn_paper_card(text, width, height, font_path, out_path, seed=0,
                              font_size=None, text_color=None,
                              word_colors=None, card_style=None,
                              skip_text=False):
    """
    Red 'torn paper' headline band — transparent outside silhouette.
    word_colors: dict of {str(word_index): "#rrggbb"} for per-word coloring.
    card_style:  dict with bgr0, bgr1, edge_h, jag, vsid, vcor, vwid overrides.
    Returns (out_path, font_size_used).
    """
    cs = card_style or {}
    # Gradient colors — new hex params; fall back to legacy bgr0/bgr1 (red-only sliders)
    _c0_hex = cs.get("card_c0", "")
    _c1_hex = cs.get("card_c1", "")
    if _c0_hex:
        _r0, _g0, _b0 = _hex_to_rgb(_c0_hex)
    else:
        bgr0 = int(cs.get("bgr0", 193))
        _r0, _g0, _b0 = bgr0, 0, 0
    if _c1_hex:
        _r1, _g1, _b1 = _hex_to_rgb(_c1_hex)
    else:
        bgr1 = int(cs.get("bgr1", 128))
        _r1, _g1, _b1 = bgr1, 0, 0
    # JS sliders store jag as 0-100, vsid as 0-80, vcor as 0-100 — divide by 100 to get 0-1
    jag_mult = float(cs.get("jag", 60)) / 100.0   # jaggedness multiplier (JS stores 10-100)
    vsid = float(cs.get("vsid", 35)) / 100.0       # vignette side strength (JS stores 0-80)
    vcor = float(cs.get("vcor", 72)) / 100.0       # vignette corner strength (JS stores 0-100)
    vwid = int(cs.get("vwid", int(width * 74 / 1080)))  # side vignette width px
    if "seed" in cs:
        seed = int(cs["seed"])

    _rnd = random.Random(seed + 7)

    _edge_override = cs.get("edge")
    EDGE = max(3, int(_edge_override)) if _edge_override else max(3, int(height * 4 / 322))
    MARGIN_X = max(32, width // 28)

    def _torn_pts(y_base, rseed):
        rng = random.Random(rseed)
        pts = [(0, y_base)]
        x = 0
        while x < width:
            noise = rng.uniform(-1.0, 1.0) * EDGE * jag_mult
            wave = (EDGE * 0.25) * math.sin(x * 0.055)
            jag_y = max(0, min(height - 1, int(y_base + noise + wave)))
            pts.append((x, jag_y))
            x += rng.randint(4, 28)
        pts.append((width, y_base))
        return pts

    top_pts = _torn_pts(EDGE, seed + 7)
    bot_pts = _torn_pts(height - EDGE, seed + 13)

    # Alpha mask — opaque only inside torn silhouette
    import numpy as np
    silhouette = top_pts + list(reversed(bot_pts))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(silhouette, fill=255)

    # Card gradient top-color → bottom-color (any color, not just red)
    grad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grad)
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(_r0 + t * (_r1 - _r0))
        g = int(_g0 + t * (_g1 - _g0))
        b = int(_b0 + t * (_b1 - _b0))
        gdraw.line([(0, y), (width - 1, y)], fill=(r, g, b, 255))

    r_ch, g_ch, b_ch, _ = grad.split()
    img = Image.merge("RGBA", (r_ch, g_ch, b_ch, mask))

    # Vignette — side + corner, controlled by vsid/vcor/vwid
    # Side: linear left/right gradients; Corner: radial gradient matching SVG vigCornerGrad
    # SVG vigCornerGrad: cx=50%, cy=50%, r=70%, stop transparent@30% → opaque@100%
    vig = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vig)
    VIG_W = max(20, vwid)
    side_max_a = int(vsid * 255)
    cor_max_a  = int(vcor * 255)

    # Side vignette: left/right linear gradient
    for x in range(VIG_W):
        a = int(side_max_a * (1.0 - x / VIG_W) ** 1.6)
        vdraw.line([(x, 0), (x, height - 1)], fill=(0, 0, 0, a))
        vdraw.line([(width - 1 - x, 0), (width - 1 - x, height - 1)], fill=(0, 0, 0, a))

    # Radial vignette: matches SVG vigCornerGrad objectBoundingBox radial gradient
    # dist_bbox = sqrt((x/W - 0.5)^2 + (y/H - 0.5)^2) in normalized bbox coords
    # gradient: transparent at offset 30% (dist = 0.30*r), opaque at offset 100% (dist = r=0.70)
    ys_arr, xs_arr = np.mgrid[0:height, 0:width].astype(np.float32)
    xs_norm = xs_arr / width - 0.5
    ys_norm = ys_arr / height - 0.5
    dist_bbox = np.sqrt(xs_norm ** 2 + ys_norm ** 2)
    r_bbox = 0.70
    t_rad = np.clip((dist_bbox / r_bbox - 0.30) / 0.70, 0.0, 1.0)
    radial_a = (t_rad * cor_max_a).astype(np.uint8)

    # Combine: max of side + radial alpha per pixel
    vig_arr = np.array(vig)
    vig_arr[:, :, 3] = np.maximum(vig_arr[:, :, 3], radial_a)
    vig = Image.fromarray(vig_arr, 'RGBA')

    vr, vg, vb, va = vig.split()
    va = Image.fromarray(np.minimum(np.array(va), np.array(mask)))
    vig = Image.merge("RGBA", (vr, vg, vb, va))
    img = Image.alpha_composite(img, vig)
    draw = ImageDraw.Draw(img)

    # White torn edge lines
    draw.line(top_pts, fill=(255, 255, 255, 255), width=6)
    draw.line(bot_pts, fill=(255, 255, 255, 255), width=6)

    # Auto-fit text
    inner_y = EDGE + 8
    inner_h = height - 2 * EDGE - 16
    text_w_max = width - 2 * MARGIN_X
    words = str(text).split()

    def _load_font(sz):
        try:
            if font_path and os.path.exists(font_path):
                return _truetype(font_path, sz)
        except Exception:
            pass
        return _best_font(sz)

    if font_size and int(font_size) > 0:
        # Use the specified font size — just wrap at that size
        fnt_t = _load_font(int(font_size))
        wrapped, line = [], []
        for w_ in words:
            test = " ".join(line + [w_])
            bbox = draw.textbbox((0, 0), test, font=fnt_t)
            if (bbox[2] - bbox[0]) > text_w_max and line:
                wrapped.append(" ".join(line))
                line = [w_]
            else:
                line.append(w_)
        if line:
            wrapped.append(" ".join(line))
        best_fs, best_lines = int(font_size), (wrapped or [str(text)])
    else:
        best_fs, best_lines = 18, [text]
        for fs in range(min(110, inner_h - 8), 17, -2):
            fnt = _load_font(fs)
            wrapped, line = [], []
            for w_ in words:
                test = " ".join(line + [w_])
                bbox = draw.textbbox((0, 0), test, font=fnt)
                if (bbox[2] - bbox[0]) > text_w_max and line:
                    wrapped.append(" ".join(line))
                    line = [w_]
                else:
                    line.append(w_)
            if line:
                wrapped.append(" ".join(line))

            GAP = 8
            asc, desc = fnt.getmetrics() if hasattr(fnt, "getmetrics") else (fs, fs // 4)
            line_h = asc + desc + GAP
            total = line_h * len(wrapped)
            if total <= inner_h:
                best_fs = fs
                best_lines = wrapped
                break

    fnt = _load_font(best_fs)
    asc, desc = fnt.getmetrics() if hasattr(fnt, "getmetrics") else (best_fs, best_fs // 4)
    line_h = asc + desc + 8
    total = line_h * len(best_lines)
    cy = inner_y + max(0, (inner_h - total) // 2)
    cy_start = int(cy)  # first line top (used by FFmpeg drawtext caller)

    # Parse text color (hex like "#ffffff" or "#ff0000")
    text_fill = (255, 255, 255, 255)
    if text_color:
        try:
            tc = text_color.lstrip("#")
            text_fill = (int(tc[0:2], 16), int(tc[2:4], 16), int(tc[4:6], 16), 255)
        except Exception:
            pass

    def _hex_to_rgba(h):
        try:
            h = h.lstrip("#")
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
        except Exception:
            return text_fill

    if skip_text:
        img.save(out_path, "PNG")
        return out_path, best_fs, best_lines, cy_start, line_h

    # ── Render text via Playwright (HarfBuzz shaping) → PIL fallback ──
    _tc_hex = text_color if (text_color and text_color.startswith('#')) else '#ffffff'
    _wc = word_colors or {}
    _txt_area_w = width - 2 * MARGIN_X
    _txt_area_h = inner_h
    _txt_tmp = out_path + '_txt.png'
    _pw_ok = _playwright_render_title(
        best_lines, _tc_hex, best_fs, font_path,
        _txt_area_w, _txt_area_h, _txt_tmp,
        word_colors=(_wc if _wc else None)
    )
    if _pw_ok and os.path.exists(_txt_tmp):
        _txt_img = Image.open(_txt_tmp).convert('RGBA')
        img.paste(_txt_img, (MARGIN_X, inner_y), _txt_img)
        try:
            os.unlink(_txt_tmp)
        except Exception:
            pass
    else:
        # PIL fallback
        global_word_idx = 0
        for ln in best_lines:
            ln_words = ln.split()
            bbox = draw.textbbox((0, 0), ln, font=fnt)
            tw = bbox[2] - bbox[0]
            lx = (width - tw) // 2
            if _wc:
                wx = lx
                for word in ln_words:
                    col = _hex_to_rgba(_wc[str(global_word_idx)]) if str(global_word_idx) in _wc else text_fill
                    wb = draw.textbbox((0, 0), word, font=fnt)
                    ww = wb[2] - wb[0]
                    draw.text((wx + 2, cy + 2), word, fill=(0, 0, 0, 180), font=fnt)
                    draw.text((wx, cy), word, fill=col, font=fnt)
                    sp_b = draw.textbbox((0, 0), " ", font=fnt)
                    wx += ww + (sp_b[2] - sp_b[0])
                    global_word_idx += 1
            else:
                draw.text((lx + 2, cy + 2), ln, fill=(0, 0, 0, 180), font=fnt)
                draw.text((lx, cy), ln, fill=text_fill, font=fnt)
                global_word_idx += len(ln_words)
            cy += line_h

    img.save(out_path, "PNG")
    return out_path, best_fs


# ═══════════════════════════════════════════════════════════
# POST-COMPOSE HELPERS  (duration enforcement + QA gate)
# ═══════════════════════════════════════════════════════════

# Platforms whose output must never exceed 180 s (hard platform limit).
_SHORT_FORM_PLATFORMS = {'youtube_short', 'instagram_reel', 'tiktok'}
_SHORT_FORM_MAX_S     = 179.5   # trim target (leave 0.5 s headroom)


def _enforce_duration(out_path: str, platform: str) -> None:
    """Stream-copy trim *out_path* to _SHORT_FORM_MAX_S if needed.

    Only acts on short-form platforms (youtube_short, instagram_reel, tiktok).
    Uses FFmpeg stream-copy (-c copy) — no re-encode.  The file is replaced
    in-place via a temporary sidecar and atomic rename.

    Parameters
    ----------
    out_path : str
        Path to the composed output file.  Modified in place if trimmed.
    platform : str
        Platform key.  Trim is skipped for platforms not in _SHORT_FORM_PLATFORMS.
    """
    import logging as _logging
    _log = _logging.getLogger("kaizer.pipeline.validator")

    if platform not in _SHORT_FORM_PLATFORMS:
        return

    # Measure actual duration via ffprobe
    try:
        result = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-print_format", "json",
             "-show_format", out_path],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout)
        dur = float(info.get("format", {}).get("duration") or 0.0)
    except Exception as exc:
        _log.warning("_enforce_duration: could not probe %s: %s", out_path, exc)
        return

    if dur <= _SHORT_FORM_MAX_S:
        return  # nothing to do

    _log.warning(
        "Output %s duration %.2fs exceeds %.1fs for platform %r — "
        "stream-copy trimming to %.1fs.",
        out_path, dur, _SHORT_FORM_MAX_S + 0.5, platform, _SHORT_FORM_MAX_S,
    )

    tmp_path = out_path + ".trim_tmp.mp4"
    try:
        trim_cmd = [
            FFMPEG_BIN, "-y",
            "-i", out_path,
            "-t", str(_SHORT_FORM_MAX_S),
            "-c", "copy",
            "-movflags", "+faststart",
            tmp_path,
        ]
        subprocess.run(trim_cmd, capture_output=True, check=True, timeout=120)
        os.replace(tmp_path, out_path)
        _log.info("Duration trim complete: %s → %.1fs", out_path, _SHORT_FORM_MAX_S)
    except Exception as exc:
        _log.error("_enforce_duration trim failed for %s: %s", out_path, exc)
        # Clean up temp file if it exists; leave original untouched.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _run_output_qa(out_path: str, platform: str,
                   expected_duration_s: float | None = None) -> None:
    """Run the QA gate on a composed output file — *advisory only*.

    QA findings are logged (warnings + errors) so users see them in the
    job log, but the pipeline never aborts on them. Real-world inputs
    have unpredictable audio levels, codecs, frame rates — a clip with
    a slightly hot true peak or an off-spec bitrate is still a usable
    deliverable. Killing the entire pipeline because of an advisory
    quality check is the wrong default.

    Defence in depth: ENCODE_ARGS_SHORT_FORM already chains alimiter
    after loudnorm to clamp peaks before they reach the muxer; the QA
    gate is the safety net that *reports* if something slipped through.

    Parameters
    ----------
    out_path : str
        Path to the composed output file.
    platform : str
        Platform key passed to validate_output().
    expected_duration_s : float | None
        If provided, QA checks that actual duration is within ±0.5 s.
    """
    import logging as _logging
    _log = _logging.getLogger("kaizer.pipeline.qa")

    try:
        from pipeline_core.qa import validate_output  # type: ignore
    except ImportError:
        try:
            from qa import validate_output  # type: ignore
        except ImportError:
            _log.warning(
                "_run_output_qa: qa module not importable — QA check skipped for %s",
                out_path,
            )
            return

    try:
        qa_result = validate_output(
            out_path,
            platform=platform,
            expected_duration_s=expected_duration_s,
        )
    except Exception as exc:
        # QA itself crashed (e.g. ffprobe failed). Don't take the
        # pipeline down with it — log and move on.
        _log.warning("QA validation crashed for %s: %s", out_path, exc)
        return

    if qa_result.warnings:
        for w in qa_result.warnings:
            _log.warning("QA warning [%s]: %s", out_path, w)
    if qa_result.errors:
        for e in qa_result.errors:
            _log.warning("QA advisory [%s]: %s", out_path, e)
        print(f"[QA advisory] {out_path}: {len(qa_result.errors)} issue(s) — "
              "delivering clip anyway. See log for details.")


# ═══════════════════════════════════════════════════════════
# COMPOSE CLIP — KAIZER_NEWS Layout (Video + Text + Image)
# ═══════════════════════════════════════════════════════════

def compose_clip(raw_clip_path, image_path, title_text, out_path, preset,
                 font_size=None, text_color=None, font_file=None, section_pct=None,
                 word_colors=None, card_style=None, platform: str = 'youtube_short'):
    """
    Compose final broadcast clip with KAIZER_NEWS 3-section layout:
      Top:    Video (49.5%)
      Middle: Torn-paper text card
      Bottom: Image (33.7%)
    """
    w, h = preset["width"], preset["height"]
    dur_info = get_video_info(raw_clip_path)
    dur = dur_info["duration"] if dur_info else 30.0

    _sp = section_pct or {}
    VIDEO_H = _e(int(h * _sp.get("video", 0.4619)))
    IMAGE_H = _e(int(h * _sp.get("image", 0.3690)))
    TEXT_H = max(20, h - VIDEO_H - IMAGE_H)

    # Overlap: card extends ov px into video above and image below (default 20)
    _cs = card_style or {}
    OV = max(0, int(_cs.get("overlap", 20)))

    clip_dir = os.path.dirname(out_path)
    clip_name = os.path.splitext(os.path.basename(out_path))[0]

    # Resolve image — fall back to a generated text card if the stock image
    # isn't available (no global branded-logo fallback).
    if not image_path or not os.path.exists(str(image_path)):
        image_path = os.path.join(clip_dir, f"_card_{clip_name}.jpg")
        generate_news_card(
            _ascii_text(title_text or "KAIZER NEWS", 40),
            image_path, w, IMAGE_H,
        )

    # Prep image to exact dimensions
    try:
        pimg = Image.open(image_path).convert("RGB")
        pimg = pimg.resize((w, IMAGE_H), Image.LANCZOS)
        prepped_img = os.path.join(clip_dir, f"_img_{clip_name}.jpg")
        pimg.save(prepped_img, "JPEG", quality=95)
        image_path = prepped_img
    except Exception:
        pass

    # Generate torn-paper text card
    _ff = font_file or "NotoSansTelugu-Bold.ttf"
    tel_font = os.path.join(FONTS_DIR, _ff)
    if not os.path.exists(tel_font):
        tel_font = os.path.join(FONTS_DIR, "Ponnala-Regular.ttf")
    if not os.path.exists(tel_font):
        tel_font = os.path.join(FONTS_DIR, "NotoSansTelugu-Bold.ttf")
    if not os.path.exists(tel_font):
        tel_font = os.path.join(FONTS_DIR, "Roboto-Bold.ttf")
    card_path = os.path.join(clip_dir, f"_txtcard_{clip_name}.png")
    _card_seed = int(_cs["seed"]) if "seed" in _cs else hash(out_path) % 100
    # Card is rendered taller by 2*OV so it bleeds into video above and image below
    CARD_H = TEXT_H + 2 * OV
    # Always render text via Playwright (HarfBuzz shaping) — FFmpeg drawtext
    # doesn't shape Telugu correctly on Windows (no HarfBuzz in Windows FFmpeg builds)
    _use_ffmpeg_text = False
    card_result = generate_torn_paper_card(
        title_text or "KAIZER NEWS",
        width=w, height=CARD_H,
        font_path=tel_font, out_path=card_path,
        seed=_card_seed,
        font_size=font_size,
        text_color=text_color,
        word_colors=word_colors,
        card_style=card_style,
        skip_text=False,
    )
    card_font_size = card_result[1] if isinstance(card_result, tuple) else 18

    img_y = VIDEO_H + TEXT_H   # image position unchanged
    card_y = VIDEO_H - OV      # card placed OV pixels into the video section

    # Build drawtext filters for card title text (HarfBuzz shaping = matches browser)
    # Only used when skip_text=True (no per-word coloring) — Playwright handles it otherwise
    _card_drawtext = []
    if _use_ffmpeg_text and len(card_result) >= 5:
        _best_lines, _cy_start, _lh = card_result[2], card_result[3], card_result[4]
        _tc = (text_color or '#ffffff').lstrip('#')

        def _esc(s):
            return (s.replace('\\', '\\\\')
                      .replace("'", "\\'")
                      .replace(':', '\\:'))

        _fnt_esc = _esc(tel_font.replace('\\', '/'))
        _fnt_sz  = card_font_size
        for i, ln in enumerate(_best_lines or []):
            ly = card_y + _cy_start + i * int(_lh)
            te = _esc(ln)
            _card_drawtext.append(
                f"drawtext=fontfile='{_fnt_esc}':text='{te}':fontsize={_fnt_sz}"
                f":fontcolor=000000@0.7:x=(w-text_w)/2+2:y={ly+2}")
            _card_drawtext.append(
                f"drawtext=fontfile='{_fnt_esc}':text='{te}':fontsize={_fnt_sz}"
                f":fontcolor={_tc}:x=(w-text_w)/2:y={ly}")

    # Logo precedence: per-clip card_style override → per-job default (from
    # channel config) → NO logo at all.  There is deliberately NO global
    # fallback — users opt in by setting a Channel.logo in the Style Profiles
    # page.  The previous Kaizer-branded default was a bug for a SaaS.
    _vlogo = (_cs.get("video_logo", "") or "") if _cs else ""
    logo_path = (_vlogo if (_vlogo and os.path.exists(_vlogo))
                 else (DEFAULT_LOGO or None))
    LOGO_W = _e(int(w * 160 / 1080))
    LOGO_H = _e(int(h * 134 / 1920))
    LOGO_MR = int(w * 24 / 1080)
    LOGO_MT = int(h * 25 / 1920)

    # Build FFmpeg command
    base = ["-i", raw_clip_path]

    def ii(*paths):
        args = []
        for p in paths:
            args += ["-loop", "1", "-framerate", "30", "-i", p]
        return args

    # Final output label: 'outv' if no text to add, 'outv_pre' if we chain drawtext after
    _final_lbl = 'outv_pre' if _card_drawtext else 'outv'

    if logo_path:
        extra = ii(image_path, card_path, logo_path)
        fc = [
            f"{_sc(0, w, VIDEO_H)}[vid]",
            f"{_sc(1, w, IMAGE_H)}[img]",
            f"[vid]pad={w}:{h}:0:0:black[vid_pad]",
            f"[vid_pad][img]overlay=x=0:y={img_y}[bg]",
            f"[3:v]scale={LOGO_W}:{LOGO_H}:force_original_aspect_ratio=decrease,"
            f"pad={LOGO_W}:{LOGO_H}:(ow-iw)/2:(oh-ih)/2:color=black@0[logo]",
            f"[bg][logo]overlay=x={w - LOGO_W - LOGO_MR}:y={LOGO_MT}[bgl]",
            f"[2:v]scale={w}:{CARD_H}:force_original_aspect_ratio=disable,setsar=1[card]",
            f"[bgl][card]overlay=x=0:y={card_y}:format=auto[{_final_lbl}]",
        ]
    else:
        extra = ii(image_path, card_path)
        fc = [
            f"{_sc(0, w, VIDEO_H)}[vid]",
            f"{_sc(1, w, IMAGE_H)}[img]",
            f"[vid]pad={w}:{h}:0:0:black[vid_pad]",
            f"[vid_pad][img]overlay=x=0:y={img_y}[bg]",
            f"[2:v]scale={w}:{CARD_H}:force_original_aspect_ratio=disable,setsar=1[card]",
            f"[bg][card]overlay=x=0:y={card_y}:format=auto[{_final_lbl}]",
        ]

    # Chain HarfBuzz-shaped title text drawtext filters as final step
    if _card_drawtext:
        fc.append(f"[outv_pre]" + ','.join(_card_drawtext) + "[outv]")

    cmd = ([FFMPEG_BIN, "-y"] + base + extra +
           ["-filter_complex", ";".join(fc),
            "-map", "[outv]", "-map", "0:a?"]
           + ENCODE_ARGS_SHORT_FORM
           + ["-shortest", out_path])

    run_ffmpeg(cmd)
    _enforce_duration(out_path, platform)
    _run_output_qa(out_path, platform)
    return {
        "font_size": card_font_size,
        "font_file": os.path.basename(tel_font),
        "card_path": card_path,
        "image_path": image_path,
    }


# ═══════════════════════════════════════════════════════════
# COMPOSE CLIP — CLEAN CARD Layout (no torn edge, framed image)
# Video on top half, solid colour band below holding the headline
# (top of band) and a centred white-bordered image (bottom of band).
# ═══════════════════════════════════════════════════════════

def compose_clip_clean_card(
    raw_clip_path, image_path, title_text, out_path, preset,
    font_size=None, text_color=None, font_file=None,
    word_colors=None,
    bg_color="#c10000",
    video_pct=0.50,
    headline_pct=0.18,
    image_h_pct=0.30,
    image_w_pct=0.80,
    image_border_px=14,
    image_border_color="#ffffff",
    platform: str = 'youtube_short',
):
    """
    Clean-card layout — same 3-block structure as torn_card but with
    straight edges and a polaroid-style frame around the bottom image.

      ┌────────────────────────────────┐
      │           VIDEO  (top)         │  video_pct (default 50%)
      │                                │
      ├────────────────────────────────┤
      │   HEADLINE TEXT (white +       │  headline_pct of band
      │   yellow-highlighted words)    │
      │                                │
      │     ┌──────────────────┐       │
      │     │  FRAMED IMAGE     │      │  image fills image_w_pct,
      │     │  (white border)   │      │  height = image_h_pct
      │     └──────────────────┘       │
      └────────────────────────────────┘
            (solid bg_color band)

    Border on the image is drawn in PIL before overlay so FFmpeg only
    sees a single rectangular RGB asset — keeps the filtergraph short
    and the QA pass simple.
    """
    w, h = preset["width"], preset["height"]

    VIDEO_H   = _e(int(h * float(video_pct)))
    HEADLINE_H = _e(int(h * float(headline_pct)))
    IMAGE_H   = _e(int(h * float(image_h_pct)))
    IMAGE_W   = _e(int(w * float(image_w_pct)))

    clip_dir = os.path.dirname(out_path)
    clip_name = os.path.splitext(os.path.basename(out_path))[0]

    # ── Solid background band (full frame, the video pads on top) ──
    r, g, b = _hex_to_rgb(bg_color)
    bg_img = Image.new("RGB", (w, h), (r, g, b))
    bg_path = os.path.join(clip_dir, f"_clean_bg_{clip_name}.png")
    bg_img.save(bg_path, "PNG")

    # ── Framed image (centred, with white polaroid border) ───────
    # Falls back to a generated text card when the news image isn't
    # available — same fallback path compose_clip uses.
    if not image_path or not os.path.exists(str(image_path)):
        _gen_card = os.path.join(clip_dir, f"_clean_card_{clip_name}.jpg")
        generate_news_card(
            _ascii_text(title_text or "KAIZER NEWS", 40),
            _gen_card, IMAGE_W, IMAGE_H,
        )
        image_path = _gen_card

    bord = max(0, int(image_border_px))
    inner_w = max(8, IMAGE_W - 2 * bord)
    inner_h = max(8, IMAGE_H - 2 * bord)
    bcr, bcg, bcb = _hex_to_rgb(image_border_color)
    framed = Image.new("RGB", (IMAGE_W, IMAGE_H), (bcr, bcg, bcb))
    try:
        inner = Image.open(image_path).convert("RGB").resize(
            (inner_w, inner_h), Image.LANCZOS,
        )
        framed.paste(inner, (bord, bord))
    except Exception:
        # Image load failed — leave the white plate so FFmpeg still
        # has a valid input. Caller sees an obviously-wrong frame
        # and can replace the image in the editor.
        pass
    framed_path = os.path.join(clip_dir, f"_clean_img_{clip_name}.jpg")
    framed.save(framed_path, "JPEG", quality=95)

    # ── Headline PNG via Playwright (HarfBuzz Telugu shaping) ────
    _ff = font_file or "NotoSansTelugu-Bold.ttf"
    tel_font = os.path.join(FONTS_DIR, _ff)
    for _fb in ("NotoSansTelugu-Bold.ttf", "Roboto-Bold.ttf"):
        if not os.path.exists(tel_font):
            tel_font = os.path.join(FONTS_DIR, _fb)
    headline_path = os.path.join(clip_dir, f"_clean_hl_{clip_name}.png")
    HEADLINE_MARGIN = max(40, w // 24)
    text_area_w = w - 2 * HEADLINE_MARGIN
    # Auto-size: aim for 2 lines max — pick the largest size that fits
    # in (text_area_w, HEADLINE_H) without overflow. compose_clip uses
    # a similar shrink-loop via PIL.textbbox; we reuse it here.
    words = str(title_text or "KAIZER NEWS").split()

    def _fit_size():
        from PIL import ImageDraw as _ID, Image as _IM
        probe = _IM.new("RGB", (8, 8))
        d = _ID.Draw(probe)
        for fs in range(min(120, HEADLINE_H), 24, -2):
            try:
                fnt = _truetype(tel_font, fs) if (tel_font and os.path.exists(tel_font)) else _best_font(fs)
            except Exception:
                fnt = _best_font(fs)
            wrapped, line = [], []
            for w_ in words:
                test = " ".join(line + [w_])
                bb = d.textbbox((0, 0), test, font=fnt)
                if (bb[2] - bb[0]) > text_area_w and line:
                    wrapped.append(" ".join(line)); line = [w_]
                else:
                    line.append(w_)
            if line:
                wrapped.append(" ".join(line))
            asc, desc = fnt.getmetrics() if hasattr(fnt, "getmetrics") else (fs, fs // 4)
            if (asc + desc + 8) * len(wrapped) <= HEADLINE_H - 8:
                return fs, wrapped
        return 28, [str(title_text or "")]

    fs_used, wrapped_lines = (int(font_size), None) if font_size else _fit_size()
    if wrapped_lines is None:
        # Caller pinned font_size — re-wrap at that size.
        from PIL import ImageDraw as _ID2, Image as _IM2
        probe2 = _IM2.new("RGB", (8, 8))
        d2 = _ID2.Draw(probe2)
        try:
            fnt2 = _truetype(tel_font, fs_used) if (tel_font and os.path.exists(tel_font)) else _best_font(fs_used)
        except Exception:
            fnt2 = _best_font(fs_used)
        wrapped_lines = []
        line = []
        for w_ in words:
            test = " ".join(line + [w_])
            bb = d2.textbbox((0, 0), test, font=fnt2)
            if (bb[2] - bb[0]) > text_area_w and line:
                wrapped_lines.append(" ".join(line)); line = [w_]
            else:
                line.append(w_)
        if line:
            wrapped_lines.append(" ".join(line))

    _tc_hex = text_color if (text_color and str(text_color).startswith("#")) else "#ffffff"
    _pw_ok = _playwright_render_title(
        wrapped_lines, _tc_hex, fs_used, tel_font,
        text_area_w, HEADLINE_H, headline_path,
        word_colors=(word_colors or None),
    )
    if not (_pw_ok and os.path.exists(headline_path)):
        # PIL fallback — render the text onto a transparent PNG.
        from PIL import ImageDraw as _IDF, Image as _IMF
        hl = _IMF.new("RGBA", (text_area_w, HEADLINE_H), (0, 0, 0, 0))
        d3 = _IDF.Draw(hl)
        try:
            fnt3 = _truetype(tel_font, fs_used) if (tel_font and os.path.exists(tel_font)) else _best_font(fs_used)
        except Exception:
            fnt3 = _best_font(fs_used)
        asc, desc = fnt3.getmetrics() if hasattr(fnt3, "getmetrics") else (fs_used, fs_used // 4)
        line_h = asc + desc + 8
        cy = max(0, (HEADLINE_H - line_h * len(wrapped_lines)) // 2)
        try:
            tr, tg, tb = _hex_to_rgb(_tc_hex); fill = (tr, tg, tb, 255)
        except Exception:
            fill = (255, 255, 255, 255)
        for ln in wrapped_lines:
            bb = d3.textbbox((0, 0), ln, font=fnt3)
            lx = (text_area_w - (bb[2] - bb[0])) // 2
            d3.text((lx + 2, cy + 2), ln, fill=(0, 0, 0, 180), font=fnt3)
            d3.text((lx, cy), ln, fill=fill, font=fnt3)
            cy += line_h
        hl.save(headline_path, "PNG")

    # ── Geometry ────────────────────────────────────────────────
    band_y       = VIDEO_H
    headline_y   = band_y + max(0, int((h - band_y - HEADLINE_H - IMAGE_H) * 0.35))
    image_x      = (w - IMAGE_W) // 2
    image_y      = headline_y + HEADLINE_H + max(0, int((h - headline_y - HEADLINE_H - IMAGE_H) * 0.5))
    # Clamp so nothing leaks past the bottom edge
    image_y      = min(image_y, h - IMAGE_H - 16)

    # Logo precedence — same rule as compose_clip
    _vlogo = ""  # caller can extend this via card_style later if needed
    logo_path = (DEFAULT_LOGO or None)
    LOGO_W = _e(int(w * 160 / 1080))
    LOGO_H = _e(int(h * 134 / 1920))
    LOGO_MR = int(w * 24 / 1080)
    LOGO_MT = int(h * 25 / 1920)

    def ii(*paths):
        args = []
        for p in paths:
            args += ["-loop", "1", "-framerate", "30", "-i", p]
        return args

    base = ["-i", raw_clip_path]

    if logo_path and os.path.exists(str(logo_path)):
        extra = ii(bg_path, framed_path, headline_path, logo_path)
        fc = [
            f"{_sc(0, w, VIDEO_H)}[vid]",
            f"[1:v]scale={w}:{h},setsar=1[bg]",
            f"[bg][vid]overlay=x=0:y=0[bv]",
            f"[2:v]scale={IMAGE_W}:{IMAGE_H},setsar=1[img]",
            f"[bv][img]overlay=x={image_x}:y={image_y}[bvi]",
            f"[3:v]scale={text_area_w}:{HEADLINE_H},setsar=1[hl]",
            f"[bvi][hl]overlay=x={HEADLINE_MARGIN}:y={headline_y}[bgh]",
            f"[4:v]scale={LOGO_W}:{LOGO_H}:force_original_aspect_ratio=decrease,"
            f"pad={LOGO_W}:{LOGO_H}:(ow-iw)/2:(oh-ih)/2:color=black@0[logo]",
            f"[bgh][logo]overlay=x={w - LOGO_W - LOGO_MR}:y={LOGO_MT}[outv]",
        ]
    else:
        extra = ii(bg_path, framed_path, headline_path)
        fc = [
            f"{_sc(0, w, VIDEO_H)}[vid]",
            f"[1:v]scale={w}:{h},setsar=1[bg]",
            f"[bg][vid]overlay=x=0:y=0[bv]",
            f"[2:v]scale={IMAGE_W}:{IMAGE_H},setsar=1[img]",
            f"[bv][img]overlay=x={image_x}:y={image_y}[bvi]",
            f"[3:v]scale={text_area_w}:{HEADLINE_H},setsar=1[hl]",
            f"[bvi][hl]overlay=x={HEADLINE_MARGIN}:y={headline_y}[outv]",
        ]

    cmd = ([FFMPEG_BIN, "-y"] + base + extra +
           ["-filter_complex", ";".join(fc),
            "-map", "[outv]", "-map", "0:a?"]
           + ENCODE_ARGS_SHORT_FORM
           + ["-shortest", out_path])

    run_ffmpeg(cmd)
    _enforce_duration(out_path, platform)
    _run_output_qa(out_path, platform)
    return {
        "font_size":  fs_used,
        "font_file":  os.path.basename(tel_font),
        "bg_color":   bg_color,
        "image_path": image_path,
        "framed_image_path": framed_path,
        "headline_path":   headline_path,
    }


# ═══════════════════════════════════════════════════════════
# COMPOSE CLIP — SPLIT FRAME Layout (Thumbnail top + Video bottom)
# ═══════════════════════════════════════════════════════════

def _make_velvet_bg(width, height, fbar_y, velvet_style=None):
    """
    Build the follow bar background (PIL RGB image) with:
      - Purple gradient top → bottom for the full frame
      - FBM domain-warped velvet texture overlaid on the follow bar zone
      - Dot grid accents (top-right + bottom-left in bar zone)

    velvet_style dict keys match the frame_followbar_design.html config keys:
      c-top, c-bot, c-vdark, c-vlight, patch-scale, octaves,
      contrast, brightness, warp, warp-scale, grain, edge-dark,
      c-dot, dot-op, dot-r, dot-sp, dot-rows, dot-cols
    """
    import numpy as np

    vs = velvet_style or {}
    def _hrgb(h, default):
        try: return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        except Exception: return default

    # ── Config (from velvet_style or reference-preset defaults) ──
    BG_TOP   = _hrgb(vs.get('c-top',   '#2d0a4e'), (0x2d, 0x0a, 0x4e))
    BG_BOT   = _hrgb(vs.get('c-bot',   '#1a0a2e'), (0x1a, 0x0a, 0x2e))
    V_DARK   = _hrgb(vs.get('c-vdark', '#0a001a'), (0x0a, 0x00, 0x1a))
    V_LIGHT  = _hrgb(vs.get('c-vlight','#3d0060'), (0x3d, 0x00, 0x60))
    PATCH_SC_BASE = int(vs.get('patch-scale', 80))
    OCTAVES  = int(vs.get('octaves', 5))
    CON      = float(vs.get('contrast',  107)) / 100.0
    BRI      = (float(vs.get('brightness', 35)) - 35) / 100.0
    WARP_AMT_BASE  = float(vs.get('warp',       55))
    WARP_SC_BASE   = float(vs.get('warp-scale', 65))
    GRAIN    = float(vs.get('grain',     14))
    EDGE_D   = float(vs.get('edge-dark', 33)) / 100.0
    DOT_COL  = _hrgb(vs.get('c-dot', '#7b3fb8'), (123, 63, 184))
    DOT_OP   = int(float(vs.get('dot-op', 38)) / 100.0 * 255)
    DOT_R_BASE  = float(vs.get('dot-r',  5))
    DOT_SP_BASE = float(vs.get('dot-sp', 18))
    DOT_ROWS = int(vs.get('dot-rows', 5))
    DOT_COLS = int(vs.get('dot-cols', 5))

    sc = width / 360.0
    psc  = PATCH_SC_BASE * sc
    wamt = WARP_AMT_BASE * sc
    wsc  = WARP_SC_BASE  * sc
    dot_r  = DOT_R_BASE  * sc
    dot_sp = DOT_SP_BASE * sc

    # ── Noise table (seeded, reproducible) ──
    NS = 512
    rng = np.random.default_rng(seed=987654321)
    NT = rng.random((NS, NS), dtype=np.float32)

    def vnoise(xs, ys):
        xi = np.mod(xs, NS).astype(np.float32)
        yi = np.mod(ys, NS).astype(np.float32)
        x0 = xi.astype(np.int32) % NS
        x1 = (x0 + 1) % NS
        y0 = yi.astype(np.int32) % NS
        y1 = (y0 + 1) % NS
        fx = xi - x0; fy = yi - y0
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        return (NT[y0, x0] * (1 - fx) * (1 - fy) +
                NT[y0, x1] * fx * (1 - fy) +
                NT[y1, x0] * (1 - fx) * fy +
                NT[y1, x1] * fx * fy)

    def warped_fbm(px2d, py2d):
        wx = vnoise(px2d / wsc + 3.7, py2d / wsc + 9.1) * wamt - wamt * 0.5
        wy = vnoise(px2d / wsc + 8.3, py2d / wsc + 2.4) * wamt - wamt * 0.5
        v = np.zeros_like(px2d, dtype=np.float32)
        amp, freq, mx = 0.55, 1.0 / psc, 0.0
        qx, qy = px2d + wx, py2d + wy
        for _ in range(OCTAVES):
            v  += vnoise(qx * freq, qy * freq) * amp
            mx += amp; amp *= 0.48; freq *= 2.1
        return v / mx

    # ── Gradient background ──
    bg_arr = np.zeros((height, width, 3), dtype=np.uint8)
    t = (np.arange(height, dtype=np.float32) / max(height - 1, 1))[:, np.newaxis]
    for ch, (top_c, bot_c) in enumerate(zip(BG_TOP, BG_BOT)):
        bg_arr[:, :, ch] = np.clip(top_c + (bot_c - top_c) * t, 0, 255).astype(np.uint8)

    # ── Velvet on follow bar zone ──
    fbar_h = height - fbar_y
    if fbar_h > 0:
        px2d, py2d = np.meshgrid(np.arange(width,         dtype=np.float32),
                                  np.arange(fbar_y, height, dtype=np.float32))
        n = warped_fbm(px2d, py2d)
        n = (n - 0.5) * CON + 0.5 + BRI
        n = np.where(n < 0.5, 2 * n * n, 1 - 2 * (1 - n) * (1 - n))
        n = np.clip(n, 0, 1)
        # edge vignette
        ex = (np.arange(width,   dtype=np.float32) / max(width   - 1, 1))[np.newaxis, :]
        ey = (np.arange(fbar_h,  dtype=np.float32) / max(fbar_h  - 1, 1))[:, np.newaxis]
        edge_f = 1 - np.minimum(1, ex * (1 - ex) * ey * (1 - ey) * 22)
        n = np.maximum(0, n - edge_f * EDGE_D * 0.6)
        # deterministic grain via sin hash
        gr_arr = (np.sin(px2d * 127.1 + py2d * 311.7) * 43758.5453 % 1.0 - 0.5) * GRAIN
        for ch, (dk, lt) in enumerate(zip(V_DARK, V_LIGHT)):
            bg_arr[fbar_y:, :, ch] = np.clip(dk + (lt - dk) * n + gr_arr, 0, 255).astype(np.uint8)

    # ── Build PIL image and draw dots ──
    img  = Image.fromarray(bg_arr, 'RGB').convert('RGBA')
    dlyr = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    dd   = ImageDraw.Draw(dlyr)
    dc   = (*DOT_COL, DOT_OP)
    pad  = dot_sp * 0.6
    # top-right dots
    for row in range(DOT_ROWS):
        for col in range(DOT_COLS):
            dx = width - pad - col * dot_sp
            dy = pad + row * dot_sp
            dd.ellipse([(dx - dot_r, dy - dot_r), (dx + dot_r, dy + dot_r)], fill=dc)
    # bottom-left dots clipped to fbar zone
    bp = dot_sp * 0.5
    for row in range(DOT_ROWS):
        for col in range(DOT_COLS):
            dx = bp + col * dot_sp
            dy = height - bp - row * dot_sp
            if dy >= fbar_y:
                dd.ellipse([(dx - dot_r, dy - dot_r), (dx + dot_r, dy + dot_r)], fill=dc)
    img.alpha_composite(dlyr)
    return img.convert('RGB')


def _playwright_render_title(title_lines, text_color, fnt_sz, font_file_path,
                             area_w, area_h, out_png_path, word_colors=None):
    """Render Telugu title text via Playwright headless browser (HarfBuzz shaping).
    Outputs a transparent RGBA PNG.  Returns True on success, False if unavailable.

    word_colors: optional dict {str(word_index): '#rrggbb'} for per-word coloring.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    import base64, html as _html

    try:
        with open(font_file_path, 'rb') as _f:
            _font_b64 = base64.b64encode(_f.read()).decode()
        _font_src = f'data:font/truetype;base64,{_font_b64}'
        _font_face = f'@font-face{{font-family:"TitleFont";src:url("{_font_src}");}}'
        _font_family = '"TitleFont",serif'
    except Exception:
        _font_face = ''
        _font_family = 'serif'

    # Build line HTML — per-word colored spans if word_colors provided
    if word_colors:
        _line_htmls = []
        _wi = 0
        for ln in title_lines:
            _spans = []
            for _w in ln.split():
                _col = word_colors.get(str(_wi), text_color)
                _spans.append(f'<span style="color:{_col}">{_html.escape(_w)}</span>')
                _wi += 1
            _line_htmls.append(' '.join(_spans))
        _title_html = '<br>'.join(_line_htmls)
    else:
        _title_html = '<br>'.join(_html.escape(ln) for ln in title_lines)

    _page_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
{_font_face}
html,body{{width:{area_w}px;height:{area_h}px;margin:0;padding:0;
  overflow:hidden;background:transparent;}}
#t{{width:100%;height:100%;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;
  font-family:{_font_family};font-size:{fnt_sz}px;font-weight:800;
  color:{text_color};line-height:1.3;word-break:break-word;padding:0 8px;
  filter:drop-shadow(2px 2px 0 rgba(0,0,0,0.7));}}
</style></head><body><div id="t">{_title_html}</div>
<script>document.fonts.ready.then(function(){{window._done=true;}});</script>
</body></html>"""

    try:
        with sync_playwright() as _p:
            _browser = _p.chromium.launch()
            _page = _browser.new_page(viewport={'width': area_w, 'height': area_h})
            _page.set_content(_page_html)
            _page.wait_for_function('window._done === true', timeout=8000)
            _page.screenshot(path=out_png_path, omit_background=True)
            _browser.close()
        return True
    except Exception:
        return False


def compose_follow_bar(raw_clip_path, out_path, preset,
                       title_text='', font_file=None,
                       text_color='#ffff00', text_size=60,
                       bg_color='#1a0a2e',
                       follow_text='FOLLOW KAIZER NEWS TELUGU',
                       follow_text_color='#ffffff',
                       social_logos=None,
                       video_logo=None,
                       velvet_style=None,
                       platform: str = 'youtube_short'):
    """
    Follow Bar layout (9:16):
      - Solid bg color (full frame)
      - Top text section: headline text on bg  (922px wide with 79px side margins)
      - Centre: 1:1 square video  (1048x1048, 16px side margins)
      - Bottom follow bar: follow text + up to 3 social media icons
      - Logo overlay on top-right of video area
    """
    w, h = preset['width'], preset['height']

    # ── Geometry (all at 1080x1920 scale, then ratio-applied) ──
    txt_mx  = _e(int(w * 79 / 1080))          # text side margin
    txt_h   = _e(int(h * 323 / 1920))         # text section height
    top_my  = _e(int(h * 30 / 1920))          # top margin
    gap     = _e(int(h * 10 / 1920))          # gap text→video
    vid_mx  = _e(int(w * 16 / 1080))          # video side margin
    vid_w   = _e(w - 2 * vid_mx)              # video width
    vid_h   = _e(vid_w)                        # 1:1 square
    vid_y   = _e(top_my + txt_h + gap)        # video top
    fbar_y  = _e(vid_y + vid_h)               # follow bar top
    fbar_h  = _e(h - fbar_y)                  # follow bar height

    clip_dir  = os.path.dirname(out_path)
    clip_name = os.path.splitext(os.path.basename(out_path))[0]

    # ── Background PNG (velvet texture) ──
    bg_img = _make_velvet_bg(w, h, fbar_y, velvet_style=velvet_style)

    # ── Resolve Telugu font ──
    _ff = font_file or 'NotoSansTelugu-Bold.ttf'
    tel_font_path = os.path.join(FONTS_DIR, _ff)
    if not os.path.exists(tel_font_path):
        tel_font_path = os.path.join(FONTS_DIR, 'NotoSansTelugu-Bold.ttf')

    # Split title into 2 balanced lines
    words = str(title_text).split()
    lines = [title_text]
    if len(words) > 1:
        best_k, best_d = 1, float('inf')
        for k in range(1, len(words)):
            d = abs(len(' '.join(words[:k])) - len(' '.join(words[k:])))
            if d < best_d:
                best_d, best_k = d, k
        lines = [' '.join(words[:best_k]), ' '.join(words[best_k:])]
    fnt_sz = max(20, text_size)

    # ── Title text: Playwright (HarfBuzz shaping) → PIL fallback ──
    _title_area_w = w - 2 * txt_mx
    _title_area_h = txt_h
    _title_tmp = os.path.join(clip_dir, f'_ttitle_{clip_name}.png')
    _pw_ok = _playwright_render_title(
        lines, text_color, fnt_sz, tel_font_path,
        _title_area_w, _title_area_h, _title_tmp
    )
    if _pw_ok and os.path.exists(_title_tmp):
        # Paste transparent title image onto velvet background
        bg_rgba = bg_img.convert('RGBA')
        _title_img = Image.open(_title_tmp).convert('RGBA')
        bg_rgba.paste(_title_img, (txt_mx, top_my), _title_img)
        bg_img = bg_rgba.convert('RGB')
        try:
            os.unlink(_title_tmp)
        except Exception:
            pass
    else:
        # PIL fallback (no HarfBuzz — conjuncts may be imperfect)
        tr, tg, tb = _hex_to_rgb(text_color)
        text_fill = (tr, tg, tb, 255)
        bg_rgba = bg_img.convert('RGBA')
        txt_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw_txt = ImageDraw.Draw(txt_layer)
        try:
            fnt = _truetype(tel_font_path, fnt_sz)
        except Exception:
            fnt = ImageFont.load_default()
        line_h = int(fnt_sz * 1.3)
        total_h = line_h * len(lines)
        cy = int(top_my + (txt_h - total_h) / 2)
        for ln in lines:
            try:
                bb = draw_txt.textbbox((0, 0), ln, font=fnt)
                tw_ln = bb[2] - bb[0]
            except Exception:
                tw_ln = len(ln) * int(fnt_sz * 0.6)
            lx = max(0, (w - tw_ln) // 2)
            draw_txt.text((lx + 2, cy + 2), ln, fill=(0, 0, 0, 180), font=fnt)
            draw_txt.text((lx, cy), ln, fill=text_fill, font=fnt)
            cy += line_h
        bg_img = Image.alpha_composite(bg_rgba, txt_layer).convert('RGB')

    draw_bg = ImageDraw.Draw(bg_img)

    # ── Draw follow bar text/icons on top of velvet background ──
    # follow text — use a Latin font so English text renders correctly (Telugu fonts have no Latin glyphs)
    _latin_font_path = os.path.join(FONTS_DIR, "Roboto-Bold.ttf")
    if not os.path.exists(_latin_font_path):
        _latin_font_path = os.path.join(FONTS_DIR, "Oswald-Bold.ttf")
    if not os.path.exists(_latin_font_path):
        _latin_font_path = tel_font_path   # last resort: title font
    # Auto-fit: start at max size and shrink until text fits within canvas width
    _max_text_w = int(w * 0.92)   # allow 4% padding each side
    _fnt_sz = max(18, int(fbar_h * 0.18))
    fbar_fnt = None
    ftw = w  # seed loop
    while _fnt_sz >= 14:
        try:
            _fnt = _truetype(_latin_font_path, _fnt_sz)
        except Exception:
            _fnt = ImageFont.load_default()
        try:
            _ftb = draw_bg.textbbox((0, 0), follow_text, font=_fnt)
            ftw = _ftb[2] - _ftb[0]
        except Exception:
            ftw = len(follow_text) * int(_fnt_sz * 0.6)
        if ftw <= _max_text_w:
            fbar_fnt = _fnt
            break
        _fnt_sz -= 2
    if fbar_fnt is None:
        try:
            fbar_fnt = _truetype(_latin_font_path, 14)
            _ftb = draw_bg.textbbox((0, 0), follow_text, font=fbar_fnt)
            ftw = _ftb[2] - _ftb[0]
        except Exception:
            fbar_fnt = ImageFont.load_default()
            ftw = len(follow_text) * 9
    fr2, fg2, fb2 = _hex_to_rgb(follow_text_color)
    tx = max(0, int((w - ftw) / 2))
    ty = fbar_y + int(fbar_h * 0.08)
    draw_bg.text((tx + 1, ty + 1), follow_text, fill=(0, 0, 0), font=fbar_fnt)
    draw_bg.text((tx, ty), follow_text, fill=(fr2, fg2, fb2), font=fbar_fnt)

    # social icons: draw circles with logos
    logos = [p for p in (social_logos or []) if p and os.path.exists(str(p))]
    if logos:
        n = len(logos)
        ico_r = int(fbar_h * 0.28)
        total_ico_w = n * ico_r * 2 + (n - 1) * int(ico_r * 0.5)
        ix0 = (w - total_ico_w) // 2
        iy0 = fbar_y + int(fbar_h * 0.42)
        for j, lp in enumerate(logos):
            cx_c = ix0 + j * (ico_r * 2 + int(ico_r * 0.5)) + ico_r
            cy_c = iy0 + ico_r
            # white circle bg
            draw_bg.ellipse([(cx_c - ico_r, cy_c - ico_r), (cx_c + ico_r, cy_c + ico_r)],
                            fill=(255, 255, 255))
            try:
                ico = Image.open(lp).convert('RGBA').resize((ico_r * 2 - 10, ico_r * 2 - 10), Image.LANCZOS)
                mask = Image.new('L', (ico_r * 2, ico_r * 2), 0)
                ImageDraw.Draw(mask).ellipse([(0, 0), (ico_r * 2 - 1, ico_r * 2 - 1)], fill=255)
                bg_img.paste(ico, (cx_c - ico_r + 5, cy_c - ico_r + 5), ico.split()[3] if ico.mode == 'RGBA' else None)
            except Exception:
                pass

    bg_path = os.path.join(clip_dir, f'_fbbg_{clip_name}.png')
    bg_img.save(bg_path, 'PNG')

    # ── Logo for video overlay ──
    # Per-clip video_logo → per-job DEFAULT_LOGO → NO overlay (no global default)
    _logo = (video_logo if (video_logo and os.path.exists(str(video_logo)))
             else (DEFAULT_LOGO or None))
    LOGO_W = _e(int(vid_w * 160 / 1080))
    LOGO_H = _e(int(vid_h * 134 / 1080))
    LOGO_MR = int(vid_w * 24 / 1080)
    LOGO_MT = int(vid_h * 25 / 1080)

    def ii(*paths):
        args = []
        for p in paths:
            args += ['-loop', '1', '-framerate', '30', '-i', p]
        return args

    base = ['-i', raw_clip_path]

    if _logo:
        extra = ii(bg_path, _logo)
        fc = [
            f'[1:v]scale={w}:{h}[bgsc]',
            f'{_sc(0, vid_w, vid_h)}[vid]',
            f'[bgsc][vid]overlay=x={vid_mx}:y={vid_y}[bv]',
            f'[2:v]scale={LOGO_W}:{LOGO_H}:force_original_aspect_ratio=decrease,'
            f'pad={LOGO_W}:{LOGO_H}:(ow-iw)/2:(oh-ih)/2:color=black@0[logo]',
            f'[bv][logo]overlay=x={vid_mx + vid_w - LOGO_W - LOGO_MR}:y={vid_y + LOGO_MT}[outv]',
        ]
    else:
        extra = ii(bg_path)
        fc = [
            f'[1:v]scale={w}:{h}[bgsc]',
            f'{_sc(0, vid_w, vid_h)}[vid]',
            f'[bgsc][vid]overlay=x={vid_mx}:y={vid_y}[outv]',
        ]

    cmd = ([FFMPEG_BIN, '-y'] + base + extra +
           ['-filter_complex', ';'.join(fc),
            '-map', '[outv]', '-map', '0:a?']
           + ENCODE_ARGS_SHORT_FORM
           + ['-shortest', out_path])

    run_ffmpeg(cmd)
    _enforce_duration(out_path, platform)
    _run_output_qa(out_path, platform)
    return {'bg_color': bg_color}


def compose_split_frame(raw_clip_path, thumbnail_path, out_path, preset, bg_color='#1a0a2e', video_logo=None, platform: str = 'youtube_short'):
    """
    Compose split-frame broadcast clip:
      - Solid background color (full frame)
      - Top:    Thumbnail image (16:9, with 60px side margins)
      - Bottom: Video clip (remaining height, same side margins)
    Layout mirrors the web editor applySplitFrame() geometry.
    """
    w, h = preset['width'], preset['height']

    # Geometry — must match web editor JS exactly
    mx = _e(int(w * 60 / 1080))          # side margin px
    my = _e(int(h * 60 / 1920))          # top/bottom margin px
    tw = _e(w - 2 * mx)                  # content width
    th = _e(int(tw * 9 / 16))            # thumbnail height (16:9)
    gap = _e(int(h * 30 / 1920))         # gap between thumbnail and video
    vy = _e(my + th + gap)               # video top y
    vh = _e(h - vy - my)                 # video height

    clip_dir = os.path.dirname(out_path)
    clip_name = os.path.splitext(os.path.basename(out_path))[0]

    # Background PNG
    r, g, b = _hex_to_rgb(bg_color)
    bg_img = Image.new('RGB', (w, h), (r, g, b))
    bg_path = os.path.join(clip_dir, f'_sfbg_{clip_name}.png')
    bg_img.save(bg_path, 'PNG')

    # Thumbnail image
    thumb_prep = bg_path  # fallback: bg only
    if thumbnail_path and os.path.exists(str(thumbnail_path)):
        try:
            timg = Image.open(thumbnail_path).convert('RGB').resize((tw, th), Image.LANCZOS)
            thumb_prep = os.path.join(clip_dir, f'_sfthumb_{clip_name}.jpg')
            timg.save(thumb_prep, 'JPEG', quality=95)
        except Exception:
            pass

    # Logo overlay — use custom logo if provided, else default
    logo_path = (video_logo if (video_logo and os.path.exists(str(video_logo)))
                 else (DEFAULT_LOGO or None))
    LOGO_W = _e(int(w * 160 / 1080))
    LOGO_H = _e(int(h * 134 / 1920))
    LOGO_MR = int(w * 24 / 1080)
    LOGO_MT = int(h * 25 / 1920)

    def ii(*paths):
        args = []
        for p in paths:
            args += ['-loop', '1', '-framerate', '30', '-i', p]
        return args

    base = ['-i', raw_clip_path]

    if logo_path:
        extra = ii(bg_path, thumb_prep, logo_path)
        fc = [
            f'{_sc(0, tw, vh)}[vid]',
            f'[1:v]scale={w}:{h}[bg]',
            f'[bg][vid]overlay=x={mx}:y={vy}[bv]',
            f'[2:v]scale={tw}:{th}[thumb]',
            f'[bv][thumb]overlay=x={mx}:y={my}[bgt]',
            f'[3:v]scale={LOGO_W}:{LOGO_H}:force_original_aspect_ratio=decrease,'
            f'pad={LOGO_W}:{LOGO_H}:(ow-iw)/2:(oh-ih)/2:color=black@0[logo]',
            f'[bgt][logo]overlay=x={w - LOGO_W - LOGO_MR}:y={LOGO_MT}[outv]',
        ]
    else:
        extra = ii(bg_path, thumb_prep)
        fc = [
            f'{_sc(0, tw, vh)}[vid]',
            f'[1:v]scale={w}:{h}[bg]',
            f'[bg][vid]overlay=x={mx}:y={vy}[bv]',
            f'[2:v]scale={tw}:{th}[thumb]',
            f'[bv][thumb]overlay=x={mx}:y={my}[outv]',
        ]

    cmd = ([FFMPEG_BIN, '-y'] + base + extra +
           ['-filter_complex', ';'.join(fc),
            '-map', '[outv]', '-map', '0:a?']
           + ENCODE_ARGS_SHORT_FORM
           + ['-shortest', out_path])

    run_ffmpeg(cmd)
    _enforce_duration(out_path, platform)
    _run_output_qa(out_path, platform)
    return {'bg_color': bg_color}


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

FRAME_LAYOUTS = {
    "torn_card":    "Torn Card   (Video + Red headline card + Image)",
    "clean_card":   "Clean Card  (Video + Headline + Framed image, no torn edge)",
    "split_frame":  "Split Frame (Thumbnail + Video on colored background)",
    "follow_bar":   "Follow Bar  (Text top + Square video + Social follow bar)",
}


def _prompt_choice(prompt_lines, options, default="1"):
    """Print numbered options and return the chosen key. Never raises."""
    print()
    for line in prompt_lines:
        print(f"  {line}")
    keys = list(options.keys())
    for i, (k, label) in enumerate(options.items(), 1):
        print(f"    {i}. {label}")
    try:
        raw = input(f"  Choice [1]: ").strip() or default
        idx = int(raw) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return keys[int(default) - 1]


def _select_platform_and_frame(platform=None, frame_layout=None):
    """
    Interactive selection: Platform first → Frame layout (Shorts only) → Confirm.

    ``youtube_full`` is the 16:9 long-form bulletin path — frame layouts
    are 9:16 Shorts-only graphics, so we skip that step entirely and
    return a sentinel frame value (callers that need a frame for Shorts
    paths never see ``youtube_full``).

    Returns ``(platform_key, frame_layout_key)``.
    """
    platform_opts = {
        k: f"{v['label']}  ({v['width']}x{v['height']})"
        for k, v in PLATFORM_PRESETS.items()
    }
    frame_opts = FRAME_LAYOUTS

    # If platform is set to long-form, the frame is meaningless — pin it
    # to the first Shorts layout key as a harmless placeholder; the
    # bulletin branch ignores frame_layout entirely.
    def _is_long_form(p: str | None) -> bool:
        return (p or "").lower() == "youtube_full"

    # If platform is already long-form (CLI), short-circuit. If both
    # are provided for a Shorts platform, skip interaction.
    if _is_long_form(platform):
        return platform, frame_layout or next(iter(frame_opts.keys()))
    if platform and frame_layout:
        return platform, frame_layout

    while True:
        # ── STEP 1: Platform ──────────────────────────────────
        if not platform:
            print()
            print("  ┌─────────────────────────────────────────┐")
            print("  │   STEP 1 — Choose Platform              │")
            print("  └─────────────────────────────────────────┘")
            p_keys = list(platform_opts.keys())
            for i, (k, label) in enumerate(platform_opts.items(), 1):
                print(f"    {i}. {label}")
            try:
                raw = input("  Choice [1]: ").strip() or "1"
                idx = int(raw) - 1
                if 0 <= idx < len(p_keys):
                    platform = p_keys[idx]
                else:
                    platform = p_keys[0]
            except (ValueError, EOFError, KeyboardInterrupt):
                platform = p_keys[0]

        # ── Long-form platforms skip the frame step entirely ──
        if _is_long_form(platform):
            preset = PLATFORM_PRESETS[platform]
            print()
            print("  ┌─────────────────────────────────────────┐")
            print("  │   Long-form selected — Confirm          │")
            print("  └─────────────────────────────────────────┘")
            print(f"    Platform   : {preset['label']}  "
                  f"({preset['width']}x{preset['height']})")
            print(f"    Layout     : long-form bulletin (TV9 broadcast layout)")
            print(f"    Frame mode : skipped (Shorts-only)")
            print()
            print("    1. ✓ Confirm — start bulletin pipeline")
            print("    2. ← Change platform")
            try:
                raw = input("  Choice [1]: ").strip() or "1"
                if raw == "2":
                    platform = None
                    continue
            except (ValueError, EOFError, KeyboardInterrupt):
                pass
            return platform, next(iter(frame_opts.keys()))

        # ── STEP 2: Frame layout (Shorts only) ────────────────
        if not frame_layout:
            print()
            print("  ┌─────────────────────────────────────────┐")
            print("  │   STEP 2 — Choose Frame Layout          │")
            print("  └─────────────────────────────────────────┘")
            f_keys = list(frame_opts.keys())
            for i, (k, label) in enumerate(frame_opts.items(), 1):
                print(f"    {i}. {label}")
            print(f"    b. ← Back  (change platform)")
            try:
                raw = input("  Choice [1]: ").strip() or "1"
                if raw.lower() == 'b':
                    platform = None
                    continue
                idx = int(raw) - 1
                if 0 <= idx < len(f_keys):
                    frame_layout = f_keys[idx]
                else:
                    frame_layout = f_keys[0]
            except (ValueError, EOFError, KeyboardInterrupt):
                frame_layout = f_keys[0]

        # ── STEP 3: Confirm ───────────────────────────────────
        preset = PLATFORM_PRESETS[platform]
        print()
        print("  ┌─────────────────────────────────────────┐")
        print("  │   Confirm Selection                     │")
        print("  └─────────────────────────────────────────┘")
        print(f"    Platform : {preset['label']}  ({preset['width']}x{preset['height']})")
        print(f"    Frame    : {FRAME_LAYOUTS[frame_layout]}")
        print()
        print("    1. ✓ Confirm — start pipeline")
        print("    2. ← Change platform")
        print("    3. ← Change frame layout")
        try:
            raw = input("  Choice [1]: ").strip() or "1"
            if raw == "2":
                platform = None
                frame_layout = None
                continue
            elif raw == "3":
                frame_layout = None
                continue
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        break

    return platform, frame_layout


def run_compound_pipeline(
    video_path: str,
    language: str = "te",
    *,
    frame_layout: str = "torn_card",
    use_default_brand_image: bool = False,
) -> dict:
    """Dedicated "Full Video + Shorts" pipeline — ONE Gemini call, both formats.

    This replaces the prior compound flow (two sequential subprocesses, each
    running the standalone single-platform pipeline). The new flow:

    1. Calls ``analyze_video_with_gemini(mode="compound")`` ONCE. Gemini
       returns ``full_video_cuts``, ``shorts_cuts``, ``image_plan`` (FULL
       VIDEO ONLY), ``shorts_headline_native``, ``bulletin_marquee_points``
       in a single JSON. ChatGPT is NOT called for titles or SEO.
    2. Writes the compound analysis to ``<job_dir>/compound_analysis.json``
       and two derivative files keyed by pass:
         - ``analysis_full.json``  → ``clips`` aliased to ``full_video_cuts``
         - ``analysis_short.json`` → ``clips`` aliased to ``shorts_cuts``
    3. Runs the full-video pass FIRST (so the bulletin image pool can later
       be reused for shorts) by calling ``run_pipeline()`` in-process with
       ``KAIZER_REUSE_ANALYSIS_FROM`` pointed at ``analysis_full.json``.
    4. Runs the shorts pass SECOND with ``analysis_short.json``. If
       ``use_default_brand_image`` is True AND the operator has set
       ``KAIZER_DEFAULT_IMAGE`` to a real file, the shorts pass uses that
       asset and skips image search/generation entirely.

    The derivative-file trick lets ``run_pipeline()`` stay 100% unchanged —
    it sees ``clips[]`` as it always has. Phase C wires the image_plan
    overlay step on top of the bulletin's output, also without touching
    ``run_pipeline``.

    Returns a dict ``{"full_meta": ..., "short_meta": ..., "compound_analysis": ...}``
    where each value is a filesystem path (or None on per-pass failure).
    """
    print()
    print(f"  ╔══════════════════════════════════════════════════════════")
    print(f"  ║ Compound pipeline — Full Video + Shorts (single Gemini call)")
    print(f"  ╚══════════════════════════════════════════════════════════")

    # Resolve language config + bring up platform presets the same way
    # run_pipeline does. We need a "preset" object to feed Gemini's
    # min/max/ideal_dur — pick the LONG-FORM preset because Gemini emits
    # both cut sets in compound mode but the duration targets are most
    # meaningful for the bulletin clips. Shorts targets are baked into the
    # NOTES ON COMPOUND MODE prose.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import languages as _langs
    lang_cfg = _langs.get(language)
    full_preset = PLATFORM_PRESETS["youtube_full"]

    # ── Output dir ─────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_stem = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem = "".join(c for c in safe_stem if c.isalnum() or c in "-_")[:40] or "kaizer"
    out_root  = os.environ.get("KAIZER_OUTPUT_ROOT") or os.path.abspath("output")
    job_dir   = os.path.join(out_root, f"compound_{timestamp}_{safe_stem}")
    os.makedirs(job_dir, exist_ok=True)
    print(f"  Output: {job_dir}")

    # ── Single Gemini call ─────────────────────────────────────
    print(f"\n  [Gemini] Compound analysis (one call, both formats) …")
    gemini_result = analyze_video_with_gemini(
        video_path, full_preset, language=language, mode="compound",
    )

    # Persist the compound analysis intact for QA / replay.
    compound_path = os.path.join(job_dir, "compound_analysis.json")
    with open(compound_path, "w", encoding="utf-8") as f:
        json.dump(gemini_result, f, ensure_ascii=False, indent=2)
    print(f"    ✓ Saved {os.path.basename(compound_path)}")

    full_cuts   = gemini_result.get("full_video_cuts") or gemini_result.get("clips") or []
    shorts_cuts = gemini_result.get("shorts_cuts")     or []
    image_plan  = gemini_result.get("image_plan")      or []
    shorts_hl   = (gemini_result.get("shorts_headline_native") or "").strip()
    marquee_pts = gemini_result.get("bulletin_marquee_points") or []

    print(f"    full_video_cuts: {len(full_cuts)}  shorts_cuts: {len(shorts_cuts)}  "
          f"image_plan: {len(image_plan)}  marquee_pts: {len(marquee_pts)}")
    if shorts_hl:
        print(f"    shorts headline: {shorts_hl}")

    # ── Build the per-pass derivative analyses ─────────────────
    # Each derivative reuses the compound JSON's metadata fields (summary,
    # entities, image_search_queries, etc.) but swaps `clips[]` for the
    # appropriate cut set so run_pipeline sees what it expects.
    full_analysis  = dict(gemini_result)
    full_analysis["clips"] = full_cuts
    full_analysis_path = os.path.join(job_dir, "analysis_full.json")
    with open(full_analysis_path, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, ensure_ascii=False, indent=2)

    short_analysis = dict(gemini_result)
    short_analysis["clips"] = shorts_cuts
    short_analysis_path = os.path.join(job_dir, "analysis_short.json")
    with open(short_analysis_path, "w", encoding="utf-8") as f:
        json.dump(short_analysis, f, ensure_ascii=False, indent=2)
    print(f"    ✓ Wrote per-pass derivatives")

    # ── Pass 1: Full video (bulletin) ──────────────────────────
    full_meta = None
    if full_cuts:
        print(f"\n  ═══ PASS 1/2 — Full Video (bulletin) ═══")
        os.environ["KAIZER_REUSE_ANALYSIS_FROM"] = full_analysis_path
        try:
            full_meta = run_pipeline(
                video_path,
                platform="youtube_full",
                frame_layout="torn_card",
                language=language,
                render_mode="bulletin",
            )
        except Exception as exc:
            print(f"  [compound] Full-video pass FAILED: {type(exc).__name__}: {exc}")
        finally:
            os.environ.pop("KAIZER_REUSE_ANALYSIS_FROM", None)

        # ── Pass 1b: image_plan content-aligned overlays ──
        # If Gemini emitted an image_plan, generate the dual-backend
        # image pool and composite the overlays onto the stitched
        # bulletin. The overlay step is purely additive: it produces
        # bulletin_with_overlays.mp4 alongside the original bulletin.mp4.
        # The editor_meta is updated to point at the overlaid version
        # so R2 upload and the UI pick it up automatically.
        if full_meta and image_plan:
            try:
                full_meta_dir = os.path.dirname(full_meta)
                bulletin_dir  = os.path.join(full_meta_dir, "bulletin")
                bulletin_mp4  = os.path.join(bulletin_dir, "bulletin.mp4")
                if os.path.exists(bulletin_mp4):
                    # Normalise the cuts so resolve_image_plan has start_sec/end_sec.
                    vinfo = get_video_info(video_path) or {}
                    src_dur = float(vinfo.get("duration") or 0.0)
                    _kept: list = []
                    for c in full_cuts:
                        try:
                            s = ts_to_sec(c.get("start", "00:00"), src_dur)
                            e = ts_to_sec(c.get("end",   "00:00"), src_dur)
                            if e > s:
                                _kept.append({
                                    **c,
                                    "start_sec":    s,
                                    "end_sec":      e,
                                    "duration_sec": e - s,
                                })
                        except Exception:
                            continue

                    print(f"  [image_plan] {len(image_plan)} entries — generating dual-backend pool …")
                    pool = generate_image_pool_from_plan(
                        image_plan, job_dir=full_meta_dir,
                        language=language, max_unique=6, dual_backend=True,
                    )
                    resolved = resolve_image_plan(
                        image_plan,
                        pool_manifest=pool,
                        kept_clips=_kept,
                        whisper_words=None,  # bulletin path doesn't produce a transcript yet
                        video_duration_sec=src_dur,
                    )
                    ready_count = sum(1 for r in resolved if r.get("status") == "ready")
                    print(f"  [image_plan] {ready_count}/{len(resolved)} entries resolved to ready")

                    if ready_count > 0:
                        overlay_out = os.path.join(bulletin_dir, "bulletin_with_overlays.mp4")
                        if os.path.exists(overlay_out) and os.path.getsize(overlay_out) > 100_000:
                            print(f"  [image_plan] {os.path.basename(overlay_out)} cached — skipping overlay pass")
                        else:
                            overlay_image_plan(bulletin_mp4, resolved, overlay_out)

                        if os.path.exists(overlay_out) and os.path.getsize(overlay_out) > 100_000:
                            print(f"  [image_plan] Final bulletin → {os.path.basename(overlay_out)}")
                            # Update editor_meta to point at the overlaid file
                            # so R2 upload + UI pick it up.
                            try:
                                with open(full_meta, "r", encoding="utf-8") as _mf:
                                    _meta = json.load(_mf)
                                for _c in _meta.get("clips") or []:
                                    if _c.get("clip_path", "").endswith("bulletin.mp4"):
                                        _c["clip_path"] = overlay_out
                                        _c["clip_path_overlay"] = overlay_out
                                        _c["clip_path_carousel_only"] = bulletin_mp4
                                with open(full_meta, "w", encoding="utf-8") as _mf:
                                    json.dump(_meta, _mf, ensure_ascii=False, indent=2)
                            except Exception as _e:
                                print(f"  [image_plan] editor_meta update skipped: {_e}")
                    # Persist the resolved manifest for QA / debugging.
                    try:
                        with open(os.path.join(bulletin_dir, "image_plan_resolved.json"),
                                  "w", encoding="utf-8") as _rf:
                            json.dump(resolved, _rf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                else:
                    print(f"  [image_plan] bulletin.mp4 not found at {bulletin_mp4} — overlay skipped")
            except Exception as _ip_exc:
                print(f"  [image_plan] FAILED ({type(_ip_exc).__name__}: {_ip_exc}) — keeping un-overlaid bulletin.mp4")
        elif full_meta and not image_plan:
            print(f"  [image_plan] Gemini emitted empty image_plan — keeping carousel-based bulletin")
    else:
        print(f"  [compound] No full_video_cuts emitted by Gemini — skipping bulletin pass")

    # ── Pass 2: Shorts ─────────────────────────────────────────
    short_meta = None
    if shorts_cuts:
        print(f"\n  ═══ PASS 2/2 — Shorts ═══")
        os.environ["KAIZER_REUSE_ANALYSIS_FROM"] = short_analysis_path
        # The default-image short-circuit already exists in run_pipeline
        # (it reads KAIZER_DEFAULT_IMAGE). If the operator asked for it,
        # surface a clear log line here; if the env var isn't set
        # because no brand image is configured, run_pipeline falls
        # through to the existing image search path.
        if use_default_brand_image:
            _dflt = os.environ.get("KAIZER_DEFAULT_IMAGE", "").strip()
            if _dflt and os.path.exists(_dflt):
                print(f"  [compound] use_default_brand_image=True — shorts will use {os.path.basename(_dflt)}")
            else:
                print(f"  [compound] use_default_brand_image=True but KAIZER_DEFAULT_IMAGE is unset/missing — "
                      f"falling back to standard shorts image path")
        try:
            short_meta = run_pipeline(
                video_path,
                platform="youtube_short",
                frame_layout=frame_layout,
                language=language,
            )
        except Exception as exc:
            print(f"  [compound] Shorts pass FAILED: {type(exc).__name__}: {exc}")
        finally:
            os.environ.pop("KAIZER_REUSE_ANALYSIS_FROM", None)
    else:
        print(f"  [compound] No shorts_cuts emitted by Gemini — skipping shorts pass")

    print()
    print(f"  ╔══════════════════════════════════════════════════════════")
    print(f"  ║ Compound pipeline DONE")
    print(f"  ║   full_meta:  {full_meta or '<skipped/failed>'}")
    print(f"  ║   short_meta: {short_meta or '<skipped/failed>'}")
    print(f"  ╚══════════════════════════════════════════════════════════")
    return {
        "full_meta":           full_meta,
        "short_meta":          short_meta,
        "compound_analysis":   compound_path,
    }


def run_pipeline(video_path: str, platform: str = None, frame_layout: str = None,
                 language: str = "te", render_mode: str = None,
                 resume_dir: str = None):
    """Complete API pipeline: Gemini → FFmpeg → ChatGPT → Pexels → Compose → Editor.

    When ``render_mode == "bulletin"``, the per-clip composition path is
    short-circuited: after Gemini analysis + FFmpeg slicing, the raw story
    slices are concatenated into a single 1–2 hour bulletin MP4 (no per-clip
    layout, no images, no titles — those land in Phases 2–4 of the long-form
    plan). All Shorts modes ignore this flag.
    """

    # Resolve language config once at the top — used for prompts, fonts, follow-bar.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import languages as _langs
    lang_cfg = _langs.get(language)

    print("=" * 60)
    print("  KAIZER NEWS — API Pipeline")
    print("=" * 60)

    # ── Validate video ─────────────────────────────────────
    if not os.path.exists(video_path):
        print(f"  Video not found: {video_path}")
        return
    vinfo = get_video_info(video_path)
    if not vinfo:
        print("  Could not read video info")
        return
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Duration: {sec_to_ts(vinfo['duration'])} | {vinfo['width']}x{vinfo['height']}")
    print(f"  Language : {lang_cfg.name_english} ({lang_cfg.name_native}, {lang_cfg.script} script)")

    # ── Platform + Frame selection (platform first, then frame, with back option) ──
    platform, frame_layout = _select_platform_and_frame(platform, frame_layout)

    # Long-form (16:9) implies the bulletin pipeline. The 9:16 frame
    # compose paths only know how to draw Shorts layouts and would
    # produce broken output on a 1920×1080 canvas. Auto-route here so
    # selecting "YouTube Full" never silently runs Shorts compose.
    if (platform or "").lower() == "youtube_full" and not (render_mode or ""):
        render_mode = "bulletin"
        print("  [auto] platform=youtube_full → render_mode=bulletin")

    preset = PLATFORM_PRESETS[platform]
    print(f"\n  Platform : {preset['label']} ({preset['width']}x{preset['height']})")
    if (render_mode or "").lower() == "bulletin":
        print(f"  Mode     : long-form bulletin (TV9 broadcast layout)")
    else:
        print(f"  Frame    : {FRAME_LAYOUTS[frame_layout]}")

    # ── Output directory ──────────────────────────────────
    # Resume mode: when --resume-dir is set we re-enter the same dir
    # an earlier run created. All idempotency checks downstream
    # (cut_video_clips, image gen, sidebar carousel, compose,
    # takeover, stitch) read existing artifacts and skip steps that
    # already finished — saving GPU time and OpenAI cost on retries.
    _resume_dir = (resume_dir or "").strip()
    if _resume_dir and os.path.isdir(_resume_dir):
        OUTPUT_DIR = os.path.abspath(_resume_dir)
        # Recover the timestamp from the dir name so R2 keys stay
        # consistent across the resumed and original runs.
        timestamp = os.path.basename(OUTPUT_DIR.rstrip(r"\/"))
        print(f"  [resume] Re-entering existing output dir: {OUTPUT_DIR}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join(OUTPUT_ROOT, platform, timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  Output: {OUTPUT_DIR}")

    TOTAL = 6

    # ════ STEP 1: Gemini Video Analysis ════
    print(f"\n  [1/{TOTAL}] Sending video to Gemini (language={lang_cfg.code}) ...")
    gemini_result = analyze_video_with_gemini(video_path, preset, language=lang_cfg.code)

    video_type = gemini_result.get("video_type", "SOLO")
    clips = gemini_result.get("clips", [])
    summary = gemini_result.get("overall_summary", "")
    # Support both new (_native) and legacy (_telugu) field names on cached data
    summary_native = (gemini_result.get("overall_summary_native")
                      or gemini_result.get("overall_summary_telugu", ""))
    keywords = gemini_result.get("image_keywords", [])
    people = gemini_result.get("key_people", [])
    topics = gemini_result.get("key_topics", [])

    print(f"    Video type: {video_type}")
    print(f"    Speakers: {gemini_result.get('total_speakers', '?')}")
    print(f"    Clips found: {len(clips)}")
    for c in clips:
        print(f"      [{c.get('index', '?')}] {c.get('start')} -> {c.get('end')} "
              f"| importance: {c.get('importance', '?')} | {c.get('summary', '')[:60]}")
    print(f"    Keywords: {', '.join(keywords[:6])}")

    # Save Gemini response
    with open(os.path.join(OUTPUT_DIR, "gemini_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(gemini_result, f, ensure_ascii=False, indent=2)

    if not clips:
        print("  Gemini returned 0 clips. Aborting.")
        return

    # ════ STEP 2: Cut Video with FFmpeg ════
    print(f"\n  [2/{TOTAL}] Cutting video with FFmpeg ...")
    cut_video_clips(video_path, clips, OUTPUT_DIR)

    # ════ Bulletin pipeline (long-form Phases 1–4) ════
    # Long-form path: every story gets the TV9 broadcast layout (Phase 2),
    # an OpenAI-generated image carousel as the sidebar + full-screen
    # photo gallery between stories (Phase 3), and an optional PiP inset
    # for stories Gemini flags as having a secondary subject (Phase 4).
    # Each layer can be disabled via env var so we can roll back to a
    # bare Phase-1 stitch on demand.
    if (render_mode or "").lower() == "bulletin":
        bulletin_dir = os.path.join(OUTPUT_DIR, "bulletin")
        os.makedirs(bulletin_dir, exist_ok=True)
        bulletin_out = os.path.join(bulletin_dir, "bulletin.mp4")

        def _flag(name: str, default: str = "1") -> bool:
            return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")

        use_overlay   = _flag("KAIZER_LONGFORM_OVERLAY", "1")
        use_sidebar   = _flag("KAIZER_CAROUSEL_SIDEBAR", "1")
        use_takeovers = _flag("KAIZER_CAROUSEL_TAKEOVERS", "1")
        use_pip       = _flag("KAIZER_LONGFORM_PIP", "0")
        # CLI flags override env (set by argparse).
        if globals().get("_BULLETIN_FLAG_OVERRIDES"):
            for k, v in _BULLETIN_FLAG_OVERRIDES.items():
                if k == "overlay":   use_overlay = v
                if k == "sidebar":   use_sidebar = v
                if k == "takeovers": use_takeovers = v
                if k == "pip":       use_pip = v

        story_paths: list[str] = []          # final segment list to stitch

        if not use_overlay:
            # Phase 1 fallback — bare slices, no graphics.
            print("  [bulletin] overlay disabled — using bare slices (Phase 1).")
            story_paths = [c.get("raw_path") for c in clips if c.get("raw_path")]
        else:
            # ── Phase 2 imports ──
            from pipeline_core.longform_compose import (
                StoryMeta, render_ticker, render_channel_bug,
                make_sidebar_placeholder, compose_bulletin_story,
                compose_pip_story, pick_pip_source,
            )
            # ── Phase 3 imports ──
            from pipeline_core.image_carousel import (
                build_sidebar_carousel, build_fullscreen_takeover, CarouselError,
            )

            # Local Gemini metadata (mirrors the Shorts path that runs after
            # this branch — kept here because we return early).
            locations = gemini_result.get("key_locations", [])
            global_kw = gemini_result.get("image_search_queries", [])

            # ── 1) Per-story image generation (OpenAI-first via search_news_images).
            #    Two optimisations vs. the original "6 images, sequential":
            #
            #      a) ADAPTIVE COUNT — scale by clip duration. A 10-second
            #         clip doesn't need 6 images cycling at 1.6 s each;
            #         users can't even register a face that fast. Rule of
            #         thumb: ~1 image per 5 seconds of clip, capped at 6.
            #         For the typical news source this saves ~40% of
            #         OpenAI calls and cost.
            #
            #      b) PARALLEL ACROSS STORIES — search_news_images() is
            #         per-story sequential, but story-N doesn't depend on
            #         story-(N-1)'s images. We run up to N_PARALLEL=3
            #         stories concurrently via ThreadPoolExecutor —
            #         well under OpenAI's image-gen rate limits — and
            #         cut wall-clock by ~3x.
            #
            #    Each story writes into its own subdir so concurrent calls
            #    don't clobber each other.

            def _adaptive_count(duration_s: float) -> int:
                if duration_s < 15:  return 2
                if duration_s < 30:  return 3
                if duration_s < 60:  return 4
                if duration_s < 120: return 5
                return 6

            # User-supplied pre-selected images take priority over OpenAI
            # gen. Same image pool applies to every story (the carousel
            # cycles through the list). When a story wants more images
            # than the user provided, we cycle the list (mod N). When
            # the user provided MORE than the story wants, we slice the
            # first N. Empty env var → existing OpenAI path.
            #
            # JOB-LEVEL POOL MODE (default since 2026-05):
            # Instead of generating per-story (e.g. 25 images for a 5-story
            # bulletin), we generate ONE small pool of N diverse, news-
            # relevant images using the FULL bulletin context, then cycle
            # that pool through every story. Pros:
            #   * ~75% cheaper ($0.24 for 6 vs $1.00 for 25)
            #   * ~75% faster (6 fits in OpenAI Tier-1's 5/min limit)
            #   * Less repetitive — per-story generation kept giving us
            #     near-duplicates because every story shared the same
            #     overall topic. Job-level uses a deduped subject pool.
            # Override with KAIZER_BULLETIN_IMAGE_MODE=per_story for the
            # legacy behaviour.
            _preselected_raw = os.environ.get("KAIZER_BULLETIN_IMAGES", "").strip()
            _preselected: list[str] = []
            if _preselected_raw:
                # Pipe-separated so Windows-style absolute paths
                # (which contain ':') don't break the parsing.
                for _p in _preselected_raw.split("|"):
                    _p = _p.strip()
                    if _p and os.path.isfile(_p):
                        _preselected.append(_p)
                if _preselected:
                    print(f"  [bulletin] using {len(_preselected)} user-supplied image(s) for every story "
                          f"(skipping OpenAI generation)")

            # ── JOB-LEVEL POOL: one diverse pool for the whole bulletin ──
            # Two-track strategy (default since 2026-05):
            #   TRACK 1 — REAL PHOTOS for named people / incidents.
            #     Gemini's `key_people` list = real public figures (politician,
            #     actor, advocate, ...). For these we want the ACTUAL web
            #     photo, not an AI render that OpenAI would refuse to draw
            #     by name anyway. Routed straight to Google CSE / DDG /
            #     Pexels via ``search_news_images(skip_openai=True)``.
            #   TRACK 2 — AI-GENERATED B-ROLL for generic context.
            #     Fills the remaining slots with topic-driven scenes
            #     (courtroom, marketplace, broadcast studio, etc.).
            #
            # The two tracks combine into the unified pool, which the
            # existing cycling code distributes across every story.
            # Override with KAIZER_BULLETIN_IMAGE_MODE=per_story for the
            # legacy per-story flow.
            _img_mode = os.environ.get("KAIZER_BULLETIN_IMAGE_MODE", "per_job").lower()
            _pool_size = max(3, min(12, int(os.environ.get("KAIZER_BULLETIN_POOL_SIZE", "6"))))
            if not _preselected and _img_mode == "per_job":
                try:
                    from pipeline_core.openai_images import (
                        generate_bulletin_image_pool, is_enabled as _openai_enabled,
                    )
                except Exception:
                    _openai_enabled = lambda: False  # noqa: E731
                    generate_bulletin_image_pool = None

                # Cap real-photo allocation so we always leave ≥1 slot
                # for generated B-roll variety. With 6-slot default and
                # 4 named people, we'd take 5 real + 1 generated.
                _real_cap = max(1, _pool_size - 1)
                _n_real = min(len(people), _real_cap) if people else 0
                _n_gen  = _pool_size - _n_real

                _pool: list[str] = []

                # ── TRACK 1: real photos for named people ────────────
                if _n_real > 0:
                    _real_dir = os.path.join(bulletin_dir, "_job_pool", "real")
                    os.makedirs(_real_dir, exist_ok=True)
                    print(f"  [bulletin] fetching {_n_real} real photo(s) "
                          f"for named subjects: {', '.join(people[:_n_real])}")
                    try:
                        _real_imgs = search_news_images(
                            search_queries=[],
                            people=people[:_n_real],
                            topics=[],
                            output_dir=_real_dir,
                            count=_n_real,
                            language=lang_cfg.code,
                            skip_openai=True,   # real photos only — NOT AI render
                        ) or []
                        _pool.extend(_real_imgs)
                        print(f"  [bulletin] real-photo track filled {len(_real_imgs)}/{_n_real}")
                    except Exception as _exc:
                        print(f"  [bulletin][warn] real-photo track failed: {_exc}")

                # ── TRACK 2: AI-generated B-roll for context ─────────
                _n_gen_actual = _pool_size - len(_pool)
                if _n_gen_actual > 0 and _openai_enabled() and generate_bulletin_image_pool is not None:
                    _job_subjects: list[str] = []
                    for _c in clips:
                        _s = (_c.get("summary") or "").strip()
                        if _s:
                            _job_subjects.append(_s[:120])
                    _gen_dir = os.path.join(bulletin_dir, "_job_pool", "generated")
                    print(f"  [bulletin] generating {_n_gen_actual} AI image(s) "
                          f"for topic/B-roll context ...")
                    try:
                        _gen_imgs = generate_bulletin_image_pool(
                            pool_size=_n_gen_actual,
                            entities=[],            # named people went to TRACK 1
                            topics=topics + locations,
                            queries=global_kw + _job_subjects,
                            language=lang_cfg.code,
                            out_dir=_gen_dir,
                        ) or []
                        _pool.extend(_gen_imgs)
                    except Exception as _exc:
                        print(f"  [bulletin][warn] AI-generated track failed: {_exc}")

                _preselected = _pool
                if _preselected:
                    print(f"  [bulletin] job-level pool ready: {len(_preselected)} image(s) "
                          f"({_n_real} real + {len(_preselected) - _n_real} AI) "
                          f"will cycle across all {len(clips)} stories")
                else:
                    print(f"  [bulletin][warn] job-level pool empty — falling back to per-story generation")

            def _gen_for_story(i: int, c: dict) -> tuple[int, list[str], str]:
                """Per-story worker. Returns (index, imgs, log_line)."""
                per_story_dir = os.path.join(bulletin_dir, f"story_{i:02d}_assets")
                os.makedirs(per_story_dir, exist_ok=True)
                duration_s = float(c.get("duration_sec") or 60.0)
                want = _adaptive_count(duration_s)

                # Short-circuit: user pre-selected images for the bulletin.
                # Cycle the pool so every story gets ``want`` images, even
                # if the user supplied fewer. No OpenAI calls, no rate-
                # limits, no $ spent.
                if _preselected:
                    imgs = [_preselected[k % len(_preselected)] for k in range(want)]
                    return (i, imgs,
                            f"    story_{i:02d}: {want} pre-selected image(s) (user pool of {len(_preselected)})")

                # Resume cache check (≥half the desired count is enough)
                images_cache_dir = os.path.join(per_story_dir, "images")
                cached = sorted([
                    os.path.join(images_cache_dir, f)
                    for f in (os.listdir(images_cache_dir)
                              if os.path.isdir(images_cache_dir) else [])
                    if f.lower().startswith("news_") and f.lower().endswith((".jpg", ".png"))
                ])
                cached = [p for p in cached if os.path.getsize(p) > 5000]
                # ``duration_s`` / ``want`` already computed at the top
                # of this worker; reuse them here.
                if len(cached) >= max(2, want - 1):
                    return (i, cached[:want],
                            f"    story_{i:02d}: {len(cached)} cached image(s) (skipping OpenAI)")

                story_summary = (c.get("summary") or "").strip()
                queries: list[str] = []
                if story_summary:
                    queries.append(story_summary[:80])
                queries += global_kw[:2]
                try:
                    imgs = search_news_images(
                        queries, people, topics + locations,
                        per_story_dir, count=want,
                        language=lang_cfg.code,
                    )
                except Exception as exc:
                    return (i, [], f"    [bulletin][warn] story_{i:02d} image fetch: {exc}")
                return (i, imgs or [],
                        f"    story_{i:02d}: {len(imgs)} image(s) (target {want}, {duration_s:.0f}s clip)")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            n_parallel = max(1, int(os.environ.get("KAIZER_BULLETIN_IMAGE_PARALLELISM", "3")))
            total_target = sum(_adaptive_count(float(c.get("duration_sec") or 60.0)) for c in clips)
            print(f"\n  [bulletin] Generating images for {len(clips)} stories "
                  f"({total_target} total, {n_parallel}-way parallel) ...")
            story_images: list[list[str]] = [[] for _ in clips]
            with ThreadPoolExecutor(max_workers=n_parallel) as _exe:
                futures = [_exe.submit(_gen_for_story, i, c) for i, c in enumerate(clips)]
                for fut in as_completed(futures):
                    try:
                        idx, imgs, log_line = fut.result()
                    except Exception as exc:
                        print(f"    [bulletin][warn] image-gen worker crashed: {exc}")
                        continue
                    story_images[idx] = imgs
                    print(log_line)

            # Write a manifest of every per-story image generated this
            # run. The runner reads this AFTER success and copies each
            # image to the user's Assets folder (folder_path="generated/")
            # so the user can browse what OpenAI produced for this job.
            # The marker line below is the canonical signal — same
            # contract as [kaizer:meta].
            try:
                _gen_manifest = []
                for _idx, _imgs in enumerate(story_images):
                    for _p in (_imgs or []):
                        if _p and os.path.isfile(_p):
                            _gen_manifest.append({
                                "story_index": _idx,
                                "path":        os.path.abspath(_p),
                                "filename":    os.path.basename(_p),
                            })
                if _gen_manifest:
                    _manifest_path = os.path.join(bulletin_dir, "_generated_images.json")
                    with open(_manifest_path, "w", encoding="utf-8") as _fh:
                        json.dump(_gen_manifest, _fh, ensure_ascii=False, indent=2)
                    print(f"[kaizer:generated_images] {os.path.abspath(_manifest_path)}")
            except Exception as _exc:
                print(f"    [bulletin][warn] generated-images manifest failed: {_exc}")

            # ── 2) Render the ticker once (shared across every story).
            # Pick native-script summary for non-English so the ticker
            # reads in the language the user picked (Telugu, Hindi, …).
            # Gemini already returns summary_native per clip alongside
            # the English summary.
            def _clip_text(c):
                if (lang_cfg.code or "en").lower() != "en":
                    native = (c.get("summary_native")
                              or c.get("summary_telugu")
                              or "").strip()
                    if native:
                        return native[:120]
                return (c.get("summary") or "").strip()[:120]

            # Compound short-circuit: if Gemini supplied a
            # bulletin_marquee_points list in compound mode, use it
            # verbatim — these are the headlines Gemini selected to
            # represent the whole source video. Falls through to the
            # per-clip-summary derivation for standalone full jobs.
            _gemini_marquee = gemini_result.get("bulletin_marquee_points") or []
            _gemini_marquee = [str(m).strip() for m in _gemini_marquee if str(m).strip()]
            if _gemini_marquee:
                all_headlines = _gemini_marquee
                print(f"    [bulletin] Using Gemini-supplied marquee "
                      f"({len(all_headlines)} points)")
            else:
                all_headlines = [
                    _clip_text(c)
                    for c in clips if _clip_text(c)
                ]
            from pipeline_core import compose_deps

            ticker_path = os.path.join(bulletin_dir, "_ticker.png")
            ticker_inputs = [lang_cfg.font_primary] if lang_cfg.font_primary else []
            ticker_extra = {
                "headlines": all_headlines or ["KAIZER NEWS"],
                "lang":      lang_cfg.code,
            }
            if compose_deps.is_fresh(ticker_path, ticker_inputs, ticker_extra, min_size=1000):
                print(f"    [bulletin] ticker cached (skipping)")
            else:
                try:
                    render_ticker(
                        all_headlines or ["KAIZER NEWS"],
                        lang_cfg.code, lang_cfg.font_primary,
                        ticker_path,
                    )
                    compose_deps.mark_built(ticker_path, ticker_inputs, ticker_extra)
                except Exception as exc:
                    print(f"    [bulletin][warn] ticker render failed: {exc}")
                    ticker_path = ""

            # ── 3) Render channel bug once.
            bug_path = os.path.join(bulletin_dir, "_bug.png")
            bug_inputs = [DEFAULT_LOGO] if DEFAULT_LOGO else []
            bug_extra = {"channel_name": "KAIZER NEWS"}
            if compose_deps.is_fresh(bug_path, bug_inputs, bug_extra, min_size=500):
                print(f"    [bulletin] channel bug cached (skipping)")
            else:
                try:
                    render_channel_bug("KAIZER NEWS", DEFAULT_LOGO or None, bug_path)
                    compose_deps.mark_built(bug_path, bug_inputs, bug_extra)
                except Exception as exc:
                    print(f"    [bulletin][warn] channel bug render failed: {exc}")
                    bug_path = None

            # ── 4) Per-story compose + (optional) takeover between stories.
            for i, c in enumerate(clips):
                raw_path = c.get("raw_path")
                if not raw_path or not os.path.isfile(raw_path):
                    print(f"    [bulletin][warn] story_{i:02d}: missing raw clip, skipping")
                    continue

                story_dur_s = float(c.get("duration_sec") or 60.0)
                imgs = story_images[i] if i < len(story_images) else []

                # Sidebar: carousel video when we have ≥2 images and feature on.
                sidebar_path: str | None = None
                sidebar_is_video = False
                if use_sidebar and len(imgs) >= 2:
                    sidebar_video = os.path.join(bulletin_dir, f"_sidebar_{i:02d}.mp4")
                    # Content-aware cache: rebuild iff any of the input
                    # images changed (e.g. user replaced one via the
                    # editor) OR the duration changed. See compose_deps.
                    sidebar_inputs = list(imgs[:5])
                    sidebar_extra = {"duration_s": round(float(story_dur_s), 3)}
                    if compose_deps.is_fresh(sidebar_video, sidebar_inputs, sidebar_extra):
                        print(f"    [bulletin] sidebar story_{i:02d} cached (skipping)")
                        sidebar_path = sidebar_video
                        sidebar_is_video = True
                    else:
                        try:
                            build_sidebar_carousel(
                                imgs[:5], story_dur_s, sidebar_video,
                                work_dir=os.path.join(bulletin_dir, f"_sidebar_work_{i:02d}"),
                            )
                            compose_deps.mark_built(sidebar_video, sidebar_inputs, sidebar_extra)
                            sidebar_path = sidebar_video
                            sidebar_is_video = True
                        except Exception as exc:
                            print(f"    [bulletin][warn] sidebar carousel story_{i:02d}: {exc}")
                            sidebar_path = None
                if sidebar_path is None:
                    sidebar_static = os.path.join(bulletin_dir, f"_sidebar_{i:02d}.png")
                    try:
                        make_sidebar_placeholder(
                            imgs[0] if imgs else None, sidebar_static,
                        )
                        sidebar_path = sidebar_static
                    except Exception as exc:
                        print(f"    [bulletin][warn] sidebar placeholder story_{i:02d}: {exc}")
                        sidebar_path = None

                # Lower-third meta from clip data — _clip_text picks
                # summary_native (Telugu/Hindi/…) when language is not
                # English, so the broadcast strap reads in-script.
                importance = int(c.get("importance") or 5)
                kicker = "BREAKING" if importance >= 8 else "NEWS"
                story_meta = StoryMeta(
                    title=(_clip_text(c) or "KAIZER NEWS")[:200],
                    kicker=kicker,
                    language=lang_cfg.code,
                    story_index=i,
                    total_stories=len(clips),
                    importance=importance,
                )

                composed_path = os.path.join(bulletin_dir, f"composed_story_{i:02d}.mp4")
                composed_ok = False

                # Build the deps fingerprint for this composed story.
                # PiP makes this branch take an extra source clip, so we
                # resolve the pip choice up-front to include in the deps.
                pip_src = pick_pip_source(clips, i) if use_pip else None
                composed_inputs = [
                    raw_path, sidebar_path, ticker_path, bug_path,
                    lang_cfg.font_primary,
                ]
                if pip_src:
                    composed_inputs.append(pip_src[0])
                composed_extra = {
                    "title":       story_meta.title,
                    "kicker":      story_meta.kicker,
                    "language":    story_meta.language,
                    "importance":  int(story_meta.importance),
                    "story_index": int(story_meta.story_index),
                    "total":       int(story_meta.total_stories),
                    "use_pip":     bool(pip_src),
                    "pip_start_s": round(float(pip_src[1]), 3) if pip_src else None,
                    "pip_dur_s":   round(float(pip_src[2]), 3) if pip_src else None,
                    "sidebar_is_video": bool(sidebar_is_video),
                }

                # Takeover helper — emits an inter-story full-screen
                # transition AFTER a story (skipped on first/last).
                def _maybe_takeover():
                    last_idx = len(clips) - 1
                    if not (use_takeovers and 0 < i < last_idx and len(imgs) >= 2):
                        return
                    takeover_dur = max(4.0, min(8.0, story_dur_s * 0.10))
                    takeover_path = os.path.join(bulletin_dir, f"takeover_{i:02d}.mp4")
                    takeover_inputs = list(imgs[:4])
                    takeover_extra = {"duration_s": round(float(takeover_dur), 3)}
                    if compose_deps.is_fresh(takeover_path, takeover_inputs, takeover_extra):
                        print(f"    [bulletin] takeover_{i:02d}.mp4 cached (skipping)")
                        story_paths.append(takeover_path)
                        return
                    try:
                        build_fullscreen_takeover(
                            imgs[:4], takeover_dur, takeover_path,
                            work_dir=os.path.join(bulletin_dir, f"_takeover_work_{i:02d}"),
                        )
                        compose_deps.mark_built(takeover_path, takeover_inputs, takeover_extra)
                        story_paths.append(takeover_path)
                        print(f"    [bulletin] takeover_{i:02d}.mp4 ({takeover_dur:.1f}s)")
                    except Exception as exc:
                        print(f"    [bulletin][warn] takeover_{i:02d}: {exc}")

                # Content-aware cache: skip rebuild only when every
                # input (raw clip, sidebar, ticker, bug, font, pip
                # source) AND every text/flag param is unchanged.
                if compose_deps.is_fresh(composed_path, composed_inputs, composed_extra):
                    print(f"    [bulletin] composed_story_{i:02d}.mp4 cached (skipping compose)")
                    composed_ok = True
                    story_paths.append(composed_path)
                    _maybe_takeover()
                    continue

                try:
                    if pip_src and sidebar_path and ticker_path:
                        pip_clip, pip_start, pip_dur = pip_src
                        compose_pip_story(
                            raw_path, story_meta, composed_path,
                            pip_clip_path=pip_clip,
                            pip_start_s=pip_start,
                            pip_duration_s=pip_dur,
                            sidebar_path=sidebar_path,
                            ticker_path=ticker_path,
                            channel_bug_path=bug_path,
                            font_path=lang_cfg.font_primary,
                            sidebar_is_video=sidebar_is_video,
                            work_dir=bulletin_dir,
                        )
                        composed_ok = True
                    elif sidebar_path and ticker_path:
                        compose_bulletin_story(
                            raw_path, story_meta, composed_path,
                            sidebar_path=sidebar_path,
                            ticker_path=ticker_path,
                            channel_bug_path=bug_path,
                            font_path=lang_cfg.font_primary,
                            sidebar_is_video=sidebar_is_video,
                            work_dir=bulletin_dir,
                        )
                        composed_ok = True
                except Exception as exc:
                    print(f"    [bulletin][warn] compose story_{i:02d}: {exc}")

                if composed_ok and os.path.isfile(composed_path):
                    compose_deps.mark_built(composed_path, composed_inputs, composed_extra)
                    story_paths.append(composed_path)
                    print(f"    [bulletin] composed_story_{i:02d}.mp4 "
                          f"({kicker}, {story_dur_s:.0f}s)")
                else:
                    # Fall back to raw slice so the story still ships.
                    story_paths.append(raw_path)
                    print(f"    [bulletin] story_{i:02d} fell back to raw slice")

                _maybe_takeover()

        # ── Stitch the assembled segment list into the final bulletin ──
        if not story_paths:
            print("  [bulletin] FAILED: no segments produced")
            return
        print(f"\n  [bulletin] Stitching {len(story_paths)} segments ...")
        from pipeline_core.bulletin_stitcher import stitch_bulletin, BulletinStitchError
        try:
            result = stitch_bulletin(
                story_paths,
                bulletin_out,
                work_dir=bulletin_dir,
            )
        except BulletinStitchError as exc:
            print(f"  [bulletin] FAILED: {exc}")
            return
        for w in result.warnings:
            print(f"    [bulletin][warn] {w}")
        print(
            f"  [bulletin] Done: {result.stories_rendered} segments, "
            f"{result.stories_skipped} skipped, "
            f"total={result.total_duration_s/60:.1f} min"
        )
        print(f"  [bulletin] Output: {bulletin_out}")

        bulletin_storage_url = ""
        bulletin_storage_key = ""
        bulletin_storage_backend = ""
        try:
            url, key, backend = _maybe_upload_final(
                bulletin_out,
                key=f"bulletin/{timestamp}/bulletin.mp4",
                content_type="video/mp4",
            )
            if url:
                print(f"  [bulletin] Uploaded → {url}")
                bulletin_storage_url = url
                bulletin_storage_key = key
                bulletin_storage_backend = backend
        except Exception as exc:
            print(f"  [bulletin] R2 upload skipped: {exc}")

        # ── Emit editor_meta.json so runner._import_clips creates ONE
        # Clip row pointing at bulletin.mp4 (NOT the per-story
        # composed_story_NN.mp4 intermediates, which would otherwise
        # show up as separate Shorts cards in the web UI). The runner
        # reads the [kaizer:meta] marker line below to locate this
        # file unambiguously.
        meta_path = ""
        try:
            meta_path = os.path.join(OUTPUT_DIR, "editor_meta.json")
            # Build a one-entry "clips" list that points at the single
            # stitched bulletin output. Total duration comes from the
            # stitcher's probe (canonical), with a fallback to the sum
            # of per-story durations for robustness.
            total_dur_s = float(getattr(result, "total_duration_s", 0.0) or 0.0)
            if total_dur_s <= 0.0:
                total_dur_s = sum(float(c.get("duration_sec") or 0.0) for c in clips)
            bulletin_text = (gemini_result.get("overall_summary_native")
                             or gemini_result.get("overall_summary_telugu")
                             or gemini_result.get("overall_summary")
                             or "Long-form bulletin")
            bulletin_meta_clip = {
                "clip_path":      os.path.abspath(bulletin_out),
                "thumb_path":     "",
                "image_path":     "",
                "duration":       total_dur_s,
                "frame_type":     "bulletin",
                "text":           bulletin_text[:500],
                "sentiment":      "",
                "entities":       people or [],
                "card_params":    {},
                "section_pct":    {},
                "follow_params":  {},
                "storage_url":    bulletin_storage_url,
                "storage_key":    bulletin_storage_key,
                "storage_backend": bulletin_storage_backend,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "render_mode":  "bulletin",
                    "platform":     platform,
                    "language":     lang_cfg.code,
                    "stories":      result.stories_rendered,
                    "skipped":      result.stories_skipped,
                    "duration_s":   total_dur_s,
                    "clips":        [bulletin_meta_clip],
                }, f, ensure_ascii=False, indent=2)
            print(f"[kaizer:meta] {meta_path}")
        except Exception as exc:
            print(f"  [bulletin][warn] editor_meta.json write failed: {exc}")
            meta_path = ""
        # Return the meta path so run_compound_pipeline can locate the
        # bulletin.mp4 it just produced and run the image_plan overlay
        # step on top of it. Standalone bulletin callers (CLI / runner)
        # ignore the return value as before.
        return meta_path

    # ════ STEP 3: Generate Title ════
    # Compound short-circuit: when run_compound_pipeline has populated
    # gemini_result["shorts_headline_native"] in a prior single Gemini
    # call, skip the ChatGPT title turn entirely and use Gemini's
    # headline. Standalone shorts jobs (no compound) keep the existing
    # ChatGPT path so behaviour is unchanged for them.
    _shorts_headline_native = (gemini_result.get("shorts_headline_native") or "").strip()
    if _shorts_headline_native:
        print(f"\n  [3/{TOTAL}] Using Gemini-supplied shorts headline (no ChatGPT call) ...")
        title_result = {
            "title_native":  _shorts_headline_native,
            "title_telugu":  _shorts_headline_native,  # legacy alias
            "title_english": "",
            "source":        "gemini-compound",
        }
    else:
        print(f"\n  [3/{TOTAL}] Generating {lang_cfg.name_english} headline ...")
        title_result = generate_title_chatgpt(summary, people, topics, language=lang_cfg.code)
    title_native = title_result.get("title_native") or title_result.get("title_telugu", "KAIZER NEWS")
    title_en = title_result.get("title_english", "")
    # Legacy alias — several downstream code paths still read title_te
    title_te = title_native
    print(f"    Title ({lang_cfg.code}): {title_native}")
    print(f"    Title (en): {title_en}")

    # Save title
    with open(os.path.join(OUTPUT_DIR, "title.json"), "w", encoding="utf-8") as f:
        json.dump(title_result, f, ensure_ascii=False, indent=2)

    # ════ STEP 4: Get Real News Images ════
    print(f"\n  [4/{TOTAL}] Searching for news images ...")
    search_queries = gemini_result.get("image_search_queries", [])
    locations = gemini_result.get("key_locations", [])

    # Short-circuit: user chose "use my default image" → skip Pexels + Gemini
    # card fallbacks entirely and reuse the uploaded asset for every clip.
    _default_img = os.environ.get("KAIZER_DEFAULT_IMAGE", "").strip()
    if _default_img and os.path.exists(_default_img):
        print(f"    ✓ Using user default image: {_default_img}")
        images = [_default_img] * (len(clips) + 2)
    else:
        images = search_news_images(
            search_queries, people, topics + locations,
            OUTPUT_DIR, count=len(clips) + 2,
            language=lang_cfg.code,
        )

    # If no real images found, generate news cards as fallback
    if not images:
        print("    No real images found — generating news card graphics ...")
        img_dir = os.path.join(OUTPUT_DIR, "images")
        os.makedirs(img_dir, exist_ok=True)
        fallback_terms = (search_queries or topics or lang_cfg.news_search_seed)
        for i, kw in enumerate(fallback_terms[:len(clips) + 2]):
            card_path = os.path.join(img_dir, f"card_{i+1:02d}.jpg")
            generate_news_card(kw, card_path, preset["width"],
                               _e(int(preset["height"] * 0.3690)))
            images.append(card_path)

    # Pad images list — repeat the last good image, or fall back to the
    # per-job default logo (only if the user configured one), not a hardcoded
    # Kaizer brand.
    while len(images) < len(clips):
        images.append(images[-1] if images else (DEFAULT_LOGO or ""))

    # ════ STEP 5: Compose Final Clips ════
    print(f"\n  [5/{TOTAL}] Composing broadcast layout ({FRAME_LAYOUTS[frame_layout]}) ...")

    # Resolve language-specific font + follow-bar text once for this job.
    # Telugu defaults to NotoSansTelugu-Bold (legible at the 80px headline
    # size, matches the editor's default selection); every other language
    # uses the bundled Noto Sans of its own script.
    if lang_cfg.code == "te":
        lang_font_basename = "NotoSansTelugu-Bold.ttf"
    else:
        lang_font_basename = os.path.basename(lang_cfg.font_primary) or "NotoSans-Bold.ttf"
    lang_follow_text = lang_cfg.follow_bar_text or "FOLLOW KAIZER NEWS"

    # ── Per-clip compose worker ─────────────────────────────────────
    # Each iteration is fully independent — different out_path, thumb,
    # storage key, and the FFmpeg subprocesses don't share any in-
    # process state. Safe to fan out across a ThreadPoolExecutor when
    # KAIZER_COMPOSE_CONCURRENCY > 1.
    #
    # Concurrency tradeoff: each FFmpeg subprocess uses all CPU cores
    # by default. Two simultaneous renders thus contend for the same
    # cores. The win is that the per-clip Python setup work
    # (Playwright HTML→PNG render of the card, PIL image prep,
    # storage upload) overlaps with another clip's FFmpeg run. Best
    # results on a 4-core box: concurrency=2, with
    # KAIZER_FFMPEG_THREADS=2 so each render gets half the cores.
    def _compose_one(i: int, clip: dict) -> Optional[dict]:
        raw_path = clip.get("raw_path")
        if not raw_path or not os.path.exists(raw_path):
            return None

        out_path = os.path.join(OUTPUT_DIR, f"clip_{i+1:02d}.mp4")
        img = images[i] if i < len(images) else images[-1]
        card_text = title_native or "KAIZER NEWS"

        print(f"    Composing clip {i+1} ({frame_layout}, font={lang_font_basename}) ...")

        if frame_layout == "split_frame":
            compose_split_frame(raw_path, img, out_path, preset, platform=platform)
            compose_meta = {}
            clip_card_params = {"font_file": lang_font_basename}
            clip_split_params = {"bg_color": "#1a0a2e"}
            clip_follow_params = {}
        elif frame_layout == "clean_card":
            compose_meta = compose_clip_clean_card(
                raw_path, img, card_text, out_path, preset,
                font_size=80,
                font_file=lang_font_basename,
                bg_color="#c10000",
                video_pct=0.50,
                headline_pct=0.18,
                image_h_pct=0.30,
                image_w_pct=0.80,
                image_border_px=14,
                image_border_color="#ffffff",
                platform=platform,
            )
            clip_card_params = {
                "font_size": compose_meta.get("font_size", 80),
                "font_file": compose_meta.get("font_file", lang_font_basename),
                "bg_color":  "#c10000",
                "image_border_px":   14,
                "image_border_color":"#ffffff",
            }
            clip_split_params  = {}
            clip_follow_params = {}
        elif frame_layout == "follow_bar":
            _default_velvet = {
                "c-top":"#2d0a4e","c-bot":"#1a0a2e","c-vdark":"#0a001a","c-vlight":"#3d0060",
                "patch-scale":80,"octaves":5,"contrast":107,"brightness":35,
                "warp":55,"warp-scale":65,"grain":14,"edge-dark":33,
                "c-dot":"#7b3fb8","dot-op":38,"dot-r":5,"dot-sp":18,"dot-rows":5,"dot-cols":5
            }
            compose_follow_bar(raw_path, out_path, preset,
                               title_text=card_text,
                               font_file=lang_font_basename,
                               text_color="#ffff00",
                               bg_color="#1a0a2e",
                               follow_text=lang_follow_text,
                               velvet_style=_default_velvet,
                               platform=platform)
            compose_meta = {}
            clip_card_params = {"font_file": lang_font_basename,
                                 "font_size": 60, "text_color": "#ffff00"}
            clip_split_params = {}
            clip_follow_params = {"bg_color": "#1a0a2e", "text_color": "#ffff00",
                                   "follow_text": lang_follow_text,
                                   "follow_text_color": "#ffffff", "social_logos": [],
                                   "velvet_style": _default_velvet}
        else:
            compose_meta = compose_clip(
                raw_path, img, card_text, out_path, preset,
                font_size=80,
                font_file=lang_font_basename,
                section_pct={"video": 0.4619, "text": 0.1691, "image": 0.3690},
                card_style={"card_c0": "#c10000", "card_c1": "#800000",
                            "edge": 9, "jag": 60, "seed": 7,
                            "vsid": 35, "vcor": 72, "vwid": 74, "overlap": 20},
                platform=platform,
            )
            clip_card_params = {
                "font_size": compose_meta.get("font_size", 80),
                "font_file": compose_meta.get("font_file", lang_font_basename),
                "card_c0": "#c10000", "card_c1": "#800000",
                "edge": 9, "jag": 60, "seed": 7,
                "vsid": 35, "vcor": 72, "vwid": 74, "overlap": 20,
            }
            clip_split_params  = {}
            clip_follow_params = {}

        # Generate thumbnail
        thumb_path = os.path.join(OUTPUT_DIR, f"thumb_{i+1:02d}.jpg")
        try:
            subprocess.run([
                FFMPEG_BIN, "-y", "-i", out_path,
                "-vframes", "1", "-q:v", "2", thumb_path
            ], capture_output=True, check=True)
        except Exception:
            thumb_path = ""

        # ── Phase 5: post-render storage upload ──────────────────────
        storage_url_ec  = ""
        storage_key_ec  = ""
        storage_bknd_ec = ""
        try:
            _clip_key = f"clips/{timestamp}/{i+1:02d}.mp4"
            storage_url_ec, storage_key_ec, storage_bknd_ec = _maybe_upload_final(
                os.path.abspath(out_path),
                _clip_key,
                content_type="video/mp4",
            )
        except Exception as _upload_err:
            print(f"  [storage] upload failed for clip {i+1}: {_upload_err}", flush=True)

        return {
            "clip_path":        os.path.abspath(out_path),
            "raw_path":         os.path.abspath(raw_path),
            "thumb_path":       os.path.abspath(thumb_path) if thumb_path else "",
            "image_path":       os.path.abspath(img),
            "text":             card_text,
            "language":         lang_cfg.code,
            "title_native":     title_native,
            "title_telugu":     title_native,  # legacy alias
            "title_english":    title_en,
            "start":            clip.get("start", ""),
            "end":              clip.get("end", ""),
            "duration":         clip.get("duration_sec", 0),
            "summary":          clip.get("summary", ""),
            "mood":             clip.get("mood", ""),
            "importance":       clip.get("importance", 5),
            "video_type":       video_type,
            "frame_type":       frame_layout,
            "card_params":      clip_card_params,
            "split_params":     clip_split_params,
            "follow_params":    clip_follow_params,
            "preset":           preset,
            # Phase 5 storage fields — empty strings when backend is local
            "storage_url":      storage_url_ec,
            "storage_key":      storage_key_ec,
            "storage_backend":  storage_bknd_ec,
        }

    # Concurrency knob — default 1 (sequential, identical to legacy
    # behaviour). Set KAIZER_COMPOSE_CONCURRENCY=2 alongside
    # KAIZER_FFMPEG_THREADS=2 on a 4-core box to roughly halve a
    # multi-clip job's wall-clock without raising peak CPU above 100%.
    _compose_n = max(1, int(os.environ.get("KAIZER_COMPOSE_CONCURRENCY", "1") or "1"))
    editor_clips: list[dict] = []
    if _compose_n <= 1 or len(clips) <= 1:
        for _i, _clip in enumerate(clips):
            ec = _compose_one(_i, _clip)
            if ec:
                editor_clips.append(ec)
    else:
        from concurrent.futures import ThreadPoolExecutor
        print(f"    [parallel] composing {len(clips)} clips with concurrency={_compose_n}")
        with ThreadPoolExecutor(max_workers=_compose_n,
                                thread_name_prefix="kaizer-compose") as _ex:
            # executor.map preserves submission order, so editor_clips
            # ends up in clip-index order without an explicit sort.
            for ec in _ex.map(lambda args: _compose_one(*args),
                              [(i, c) for i, c in enumerate(clips)]):
                if ec:
                    editor_clips.append(ec)

    # ════ STEP 6: Save Metadata + Launch Editor ════
    print(f"\n  [6/{TOTAL}] Saving metadata & launching editor ...")

    editor_meta = {
        "video_path": os.path.abspath(video_path),
        "platform": platform,
        "frame_layout": frame_layout,
        "language":    lang_cfg.code,
        "language_english": lang_cfg.name_english,
        "language_native": lang_cfg.name_native,
        "script":      lang_cfg.script,
        "preset": preset,
        "video_type": video_type,
        "title_native":  title_native,
        "title_telugu":  title_native,  # legacy alias
        "title_english": title_en,
        "summary": summary,
        "summary_native": summary_native,
        "summary_telugu": summary_native,  # legacy alias
        "people": people,
        "topics": topics,
        "keywords": keywords,
        "clips": editor_clips,
        "created": timestamp,
    }

    meta_path = os.path.join(OUTPUT_DIR, "editor_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(editor_meta, f, ensure_ascii=False, indent=2)

    # Machine-parseable marker line for runner.py — canonical source for the
    # post-pipeline clip-import step.  ASCII-only so it survives any encoding.
    print(f"[kaizer:meta] {os.path.abspath(meta_path)}", flush=True)

    # Print summary report
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Frame layout: {FRAME_LAYOUTS[frame_layout]}")
    print(f"  Platform    : {preset['label']}")
    print(f"  Language    : {lang_cfg.name_english} ({lang_cfg.script})")
    print(f"  Video type  : {video_type}")
    print(f"  Clips       : {len(editor_clips)}")
    print(f"  Title ({lang_cfg.code}): {title_native}")
    print(f"  Title (en)  : {title_en}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"  Metadata   : {meta_path}")

    # Save readable report
    report_lines = [
        "KAIZER NEWS — API Pipeline Report",
        "=" * 50,
        f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Video      : {os.path.basename(video_path)}",
        f"Duration   : {sec_to_ts(vinfo['duration'])}",
        f"Platform   : {preset['label']}",
        f"Frame      : {FRAME_LAYOUTS[frame_layout]}",
        f"Video type : {video_type}",
        f"Speakers   : {gemini_result.get('total_speakers', '?')}",
        f"Language   : {lang_cfg.name_english} ({lang_cfg.script}) — detected: {gemini_result.get('language', '?')}",
        "",
        f"Title ({lang_cfg.code}) : {title_native}",
        f"Title (en) : {title_en}",
        "",
        "CLIPS:",
    ]
    for i, c in enumerate(editor_clips, 1):
        report_lines.append(f"  Clip {i}: {c['start']} -> {c['end']} ({c['duration']:.1f}s)")
        report_lines.append(f"    Summary: {c['summary']}")
        report_lines.append(f"    Mood: {c['mood']} | Importance: {c['importance']}")
    report_lines.append("")
    report_lines.append(f"Overall summary: {summary}")

    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Launch web editor — kill any existing server on port 7654 first
    try:
        if sys.platform == "win32":
            _kill = subprocess.run(
                ["powershell", "-Command",
                 "Get-NetTCPConnection -LocalPort 7654 -ErrorAction SilentlyContinue | "
                 "Select-Object -ExpandProperty OwningProcess | "
                 "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"],
                capture_output=True
            )
        else:
            subprocess.run(["fuser", "-k", "7654/tcp"], capture_output=True)
        time.sleep(0.8)
    except Exception:
        pass

    try:
        editor_script = os.path.join(BASE_DIR, "scripts", "12_web_editor.py")
        if os.path.exists(editor_script):
            subprocess.Popen(
                [sys.executable, editor_script, meta_path],
                creationflags=0x00000008 if sys.platform == "win32" else 0
            )
            time.sleep(1.5)  # give server a moment to start
            import webbrowser
            webbrowser.open("http://localhost:7654")
            print("  Web editor launching at http://localhost:7654")
        else:
            print(f"  Editor not found at {editor_script}")
            print(f"  Run manually: python scripts/12_web_editor.py \"{meta_path}\"")
    except Exception as e:
        print(f"  Could not auto-launch editor: {e}")


    print("=" * 60)
    return meta_path


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="KAIZER NEWS Pipeline")
    _parser.add_argument("video", nargs="?", help="Path to video file")
    _parser.add_argument("--platform", default=None, choices=list(PLATFORM_PRESETS.keys()),
                         help="Platform preset key")
    _parser.add_argument("--frame", default=None, choices=list(FRAME_LAYOUTS.keys()),
                         help="Frame layout key")
    _parser.add_argument("--render-mode", default=None,
                         choices=["bulletin"],
                         help="Long-form render mode. 'bulletin' stitches all "
                              "Gemini-extracted stories into one 1-2hr MP4 for "
                              "YouTube Full. Omit for the default per-clip "
                              "Shorts pipeline.")
    _parser.add_argument("--resume-dir", default=None,
                         help="Re-enter an existing OUTPUT_DIR instead of "
                              "creating a new timestamped one. Useful when a "
                              "previous run failed late (e.g. final stitch) — "
                              "already-cut slices, OpenAI images, sidebar "
                              "carousels, composed stories, and takeovers are "
                              "reused, saving cost + time. Path can be "
                              "relative or absolute.")
    # Bulletin-only feature toggles (silently ignored for Shorts):
    _parser.add_argument("--no-broadcast-overlay", action="store_true",
                         help="bulletin: skip TV9 graphics; emit bare slice stitch.")
    _parser.add_argument("--no-carousel-sidebar", action="store_true",
                         help="bulletin: skip Ken-Burns image carousel sidebar.")
    _parser.add_argument("--no-carousel-takeovers", action="store_true",
                         help="bulletin: skip full-screen image takeovers between stories.")
    _parser.add_argument("--pip", action="store_true",
                         help="bulletin: enable picture-in-picture inset on stories "
                              "where Gemini provides a usable secondary subject.")
    _parser.add_argument("--language", default="te",
                         help="Language code (te|hi|ta|kn|ml|bn|mr|gu|en). Default: te")
    _parser.add_argument("--default-image", default="",
                         help="Absolute path to an image the user uploaded; when set, "
                              "every clip uses THIS image instead of a fetched stock photo.")
    _parser.add_argument("--compound", action="store_true",
                         help="Run the dedicated 'Full Video + Shorts' compound "
                              "pipeline: ONE Gemini call returns both cut sets + "
                              "image_plan + shorts_headline_native + bulletin marquee, "
                              "then the bulletin and shorts passes run in-process "
                              "back-to-back. Replaces the legacy two-subprocess fan-out.")
    _parser.add_argument("--use-default-brand-image", action="store_true",
                         help="Compound mode only: when set, the shorts pass uses the "
                              "image at --default-image as its lower-panel art and "
                              "skips image search/generation. Pairs with --compound.")
    _args = _parser.parse_args()
    _video = _args.video
    if _args.default_image:
        os.environ["KAIZER_DEFAULT_IMAGE"] = _args.default_image

    if not _video:
        for ext in ("*.mp4", "*.MP4", "*.mkv", "*.avi"):
            import glob as _gl
            found = _gl.glob(os.path.join(BASE_DIR, ext))
            if found:
                print("  Available videos:")
                for i, f in enumerate(found, 1):
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    print(f"    {i}. {os.path.basename(f)} ({size_mb:.0f} MB)")
                try:
                    choice = input("  Select video [1]: ").strip() or "1"
                    _video = found[int(choice) - 1]
                except (ValueError, IndexError, EOFError, KeyboardInterrupt):
                    _video = found[0]
                break

    if not _video:
        print("Usage: python scripts/11_api_pipeline.py <video_path> [--platform X] [--frame Y]")
        sys.exit(1)

    # Bulletin feature toggles — flow into the bulletin branch of run_pipeline
    # via a module-level dict (only honoured when --render-mode bulletin).
    _BULLETIN_FLAG_OVERRIDES = {
        "overlay":   not _args.no_broadcast_overlay,
        "sidebar":   not _args.no_carousel_sidebar,
        "takeovers": not _args.no_carousel_takeovers,
        "pip":       bool(_args.pip),
    }

    if _args.compound:
        # Compound dedicated pipeline — ONE Gemini call, both formats.
        run_compound_pipeline(
            os.path.abspath(_video),
            language=_args.language,
            frame_layout=_args.frame or "torn_card",
            use_default_brand_image=bool(_args.use_default_brand_image),
        )
    else:
        run_pipeline(os.path.abspath(_video), platform=_args.platform,
                     frame_layout=_args.frame, language=_args.language,
                     render_mode=_args.render_mode,
                     resume_dir=_args.resume_dir)
