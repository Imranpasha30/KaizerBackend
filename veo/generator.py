"""Veo 3 video generation — submit, poll, download.

Uses the `google-genai` SDK (pip install google-genai).  Requires billing
enabled on the Google Cloud project that owns GEMINI_API_KEY.

Operations are asynchronous — submission returns an `operation` object that
must be polled until `done`.  Typical wall time: 30-180 seconds.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional


# Model names — change VEO_MODEL via env if Google ships a newer one.
VEO_MODEL = os.getenv("KAIZER_VEO_MODEL", "veo-3.0-generate-preview")
POLL_INTERVAL_SEC = 6
POLL_TIMEOUT_SEC  = 10 * 60   # 10 minutes is plenty for an 8s clip


class VeoError(Exception):
    """Raised on terminal Veo API failures."""


def _client():
    """Build a google-genai client bound to our API key."""
    try:
        from google import genai as _genai
    except ImportError as e:
        raise VeoError(
            "google-genai SDK not installed. Run: venv/Scripts/pip install google-genai"
        ) from e
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise VeoError("GEMINI_API_KEY is not set")
    return _genai.Client(api_key=api_key)


def generate_video(
    prompt: str,
    out_path: str | Path,
    *,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 8,
    negative_prompt: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate a video with Veo 3 and write it to `out_path` as MP4.

    Returns a dict with metadata (model, prompt, duration, aspect, path, bytes).
    Raises `VeoError` on failure (non-billing project, invalid prompt, timeout).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        print(f"[veo] {msg}", flush=True)
        if progress_cb:
            try: progress_cb(msg)
            except Exception: pass

    client = _client()

    from google.genai import types as _types

    cfg = _types.GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        duration_seconds=int(duration_seconds),
        # Veo 3 defaults to 1 video per call; we don't need more.
        number_of_videos=1,
        person_generation="allow_all",     # required for news-style b-roll
    )
    if negative_prompt:
        cfg.negative_prompt = negative_prompt

    _log(f"Submitting to {VEO_MODEL} ({aspect_ratio}, {duration_seconds}s) …")
    op = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt,
        config=cfg,
    )

    # Poll until done
    started = time.time()
    while not getattr(op, "done", False):
        if time.time() - started > POLL_TIMEOUT_SEC:
            raise VeoError(f"Veo operation timed out after {POLL_TIMEOUT_SEC}s")
        _log(f"polling… ({int(time.time() - started)}s elapsed)")
        time.sleep(POLL_INTERVAL_SEC)
        op = client.operations.get(op)

    # Error branch
    err = getattr(op, "error", None)
    if err:
        raise VeoError(f"Veo generation failed: {err}")

    # Success — pull the first generated video
    resp = getattr(op, "response", None)
    videos = getattr(resp, "generated_videos", None) or []
    if not videos:
        raise VeoError("Veo returned no videos — check prompt for policy flags.")

    v = videos[0]
    # The SDK gives us either `.video.video_bytes` (bytes) OR a `.video.uri`
    # (Cloud Storage link).  Prefer bytes; fall back to download.
    blob = getattr(getattr(v, "video", None), "video_bytes", None)
    if blob is None:
        uri = getattr(getattr(v, "video", None), "uri", "")
        if not uri:
            raise VeoError("Veo response had neither inline bytes nor a URI.")
        import urllib.request
        with urllib.request.urlopen(uri) as r:
            blob = r.read()

    out.write_bytes(blob)
    _log(f"wrote {len(blob)} bytes → {out}")

    return {
        "model":           VEO_MODEL,
        "prompt":          prompt,
        "aspect_ratio":    aspect_ratio,
        "duration_seconds": duration_seconds,
        "path":            str(out.resolve()),
        "bytes":           len(blob),
    }
