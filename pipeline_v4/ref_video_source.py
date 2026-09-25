"""Reference-video AUTO-SOURCE fallback (Phase 3c).

When the operator hasn't uploaded a B-roll clip for a moment, this can fetch
ONE short related clip via yt-dlp so the AI still has something to cut to.

⚠️ OFF BY DEFAULT and OPT-IN. Enable with ``KAIZER_V4_AUTOSOURCE_VIDEO=1``.
COPYRIGHT: downloaded third-party footage may be copyrighted; inserting it into
a published video is the operator's legal responsibility. The uploaded-clip path
(a tagged UserAsset) is the safe, primary source — this is only a best-effort
fallback. Every failure is soft (returns None → no cutaway, render unaffected).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def autosource_enabled() -> bool:
    return (os.environ.get("KAIZER_V4_AUTOSOURCE_VIDEO") or "").strip().lower() in (
        "1", "on", "true", "yes")


def _yt_dlp_bin() -> Optional[str]:
    return (shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
            or shutil.which("youtube-dl"))


def autosource_clip(
    query: str, out_dir: Path, *, label: str = "",
    max_seconds: float = 8.0, timeout: int = 120,
    max_source_seconds: int = 900,
) -> Optional[dict]:
    """Fetch ONE short related clip for ``query`` into ``out_dir`` and trim it
    to ``max_seconds``. Returns ``{"filename", "label", "source_url"}`` (the
    clip lives in out_dir) or None on any failure / when disabled.

    Guards: gated by KAIZER_V4_AUTOSOURCE_VIDEO; yt-dlp must be installed;
    only the FIRST search hit; source must be < ``max_source_seconds`` (skip
    long uploads); ≤720p; hard subprocess timeouts; trimmed to max_seconds.
    """
    if not autosource_enabled():
        return None
    q = (query or "").strip()
    if not q:
        return None
    ytdlp = _yt_dlp_bin()
    if not ytdlp:
        print("[v4/autosource] yt-dlp not installed — skipping", flush=True)
        return None
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "_autosrc_raw.%(ext)s"
    try:
        # First search hit, ≤720p mp4, skip playlists + over-long uploads.
        dl = [
            ytdlp, f"ytsearch1:{q}", "--no-playlist", "--no-warnings",
            "--match-filter", f"duration < {int(max_source_seconds)}",
            "-f", "mp4[height<=720]/best[height<=720]/best",
            "--max-filesize", "150M",
            "-o", str(raw),
        ]
        subprocess.run(dl, capture_output=True, text=True, timeout=timeout, check=True)
    except Exception as exc:
        print(f"[v4/autosource] fetch failed for {q!r}: "
              f"{str(exc)[:160]}", flush=True)
        return None
    # Find whatever file yt-dlp wrote.
    got = next((p for p in out_dir.glob("_autosrc_raw.*") if p.is_file()), None)
    if not got or got.stat().st_size == 0:
        return None
    safe = "".join(c for c in (label or q)[:40] if c.isalnum() or c in " -_").strip()
    safe = (safe or "refclip").replace(" ", "_")
    final = out_dir / f"_autoref_{safe}.mp4"
    try:
        # Trim to a short cutaway + normalize to a safe encode.
        cut = [
            ffmpeg, "-y", "-v", "error", "-i", str(got),
            "-t", f"{float(max_seconds):.2f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(final),
        ]
        subprocess.run(cut, capture_output=True, text=True, timeout=timeout, check=True)
    except Exception as exc:
        print(f"[v4/autosource] trim failed for {q!r}: {str(exc)[:160]}", flush=True)
        try:
            got.unlink()
        except OSError:
            pass
        return None
    try:
        got.unlink()
    except OSError:
        pass
    if not final.is_file() or final.stat().st_size == 0:
        return None
    print(f"[v4/autosource] fetched clip for {q!r} -> {final.name}", flush=True)
    return {"filename": final.name, "label": (label or q)[:120], "source_url": q}
