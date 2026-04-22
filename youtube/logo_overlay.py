"""FFmpeg per-destination logo overlay.

Every YouTube account (OAuthToken) can carry its own overlay logo.  The
pipeline renders a CLEAN master — this helper re-encodes it just before
upload so the Auto Wala video gets Auto Wala's logo and the Cyber Sphere
video gets Cyber Sphere's.

Geometry matches the pipeline's baked-in overlay so users who were
previously relying on the old single-logo behavior see the same look.

Returns the path to the overlaid file (caller owns cleanup), or the
original path if overlay fails / no logo configured.  Never raises — a
failed overlay falls back to uploading the clean master.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _find_ffmpeg() -> str:
    """Same lookup strategy as pipeline_core: PATH first, then Railway paths."""
    which = shutil.which("ffmpeg")
    if which:
        return which
    for candidate in (
        "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg",
        "/app/.apt/usr/bin/ffmpeg",        # Railway nixpacks layout
    ):
        if os.path.exists(candidate):
            return candidate
    return "ffmpeg"   # last-ditch — hope it's on PATH


def overlay_logo(
    clip_path: str,
    logo_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Create a copy of `clip_path` with `logo_path` overlaid top-right.

    Returns the absolute path to the overlaid file (stored in a temp dir the
    caller is expected to delete once upload completes).  Returns the
    original `clip_path` unchanged when the logo is missing/empty or FFmpeg
    fails — uploads NEVER block on overlay failure.

    Geometry (matches pipeline_core/pipeline.py):
      - Logo width:   160 / 1080 of video width
      - Logo height:  134 / 1920 of video height (or 1080 for landscape)
      - Margin right: 24 / 1080 of video width
      - Margin top:   25 / 1920 of video height
    """
    if not logo_path or not os.path.exists(logo_path):
        return clip_path
    if not clip_path or not os.path.exists(clip_path):
        return clip_path

    # Compute overlay geometry from actual video dimensions, with sensible
    # defaults matching the pipeline's torn_card layout.
    vw = int(width) if width else 1080
    vh = int(height) if height else 1920
    # Aspect-aware denominator: landscape (16:9) uses 1080 as vh denominator.
    vh_denom = 1080 if vh < vw else 1920

    logo_w = max(32, int(vw * 160 / 1080))
    logo_h = max(32, int(vh * 134 / vh_denom))
    margin_r = int(vw * 24 / 1080)
    margin_t = int(vh * 25 / vh_denom)

    # Temp output path in system temp dir; caller cleans up after upload.
    td = tempfile.mkdtemp(prefix="kaizer_logo_")
    stem = Path(clip_path).stem
    out_path = str(Path(td) / f"{stem}_branded.mp4")

    ffmpeg = _find_ffmpeg()
    # ultrafast + crf 23 — favor wall-time over bitrate savings; YouTube
    # re-encodes anyway, so burning cycles here is wasteful.
    cmd = [
        ffmpeg, "-y",
        "-i", clip_path,
        "-i", logo_path,
        "-filter_complex",
        (
            f"[1:v]scale={logo_w}:{logo_h}:force_original_aspect_ratio=decrease,"
            f"pad={logo_w}:{logo_h}:(ow-iw)/2:(oh-ih)/2:color=black@0[logo];"
            f"[0:v][logo]overlay=x=W-w-{margin_r}:y={margin_t}"
        ),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "copy",   # passthrough audio — no re-encode
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not os.path.exists(out_path):
            print(f"[logo-overlay] FFmpeg failed (rc={r.returncode}): {r.stderr[-400:]}")
            try: shutil.rmtree(td, ignore_errors=True)
            except Exception: pass
            return clip_path
        return out_path
    except subprocess.TimeoutExpired:
        print("[logo-overlay] FFmpeg timed out after 180s")
        try: shutil.rmtree(td, ignore_errors=True)
        except Exception: pass
        return clip_path
    except Exception as e:
        print(f"[logo-overlay] unexpected error: {e}")
        try: shutil.rmtree(td, ignore_errors=True)
        except Exception: pass
        return clip_path


def cleanup_overlay(path: str, original: str) -> None:
    """Delete the overlaid file + its parent temp dir, ONLY if it's not the
    original file (overlay failed → we returned original → nothing to clean)."""
    if not path or path == original:
        return
    try:
        parent = Path(path).parent
        Path(path).unlink(missing_ok=True)
        # Remove the temp dir if it's empty (prefix `kaizer_logo_` is unique)
        if parent.name.startswith("kaizer_logo_"):
            try:
                parent.rmdir()
            except OSError:
                # Still has files — leave it
                pass
    except Exception as e:
        print(f"[logo-overlay] cleanup failed: {e}")
