"""
kaizer.pipeline.cta_overlay
============================
End-card / CTA overlay renderer for the Kaizer News video pipeline.

Applies a mode-appropriate call-to-action to the last N seconds of a clip.
Uses ``captions.render_caption`` for Indic-safe text rendering and FFmpeg's
``overlay`` filter for compositing — so the CTA occupies the lower third of
the safe zone, with audio untouched.

Usage
-----
    from pipeline_core.cta_overlay import apply_cta, CTAResult

    result = apply_cta(
        "/path/to/clip.mp4",
        cta_style="soft_follow",
        output_path="/path/to/output.mp4",
        platform="youtube_short",
        cta_duration_s=3.0,
        source_language="te",
    )
    print(result.cta_style, result.cta_start_s)

CTAResult fields
----------------
  output_path    : str
  cta_style      : str
  cta_start_s    : float
  cta_duration_s : float   — may differ from the requested value if the clip
                             was too short (see shrink-to-fit logic below)
  warnings       : list[str]
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

from pipeline_core.pipeline import FFMPEG_BIN, ENCODE_ARGS_SHORT_FORM
from pipeline_core.captions import render_caption, safe_zone, detect_script

logger = logging.getLogger("kaizer.pipeline.cta_overlay")

# ── Valid CTA styles ──────────────────────────────────────────────────────────

_VALID_STYLES: frozenset[str] = frozenset({
    "soft_follow",
    "related_video",
    "next_part",
    "url_overlay",
    "none",
})

# Default Telugu translations for soft_follow (the most locale-sensitive style)
_TELUGU_SOFT_FOLLOW = "మరిన్నింటికి Follow"

# Font-size at reference width of 1080 px.  Scales proportionally for other widths.
_REFERENCE_FONT_SIZE = 64
_REFERENCE_WIDTH = 1080


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class CTAResult:
    """Result of a single :func:`apply_cta` call.

    Attributes
    ----------
    output_path : str
        Absolute path to the rendered output file.
    cta_style : str
        The CTA style that was applied (may be 'none' for pass-throughs).
    cta_start_s : float
        Timestamp (in seconds) at which the CTA overlay becomes visible.
    cta_duration_s : float
        How long (seconds) the CTA overlay is visible.  May differ from the
        requested value when the clip was shorter than ``cta_duration_s``.
    warnings : list[str]
        Non-fatal issues collected during processing.
    """

    output_path: str
    cta_style: str
    cta_start_s: float
    cta_duration_s: float
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _probe_duration(video_path: str) -> float:
    """Return duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        from pipeline_core.qa import FFPROBE_BIN as _ffprobe  # type: ignore
    except Exception:
        import shutil as _sh
        _ffprobe = _sh.which("ffprobe") or "ffprobe"

    cmd = [
        _ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            import json
            data = json.loads(proc.stdout)
            return float(data.get("format", {}).get("duration", 0.0))
    except Exception as exc:
        logger.warning("cta_overlay: ffprobe duration probe failed: %s", exc)
    return 0.0


def _probe_video_size(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream. Falls back to (1080, 1920)."""
    try:
        from pipeline_core.qa import FFPROBE_BIN as _ffprobe  # type: ignore
    except Exception:
        import shutil as _sh
        _ffprobe = _sh.which("ffprobe") or "ffprobe"

    cmd = [
        _ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]
    try:
        import json
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            if streams:
                w = int(streams[0].get("width") or 1080)
                h = int(streams[0].get("height") or 1920)
                return w, h
    except Exception as exc:
        logger.warning("cta_overlay: ffprobe size probe failed: %s", exc)
    return 1080, 1920


def _resolve_cta_text(
    cta_style: str,
    text: str | None,
    source_language: str | None,
) -> str:
    """Return the display text for a given cta_style.

    For 'soft_follow', picks a Telugu default when source_language is 'te'.
    For 'url_overlay', `text` is REQUIRED.
    """
    if cta_style == "none":
        return ""

    if cta_style == "url_overlay":
        if not text:
            raise ValueError(
                "cta_style='url_overlay' requires `text` to be the URL string."
            )
        return text

    if text:
        return text

    # Defaults keyed by style
    _defaults: dict[str, str] = {
        "soft_follow": "Follow for more",
        "related_video": "Watch full video on my channel →",
        "next_part": "Part continues →",
    }

    default = _defaults.get(cta_style, "Follow for more")

    # Locale override for Telugu
    if cta_style == "soft_follow" and source_language in ("te", "tel"):
        default = _TELUGU_SOFT_FOLLOW

    return default


# ── Public API ────────────────────────────────────────────────────────────────

def apply_cta(
    video_path: str,
    *,
    cta_style: str,
    output_path: str,
    text: str | None = None,
    sub_text: str | None = None,
    platform: str = "youtube_short",
    cta_duration_s: float = 3.0,
    source_language: str | None = None,
) -> CTAResult:
    """Render a CTA PNG via :func:`captions.render_caption`, overlay it on the
    last *cta_duration_s* seconds of *video_path* via FFmpeg's overlay filter,
    keep audio unchanged, re-encode once using ``ENCODE_ARGS_SHORT_FORM``.

    Parameters
    ----------
    video_path : str
        Absolute path to the source clip.
    cta_style : str
        One of: ``'soft_follow'``, ``'related_video'``, ``'next_part'``,
        ``'url_overlay'``, ``'none'``.
    output_path : str
        Absolute path to write the composited MP4.
    text : str | None
        Main CTA text.  Required for 'related_video', 'next_part',
        'url_overlay'.  Optional for others (falls back to defaults).
    sub_text : str | None
        Optional second line (e.g. a URL shown below the main CTA).
    platform : str
        Platform name passed to :func:`captions.safe_zone` for positioning.
    cta_duration_s : float
        Duration (seconds) of the overlay.  Shrunk automatically when the
        video is shorter.
    source_language : str | None
        ISO 639-1 language hint for font/text selection.

    Returns
    -------
    CTAResult

    Raises
    ------
    ValueError
        If *cta_style* is not valid or if 'url_overlay' is used without *text*.
    RuntimeError
        If the FFmpeg encode step fails.
    """
    # Normalise path separators
    video_path = video_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")

    warnings: list[str] = []

    if cta_style not in _VALID_STYLES:
        raise ValueError(
            f"Unknown cta_style {cta_style!r}. "
            f"Valid styles: {sorted(_VALID_STYLES)}."
        )

    # ── 'none' pass-through: copy via ffmpeg -c copy ──────────────────────────
    if cta_style == "none":
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", video_path,
            "-c", "copy",
            output_path,
        ]
        logger.debug("CTA none pass-through: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg copy failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-500:]}"
            )
        logger.info("CTA none pass-through: %s → %s", video_path, output_path)
        return CTAResult(
            output_path=output_path,
            cta_style="none",
            cta_start_s=0.0,
            cta_duration_s=0.0,
            warnings=warnings,
        )

    # ── Probe video duration + dimensions ─────────────────────────────────────
    dur = _probe_duration(video_path)
    if dur <= 0.0:
        warnings.append(
            f"Could not determine video duration for {video_path!r}; "
            "assuming 30 s for CTA timing."
        )
        dur = 30.0

    vid_w, vid_h = _probe_video_size(video_path)

    # ── Shrink cta_duration_s if the clip is shorter ──────────────────────────
    if cta_duration_s >= dur:
        new_dur = max(1.0, dur * 0.3)
        warnings.append(
            f"Video duration ({dur:.2f}s) <= requested cta_duration_s "
            f"({cta_duration_s:.2f}s). Shrinking CTA to {new_dur:.2f}s."
        )
        cta_duration_s = new_dur

    cta_start_s = max(0.0, dur - cta_duration_s)

    # ── Resolve CTA text ──────────────────────────────────────────────────────
    display_text = _resolve_cta_text(cta_style, text, source_language)

    # Combine main text + sub_text
    full_text = display_text
    if sub_text:
        full_text = f"{display_text}\n{sub_text}"

    # ── Detect script for font selection ──────────────────────────────────────
    script_hint: str | None = None
    if source_language in ("te", "tel"):
        script_hint = "telugu"
    elif source_language in ("hi", "hin", "mr", "mar"):
        script_hint = "devanagari"
    elif source_language in ("ta", "tam"):
        script_hint = "tamil"
    elif source_language in ("bn", "ben"):
        script_hint = "bengali"
    elif source_language in ("kn", "kan"):
        script_hint = "kannada"
    elif source_language in ("ml", "mal"):
        script_hint = "malayalam"
    elif source_language in ("gu", "guj"):
        script_hint = "gujarati"
    else:
        # Auto-detect from text content
        detected = detect_script(full_text)
        script_hint = detected if detected != "latin" else None

    # ── Compute font size + safe zone ─────────────────────────────────────────
    font_size = max(
        24,
        int(_REFERENCE_FONT_SIZE * vid_w / _REFERENCE_WIDTH),
    )

    try:
        safe_x, safe_y, safe_w, safe_h = safe_zone(platform, vid_w, vid_h)
    except ValueError:
        # Unknown platform — use generous defaults
        warnings.append(
            f"Unknown platform {platform!r} for safe_zone; using full-frame defaults."
        )
        safe_x = int(vid_w * 0.05)
        safe_y = int(vid_h * 0.05)
        safe_w = int(vid_w * 0.90)
        safe_h = int(vid_h * 0.90)

    # Lower-third: position CTA in bottom 20% of the safe zone
    caption_max_w = safe_w

    # ── Render CTA PNG via captions.render_caption ────────────────────────────
    render_kwargs: dict = dict(
        max_width=caption_max_w,
        font_size=font_size,
        color="#FFFFFF",
        stroke_color="#000000",
        stroke_width=max(2, font_size // 20),
        bg_color="#00000099",
        bg_padding=max(8, font_size // 8),
        bg_radius=12,
        align="center",
    )
    if script_hint:
        render_kwargs["script"] = script_hint

    caption_result = render_caption(full_text, **render_kwargs)
    warnings.extend(caption_result.warnings)

    # ── Compute overlay (x, y) — centred horizontally in safe zone, lower third
    overlay_w = caption_result.width
    overlay_h = caption_result.height

    # Horizontal: centre within safe zone
    overlay_x = safe_x + max(0, (safe_w - overlay_w) // 2)

    # Vertical: lower-third of safe zone (top of caption sits 80% down the safe zone)
    lower_third_y = safe_y + int(safe_h * 0.80)
    # Clamp so caption doesn't go below the frame
    overlay_y = min(lower_third_y, vid_h - overlay_h - 10)
    overlay_y = max(0, overlay_y)

    # ── Write temporary PNG ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_png = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_png = tf.name.replace("\\", "/")
        caption_result.image.save(tmp_png)
        logger.debug(
            "CTA PNG: %dx%d saved to %s (style=%s, start=%.2fs, dur=%.2fs)",
            overlay_w, overlay_h, tmp_png, cta_style, cta_start_s, cta_duration_s,
        )

        # ── Build FFmpeg filter_complex command ───────────────────────────────
        # NOTE: ENCODE_ARGS_SHORT_FORM already contains -c:a aac + loudnorm.
        # We override -c:a to 'copy' so audio is untouched (already normalised
        # by the upstream slice encode or source clip).
        # Build video-only encode args (strip any -c:a and -af entries from
        # ENCODE_ARGS_SHORT_FORM so we can append our own audio handling).
        video_encode_args: list[str] = []
        skip_next = False
        for i, arg in enumerate(ENCODE_ARGS_SHORT_FORM):
            if skip_next:
                skip_next = False
                continue
            if arg in ("-c:a", "-b:a", "-ar"):
                skip_next = True
                continue
            if arg == "-af":
                skip_next = True
                continue
            video_encode_args.append(arg)

        filter_str = (
            f"[1]format=rgba[cta];"
            f"[0][cta]overlay="
            f"x={overlay_x}:y={overlay_y}:"
            f"enable='between(t,{cta_start_s:.6f},{cta_start_s + cta_duration_s:.6f})'"
        )

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", video_path,
            "-i", tmp_png,
            "-filter_complex", filter_str,
        ] + video_encode_args + [
            "-c:a", "copy",
            output_path,
        ]

        logger.debug("CTA ffmpeg cmd: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            raise RuntimeError(
                f"FFmpeg CTA overlay failed (rc={proc.returncode}): {stderr_tail}"
            )

        logger.info(
            "CTA overlay applied: style=%s start=%.2fs dur=%.2fs → %s",
            cta_style, cta_start_s, cta_duration_s, output_path,
        )

    finally:
        if tmp_png and os.path.isfile(tmp_png):
            try:
                os.remove(tmp_png)
            except OSError as exc:
                logger.warning("Could not remove temp CTA PNG %s: %s", tmp_png, exc)

    return CTAResult(
        output_path=output_path,
        cta_style=cta_style,
        cta_start_s=cta_start_s,
        cta_duration_s=cta_duration_s,
        warnings=warnings,
    )
