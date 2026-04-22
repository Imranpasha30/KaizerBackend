"""
kaizer.pipeline.variants
=========================
Turns ONE rendered master clip → three platform-specific MP4 variants,
each differing enough to dodge Instagram's visual-similarity dedupe.

Usage
-----
    from pipeline_core.variants import generate_variants, PLATFORM_VARIANT_SPECS

    variants = generate_variants(
        "/path/to/master.mp4",
        platforms=["youtube_short", "instagram_reel", "tiktok"],
        output_dir="/path/to/out",
        caption_text="Breaking news headline",
        source_language="te",
    )
    for v in variants:
        print(v.platform, v.qa_ok, v.originality_vs_source)

PlatformVariant fields
----------------------
  platform               : str
  output_path            : str
  bitrate_kbps           : float
  fps                    : float
  loop_score             : LoopScore | None
  caption_overlay_applied: bool
  cta_applied            : str
  originality_vs_source  : float     — [0, 1]; ~0 = identical, 1 = maximally different
  qa_ok                  : bool
  warnings               : list[str]
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pipeline_core.loop_score import LoopScore, score_loop
from pipeline_core.pipeline import FFMPEG_BIN, ENCODE_ARGS_INTERMEDIATE
from pipeline_core.captions import render_caption, safe_zone as captions_safe_zone
from pipeline_core.cta_overlay import apply_cta
from pipeline_core.qa import validate_output
from pipeline_core.validator import validate_input

logger = logging.getLogger("kaizer.pipeline.variants")

# ── Supported platforms ───────────────────────────────────────────────────────

_SUPPORTED_PLATFORMS: frozenset[str] = frozenset({
    "youtube_short",
    "instagram_reel",
    "tiktok",
})


# ── VariantSpec ───────────────────────────────────────────────────────────────

@dataclass
class VariantSpec:
    """Static configuration for a single platform variant."""

    platform: str               # 'youtube_short' | 'instagram_reel' | 'tiktok'
    label: str
    bitrate_kbps: int
    maxrate_kbps: int
    bufsize_kbps: int
    fps: Optional[int]          # None = preserve source
    safe_zone: tuple[int, int, int, int]   # (x, y, w, h) for 1080x1920
    caption_style: str          # 'audio_forward' | 'caption_heavy' | 'trending_overlay'
    cta_style: str
    cta_text_default: str
    requires_loop: bool


PLATFORM_VARIANT_SPECS: dict[str, VariantSpec] = {
    "youtube_short": VariantSpec(
        platform="youtube_short",
        label="YT Shorts",
        bitrate_kbps=8000,
        maxrate_kbps=10000,
        bufsize_kbps=16000,
        fps=None,
        safe_zone=(60, 120, 900, 1520),
        caption_style="audio_forward",
        cta_style="related_video",
        cta_text_default="Watch full video on my channel →",
        requires_loop=False,
    ),
    "instagram_reel": VariantSpec(
        platform="instagram_reel",
        label="IG Reels",
        bitrate_kbps=7500,
        maxrate_kbps=10000,
        bufsize_kbps=16000,
        fps=30,
        safe_zone=(60, 250, 840, 1350),
        caption_style="caption_heavy",
        cta_style="soft_follow",
        cta_text_default="Send this to someone who needs to see it →",
        requires_loop=True,
    ),
    "tiktok": VariantSpec(
        platform="tiktok",
        label="TikTok",
        bitrate_kbps=6000,
        maxrate_kbps=10000,
        bufsize_kbps=12000,
        fps=None,
        safe_zone=(60, 150, 860, 1420),
        caption_style="trending_overlay",
        cta_style="url_overlay",
        cta_text_default="Follow for more ⇢",
        requires_loop=False,
    ),
}


# ── PlatformVariant ───────────────────────────────────────────────────────────

@dataclass
class PlatformVariant:
    """Result of processing one platform in generate_variants()."""

    platform: str
    output_path: str
    bitrate_kbps: float
    fps: float
    loop_score: Optional[LoopScore]
    caption_overlay_applied: bool
    cta_applied: str
    originality_vs_source: float
    qa_ok: bool
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _originality_delta(variant_path: str, master_path: str) -> float:
    """Sample ~10 evenly-spaced frames from each file, resize to 256x256,
    compute mean absolute pixel difference / 255.

    Returns a value in [0, 1].  Identical files → ~0.  Uses cv2.
    Degrades gracefully: returns 0.0 if either file cannot be opened.
    """
    def _sample_frames(path: str, n_frames: int = 10) -> list[np.ndarray]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning("_originality_delta: cannot open %s", path)
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return []
        indices = [int(total * i / n_frames) for i in range(n_frames)]
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                frames.append(resized.astype(np.float32))
        cap.release()
        return frames

    variant_frames = _sample_frames(variant_path)
    master_frames = _sample_frames(master_path)

    if not variant_frames or not master_frames:
        logger.warning(
            "_originality_delta: could not sample frames from variant=%s or master=%s",
            variant_path,
            master_path,
        )
        return 0.0

    n = min(len(variant_frames), len(master_frames))
    diffs: list[float] = []
    for v_frame, m_frame in zip(variant_frames[:n], master_frames[:n]):
        mean_diff = float(np.mean(np.abs(v_frame - m_frame))) / 255.0
        diffs.append(mean_diff)

    result = float(np.mean(diffs)) if diffs else 0.0
    return min(1.0, max(0.0, result))


def _probe_fps(path: str) -> float:
    """Return the fps of the first video stream via ffprobe. Falls back to 0.0."""
    try:
        from pipeline_core.qa import FFPROBE_BIN as _ffprobe  # already resolved
    except Exception:
        import shutil as _sh
        _ffprobe = _sh.which("ffprobe") or "ffprobe"

    import json as _json
    cmd = [
        _ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = _json.loads(proc.stdout)
            streams = data.get("streams", [])
            if streams:
                r = streams[0].get("r_frame_rate") or streams[0].get("avg_frame_rate") or "0/1"
                parts = r.split("/")
                if len(parts) == 2:
                    num, den = float(parts[0]), float(parts[1])
                    return num / den if den != 0.0 else 0.0
                return float(r)
    except Exception as exc:
        logger.warning("_probe_fps failed for %s: %s", path, exc)
    return 0.0


def _probe_bitrate_kbps(path: str) -> float:
    """Return best-effort video bitrate in kbps via ffprobe."""
    try:
        from pipeline_core.qa import FFPROBE_BIN as _ffprobe
    except Exception:
        import shutil as _sh
        _ffprobe = _sh.which("ffprobe") or "ffprobe"

    import json as _json
    cmd = [
        _ffprobe,
        "-v", "error",
        "-show_entries", "format=bit_rate",
        "-of", "json",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = _json.loads(proc.stdout)
            br = data.get("format", {}).get("bit_rate")
            if br:
                return float(br) / 1000.0
    except Exception as exc:
        logger.warning("_probe_bitrate_kbps failed for %s: %s", path, exc)
    return 0.0


def _build_ffmpeg_encode_cmd(
    input_path: str,
    output_path: str,
    cfg: VariantSpec,
) -> list[str]:
    """Build the FFmpeg re-encode command for a platform variant.

    Uses:
      -b:v {bitrate}k -maxrate {maxrate}k -bufsize {bufsize}k
      optional -r {fps}
      -pix_fmt yuv420p + bt709 colour tags
      -profile:v high -level 4.1 -movflags +faststart
      -c:a copy  (audio was already normalised in master)
    """
    bitrate_str   = f"{cfg.bitrate_kbps}k"
    maxrate_str   = f"{cfg.maxrate_kbps}k"
    bufsize_str   = f"{cfg.bufsize_kbps}k"

    cmd: list[str] = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
    ]

    # Optional fps override
    if cfg.fps is not None:
        cmd += ["-r", str(cfg.fps)]

    cmd += [
        # Video codec
        "-c:v",            "libx264",
        "-preset",         "medium",
        "-b:v",            bitrate_str,
        "-maxrate",        maxrate_str,
        "-bufsize",        bufsize_str,
        # Colour / pixel format
        "-pix_fmt",        "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc",      "bt709",
        "-colorspace",     "bt709",
        # Profile / container
        "-profile:v",      "high",
        "-level",          "4.1",
        "-movflags",       "+faststart",
        # Audio: copy — master audio was already normalised
        "-c:a",            "copy",
        output_path,
    ]
    return cmd


def _apply_caption_overlay(
    input_path: str,
    output_path: str,
    caption_text: str,
    cfg: VariantSpec,
    warnings: list[str],
    source_language: Optional[str],
) -> bool:
    """Render caption_text as a PNG and FFmpeg-overlay it at the safe zone's
    lower-third.  Returns True on success, False on failure (appends warning).

    The PNG is rendered via captions.render_caption (Indic-aware) then
    composited via FFmpeg's overlay filter.
    """
    # Determine safe zone (x, y, w, h)
    sz_x, sz_y, sz_w, sz_h = cfg.safe_zone

    # Font size: scale from reference width 1080
    font_size = max(32, int(64 * sz_w / 900))

    # Script hint from source_language
    script_hint: Optional[str] = None
    if source_language:
        _lang_script = {
            "te": "telugu", "tel": "telugu",
            "hi": "devanagari", "hin": "devanagari",
            "mr": "devanagari", "mar": "devanagari",
            "ta": "tamil",    "tam": "tamil",
            "bn": "bengali",  "ben": "bengali",
            "kn": "kannada",  "kan": "kannada",
            "ml": "malayalam","mal": "malayalam",
            "gu": "gujarati", "guj": "gujarati",
        }
        script_hint = _lang_script.get(source_language.lower())

    render_kwargs: dict = dict(
        max_width=sz_w,
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

    try:
        caption_result = render_caption(caption_text, **render_kwargs)
        warnings.extend(caption_result.warnings)
    except Exception as exc:
        warnings.append(f"caption render failed for {cfg.platform!r}: {exc}")
        logger.warning("Caption render error platform=%s: %s", cfg.platform, exc)
        return False

    # Position: lower-third of safe zone
    overlay_w = caption_result.width
    overlay_h = caption_result.height
    overlay_x = sz_x + max(0, (sz_w - overlay_w) // 2)
    lower_third_y = sz_y + int(sz_h * 0.80)
    overlay_y = min(lower_third_y, 1920 - overlay_h - 10)
    overlay_y = max(0, overlay_y)

    tmp_png: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_png = tf.name.replace("\\", "/")
        caption_result.image.save(tmp_png)

        filter_str = (
            "[1]format=rgba[cap];"
            f"[0][cap]overlay=x={overlay_x}:y={overlay_y}"
        )

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", input_path,
            "-i", tmp_png,
            "-filter_complex", filter_str,
            "-c:v", "libx264",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-profile:v", "high",
            "-level", "4.1",
            "-movflags", "+faststart",
            "-c:a", "copy",
            output_path,
        ]
        logger.debug("Caption overlay FFmpeg cmd: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            warnings.append(
                f"Caption overlay FFmpeg failed for {cfg.platform!r} "
                f"(rc={proc.returncode}): {stderr_tail}"
            )
            logger.warning(
                "Caption overlay failed platform=%s rc=%d",
                cfg.platform, proc.returncode,
            )
            return False

        logger.info(
            "Caption overlay applied: platform=%s %dx%d at (%d,%d) → %s",
            cfg.platform, overlay_w, overlay_h, overlay_x, overlay_y, output_path,
        )
        return True

    except Exception as exc:
        warnings.append(
            f"Caption overlay error for {cfg.platform!r}: {exc}"
        )
        logger.warning("Caption overlay exception platform=%s: %s", cfg.platform, exc)
        return False

    finally:
        if tmp_png and os.path.isfile(tmp_png):
            try:
                os.remove(tmp_png)
            except OSError as rm_exc:
                logger.warning(
                    "Could not remove temp caption PNG %s: %s", tmp_png, rm_exc
                )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_variants(
    master_clip_path: str,
    *,
    platforms: list[str],
    output_dir: str,
    caption_text: Optional[str] = None,
    cta_text_override: Optional[dict] = None,
    source_language: Optional[str] = None,
) -> list[PlatformVariant]:
    """Produce per-platform variants in input order.

    Parameters
    ----------
    master_clip_path : str
        Absolute path to the rendered master clip (input to all platforms).
    platforms : list[str]
        Ordered list of platforms to process.  Empty list returns [].
        Unknown platform name raises ValueError.
    output_dir : str
        Directory to write final variant MP4 files.
    caption_text : str | None
        If given, rendered as a caption overlay at the safe-zone lower-third.
    cta_text_override : dict | None
        Mapping of platform → CTA text string.  Falls back to each
        VariantSpec's cta_text_default when the platform key is absent.
    source_language : str | None
        ISO 639-1 language hint for font/text selection (e.g. 'te', 'hi').

    Returns
    -------
    list[PlatformVariant]
        One entry per requested platform, in input order.

    Raises
    ------
    ValueError
        If any requested platform is not in PLATFORM_VARIANT_SPECS.
    """
    if not platforms:
        return []

    # Validate requested platform names up-front
    unknown = [p for p in platforms if p not in _SUPPORTED_PLATFORMS]
    if unknown:
        raise ValueError(
            f"Unknown platform(s): {unknown!r}. "
            f"Supported: {sorted(_SUPPORTED_PLATFORMS)}."
        )

    # Normalise paths to forward slashes
    master_clip_path = master_clip_path.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)

    cta_text_override = cta_text_override or {}

    results: list[PlatformVariant] = []

    for platform in platforms:
        cfg = PLATFORM_VARIANT_SPECS[platform]
        warnings: list[str] = []

        logger.info(
            "generate_variants: starting platform=%s master=%s",
            platform, master_clip_path,
        )

        # ── 1. validate_input ─────────────────────────────────────────────────
        val = validate_input(master_clip_path)
        if not val.ok:
            raise RuntimeError(
                f"Master clip failed input validation for platform {platform!r}: "
                f"{val.errors}"
            )
        warnings.extend(val.warnings)

        # ── 2. FFmpeg re-encode into temp dir ─────────────────────────────────
        tmp_dir = tempfile.mkdtemp(prefix="kaizer_variant_").replace("\\", "/")
        encoded_path = f"{tmp_dir}/{platform}_encoded.mp4"

        encode_cmd = _build_ffmpeg_encode_cmd(master_clip_path, encoded_path, cfg)
        logger.debug("Re-encode cmd: %s", " ".join(encode_cmd))
        try:
            proc = subprocess.run(
                encode_cmd, capture_output=True, text=True, timeout=600
            )
            if proc.returncode != 0:
                stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
                raise RuntimeError(
                    f"FFmpeg re-encode failed for {platform!r} "
                    f"(rc={proc.returncode}): {stderr_tail}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"FFmpeg re-encode timed out for platform {platform!r}."
            )

        logger.info(
            "Re-encode done: platform=%s → %s", platform, encoded_path
        )

        # ── 3. Caption overlay ────────────────────────────────────────────────
        caption_overlay_applied = False
        current_path = encoded_path

        if caption_text:
            cap_out = f"{tmp_dir}/{platform}_captioned.mp4"
            caption_overlay_applied = _apply_caption_overlay(
                current_path,
                cap_out,
                caption_text,
                cfg,
                warnings,
                source_language,
            )
            if caption_overlay_applied and os.path.isfile(cap_out):
                current_path = cap_out

        # ── 4. CTA overlay ────────────────────────────────────────────────────
        cta_text = cta_text_override.get(platform) or cfg.cta_text_default
        cta_out = f"{tmp_dir}/{platform}_cta.mp4"

        cta_applied_style = cfg.cta_style
        try:
            cta_result = apply_cta(
                current_path,
                cta_style=cfg.cta_style,
                output_path=cta_out,
                text=cta_text,
                platform=platform,
                cta_duration_s=3.0,
                source_language=source_language,
            )
            warnings.extend(cta_result.warnings)
            current_path = cta_out
            cta_applied_style = cta_result.cta_style
        except Exception as exc:
            warnings.append(
                f"CTA overlay failed for {platform!r}: {exc}. Proceeding without CTA."
            )
            cta_applied_style = "none"
            logger.warning(
                "CTA overlay exception platform=%s: %s", platform, exc
            )

        # ── 5. Loop scoring (IG Reels only) ───────────────────────────────────
        loop_score_result: Optional[LoopScore] = None
        if cfg.requires_loop:
            try:
                loop_score_result = score_loop(current_path)
                logger.info(
                    "Loop score platform=%s overall=%.1f phash=%d xcorr=%.2f motion=%.2f",
                    platform,
                    loop_score_result.overall,
                    loop_score_result.visual_phash_distance,
                    loop_score_result.audio_xcorr,
                    loop_score_result.motion_continuity,
                )
                if loop_score_result.suggestions:
                    for suggestion in loop_score_result.suggestions:
                        warnings.append(f"loop_score suggestion: {suggestion}")
                if loop_score_result.overall < 60:
                    warnings.append(
                        f"Loop score {loop_score_result.overall:.1f}/100 is below 60 for "
                        f"{platform!r}. Consider crossfade or beat-aligned trim (v2 fix)."
                    )
            except Exception as exc:
                warnings.append(
                    f"Loop scoring failed for {platform!r}: {exc}"
                )
                logger.warning(
                    "Loop score exception platform=%s: %s", platform, exc
                )

        # ── Copy to final output dir ──────────────────────────────────────────
        final_path = f"{output_dir}/{platform}.mp4"
        try:
            import shutil as _shutil
            _shutil.copy2(current_path, final_path)
            final_path = final_path.replace("\\", "/")
        except Exception as exc:
            raise RuntimeError(
                f"Could not copy variant to output dir for {platform!r}: {exc}"
            ) from exc

        # ── 6. Originality delta ──────────────────────────────────────────────
        originality = _originality_delta(final_path, master_clip_path)
        if originality < 0.15:
            warnings.append(
                f"Originality score {originality:.3f} is below 0.15 for {platform!r}. "
                "Consider adding extra transforms (colour grade, letterbox, animated text) "
                "to reduce IG visual-similarity dedupe risk."
            )
            logger.warning(
                "Low originality for platform=%s score=%.3f", platform, originality
            )

        # ── 7. QA ─────────────────────────────────────────────────────────────
        qa_ok = False
        try:
            qa_result = validate_output(final_path, platform=platform)
            qa_ok = qa_result.ok
            warnings.extend(qa_result.warnings)
            if qa_result.errors:
                for err in qa_result.errors:
                    warnings.append(f"QA error: {err}")
            logger.info(
                "QA %s platform=%s ok=%s",
                "PASSED" if qa_ok else "FAILED", platform, qa_ok,
            )
        except Exception as exc:
            warnings.append(f"QA validation raised exception for {platform!r}: {exc}")
            logger.warning(
                "QA exception platform=%s: %s", platform, exc
            )

        # ── Probe final fps + bitrate for variant metadata ────────────────────
        final_fps = _probe_fps(final_path)
        final_bitrate_kbps = _probe_bitrate_kbps(final_path)

        # ── 8. Clean temp files ───────────────────────────────────────────────
        try:
            import shutil as _shutil
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning(
                "Could not remove temp dir %s: %s", tmp_dir, exc
            )

        # ── Assemble result ───────────────────────────────────────────────────
        variant = PlatformVariant(
            platform=platform,
            output_path=final_path,
            bitrate_kbps=final_bitrate_kbps,
            fps=final_fps,
            loop_score=loop_score_result,
            caption_overlay_applied=caption_overlay_applied,
            cta_applied=cta_applied_style,
            originality_vs_source=originality,
            qa_ok=qa_ok,
            warnings=warnings,
        )
        results.append(variant)
        logger.info(
            "generate_variants: finished platform=%s output=%s qa_ok=%s originality=%.3f warnings=%d",
            platform,
            final_path,
            qa_ok,
            originality,
            len(warnings),
        )

    return results
