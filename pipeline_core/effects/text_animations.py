"""
kaizer.pipeline.effects.text_animations
=========================================
Animated text overlay renderer.

Produces per-frame RGBA PIL images that can be baked onto a video clip via
FFmpeg overlay. Five animation types are supported:

  typewriter  — characters revealed one at a time; last 0.5 s holds full string
  word_pop    — each word scales from 0.8× → 1.0× with 0.1 s stagger
  bounce_in   — full text bounces in from below with ease-out
  slide_up    — full text slides from below-canvas to final position
  karaoke     — per-word colour highlight as spoken; requires word_timings

Usage
-----
    from pipeline_core.effects.text_animations import TextAnimationSpec, apply_text_animation

    spec = TextAnimationSpec(
        name='bounce_in', text='Breaking News', duration_s=1.5, start_s=0.2,
        font_size=72, color='#FFFFFF', position=(540, 1600), platform='youtube_short',
    )
    out = apply_text_animation('/tmp/clip.mp4', spec, output_path='/tmp/out.mp4')
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pipeline_core.pipeline import FFMPEG_BIN, ENCODE_ARGS_SHORT_FORM
from pipeline_core.captions import render_caption, safe_zone, detect_script

logger = logging.getLogger("kaizer.pipeline.editor_pro.text_animations")

# ── Public constants ──────────────────────────────────────────────────────────

SUPPORTED_ANIMATIONS: tuple[str, ...] = (
    "typewriter",
    "word_pop",
    "bounce_in",
    "slide_up",
    "karaoke",
)

# Default colours
_HIGHLIGHT_COLOR = "#FFFF00"   # karaoke highlight
_DEFAULT_BG_COLOR = "#00000088"


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class TextAnimationSpec:
    """Specification for a single animated text overlay.

    Attributes
    ----------
    name : str
        One of SUPPORTED_ANIMATIONS.
    text : str
        The full text string to animate.
    duration_s : float
        Total duration of the animation in seconds.
    start_s : float
        When (in clip time) the animation begins.
    font_size : int
        Base font size in pixels at the reference canvas size.
    color : str
        Primary text colour as a hex string (e.g. '#FFFFFF').
    position : tuple[int, int]
        (x, y) top-left origin of the text region on the canvas.
    platform : str
        Platform identifier — passed to captions.safe_zone for boundary checks.
    params : dict
        Extra per-animation parameters (e.g. 'word_timings' for karaoke).
    """

    name: str
    text: str
    duration_s: float
    start_s: float
    font_size: int
    color: str
    position: tuple[int, int]
    platform: str = "youtube_short"
    params: dict = field(default_factory=dict)


# ── Easing functions ──────────────────────────────────────────────────────────

def _ease_out_cubic(t: float) -> float:
    """Ease-out cubic: fast start, slow finish.  t ∈ [0, 1]."""
    return 1.0 - (1.0 - max(0.0, min(1.0, t))) ** 3


def _ease_out_bounce(t: float) -> float:
    """Approximate ease-out bounce for the bounce_in effect."""
    t = max(0.0, min(1.0, t))
    n1, d1 = 7.5625, 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    elif t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


# ── Caption rendering helper ──────────────────────────────────────────────────

def _render_text_image(
    text: str,
    font_size: int,
    color: str,
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    """Render *text* to a transparent RGBA image using captions.render_caption.

    Falls back to a minimal Pillow draw if render_caption raises.
    """
    max_w = max(100, canvas_w - 80)
    try:
        result = render_caption(
            text,
            max_width=max_w,
            font_size=font_size,
            color=color,
            stroke_color="#000000",
            stroke_width=max(1, font_size // 20),
        )
        return result.image.convert("RGBA")
    except Exception as exc:
        logger.warning("_render_text_image: render_caption failed (%s); using fallback", exc)
        # Minimal fallback
        img = Image.new("RGBA", (max_w, font_size + 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((4, 4), text, fill=color, font=font)
        return img


def _render_word_image(
    word: str,
    font_size: int,
    color: str,
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    """Render a single word, returning a tight RGBA image."""
    return _render_text_image(word, font_size, color, canvas_w, canvas_h)


# ── Per-animation frame generators ───────────────────────────────────────────

def _frames_typewriter(
    spec: TextAnimationSpec,
    canvas_size: tuple[int, int],
    fps: int,
    n_frames: int,
) -> list[Image.Image]:
    """Reveal one character per frame until full string, hold last 0.5 s."""
    canvas_w, canvas_h = canvas_size
    frames: list[Image.Image] = []

    text = spec.text
    total_chars = max(1, len(text))
    hold_frames = int(0.5 * fps)
    reveal_frames = max(1, n_frames - hold_frames)

    x, y = spec.position

    for i in range(n_frames):
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        if i < reveal_frames:
            progress = (i + 1) / reveal_frames
            visible = max(1, int(progress * total_chars))
        else:
            visible = total_chars

        partial = text[:visible]
        if partial:
            glyph = _render_text_image(partial, spec.font_size, spec.color, canvas_w, canvas_h)
            px = max(0, min(x, canvas_w - glyph.width))
            py = max(0, min(y, canvas_h - glyph.height))
            canvas.paste(glyph, (px, py), glyph)
        frames.append(canvas)

    return frames


def _frames_word_pop(
    spec: TextAnimationSpec,
    canvas_size: tuple[int, int],
    fps: int,
    n_frames: int,
) -> list[Image.Image]:
    """Each word scales 0.8× → 1.0× with 0.1 s stagger between words."""
    canvas_w, canvas_h = canvas_size
    frames: list[Image.Image] = []

    words = spec.text.split()
    if not words:
        return [Image.new("RGBA", canvas_size, (0, 0, 0, 0))] * n_frames

    stagger_frames = max(1, int(0.1 * fps))
    pop_frames = max(1, int(0.2 * fps))   # each word animates over ~0.2 s
    x_base, y_base = spec.position

    # Pre-render each word at full size
    word_images: list[Image.Image] = [
        _render_word_image(w, spec.font_size, spec.color, canvas_w, canvas_h)
        for w in words
    ]

    # Compute word layout (left-to-right, wrap on canvas boundary)
    positions: list[tuple[int, int]] = []
    cx, cy = x_base, y_base
    for img in word_images:
        if cx + img.width > canvas_w - 40:
            cx = x_base
            cy += img.height + 8
        positions.append((cx, cy))
        cx += img.width + max(4, spec.font_size // 8)

    for frame_idx in range(n_frames):
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        for wi, (word_img, (wx, wy)) in enumerate(zip(word_images, positions)):
            word_start_frame = wi * stagger_frames
            elapsed = frame_idx - word_start_frame
            if elapsed < 0:
                continue  # word hasn't appeared yet
            t = min(1.0, elapsed / max(1, pop_frames))
            scale = 0.8 + 0.2 * _ease_out_cubic(t)

            new_w = max(1, int(word_img.width * scale))
            new_h = max(1, int(word_img.height * scale))
            scaled = word_img.resize((new_w, new_h), Image.LANCZOS)

            # Centre the scaled image on the word's target position
            paste_x = max(0, wx + (word_img.width - new_w) // 2)
            paste_y = max(0, wy + (word_img.height - new_h) // 2)
            canvas.paste(scaled, (paste_x, paste_y), scaled)
        frames.append(canvas)

    return frames


def _frames_bounce_in(
    spec: TextAnimationSpec,
    canvas_size: tuple[int, int],
    fps: int,
    n_frames: int,
) -> list[Image.Image]:
    """Full text bounces in from below with ease-out bounce over first 0.6 s, then holds."""
    canvas_w, canvas_h = canvas_size
    frames: list[Image.Image] = []

    glyph = _render_text_image(spec.text, spec.font_size, spec.color, canvas_w, canvas_h)
    target_x, target_y = spec.position
    off_screen_y = canvas_h + glyph.height

    bounce_frames = max(1, int(0.6 * fps))

    for i in range(n_frames):
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        t = min(1.0, i / max(1, bounce_frames - 1))
        ease = _ease_out_bounce(t)

        cur_y = int(off_screen_y + (target_y - off_screen_y) * ease)
        px = max(0, min(target_x, canvas_w - glyph.width))
        py = max(0, min(cur_y, canvas_h - glyph.height))
        canvas.paste(glyph, (px, py), glyph)
        frames.append(canvas)

    return frames


def _frames_slide_up(
    spec: TextAnimationSpec,
    canvas_size: tuple[int, int],
    fps: int,
    n_frames: int,
) -> list[Image.Image]:
    """Full text slides from below canvas to final position with ease-out cubic."""
    canvas_w, canvas_h = canvas_size
    frames: list[Image.Image] = []

    glyph = _render_text_image(spec.text, spec.font_size, spec.color, canvas_w, canvas_h)
    target_x, target_y = spec.position
    start_y = canvas_h + glyph.height

    slide_frames = max(1, int(0.5 * fps))

    for i in range(n_frames):
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        t = min(1.0, i / max(1, slide_frames - 1))
        ease = _ease_out_cubic(t)

        cur_y = int(start_y + (target_y - start_y) * ease)
        px = max(0, min(target_x, canvas_w - glyph.width))
        py = min(cur_y, canvas_h - 1)
        if py >= -glyph.height:
            canvas.paste(glyph, (px, py), glyph)
        frames.append(canvas)

    return frames


def _frames_karaoke(
    spec: TextAnimationSpec,
    canvas_size: tuple[int, int],
    fps: int,
    n_frames: int,
) -> list[Image.Image]:
    """Highlight each word as it's spoken, based on word_timings in spec.params."""
    canvas_w, canvas_h = canvas_size
    frames: list[Image.Image] = []

    word_timings: list[dict[str, Any]] = spec.params.get("word_timings", [])
    words = spec.text.split() if spec.text else []

    if not word_timings:
        # Infer uniform timing from the words
        total_dur = spec.duration_s
        word_timings = [
            {"word": w, "start": i * total_dur / max(1, len(words))}
            for i, w in enumerate(words)
        ]

    x_base, y_base = spec.position
    inactive_color = spec.color
    highlight_color = spec.params.get("highlight_color", _HIGHLIGHT_COLOR)

    # Pre-render both inactive and highlighted versions of each word
    word_imgs_inactive: list[Image.Image] = [
        _render_word_image(w, spec.font_size, inactive_color, canvas_w, canvas_h)
        for w in words
    ]
    word_imgs_highlight: list[Image.Image] = [
        _render_word_image(w, spec.font_size, highlight_color, canvas_w, canvas_h)
        for w in words
    ]

    # Compute layout
    positions: list[tuple[int, int]] = []
    cx, cy = x_base, y_base
    for img in word_imgs_inactive:
        if cx + img.width > canvas_w - 40:
            cx = x_base
            cy += img.height + 8
        positions.append((cx, cy))
        cx += img.width + max(4, spec.font_size // 8)

    # Build timing index: which word is active at each frame
    def _active_word_index(t_s: float) -> int:
        active = -1
        for wi, wt in enumerate(word_timings):
            wstart = float(wt.get("start", 0.0))
            if t_s >= wstart - spec.start_s:
                active = wi
            else:
                break
        return active

    for fi in range(n_frames):
        t_abs = spec.start_s + fi / fps
        active_wi = _active_word_index(t_abs)
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        for wi, (px, py) in enumerate(positions):
            if wi <= active_wi:
                img = word_imgs_highlight[wi]
            else:
                img = word_imgs_inactive[wi]
            ppx = max(0, min(px, canvas_w - img.width))
            ppy = max(0, min(py, canvas_h - img.height))
            canvas.paste(img, (ppx, ppy), img)
        frames.append(canvas)

    return frames


# ── Public API ────────────────────────────────────────────────────────────────

def render_animation_frames(
    spec: TextAnimationSpec,
    *,
    canvas_size: tuple[int, int] = (1080, 1920),
    fps: int = 30,
) -> list[Image.Image]:
    """Produce per-frame RGBA PNG overlays representing the animation.

    Returns a list with exactly ``ceil(duration_s * fps)`` frames.
    Each frame is an RGBA PIL Image at *canvas_size*.

    Parameters
    ----------
    spec : TextAnimationSpec
        Animation specification.
    canvas_size : tuple[int, int]
        (width, height) of each output frame in pixels.
    fps : int
        Frames per second.

    Returns
    -------
    list[PIL.Image.Image]
        RGBA frames.

    Raises
    ------
    ValueError
        If ``spec.name`` is not in SUPPORTED_ANIMATIONS.
    """
    if spec.name not in SUPPORTED_ANIMATIONS:
        raise ValueError(
            f"Unknown animation {spec.name!r}. "
            f"Valid: {SUPPORTED_ANIMATIONS}"
        )

    n_frames = math.ceil(spec.duration_s * fps)
    n_frames = max(1, n_frames)

    dispatchers = {
        "typewriter": _frames_typewriter,
        "word_pop":   _frames_word_pop,
        "bounce_in":  _frames_bounce_in,
        "slide_up":   _frames_slide_up,
        "karaoke":    _frames_karaoke,
    }

    try:
        frames = dispatchers[spec.name](spec, canvas_size, fps, n_frames)
    except Exception as exc:
        logger.warning(
            "render_animation_frames: %s failed (%s); returning blank frames",
            spec.name, exc,
        )
        frames = [Image.new("RGBA", canvas_size, (0, 0, 0, 0))] * n_frames

    # Ensure exact frame count
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else Image.new("RGBA", canvas_size, (0, 0, 0, 0)))
    frames = frames[:n_frames]

    return frames


def apply_text_animation(
    video_path: str,
    spec: TextAnimationSpec,
    *,
    output_path: str,
) -> str:
    """Bake a TextAnimationSpec onto *video_path*.

    Renders frames via render_animation_frames, writes them to a temp PNG
    sequence, then uses FFmpeg's movie source + overlay filter to composite
    the animation onto the video at the specified start time.

    Audio is preserved unchanged (-c:a copy).

    Parameters
    ----------
    video_path : str
        Absolute path to the source video.
    spec : TextAnimationSpec
        Animation specification.
    output_path : str
        Absolute path for the output MP4.

    Returns
    -------
    str
        *output_path* on success.

    Raises
    ------
    ValueError
        If ``spec.name`` is not in SUPPORTED_ANIMATIONS.
    RuntimeError
        If FFmpeg fails.  Callers should catch and record as a warning.
    """
    if spec.name not in SUPPORTED_ANIMATIONS:
        raise ValueError(
            f"Unknown animation {spec.name!r}. "
            f"Valid: {SUPPORTED_ANIMATIONS}"
        )

    video_path  = video_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fps = 30
    frames = render_animation_frames(spec, canvas_size=(1080, 1920), fps=fps)

    tmp_dir = tempfile.mkdtemp(prefix="kaizer_textanim_")
    try:
        # Write PNG sequence
        for idx, frame in enumerate(frames):
            frame_path = os.path.join(tmp_dir, f"frame_{idx:04d}.png")
            frame.save(frame_path)

        n_frames = len(frames)
        anim_dur = n_frames / fps

        # Build overlay command using FFmpeg's image2 demuxer for the PNG
        # sequence, then overlay=enable='between(t,...)'
        seq_pattern = os.path.join(tmp_dir, "frame_%04d.png").replace("\\", "/")
        start_s = float(spec.start_s)
        end_s   = start_s + anim_dur

        filter_complex = (
            f"[1:v]setpts=PTS-STARTPTS+{start_s:.4f}/TB[anim];"
            f"[0:v][anim]overlay=0:0:enable='between(t,{start_s:.4f},{end_s:.4f})'[vout]"
        )

        cmd: list[str] = [
            FFMPEG_BIN, "-y",
            "-i", video_path,
            "-framerate", str(fps),
            "-i", seq_pattern,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a?",
            "-c:a", "copy",
        ] + [a for a in ENCODE_ARGS_SHORT_FORM
             if a not in ("-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                          "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                          "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.85:level=disabled")] + [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        # Simplified + safe cmd — override with clean explicit list
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", video_path,
            "-framerate", str(fps),
            "-i", seq_pattern,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        logger.debug("apply_text_animation(%s): %s", spec.name, " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            raise RuntimeError(
                f"apply_text_animation FFmpeg failed (rc={proc.returncode}) "
                f"animation={spec.name!r}:\n{stderr_tail}"
            )

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning("apply_text_animation: could not clean tmp dir %s: %s", tmp_dir, exc)

    logger.info(
        "apply_text_animation: %s @ %.2fs → %s",
        spec.name, spec.start_s, os.path.basename(output_path),
    )
    return output_path
