"""cutClip — Python port of teammate's main Shorts renderer.

Orchestrates a single 1080×1920 Shorts cut:

  1. Wrap the user/Claude title text to ≤2 lines (news layout).
  2. Render the Telugu title PNG (rsvg-convert if available, else
     Pillow). Computes dynamic font size to fill ~95% panel width.
  3. Resolve the inset photo: custom upload / AI-generated / extracted
     video frame midpoint.
  4. Build the news_panel filter graph + wire input indices.
  5. Spawn ffmpeg with the right input order, hard 5-minute timeout,
     CRF 20 libx264 (matches teammate).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from . import dot_pattern, hw_accel, news_panel, telugu_title


# ── ffmpeg helpers ──────────────────────────────────────────────────

def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def _ffprobe_bin() -> str:
    return os.environ.get("FFPROBE_BIN", "ffprobe")


class CutClipError(RuntimeError):
    """ffmpeg cutClip failed, timed out, or produced no output."""


def ffprobe_dims(input_path: str, timeout_s: int = 15) -> Optional[dict]:
    """Return ``{width, height}`` for the first video stream, or
    None if probe fails. Port of teammate's ffprobeDims."""
    try:
        proc = subprocess.run(
            [_ffprobe_bin(), "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0",
             input_path],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.match(r"^(\d+)x(\d+)", (proc.stdout or "").strip())
    if not m:
        return None
    return {"width": int(m.group(1)), "height": int(m.group(2))}


def extract_frame(input_path: str, output_path: str, at_sec: float,
                  timeout_s: int = 60) -> str:
    """Pull a single JPEG frame from the source at ``at_sec``. Port
    of teammate's extractFrame."""
    proc = subprocess.run(
        [_ffmpeg_bin(),
         "-ss", str(at_sec),
         "-i",  input_path,
         "-frames:v", "1",
         "-q:v",      "2",
         "-y", output_path],
        capture_output=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[-300:]
        raise CutClipError(f"frame extract failed: {stderr}")
    return output_path


# ── Text wrapping ──────────────────────────────────────────────────

def wrap_for_overlay(text: str, max_chars_per_line: int = 18,
                     max_lines: int = 3) -> str:
    """Greedy line-wrap. Direct port of teammate's wrapForOverlay."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return ""
    words = t.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(lines) >= max_lines:
            break
        candidate = (cur + " " + w) if cur else w
        if len(candidate) <= max_chars_per_line:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            if len(w) > max_chars_per_line:
                rest = w
                while len(rest) > max_chars_per_line and len(lines) < max_lines:
                    lines.append(rest[:max_chars_per_line])
                    rest = rest[max_chars_per_line:]
                cur = rest
            else:
                cur = w
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return "\n".join(lines)


def _escape_for_drawtext(s: str) -> str:
    return (
        (s or "")
        .replace("\\", "\\\\")
        .replace(":",  r"\:")
        .replace("'",  r"\'")
    )


# ── The main entry point ───────────────────────────────────────────

def cut_clip(
    *,
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    title: str = "",
    hook: str = "",
    logo_path: Optional[str] = None,
    panel_color: str = "#dc2626",
    footer_text: str = "KAIZER NEWS NETWORK",
    custom_snap_path: Optional[str] = None,
    color_grade: str = "subtle",
    cinematic_edit: bool = False,
    timeout_s: int = 5 * 60,
    layout: str = "news",                       # news | branded
    logo_corner: str = "top-right",             # branded layout only
) -> str:
    """Cut + render a single Shorts. Returns ``output_path`` on success.

    Inputs to ffmpeg (in this fixed order, indices computed below):
      [0] source video (always)
      [1] logo PNG     (if has_logo)
      [2] snap photo   (if has_snap)
      [3] title PNG    (if has_title_png)
      [4] texture PNG  (if has_texture — currently unimplemented,
                        defer to Session 3)
    """
    if not os.path.isfile(input_path):
        raise CutClipError(f"input not found: {input_path}")
    if end_sec - start_sec < 0.5:
        raise CutClipError(f"clip too short ({end_sec - start_sec:.2f}s)")

    has_logo = bool(logo_path and os.path.isfile(logo_path))
    cleanups: list[str] = []

    # ── 1) Wrap title / hook for the 2-line news panel ─────────────
    raw = (hook or title or "").strip().replace("\\n", "\n").replace("\r", "")
    if "\n" in raw:
        explicit = [s.strip() for s in raw.split("\n") if s.strip()]
        if len(explicit) <= 2:
            wrapped = "\n".join(explicit)
        else:
            wrapped = wrap_for_overlay(" ".join(explicit), 22, 2)
    else:
        wrapped = wrap_for_overlay(raw, 22, 2)

    # ── 2) Compute dynamic font size + render title PNG ──────────
    title_png_path: Optional[str] = None
    title_png_height: int = 0
    dyn_title_size = 82

    if wrapped:
        # Strip bomb markers when measuring — they don't render to glyphs.
        longest = max(
            (len(line.replace("*", "")) for line in wrapped.split("\n")),
            default=1,
        )
        PANEL_W = 1020         # ~95% of 1080
        PER_CHAR_AT_82 = 48    # empirical Telugu Bold advance
        ideal = int((PANEL_W / max(longest, 1)) * (82 / PER_CHAR_AT_82))
        dyn_title_size = max(88, min(140, ideal))

        png_path = output_path + ".title.png"
        try:
            r = telugu_title.render_title_png(
                text=wrapped,
                output_path=png_path,
                font_size=dyn_title_size,
                fill_color="#ffffff",
                emphasis_color="#fde047",
                stroke_color="#000000",
                stroke_width=9,
            )
        except Exception as exc:
            print(f"[cut_clip] title render exception: {exc}")
            r = None
        if r:
            title_png_path   = r["path"]
            title_png_height = int(r["height"] or 0)
            cleanups.append(title_png_path)

    # ── 3) Inset photo: custom / AI / midpoint frame ─────────────
    snap_path: Optional[str] = None
    if custom_snap_path and os.path.isfile(custom_snap_path):
        snap_path = custom_snap_path   # don't add to cleanups
    else:
        snap_path = output_path + ".snap.jpg"
        try:
            mid = start_sec + (end_sec - start_sec) / 2
            extract_frame(input_path, snap_path, mid)
            cleanups.append(snap_path)
        except CutClipError:
            snap_path = None

    has_snap = bool(snap_path and os.path.isfile(snap_path))

    # ── 4) Dot-pattern texture overlay (idempotent — cached on disk) ──
    texture_path = dot_pattern.ensure_dot_pattern()
    has_texture  = bool(texture_path and os.path.isfile(texture_path))

    # ── 5) Compute input indices the builder needs ──────────────
    next_idx = 1
    logo_idx = -1
    snap_idx = -1
    title_png_input_idx = -1
    texture_input_idx   = -1
    if has_logo:
        logo_idx = next_idx; next_idx += 1
    if has_snap:
        snap_idx = next_idx; next_idx += 1
    has_title_png = bool(title_png_path and os.path.isfile(title_png_path))
    if has_title_png:
        title_png_input_idx = next_idx; next_idx += 1
    if has_texture:
        texture_input_idx = next_idx; next_idx += 1

    font_path = telugu_title.telugu_font_path()

    # ── 6) Build filter_complex (per chosen layout) ───────────
    # Indices are computed in step 5 with the actual ffmpeg input
    # positions the inputs will land at. Passing them through means
    # there's no post-hoc string substitution — which previously had
    # a nasty bug: a global ``[2:v]`` replace also clobbered the title
    # PNG's reference when both shared input index 2.
    if layout == "branded":
        filter_str, out_tag = news_panel.build_branded_layout_filter(
            has_font=bool(font_path),
            has_logo=has_logo,
            title=wrapped,
            logo_corner=logo_corner,
            font_path=font_path,
            logo_input_idx=logo_idx,
        )
        # Branded layout doesn't consume snap/title/texture — drop them
        # from the input list (step 7) so ffmpeg's arg count matches
        # the actual references in the graph.
        has_snap            = False
        has_title_png       = False
        has_texture         = False
        title_png_path      = None
        snap_path           = None
        texture_path        = None
    else:
        filter_str, out_tag = news_panel.build_news_layout_filter(
            has_font=bool(font_path),
            has_logo=has_logo,
            has_snap=has_snap,
            panel_color=panel_color,
            footer_text=footer_text,
            title_font_size=dyn_title_size,
            title_png_path=title_png_path,
            title_png_input_idx=title_png_input_idx,
            title_png_height=title_png_height,
            snap_input_idx=snap_idx,
            logo_input_idx=logo_idx,
            texture_input_idx=texture_input_idx,
            color_grade=color_grade,
            cinematic_edit=cinematic_edit,
            clip_duration_sec=max(0.5, end_sec - start_sec),
            font_path=font_path,
        )

    # Replace the sentinel for the footer (drawtext path) with the
    # safely-escaped footer string. The title-PNG path is the primary
    # one — drawtext is only used if PNG rendering failed.
    filter_str = filter_str.replace(
        "__FOOTER__", _escape_for_drawtext(footer_text or "")
    )
    # If the drawtext title path is in use (no PNG), we'd need to
    # write the wrapped title to a file and substitute __TITLE_FILE__.
    # The PNG path is preferred — only do file substitution when the
    # sentinel is actually present.
    if "__TITLE_FILE__" in filter_str and wrapped:
        title_file = output_path + ".title.txt"
        Path(title_file).write_text(wrapped, encoding="utf-8")
        cleanups.append(title_file)
        # ffmpeg textfile= needs the path WITHOUT escape backslashes
        # but with forward slashes on Windows so colons in the path
        # don't break the option parser. Quote it.
        # Same escaping rule as the font path: forward slashes for
        # separators AND backslash-escape the drive-letter colon so
        # ffmpeg's filter-option parser doesn't treat ``E:`` as a
        # delimiter.
        escaped = title_file.replace("\\", "/").replace(":", r"\:")
        filter_str = filter_str.replace("__TITLE_FILE__", escaped)

    # NOTE: previously this point had a post-hoc filter-string rewrite
    # (``filter_str.replace("[2:v]", ...)``) to fix snap/logo indices.
    # That was buggy — it also clobbered the title PNG when title and
    # snap shared input index 2 (e.g. no logo + has_snap=True).
    # The builders now take ``snap_input_idx`` / ``logo_input_idx`` as
    # explicit parameters so the references are correct by construction.

    # ── 7) Compose ffmpeg args ───────────────────────────────
    args: list[str] = [
        _ffmpeg_bin(),
        "-ss", f"{start_sec}",
        "-to", f"{end_sec}",
        "-i",  input_path,
    ]
    if has_logo:
        args += ["-i", logo_path]
    if has_snap:
        args += ["-i", snap_path]
    if has_title_png:
        args += ["-i", title_png_path]
    if has_texture:
        # Static PNG — loop it for the clip duration so ffmpeg sees a
        # video stream of matching length instead of EOF after frame 1.
        args += ["-loop", "1", "-i", texture_path]

    args += [
        "-filter_complex", filter_str,
        "-map", f"[{out_tag}]",
        "-map", "0:a?",
        # Video encoder picked at startup — h264_nvenc when an NVIDIA
        # GPU is present (5-10× faster), libx264 fallback otherwise.
        *hw_accel.encoder_args(),
        "-r", "30",
        "-c:a", "aac",
        "-b:a", "160k",
        "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        "-y", output_path,
    ]
    print(f"[cut_clip] encoder={hw_accel.active_encoder_label()} layout={layout} duration={end_sec-start_sec:.1f}s")

    # ── 8) Spawn + cleanup ───────────────────────────────────
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        for f in cleanups:
            try: Path(f).unlink(missing_ok=True)
            except OSError: pass
        raise CutClipError(f"ffmpeg cut timed out after {timeout_s}s") from exc

    for f in cleanups:
        try: Path(f).unlink(missing_ok=True)
        except OSError: pass

    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[-600:]
        raise CutClipError(f"ffmpeg cut failed: {stderr}")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1000:
        raise CutClipError("ffmpeg exited 0 but output is empty/missing")

    return output_path
