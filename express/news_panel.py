"""buildNewsLayoutFilter — Python port of the teammate's TV news split
panel.

Layout (1080×1920 = 9:16 vertical):
  top 960 px    = source video (scaled+cropped to fill, no bars)
  white divider = 6 px line at y=960
  bottom 960 px = colored panel containing
                    - 2-line Telugu title (overlay PNG) near the top
                    - white-bordered inset photo below the title
                    - "KAIZER NEWS NETWORK" footer near the bottom
  channel logo  = top-right of the source band

Returns ``(filter_complex_str, out_tag)`` where ``out_tag`` is the
final stream label the caller passes to ``-map`` in ffmpeg.

Input layout the caller must respect when adding ``-i`` flags in order:
  [0] = source video (always)
  [N] = logo PNG, then snap photo, then title PNG, then texture PNG
        depending on which ``has_*`` / ``*_input_idx`` are set.
"""
from __future__ import annotations

from typing import Optional

from . import color_grade as cg


_W, _H = 1080, 1920
_TOP_H = 960
_BOT_H = _H - _TOP_H

_TITLE_Y    = _TOP_H + 24
_FOOTER_Y   = _TOP_H + 880          # 1840 — slightly above bottom edge
_DIVIDER_H  = 6


def build_news_layout_filter(
    *,
    has_font: bool,
    has_logo: bool,
    has_snap: bool,
    panel_color: str = "#dc2626",
    footer_text: str = "",
    title_font_size: int = 82,
    title_png_path: Optional[str] = None,
    title_png_input_idx: int = -1,
    title_png_height: int = 0,
    snap_input_idx: int = -1,                # NEW — explicit, not hard-coded [2:v]
    logo_input_idx: int = -1,                # NEW — explicit, not hard-coded [1:v]
    texture_input_idx: int = -1,
    color_grade: str = "subtle",
    cinematic_edit: bool = False,
    clip_duration_sec: float = 30.0,
    font_path: Optional[str] = None,        # for the footer drawtext only
) -> tuple[str, str]:
    """Build the filter_complex graph. See module docstring for layout.

    The caller is responsible for:
      - Wiring ``-i`` flags in the same order (video, logo, snap,
        titlePng, texture) and passing the resulting indices via
        ``title_png_input_idx`` / ``texture_input_idx``.
      - Setting ``has_logo`` + adding the logo as input 1 (the filter
        references ``[1:v]``).
      - Setting ``has_snap`` + adding the snap as input 2 (the filter
        references ``[2:v]``). cut_clip rewires these post-hoc when
        the actual indices differ.
    """
    # Compute snap geometry — sit 28 px below the title bottom edge so
    # text and photo never overlap. Cap at 470 high so it leaves room
    # for the footer.
    title_end = _TITLE_Y + title_png_height if title_png_height > 0 else (_TOP_H + 320)
    snap_y_raw   = title_end + 28
    snap_bottom  = _FOOTER_Y - 26
    snap_w       = 700
    snap_border  = 8
    avail_h      = snap_bottom - snap_y_raw
    snap_h       = max(220, min(470, avail_h))
    snap_y       = snap_y_raw

    parts: list[str] = []

    # ffmpeg's filter-graph parser treats ``\`` as an escape char and
    # ``:`` as an option separator, so Windows paths like
    # ``E:\path\NotoSansTelugu-Bold.ttf`` corrupt the graph in TWO
    # places: the backslashes (each `\K`, `\n`, etc consumed) and the
    # ``E:`` drive-letter colon (read as an option terminator).
    # Fix: forward-slash the separators AND backslash-escape the
    # remaining drive-letter colon. ffmpeg accepts ``E\:/path/font.ttf``.
    if font_path:
        font_path = font_path.replace("\\", "/").replace(":", r"\:")

    # ── Source band: scale+crop, optional Ken Burns + grain, color grade
    grade_chain = cg.grade_chain(color_grade)
    source_filters = [
        f"scale={_W}:{_TOP_H}:force_original_aspect_ratio=increase",
        f"crop={_W}:{_TOP_H}",
    ]
    if cinematic_edit:
        frames = max(2, round(clip_duration_sec * 30))
        source_filters.extend([
            f"scale={round(_W * 1.25)}:{round(_TOP_H * 1.25)}:force_original_aspect_ratio=increase",
            f"crop={round(_W * 1.25)}:{round(_TOP_H * 1.25)}",
            f"zoompan=z='min(zoom+0.0008,1.04)':d={frames}:s={_W}x{_TOP_H}:fps=30",
        ])
    if grade_chain:
        source_filters.append(grade_chain)
    if cinematic_edit:
        source_filters.append("noise=alls=6:allf=t")
    parts.append(f"[0:v]{','.join(source_filters)}[top]")

    # ── Panel: solid color + optional dot texture overlay
    parts.append(
        f"color=c={panel_color}:s={_W}x{_BOT_H}:r=30,format=yuv420p,"
        f"eq=saturation=1.1:contrast=1.03[panelBase]"
    )
    if texture_input_idx >= 0:
        parts.append(
            f"[{texture_input_idx}:v]format=rgba,scale={_W}:{_BOT_H}[panelTex]"
        )
        parts.append("[panelBase][panelTex]overlay=0:0:format=auto[panel]")
    else:
        parts.append("[panelBase]copy[panel]")

    parts.append("[top][panel]vstack=inputs=2[stacked]")

    # ── Thin white divider at y=960 (between source and panel)
    parts.append(
        f"[stacked]drawbox=x=0:y={_TOP_H - _DIVIDER_H // 2}:w={_W}:h={_DIVIDER_H}:"
        "color=white@0.95:t=fill[stage]"
    )
    last_tag = "stage"

    # ── Title overlay: prefer pre-rendered PNG, fall back to drawtext
    if title_png_path and title_png_input_idx >= 0:
        max_title_w = _W
        parts.append(
            f"[{title_png_input_idx}:v]scale="
            f"'if(gt(iw,{max_title_w}),{max_title_w},iw)':-1:flags=lanczos,"
            "format=rgba[titlepng]"
        )
        parts.append(f"[{last_tag}][titlepng]overlay=(W-w)/2:{_TITLE_Y}[titled]")
        last_tag = "titled"
    elif has_font and font_path:
        # drawtext fallback — minimal shaping, no bomb-word color.
        # Uses a sentinel that cut_clip replaces with the actual file
        # path to keep escaping simple.
        parts.append(
            f"[{last_tag}]drawtext="
            f"fontfile='{font_path}':"
            "textfile='__TITLE_FILE__':"
            f"fontsize={title_font_size}:"
            "fontcolor=white:"
            "bordercolor=black:borderw=5:"
            f"line_spacing={round(title_font_size * 0.3)}:"
            "x=(w-text_w)/2:"
            f"y={_TITLE_Y}[titled]"
        )
        last_tag = "titled"

    # ── Inset photo with white border. snap_input_idx is the actual
    # ffmpeg input position resolved by the caller (cut_clip).
    if has_snap and snap_input_idx >= 0:
        frame_w = snap_w + snap_border * 2
        frame_h = snap_h + snap_border * 2
        parts.append(
            f"[{snap_input_idx}:v]scale={snap_w}:{snap_h}:force_original_aspect_ratio=increase,"
            f"crop={snap_w}:{snap_h},"
            "eq=saturation=1.1:contrast=1.04[snap0]"
        )
        parts.append(
            f"[snap0]pad={frame_w}:{frame_h}:{snap_border}:{snap_border}:white[snap]"
        )
        parts.append(f"[{last_tag}][snap]overlay=(W-w)/2:{snap_y}[withSnap]")
        last_tag = "withSnap"

    # ── Footer
    if has_font and footer_text and font_path:
        parts.append(
            f"[{last_tag}]drawtext="
            f"fontfile='{font_path}':"
            "text='__FOOTER__':"
            "fontsize=34:"
            "fontcolor=white@0.95:"
            "x=(w-text_w)/2:"
            f"y={_FOOTER_Y}[footed]"
        )
        last_tag = "footed"

    # ── Logo top-right of source band
    if has_logo and logo_input_idx >= 0:
        parts.append(f"[{logo_input_idx}:v]scale=90:-1[logo]")
        parts.append(f"[{last_tag}][logo]overlay=W-w-32:32[final]")
        last_tag = "final"

    return ";".join(parts), last_tag


# ─── Branded layout (the teammate's 2nd Shorts layout) ─────────────
#
# Port of ``buildBrandedFilter`` (server.js:1329). Vertical reframe
# with blurred-bg fill + foreground video centered below a title band
# + logo at the chosen corner. Cleaner / less "TV news" looking;
# useful when the user wants a simpler influencer-style Short.

_BRAND_TOP_BAND = 360         # reserved for title at the top
_BRAND_FG_Y     = _BRAND_TOP_BAND + 40


def build_branded_layout_filter(
    *,
    has_font: bool,
    has_logo: bool,
    title: str = "",
    logo_corner: str = "top-right",   # top-right | top-left | bottom-right | bottom-left
    font_path: Optional[str] = None,
    logo_input_idx: int = -1,         # NEW — explicit, not hard-coded [1:v]
) -> tuple[str, str]:
    """1080×1920 reframe with blurred background + title band.

    Caller's ffmpeg input order:
      [0] = source video (always)
      [1] = logo PNG (only when has_logo)

    Replace ``__TITLE_FILE__`` in the returned filter with the path
    to a UTF-8 text file holding the wrapped title (drawtext textfile
    contract), and apply the same Windows path-escape (``replace `\\`
    with `/` and `:` with `\\:``) that cut_clip applies for news.
    """
    if font_path:
        font_path = font_path.replace("\\", "/").replace(":", r"\:")

    parts: list[str] = []
    # Background = blurred copy of the source filling the canvas.
    # Foreground = source scaled to width, centered under the title.
    parts.append("[0:v]split=2[fg0][bg0]")
    parts.append(
        f"[bg0]scale={_W}:{_H}:force_original_aspect_ratio=increase,"
        f"crop={_W}:{_H},boxblur=22:1,eq=brightness=-0.08:saturation=0.7[bg]"
    )
    parts.append(f"[fg0]scale={_W}:-2:force_original_aspect_ratio=decrease[fg]")
    parts.append(f"[bg][fg]overlay=(W-w)/2:{_BRAND_FG_Y}[stage]")
    last_tag = "stage"

    if has_font and title and font_path:
        # textfile= sentinel; cut_clip writes the wrapped string to a
        # temp file and substitutes the path post-build.
        parts.append(
            f"[{last_tag}]drawtext="
            f"fontfile='{font_path}':"
            "textfile='__TITLE_FILE__':"
            "fontsize=78:"
            "fontcolor=white:"
            "bordercolor=black:borderw=6:"
            "line_spacing=14:"
            "x=(w-text_w)/2:"
            "y=80[titled]"
        )
        last_tag = "titled"

    if has_logo and logo_input_idx >= 0:
        parts.append(f"[{logo_input_idx}:v]scale=200:-1[logo]")
        # 36 px margin from the chosen corner.
        if logo_corner == "top-left":
            pos = "36:36"
        elif logo_corner == "bottom-left":
            pos = "36:H-h-36"
        elif logo_corner == "bottom-right":
            pos = "W-w-36:H-h-36"
        else:                            # top-right (default)
            pos = "W-w-36:36"
        parts.append(f"[{last_tag}][logo]overlay={pos}[final]")
        last_tag = "final"

    return ";".join(parts), last_tag
