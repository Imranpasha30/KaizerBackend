"""
kaizer.pipeline.captions
========================
Indic-script caption renderer for the Kaizer News video pipeline.

Renders text to a PIL RGBA ``Image`` at arbitrary size, with no clipping.
Handles all major Indic scripts via ``uharfbuzz`` shaping + Pillow rasterisation.
Latin text is rendered with Pillow's FreeType BASIC engine directly.

Usage
-----
    from pipeline_core.captions import render_caption, detect_script, safe_zone, CaptionResult

    result = render_caption(
        "తెలుగు వార్తలు",
        max_width=900,
        font_size=56,
        color="#FFFFFF",
        stroke_color="#000000",
        stroke_width=4,
        bg_color="#00000088",
    )
    if result.warnings:
        logger.warning("Caption warnings: %s", result.warnings)
    result.image.save("/tmp/caption.png")

CaptionResult fields
--------------------
  image      : PIL.Image.Image   — RGBA image with transparent background.
  width      : int               — Pixel width of the rendered image.
  height     : int               — Pixel height of the rendered image.
  font_path  : str               — Absolute path to the font actually used.
  script     : str               — Detected (or overridden) script name.
  warnings   : list[str]         — Non-fatal issues (missing font, fallback used …).

Script detection
----------------
Returns one of: 'latin', 'devanagari', 'telugu', 'tamil', 'bengali',
'kannada', 'malayalam', 'gujarati', 'mixed'.

The dominant Unicode block in the text determines the script. If significant
characters from two or more Indic scripts are present, 'mixed' is returned.

Shaping
-------
Indic scripts (telugu, devanagari, tamil, bengali, kannada, malayalam,
gujarati) are shaped with uharfbuzz before rendering.  HarfBuzz applies
GSUB/GPOS lookups (conjunct consonants, matra reordering, etc.) and returns
per-cluster advances and offsets. Each cluster is then composited onto the
canvas using those advances, while Pillow FreeType rasterises the individual
cluster strings at the pixel level.

Latin text uses Pillow's standard text-rendering path with stroke support
via the ``stroke_width`` / ``stroke_fill`` parameters.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union

from PIL import Image, ImageColor, ImageDraw, ImageFont

logger = logging.getLogger("kaizer.pipeline.captions")

# ── Constants ─────────────────────────────────────────────────────────────────

# Absolute path to the resources/fonts directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR: str = os.path.normpath(
    os.path.join(_HERE, "..", "resources", "fonts")
).replace("\\", "/")

# Unicode block ranges used for script detection.
# Each entry: (start_codepoint, end_codepoint, script_name)
_UNICODE_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "devanagari"),   # Devanagari (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF, "bengali"),      # Bengali / Assamese
    (0x0A80, 0x0AFF, "gujarati"),     # Gujarati
    (0x0B00, 0x0B7F, "odia"),         # Odia — mapped to 'mixed' if alone
    (0x0B80, 0x0BFF, "tamil"),        # Tamil
    (0x0C00, 0x0C7F, "telugu"),       # Telugu
    (0x0C80, 0x0CFF, "kannada"),      # Kannada
    (0x0D00, 0x0D7F, "malayalam"),    # Malayalam
]

# Minimum fraction of non-ASCII characters that must belong to one script
# for it to be declared the dominant script.
_SCRIPT_DOMINANCE_THRESHOLD = 0.5

# Supported Indic scripts (require HarfBuzz shaping).
_INDIC_SCRIPTS: frozenset[str] = frozenset({
    "devanagari", "telugu", "tamil", "bengali",
    "kannada", "malayalam", "gujarati",
})

# Font fallback map: script → ordered list of candidate filenames (TTF only;
# WOFF/WOFF2 files are skipped — see _load_font_for_script for the guard).
_FONT_FALLBACK_MAP: dict[str, list[str]] = {
    "telugu": [
        "NotoSansTelugu-Bold.ttf",
        "NotoSansTelugu-Regular.ttf",
        "NotoSerifTelugu-Bold.ttf",
        "NotoSerifTelugu-Regular.ttf",
        # Telugu Google Fonts (may be WOFF, filtered at load time)
        "Dhurjati-Regular.ttf",
        "Gurajada-Regular.ttf",
        "HindGuntur-Regular.ttf",
        "HindGuntur-Bold.ttf",
        "Mallanna-Regular.ttf",
        "Mandali-Regular.ttf",
        "NTR-Regular.ttf",
        "Ponnala-Regular.ttf",
        "Ramabhadra-Regular.ttf",
        "Ramaraja-Regular.ttf",
        "TenaliRamakrishna-Regular.ttf",
        "Timmana-Regular.ttf",
    ],
    "devanagari": [
        "NotoSansDevanagari-Regular.ttf",
        "NotoSansDevanagari-Bold.ttf",
        "Laila-Regular.ttf",
        "Laila-Bold.ttf",
    ],
    "tamil": [
        "NotoSansTamil-Regular.ttf",
        "NotoSansTamil-Bold.ttf",
    ],
    "bengali": [
        "NotoSansBengali-Regular.ttf",
        "NotoSansBengali-Bold.ttf",
    ],
    "kannada": [
        "NotoSansKannada-Regular.ttf",
        "NotoSansKannada-Bold.ttf",
    ],
    "malayalam": [
        "NotoSansMalayalam-Regular.ttf",
        "NotoSansMalayalam-Bold.ttf",
    ],
    "gujarati": [
        "NotoSansGujarati-Regular.ttf",
        "NotoSansGujarati-Bold.ttf",
    ],
    "latin": [
        "NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf",
        "Roboto-Bold.ttf",
        "Oswald-Bold.ttf",
    ],
    "mixed": [
        "NotoSans-Regular.ttf",
    ],
}

# Magic bytes for valid OpenType / TrueType fonts.
_VALID_FONT_MAGIC: frozenset[bytes] = frozenset({
    b"\x00\x01\x00\x00",  # TrueType
    b"OTTO",              # OpenType CFF
    b"true",              # Apple TrueType
    b"typ1",              # PostScript Type 1 (rare)
})


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class CaptionResult:
    """Result of a render_caption() call.

    Attributes
    ----------
    image : PIL.Image.Image
        RGBA image with transparent background containing the rendered caption.
    width : int
        Pixel width of *image*.
    height : int
        Pixel height of *image*.
    font_path : str
        Absolute path to the font file actually used (for debugging).
    script : str
        Script detected in the input text (or the caller-supplied override).
    warnings : list[str]
        Non-fatal issues collected during rendering (missing fonts, fallbacks …).
    """

    image: Image.Image
    width: int
    height: int
    font_path: str
    script: str
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_valid_ttf(path: str) -> bool:
    """Return True if *path* begins with a recognised OpenType/TrueType magic."""
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
        return magic in _VALID_FONT_MAGIC
    except OSError:
        return False


def _resolve_color(
    color: Union[str, tuple],
    default_alpha: int = 255,
) -> tuple[int, int, int, int]:
    """Parse *color* (hex string or tuple) into an RGBA 4-tuple."""
    if isinstance(color, tuple):
        if len(color) == 3:
            return (int(color[0]), int(color[1]), int(color[2]), default_alpha)
        if len(color) == 4:
            return (int(color[0]), int(color[1]), int(color[2]), int(color[3]))
        raise ValueError(f"Color tuple must have 3 or 4 elements, got {len(color)}: {color!r}")
    rgba = ImageColor.getrgb(color)
    if len(rgba) == 3:
        return (rgba[0], rgba[1], rgba[2], default_alpha)
    return (rgba[0], rgba[1], rgba[2], rgba[3])


def _load_font_for_script(
    script: str,
    font_size: int,
    warnings: list[str],
) -> tuple[ImageFont.FreeTypeFont, str]:
    """Load the best available PIL font for *script* at *font_size*.

    Works through _FONT_FALLBACK_MAP for the given script, then falls back to
    any TTF in FONTS_DIR that contains the script name (case-insensitive), then
    falls back to the Pillow built-in default font (bitmap, no sizing).

    Returns
    -------
    (pil_font, font_path)
        *font_path* is the absolute path of the loaded font, or the string
        ``"<pillow-default>"`` if the fallback bitmap font was used.
    """
    candidates: list[str] = list(_FONT_FALLBACK_MAP.get(script, []))

    # Generic Noto fallback for any script not in the map
    if not candidates:
        candidates = _FONT_FALLBACK_MAP.get("latin", [])
        warnings.append(
            f"No font candidates defined for script {script!r}; "
            "falling back to Latin font list."
        )

    # Try each candidate in order
    for fname in candidates:
        path = os.path.join(FONTS_DIR, fname).replace("\\", "/")
        if os.path.isfile(path) and _is_valid_ttf(path):
            try:
                font = ImageFont.truetype(path, font_size)
                logger.debug("Loaded font %s for script %r", fname, script)
                return font, path
            except Exception as exc:
                logger.warning(
                    "Failed to load font %s for script %r: %s", path, script, exc
                )
                warnings.append(f"Could not load font {fname!r}: {exc}")

    # Wildcard scan: any .ttf in FONTS_DIR whose name contains the script name
    script_keyword = script.lower()
    try:
        for entry in sorted(os.listdir(FONTS_DIR)):
            if not entry.lower().endswith(".ttf"):
                continue
            if script_keyword not in entry.lower():
                continue
            path = os.path.join(FONTS_DIR, entry).replace("\\", "/")
            if _is_valid_ttf(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    warnings.append(
                        f"Primary font candidates unavailable for script {script!r}; "
                        f"using wildcard match {entry!r}."
                    )
                    logger.warning(
                        "Using wildcard font %s for script %r", path, script
                    )
                    return font, path
                except Exception:
                    pass
    except OSError as exc:
        logger.warning("Could not scan FONTS_DIR %s: %s", FONTS_DIR, exc)

    # Last resort: Pillow default (bitmap, ignores font_size)
    warnings.append(
        f"No usable font found for script {script!r} in {FONTS_DIR!r}. "
        "Using Pillow built-in default font — quality will be poor."
    )
    logger.warning(
        "Falling back to Pillow default font for script %r (font_size will be ignored)",
        script,
    )
    return ImageFont.load_default(), "<pillow-default>"


def _load_hb_font(font_path: str) -> tuple["hb.Font", int, float] | None:  # type: ignore[name-defined]
    """Load a uharfbuzz Font from *font_path*.

    Returns
    -------
    (hb_font, upem, scale_factor) or None if uharfbuzz is unavailable or
    the font cannot be shaped (e.g. WOFF files slip through).
    """
    try:
        import uharfbuzz as hb
    except ImportError:
        return None

    if not _is_valid_ttf(font_path):
        return None

    try:
        with open(font_path, "rb") as fh:
            font_data = fh.read()
        face = hb.Face(font_data)
        if face.glyph_count == 0:
            logger.warning(
                "HarfBuzz loaded %s but glyph_count=0 — shaping disabled for this font.",
                font_path,
            )
            return None
        font = hb.Font(face)
        hb.ot_font_set_funcs(font)
        upem: int = face.upem
        return font, upem, 1.0  # scale_factor resolved at call time
    except Exception as exc:
        logger.warning("Could not load HarfBuzz font from %s: %s", font_path, exc)
        return None


def _shape_text(
    text: str,
    hb_font: "hb.Font",  # type: ignore[name-defined]
    upem: int,
    font_size: int,
) -> tuple[list, list, list[int], float]:
    """Shape *text* with HarfBuzz.

    Returns
    -------
    (glyph_infos, glyph_positions, codepoints, scale_factor)
        *scale_factor* converts HarfBuzz design units (upem) to pixels.
    """
    import uharfbuzz as hb  # already imported by caller

    codepoints = [ord(c) for c in text]
    buf = hb.Buffer()
    buf.add_codepoints(codepoints)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    scale_factor = font_size / upem
    return buf.glyph_infos, buf.glyph_positions, codepoints, scale_factor


def _measure_shaped_width(
    glyph_positions: list,
    scale_factor: float,
) -> float:
    """Return total pixel advance width from HarfBuzz glyph positions."""
    return sum(pos.x_advance * scale_factor for pos in glyph_positions)


def _measure_line_width_pil(text: str, pil_font: ImageFont.FreeTypeFont) -> float:
    """Return text width in pixels using PIL (for Latin / fallback)."""
    try:
        return pil_font.getlength(text)
    except AttributeError:
        # Older Pillow
        w, _ = pil_font.getsize(text)  # type: ignore[attr-defined]
        return float(w)


def _word_wrap(
    text: str,
    max_width: int,
    pil_font: ImageFont.FreeTypeFont,
    hb_font_data: tuple | None,
    font_size: int,
    warnings: list[str],
) -> list[str]:
    """Wrap *text* into lines of at most *max_width* pixels.

    Uses HarfBuzz advances for Indic scripts when *hb_font_data* is not None;
    falls back to PIL measurement for Latin / fallback fonts.

    Handles the edge case where a single word exceeds *max_width* by
    character-wrapping that word.

    Parameters
    ----------
    hb_font_data : (hb_font, upem) | None
    """
    def _measure(s: str) -> float:
        if hb_font_data is not None and s:
            hb_font, upem = hb_font_data
            try:
                _, glyph_positions, _, scale_factor = _shape_text(
                    s, hb_font, upem, font_size
                )
                return _measure_shaped_width(glyph_positions, scale_factor)
            except Exception:
                pass
        return _measure_line_width_pil(s, pil_font)

    words = text.split(" ")
    lines: list[str] = []
    current_line = ""

    for word in words:
        trial = (current_line + " " + word).strip() if current_line else word

        if _measure(trial) <= max_width:
            current_line = trial
        else:
            # Current line is full — commit it (unless empty)
            if current_line:
                lines.append(current_line)
                current_line = ""

            # Check if *word* alone fits
            if _measure(word) <= max_width:
                current_line = word
            else:
                # Single word exceeds max_width: char-wrap
                char_buf = ""
                for ch in word:
                    trial_char = char_buf + ch
                    if _measure(trial_char) <= max_width:
                        char_buf = trial_char
                    else:
                        if char_buf:
                            lines.append(char_buf)
                        char_buf = ch
                if char_buf:
                    current_line = char_buf

    if current_line:
        lines.append(current_line)

    if not lines:
        lines = [""]

    return lines


def _render_line_shaped(
    line: str,
    pil_font: ImageFont.FreeTypeFont,
    hb_font: "hb.Font",  # type: ignore[name-defined]
    upem: int,
    font_size: int,
    fill_rgba: tuple[int, int, int, int],
    stroke_rgba: tuple[int, int, int, int] | None,
    stroke_width: int,
    canvas_width: int,
    line_height: int,
    ascender_px: int,
) -> Image.Image:
    """Render a single shaped (Indic) line onto a transparent canvas.

    The canvas is sized to *canvas_width* × *line_height*.  Text is drawn
    at x=0, baseline = *ascender_px* from the top.

    For each HarfBuzz cluster, the original Unicode characters for that
    cluster are rendered at the cluster's HarfBuzz-derived x position using
    PIL FreeType.  This gives correct advance widths (from HarfBuzz) while
    using FreeType for pixel-level rasterisation.
    """
    scale_factor = font_size / upem

    try:
        glyph_infos, glyph_positions, codepoints, _ = _shape_text(
            line, hb_font, upem, font_size
        )
    except Exception as exc:
        logger.warning("HarfBuzz shaping failed for line %r: %s — using PIL fallback", line, exc)
        return _render_line_pil(
            line, pil_font, fill_rgba, stroke_rgba, stroke_width,
            canvas_width, line_height, ascender_px,
        )

    # Build cluster → list of (glyph_info, glyph_pos) mapping
    cluster_glyph_map: dict[int, list] = defaultdict(list)
    for info, pos in zip(glyph_infos, glyph_positions):
        cluster_glyph_map[info.cluster].append((info, pos))

    # Determine original characters per cluster
    clusters_ordered = sorted(cluster_glyph_map.keys())
    cluster_chars: dict[int, str] = {}
    for idx, cluster_start in enumerate(clusters_ordered):
        cluster_end = (
            clusters_ordered[idx + 1] if idx + 1 < len(clusters_ordered) else len(codepoints)
        )
        chars = "".join(chr(codepoints[j]) for j in range(cluster_start, cluster_end))
        cluster_chars[cluster_start] = chars

    img = Image.new("RGBA", (canvas_width, line_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_cursor = 0.0
    baseline_top = 0  # text drawn from y=0; baseline implicit in FreeType metrics

    for cluster_start in clusters_ordered:
        glyphs_in_cluster = cluster_glyph_map[cluster_start]
        chars = cluster_chars.get(cluster_start, "")

        # Total x advance for this cluster (sum of all glyphs in cluster)
        cluster_adv_px = sum(pos.x_advance * scale_factor for _, pos in glyphs_in_cluster)
        # x offset for first glyph (usually 0 for LTR Indic)
        x_off_px = glyphs_in_cluster[0][1].x_offset * scale_factor
        y_off_px = glyphs_in_cluster[0][1].y_offset * scale_factor

        draw_x = x_cursor + x_off_px
        draw_y = float(baseline_top) - y_off_px  # y_offset is positive-up in HB

        if chars:
            draw_kwargs: dict = {
                "font": pil_font,
                "fill": fill_rgba,
            }
            if stroke_rgba is not None and stroke_width > 0:
                draw_kwargs["stroke_width"] = stroke_width
                draw_kwargs["stroke_fill"] = stroke_rgba

            draw.text((draw_x, draw_y), chars, **draw_kwargs)

        x_cursor += cluster_adv_px

    return img


def _render_line_pil(
    line: str,
    pil_font: ImageFont.FreeTypeFont,
    fill_rgba: tuple[int, int, int, int],
    stroke_rgba: tuple[int, int, int, int] | None,
    stroke_width: int,
    canvas_width: int,
    line_height: int,
    ascender_px: int,
) -> Image.Image:
    """Render a single line with PIL's standard text rendering (Latin / fallback)."""
    img = Image.new("RGBA", (canvas_width, line_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw_kwargs: dict = {
        "font": pil_font,
        "fill": fill_rgba,
    }
    if stroke_rgba is not None and stroke_width > 0:
        draw_kwargs["stroke_width"] = stroke_width
        draw_kwargs["stroke_fill"] = stroke_rgba

    draw.text((0, 0), line, **draw_kwargs)
    return img


def _get_line_metrics(
    pil_font: ImageFont.FreeTypeFont,
    hb_font_data: tuple | None,
    font_size: int,
) -> tuple[int, int, int]:
    """Return (ascender_px, descender_px, line_height_px).

    Prefers HarfBuzz font extents; falls back to PIL ascent/descent.
    """
    if hb_font_data is not None:
        hb_font, upem = hb_font_data
        scale = font_size / upem
        try:
            extents = hb_font.get_font_extents("ltr")
            ascender_px = max(1, int(extents.ascender * scale))
            descender_px = max(1, int(abs(extents.descender) * scale))
            line_gap = max(0, int(getattr(extents, "line_gap", 0) * scale))
            line_height_px = ascender_px + descender_px + line_gap
            return ascender_px, descender_px, line_height_px
        except Exception as exc:
            logger.debug("HarfBuzz font extents failed: %s", exc)

    # PIL fallback
    internal = pil_font.font
    ascender_px = max(1, int(getattr(internal, "ascent", font_size)))
    descender_px = max(1, int(getattr(internal, "descent", font_size // 5)))
    line_height_px = ascender_px + descender_px
    return ascender_px, descender_px, line_height_px


# ── Public API ────────────────────────────────────────────────────────────────

def detect_script(text: str) -> str:
    """Detect the dominant Unicode script in *text*.

    Parameters
    ----------
    text : str
        Input text of any length.

    Returns
    -------
    str
        One of: 'latin', 'devanagari', 'telugu', 'tamil', 'bengali',
        'kannada', 'malayalam', 'gujarati', 'mixed'.

    Notes
    -----
    - ASCII / Basic Latin characters are ignored in the script count so
      punctuation and digits do not bias the result toward 'latin'.
    - If the non-ASCII character count is below 4 *or* if more than one
      Indic script exceeds 20% of the non-ASCII count, 'mixed' is returned.
    - If no non-ASCII characters are found, 'latin' is returned.
    """
    if not text:
        return "latin"

    # Count codepoints per script block (ignoring ASCII)
    script_counts: dict[str, int] = {}
    non_ascii_total = 0

    for ch in text:
        cp = ord(ch)
        if cp < 0x0100:
            # Basic Latin / Latin-1 Supplement — ignore for Indic detection
            continue
        non_ascii_total += 1
        for start, end, sname in _UNICODE_SCRIPT_RANGES:
            if start <= cp <= end:
                script_counts[sname] = script_counts.get(sname, 0) + 1
                break

    if non_ascii_total == 0:
        return "latin"

    if non_ascii_total < 4:
        # Too few non-ASCII chars to be confident — call it mixed
        return "mixed"

    if not script_counts:
        # All non-ASCII characters are outside Indic blocks (e.g. Arabic, CJK)
        return "mixed"

    # Find dominant script
    dominant_script = max(script_counts, key=lambda s: script_counts[s])
    dominant_count = script_counts[dominant_script]
    dominant_fraction = dominant_count / non_ascii_total

    if dominant_fraction < _SCRIPT_DOMINANCE_THRESHOLD:
        return "mixed"

    # Check for contamination: another script with > 20% share
    other_indic = [
        s for s in script_counts
        if s != dominant_script and script_counts[s] / non_ascii_total > 0.2
    ]
    if other_indic:
        return "mixed"

    return dominant_script


def safe_zone(
    platform: str,
    w: int = 1080,
    h: int = 1920,
) -> tuple[int, int, int, int]:
    """Return the caption-safe rectangle for *platform*.

    Parameters
    ----------
    platform : str
        One of: ``'youtube_short'``, ``'instagram_reel'``, ``'tiktok'``.
    w : int
        Frame width in pixels (default 1080).
    h : int
        Frame height in pixels (default 1920).

    Returns
    -------
    (x, y, width, height)
        Top-left corner and dimensions of the safe rectangle in pixels.

    Raises
    ------
    ValueError
        If *platform* is not one of the three recognised names.

    Notes
    -----
    Safe zone constants (for 1080×1920 frame):

    * **Instagram Reels** — top=250 bottom=320 left=60 right=180
      → safe = 840×1350 at (60, 250)
    * **YouTube Shorts** — top=120 bottom=280 left=60 right=120
      → safe = 900×1520 at (60, 120)
    * **TikTok**         — top=150 bottom=350 left=60 right=160
      → safe = 860×1420 at (60, 150)
    """
    _SAFE_ZONES: dict[str, dict[str, int]] = {
        "instagram_reel": {"top": 250, "bottom": 320, "left": 60, "right": 180},
        "youtube_short":  {"top": 120, "bottom": 280, "left": 60, "right": 120},
        "tiktok":         {"top": 150, "bottom": 350, "left": 60, "right": 160},
    }

    key = platform.lower().replace("-", "_")
    if key not in _SAFE_ZONES:
        raise ValueError(
            f"Unknown platform {platform!r}. "
            f"Valid platforms: {sorted(_SAFE_ZONES.keys())}."
        )

    # Scale margins proportionally if frame size differs from 1080×1920
    ref_w, ref_h = 1080, 1920
    margins = _SAFE_ZONES[key]

    top    = int(margins["top"]    * h / ref_h)
    bottom = int(margins["bottom"] * h / ref_h)
    left   = int(margins["left"]   * w / ref_w)
    right  = int(margins["right"]  * w / ref_w)

    safe_x = left
    safe_y = top
    safe_w = w - left - right
    safe_h = h - top - bottom

    return (safe_x, safe_y, safe_w, safe_h)


def render_caption(
    text: str,
    *,
    max_width: int,
    font_size: int = 56,
    color: Union[str, tuple] = "#FFFFFF",
    stroke_color: Union[str, None] = "#000000",
    stroke_width: int = 4,
    bg_color: Union[str, None] = None,
    bg_padding: int = 16,
    bg_radius: int = 12,
    line_spacing: float = 1.15,
    align: str = "center",
    script: Union[str, None] = None,
) -> CaptionResult:
    """Render *text* as a transparent-background RGBA PIL Image.

    Parameters
    ----------
    text : str
        Caption text to render.
    max_width : int
        Maximum line width in pixels.  Text is word-wrapped to fit.
    font_size : int
        Font size in points (default 56).
    color : str | tuple
        Text fill colour as a CSS hex string (e.g. ``"#FFFFFF"``,
        ``"#FFFFFF88"``) or an RGBA 4-tuple. Default: white.
    stroke_color : str | None
        Stroke (outline) colour, or None to disable stroke. Default: black.
    stroke_width : int
        Stroke width in pixels (default 4).  Ignored when *stroke_color* is None.
    bg_color : str | None
        Background box colour (e.g. ``"#00000088"`` for 53% opaque black).
        None = no background box.
    bg_padding : int
        Padding inside the background box in pixels (default 16).
    bg_radius : int
        Corner radius of the background box in pixels (default 12).
    line_spacing : float
        Multiplier on the line height (default 1.15 = 15% extra spacing).
    align : str
        Horizontal text alignment: ``'left'``, ``'center'``, or ``'right'``.
    script : str | None
        Override the auto-detected script. One of the values returned by
        :func:`detect_script`.  None = auto-detect.

    Returns
    -------
    CaptionResult
        The rendered RGBA image and associated metadata.

    Raises
    ------
    ValueError
        If *align* is not one of 'left', 'center', 'right'.
    """
    warnings: list[str] = []

    if align not in ("left", "center", "right"):
        raise ValueError(
            f"Invalid align value {align!r}. Must be 'left', 'center', or 'right'."
        )

    # ── Script detection ──────────────────────────────────────────────────────
    detected_script: str = script if script is not None else detect_script(text)
    logger.debug("render_caption: script=%r text=%r…", detected_script, text[:40])

    # ── Font loading ──────────────────────────────────────────────────────────
    pil_font, font_path = _load_font_for_script(detected_script, font_size, warnings)

    # ── HarfBuzz setup (Indic only) ───────────────────────────────────────────
    hb_font_data: tuple | None = None
    is_indic = detected_script in _INDIC_SCRIPTS

    if is_indic:
        hb_result = _load_hb_font(font_path)
        if hb_result is not None:
            hb_font_obj, upem, _ = hb_result
            hb_font_data = (hb_font_obj, upem)
            logger.debug("HarfBuzz shaping enabled: font=%s upem=%d", font_path, upem)
        else:
            warnings.append(
                f"HarfBuzz shaping unavailable for {font_path!r}; "
                "rendering may have degraded Indic glyph quality."
            )

    # ── Parse colours ─────────────────────────────────────────────────────────
    fill_rgba = _resolve_color(color)
    stroke_rgba: tuple[int, int, int, int] | None = None
    if stroke_color is not None and stroke_width > 0:
        stroke_rgba = _resolve_color(stroke_color)

    bg_rgba: tuple[int, int, int, int] | None = None
    if bg_color is not None:
        bg_rgba = _resolve_color(bg_color)

    # ── Line metrics ──────────────────────────────────────────────────────────
    ascender_px, descender_px, base_line_height = _get_line_metrics(
        pil_font, hb_font_data, font_size
    )
    # Account for stroke expanding the bounding box
    stroke_extra = stroke_width if stroke_rgba is not None else 0
    ascender_px += stroke_extra
    descender_px += stroke_extra
    base_line_height = ascender_px + descender_px

    line_height = int(base_line_height * line_spacing)

    # ── Word-wrap ─────────────────────────────────────────────────────────────
    lines = _word_wrap(
        text.strip(),
        max_width=max_width - stroke_extra * 2,
        pil_font=pil_font,
        hb_font_data=hb_font_data,
        font_size=font_size,
        warnings=warnings,
    )
    logger.debug("render_caption: wrapped to %d lines", len(lines))

    # ── Compute canvas size ───────────────────────────────────────────────────
    # Per-line widths (for alignment)
    line_widths: list[float] = []
    for ln in lines:
        if hb_font_data is not None and ln:
            try:
                hb_font_obj, upem = hb_font_data
                _, glyph_positions, _, scale_factor = _shape_text(
                    ln, hb_font_obj, upem, font_size
                )
                lw = _measure_shaped_width(glyph_positions, scale_factor)
            except Exception:
                lw = _measure_line_width_pil(ln, pil_font)
        else:
            lw = _measure_line_width_pil(ln, pil_font)
        line_widths.append(lw)

    text_width = int(max(line_widths, default=0)) + stroke_extra * 2 + 2
    text_height = line_height * len(lines)

    if bg_rgba is not None:
        canvas_w = text_width + bg_padding * 2
        canvas_h = text_height + bg_padding * 2
        text_x_origin = bg_padding
        text_y_origin = bg_padding
    else:
        canvas_w = text_width
        canvas_h = text_height
        text_x_origin = 0
        text_y_origin = 0

    canvas_w = max(canvas_w, 1)
    canvas_h = max(canvas_h, 1)

    # ── Create canvas ─────────────────────────────────────────────────────────
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # ── Background box ────────────────────────────────────────────────────────
    if bg_rgba is not None:
        draw_bg = ImageDraw.Draw(canvas)
        draw_bg.rounded_rectangle(
            (0, 0, canvas_w - 1, canvas_h - 1),
            radius=bg_radius,
            fill=bg_rgba,
        )

    # ── Render each line ──────────────────────────────────────────────────────
    for i, (ln, lw) in enumerate(zip(lines, line_widths)):
        line_top = text_y_origin + i * line_height

        # Horizontal alignment offset
        if align == "center":
            x_offset = int((text_width - lw) / 2)
        elif align == "right":
            x_offset = int(text_width - lw)
        else:
            x_offset = 0

        paste_x = text_x_origin + x_offset

        if is_indic and hb_font_data is not None:
            # Render with HarfBuzz cluster positioning
            hb_font_obj, upem = hb_font_data
            line_img = _render_line_shaped(
                ln,
                pil_font=pil_font,
                hb_font=hb_font_obj,
                upem=upem,
                font_size=font_size,
                fill_rgba=fill_rgba,
                stroke_rgba=stroke_rgba,
                stroke_width=stroke_width,
                canvas_width=text_width + stroke_extra * 2,
                line_height=line_height,
                ascender_px=ascender_px,
            )
        else:
            # Latin / fallback: plain PIL render
            line_img = _render_line_pil(
                ln,
                pil_font=pil_font,
                fill_rgba=fill_rgba,
                stroke_rgba=stroke_rgba,
                stroke_width=stroke_width,
                canvas_width=text_width + stroke_extra * 2,
                line_height=line_height,
                ascender_px=ascender_px,
            )

        canvas.alpha_composite(line_img, dest=(paste_x, line_top))

    logger.info(
        "render_caption: script=%r font=%s size=%d lines=%d canvas=%dx%d warnings=%d",
        detected_script,
        os.path.basename(font_path),
        font_size,
        len(lines),
        canvas_w,
        canvas_h,
        len(warnings),
    )

    return CaptionResult(
        image=canvas,
        width=canvas_w,
        height=canvas_h,
        font_path=font_path,
        script=detected_script,
        warnings=warnings,
    )
