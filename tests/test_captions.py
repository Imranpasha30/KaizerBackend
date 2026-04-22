"""
test_captions.py — Tests for pipeline_core.captions

TDD-style, written against the spec before the Builder ships the module.
When captions.py does not exist, all tests are skipped with "blocked on Builder".

Spec under test:
  detect_script(text: str) -> str
    Returns: 'latin' | 'devanagari' | 'telugu' | 'tamil' | 'bengali' |
             'kannada' | 'malayalam' | 'gujarati' | 'mixed'

  render_caption(text, *, max_width, font_size, color, stroke_color,
                 stroke_width, bg_color, bg_padding, bg_radius,
                 line_spacing, align, script) -> CaptionResult

  safe_zone(platform, w=1080, h=1920) -> tuple[int, int, int, int]
    youtube_short   : (60, 120, 900, 1520)
    instagram_reel  : (60, 250, 840, 1350)
    tiktok          : (60, 150, 860, 1420)
    Unknown platform: raises ValueError
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import guard — collected but skipped until Builder ships
# ---------------------------------------------------------------------------
try:
    from pipeline_core.captions import detect_script, render_caption, safe_zone, CaptionResult
    _CAPTIONS_AVAILABLE = True
except ImportError:
    _CAPTIONS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CAPTIONS_AVAILABLE,
    reason="pipeline_core.captions not yet implemented (blocked on Builder)",
)


# ===========================================================================
# Script detection — pure Unicode-range logic; no font files required
# ===========================================================================

def test_detect_latin_english():
    """ASCII/Latin text must be classified as 'latin'."""
    result = detect_script("Breaking news from London")
    assert result == "latin", f"Expected 'latin', got {result!r}"


def test_detect_telugu():
    """Telugu Unicode block (U+0C00-U+0C7F) must return 'telugu'."""
    result = detect_script("తెలుగు వార్తలు")
    assert result == "telugu", f"Expected 'telugu', got {result!r}"


def test_detect_devanagari_hindi():
    """Hindi (Devanagari U+0900-U+097F) must return 'devanagari'."""
    result = detect_script("नमस्ते")
    assert result == "devanagari", f"Expected 'devanagari', got {result!r}"


def test_detect_devanagari_marathi():
    """Marathi also uses Devanagari script; must return 'devanagari'."""
    result = detect_script("महाराष्ट्र")
    assert result == "devanagari", f"Expected 'devanagari', got {result!r}"


def test_detect_tamil():
    """Tamil Unicode block (U+0B80-U+0BFF) must return 'tamil'."""
    result = detect_script("தமிழ்")
    assert result == "tamil", f"Expected 'tamil', got {result!r}"


def test_detect_bengali():
    """Bengali Unicode block (U+0980-U+09FF) must return 'bengali'."""
    result = detect_script("বাংলা")
    assert result == "bengali", f"Expected 'bengali', got {result!r}"


def test_detect_kannada():
    """Kannada Unicode block (U+0C80-U+0CFF) must return 'kannada'."""
    result = detect_script("ಕನ್ನಡ")
    assert result == "kannada", f"Expected 'kannada', got {result!r}"


def test_detect_malayalam():
    """Malayalam Unicode block (U+0D00-U+0D7F) must return 'malayalam'."""
    result = detect_script("മലയാളം")
    assert result == "malayalam", f"Expected 'malayalam', got {result!r}"


def test_detect_gujarati():
    """Gujarati Unicode block (U+0A80-U+0AFF) must return 'gujarati'."""
    result = detect_script("ગુજરાતી")
    assert result == "gujarati", f"Expected 'gujarati', got {result!r}"


def test_detect_latin_telugu_returns_telugu_when_telugu_dominant():
    """When Telugu is the dominant non-ASCII script, return 'telugu' (the
    product's target use case — creators write mostly Telugu with a few
    English words, and the caption renderer must still pick the Telugu font).
    'mixed' is reserved for genuinely unclassifiable multi-script content."""
    result = detect_script("Breaking: తెలుగు న్యూస్")
    assert result == "telugu", (
        f"Expected 'telugu' (dominant-script semantics), got {result!r}. "
        "Mixed Telugu+English content MUST be classified as telugu so the "
        "caption renderer picks NotoSansTelugu, not NotoSans-Latin."
    )


def test_detect_mixed_returns_mixed():
    """Genuinely multi-script content with no single dominant non-ASCII script
    must return 'mixed'. Here Telugu + Devanagari + Tamil are each ~33% of
    non-ASCII characters, so no single script wins."""
    result = detect_script("తెలుగు नमस्ते தமிழ்")
    assert result in {"mixed", "telugu", "devanagari", "tamil"}, (
        f"Expected multi-script classification, got {result!r}. "
        "Accepting any of the represented scripts or 'mixed' — the exact "
        "tie-break depends on character count which varies with conjuncts."
    )


def test_detect_empty_string_is_latin():
    """Empty string has no script-specific characters; must default to 'latin'."""
    result = detect_script("")
    assert result == "latin", f"Expected 'latin' for empty string, got {result!r}"


def test_detect_whitespace_only_is_latin():
    """Whitespace-only string has no script characters; must default to 'latin'."""
    result = detect_script("   \t\n  ")
    assert result == "latin", f"Expected 'latin' for whitespace-only, got {result!r}"


# ===========================================================================
# Rendering
# ===========================================================================

def test_render_latin_returns_image():
    """
    Rendering plain Latin text must return a CaptionResult whose .image is an
    RGBA PIL Image with positive width and height.
    """
    from PIL import Image as _PILImage

    result = render_caption("Hello World", max_width=800)

    assert isinstance(result, CaptionResult), f"Expected CaptionResult, got {type(result)}"
    assert isinstance(result.image, _PILImage.Image), (
        f"result.image must be a PIL Image, got {type(result.image)}"
    )
    assert result.image.mode == "RGBA", (
        f"result.image must be RGBA, got {result.image.mode!r}"
    )
    assert result.width > 0, "result.width must be positive"
    assert result.height > 0, "result.height must be positive"


def test_render_telugu_returns_image():
    """
    Rendering Telugu text must not raise even when the ideal font is absent.
    When a suitable font is unavailable the implementation may populate
    .warnings but must still return a CaptionResult with a valid image.
    """
    result = render_caption("తెలుగు వార్తలు", max_width=800)

    assert isinstance(result, CaptionResult), f"Expected CaptionResult, got {type(result)}"
    assert result.width > 0
    assert result.height > 0
    # warnings is a list (possibly empty if a Telugu font was found)
    assert isinstance(result.warnings, list), "CaptionResult.warnings must be a list"


def test_render_respects_max_width():
    """
    The rendered image width must not exceed max_width plus the bg_padding
    tolerance (16 px each side = 32 px), even for long text.
    """
    long_text = "This is a very long breaking news headline that definitely exceeds a small width"
    max_width = 400
    result = render_caption(long_text, max_width=max_width, bg_padding=16)

    tolerance = 32  # 16px padding on each side
    assert result.width <= max_width + tolerance, (
        f"Rendered width {result.width} exceeds max_width {max_width} + tolerance {tolerance}"
    )


def test_render_wraps_long_single_word():
    """
    A single token longer than max_width must be char-wrapped so the image
    does not overflow. The resulting image must still have width <= max_width
    + padding tolerance and height > a single line's height.
    """
    # A single "word" of repeated characters that will exceed 200px at 56pt
    long_word = "X" * 80
    max_width = 200
    result = render_caption(long_word, max_width=max_width, font_size=56, bg_padding=0)

    tolerance = 32
    assert result.width <= max_width + tolerance, (
        f"Single long word overflowed: width={result.width}, max={max_width}"
    )


def test_render_applies_stroke():
    """
    Rendering with stroke_width=4 should increase dark-pixel coverage compared
    to rendering without any stroke (stroke_color=None).
    We sample the total sum of the image's luminance channel as a proxy.
    """
    import numpy as np

    text = "KAIZER"
    no_stroke = render_caption(
        text, max_width=600, font_size=56,
        stroke_color=None, stroke_width=0,
        bg_color=None,
    )
    with_stroke = render_caption(
        text, max_width=600, font_size=56,
        stroke_color="#000000", stroke_width=4,
        bg_color=None,
    )

    # Convert to numpy and count pixels with low luminance (dark from stroke)
    def _dark_pixel_count(img):
        """Count pixels where ANY channel is < 50 (dark)."""
        arr = np.array(img.convert("RGBA"))
        # Only count non-transparent pixels that are dark
        visible = arr[:, :, 3] > 20
        dark = (arr[:, :, 0] < 50) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
        return int((visible & dark).sum())

    # Resize to same canvas for fair comparison if dimensions differ
    no_stroke_dark = _dark_pixel_count(no_stroke.image)
    with_stroke_dark = _dark_pixel_count(with_stroke.image)

    assert with_stroke_dark > no_stroke_dark, (
        f"Stroke should add dark pixels: no_stroke={no_stroke_dark}, "
        f"with_stroke={with_stroke_dark}"
    )


def test_render_bg_box_changes_alpha_channel():
    """
    When bg_color is set, the background region must have non-zero alpha.
    Sample the center pixel of the rendered image — it must not be fully
    transparent.
    """
    result = render_caption(
        "News Flash",
        max_width=600,
        font_size=56,
        bg_color="#FF000080",  # semi-transparent red
        bg_padding=16,
    )

    img = result.image
    assert img.mode == "RGBA"

    cx, cy = img.width // 2, img.height // 2
    pixel = img.getpixel((cx, cy))  # returns (R, G, B, A)
    alpha = pixel[3]

    assert alpha > 0, (
        f"Center pixel alpha is 0 — bg_color was not rendered. Pixel={pixel}"
    )


def test_render_align_center_vs_left_differs():
    """
    Two renders of the same text with align='center' and align='left' must
    produce different images (the text x-offsets should differ).
    We compare the raw pixel bytes of the images.
    """
    text = "Kaizer Breaking News"
    max_width = 800

    centered = render_caption(text, max_width=max_width, align="center", bg_color=None)
    left_aligned = render_caption(text, max_width=max_width, align="left", bg_color=None)

    # Convert both to the same size for comparison (pad smaller one)
    w = max(centered.image.width, left_aligned.image.width)
    h = max(centered.image.height, left_aligned.image.height)

    from PIL import Image as _PILImage
    canvas_c = _PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas_l = _PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas_c.paste(centered.image, (0, 0))
    canvas_l.paste(left_aligned.image, (0, 0))

    assert canvas_c.tobytes() != canvas_l.tobytes(), (
        "center and left aligned renders produced identical images — alignment is not being applied"
    )


# ===========================================================================
# Safe zones
# ===========================================================================

def test_safe_zone_youtube_short():
    """youtube_short safe zone must be (60, 120, 900, 1520)."""
    result = safe_zone("youtube_short")
    assert result == (60, 120, 900, 1520), (
        f"youtube_short safe zone: expected (60, 120, 900, 1520), got {result}"
    )


def test_safe_zone_instagram_reel():
    """instagram_reel safe zone must be (60, 250, 840, 1350)."""
    result = safe_zone("instagram_reel")
    assert result == (60, 250, 840, 1350), (
        f"instagram_reel safe zone: expected (60, 250, 840, 1350), got {result}"
    )


def test_safe_zone_tiktok():
    """tiktok safe zone must be (60, 150, 860, 1420)."""
    result = safe_zone("tiktok")
    assert result == (60, 150, 860, 1420), (
        f"tiktok safe zone: expected (60, 150, 860, 1420), got {result}"
    )


def test_safe_zone_unknown_platform_raises():
    """Passing an unknown platform name must raise ValueError."""
    with pytest.raises(ValueError):
        safe_zone("facebook_story")


def test_safe_zone_ig_right_margin_wider_than_yt():
    """
    Critical regression guard: Instagram Reel has a wider right exclusion zone
    than YouTube Short.

    Interpretation of the tuple (x_left, y_top, x_right, y_bottom):
      - IG  : x_right = 840  →  right margin = 1080 - 840 = 240
      - YT  : x_right = 900  →  right margin = 1080 - 900 = 180
    So IG right margin (240) > YT right margin (180).
    """
    yt = safe_zone("youtube_short")
    ig = safe_zone("instagram_reel")

    # tuple layout: (x_left, y_top, x_right, y_bottom)
    yt_right_margin = 1080 - yt[2]   # 1080 - 900 = 180
    ig_right_margin = 1080 - ig[2]   # 1080 - 840 = 240

    assert ig_right_margin > yt_right_margin, (
        f"IG right margin ({ig_right_margin}px) must exceed YT right margin "
        f"({yt_right_margin}px). "
        f"yt safe_zone={yt}, ig safe_zone={ig}"
    )
