"""ffmpeg color grade chains — direct port of the teammate's
``colorGradeChain()``.

Returns a comma-separated ffmpeg filter chain (or None for "off") that
can be inlined inside a larger filter_complex graph. The shapes match
the teammate's exactly so visual A/B is meaningful.

Presets
-------
- ``subtle``      light pro polish, won't change skin tones (default)
- ``cinematic``   teal-and-orange film look with vignette
- ``news-vivid``  punchy broadcast-news grade (BIG TV / NTV style)
- ``warm``        golden-hour bias
- ``cool``        bluish bias
- ``off``         no grade, returns None
"""
from __future__ import annotations

from typing import Optional


_PRESETS: dict[str, Optional[str]] = {
    "off":      None,

    "subtle": ",".join([
        "curves=preset=increase_contrast",
        "eq=saturation=1.08:contrast=1.04:brightness=0.01:gamma=0.98",
        "unsharp=3:3:0.5:3:3:0.0",
        "format=yuv420p",
    ]),

    "cinematic": ",".join([
        "curves=preset=increase_contrast",
        "eq=saturation=1.18:contrast=1.10:brightness=0.005:gamma=0.95",
        # colorbalance rs/gs/bs = shadows red/green/blue; rm/gm/bm =
        # midtones; rh/gh/bh = highlights. Negative blue in shadows
        # warms them; positive blue in highlights cools the sky.
        "colorbalance=rs=-0.02:gs=-0.02:bs=0.04:rm=0.02:gm=-0.01:bm=-0.02:rh=0.04:gh=0.02:bh=-0.04",
        "unsharp=3:3:0.7:3:3:0.0",
        "vignette=PI/5",
        "format=yuv420p",
    ]),

    "news-vivid": ",".join([
        "curves=preset=strong_contrast",
        "eq=saturation=1.28:contrast=1.14:brightness=0.0:gamma=0.92",
        "colorbalance=rs=0.02:gs=0:bs=-0.02:rm=0.03:gm=0:bm=-0.02:rh=0.04:gh=0.02:bh=-0.03",
        "unsharp=5:5:0.8:5:5:0.0",
        "format=yuv420p",
    ]),

    "warm": ",".join([
        "eq=saturation=1.10:contrast=1.05:brightness=0.02:gamma=0.97",
        "colorbalance=rm=0.05:gm=0.02:bm=-0.04:rh=0.03:bh=-0.03",
        "unsharp=3:3:0.4:3:3:0.0",
        "format=yuv420p",
    ]),

    "cool": ",".join([
        "eq=saturation=1.08:contrast=1.06:brightness=0.0:gamma=0.98",
        "colorbalance=rs=-0.02:bs=0.05:rm=-0.01:bm=0.03",
        "unsharp=3:3:0.5:3:3:0.0",
        "format=yuv420p",
    ]),
}


VALID_PRESETS = tuple(_PRESETS.keys())


def grade_chain(preset: Optional[str]) -> Optional[str]:
    """Return the ffmpeg filter chain for ``preset``, or None when
    grading is disabled / preset unknown. Unknown presets fall back
    to ``subtle`` to match the teammate's default behavior."""
    if not preset:
        return _PRESETS["subtle"]
    return _PRESETS.get(preset.lower().strip(), _PRESETS["subtle"])
