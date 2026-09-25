# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/theme.py.
# Changes from upstream: none (verbatim copy; origin header added).
"""Language -> Remotion theme mapping + localized promo CTA text.

PURE (no I/O). The Remotion design wave selects its visual theme from the
``theme`` key in the EDL (see ``remotion/src/edl.ts`` -> ``EdlSchema.theme``).
We map the job language to a theme KEY the design wave understands:

  * Telugu (``te``) -> ``'telugu'``   (Telugu-script-aware theme)
  * everything else -> ``'kaizerDark'`` (the default dark theme)

We also thread the raw ``language`` code through so whichever field the
Remotion side keys on (``language`` or ``theme``) is populated.
"""

from __future__ import annotations

# Theme keys the Remotion design wave registers.
THEME_TELUGU = "telugu"
THEME_DEFAULT = "kaizerDark"

# Localized "watch the full episode" CTA for the promo end-card. Telugu is in
# Telugu script; all other languages fall back to English.
_CTA_TELUGU = "పూర్తి వీడియో చూడండి"  # "poorti video choodandi"
_CTA_DEFAULT = "Watch the full episode"


def theme_for_language(language: str | None) -> str:
    """Map a language code to a Remotion theme key.

    ``te`` (Telugu) -> ``'telugu'``; anything else (incl. None) -> ``'kaizerDark'``.
    Case/whitespace tolerant; accepts region-tagged codes like ``te-IN``.
    """
    code = (language or "").strip().lower()
    base = code.split("-", 1)[0].split("_", 1)[0]
    return THEME_TELUGU if base == "te" else THEME_DEFAULT


def end_card_cta_for_language(language: str | None) -> str:
    """Localized end-card CTA text for the promo. Telugu script for ``te``."""
    code = (language or "").strip().lower()
    base = code.split("-", 1)[0].split("_", 1)[0]
    return _CTA_TELUGU if base == "te" else _CTA_DEFAULT


__all__ = [
    "THEME_TELUGU",
    "THEME_DEFAULT",
    "theme_for_language",
    "end_card_cta_for_language",
]
