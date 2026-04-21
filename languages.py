"""Central language config — single source of truth for all locale-specific behavior.

Adding a new language = add one entry to LANGUAGES. Every consumer reads from
this module (prompts, fonts, follow-bar text, search queries) so nothing
is Telugu-hardcoded anymore.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

FONTS_DIR = Path(__file__).parent / "resources" / "fonts"


@dataclass(frozen=True)
class LanguageConfig:
    code: str                    # ISO 639-1: te, hi, ta, kn, ml, bn, mr, gu, en
    name_english: str            # "Telugu"
    name_native: str             # "తెలుగు"
    script: str                  # "Telugu" | "Devanagari" | "Tamil" | ...
    direction: str = "ltr"
    font_primary: str = ""       # absolute path to bold TTF (headline/card)
    font_secondary: str = ""     # absolute path to regular TTF (body text)
    font_fallback: str = ""      # Latin for mixed content
    follow_bar_text: str = ""    # translated "FOLLOW KAIZER NEWS"
    news_search_seed: List[str] = field(default_factory=list)  # base image-search terms
    prompt_language_phrase: str = ""  # for Gemini prompt: "Telugu news"
    title_style_hint: str = ""   # extra guidance for title generation


def _p(name: str) -> str:
    """Absolute path to a bundled font file, or '' if missing."""
    path = FONTS_DIR / name
    return str(path) if path.exists() else ""


# ── Primary & fallback fonts are resolved lazily so a missing file degrades
# ──   gracefully to the Telugu bundle (which was the original default).
_LATIN_BOLD    = _p("NotoSans-Bold.ttf")    or _p("Roboto-Bold.ttf")
_LATIN_REG     = _p("NotoSans-Regular.ttf") or _p("Roboto-Bold.ttf")
_FALLBACK_BOLD = _p("NotoSansTelugu-Bold.ttf")
_FALLBACK_REG  = _p("NotoSansTelugu-Regular.ttf")


LANGUAGES: Dict[str, LanguageConfig] = {
    "te": LanguageConfig(
        code="te", name_english="Telugu", name_native="తెలుగు", script="Telugu",
        font_primary   = _p("NotoSansTelugu-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansTelugu-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="FOLLOW KAIZER NEWS TELUGU",
        news_search_seed=["Telugu news today", "తెలుగు వార్తలు", "Andhra Telangana news"],
        prompt_language_phrase="Telugu",
        title_style_hint="Short, punchy Telugu news ticker headline. Use natural Telugu — no transliteration.",
    ),
    "hi": LanguageConfig(
        code="hi", name_english="Hindi", name_native="हिन्दी", script="Devanagari",
        font_primary   = _p("NotoSansDevanagari-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansDevanagari-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="फॉलो कैज़र न्यूज़",
        news_search_seed=["Hindi news today", "हिंदी समाचार", "Bharat news"],
        prompt_language_phrase="Hindi",
        title_style_hint="Short, punchy Hindi news ticker headline in Devanagari script. No English transliteration.",
    ),
    "ta": LanguageConfig(
        code="ta", name_english="Tamil", name_native="தமிழ்", script="Tamil",
        font_primary   = _p("NotoSansTamil-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansTamil-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="கைசர் செய்தியை பின்தொடருங்கள்",
        news_search_seed=["Tamil news today", "தமிழ் செய்திகள்", "Chennai news"],
        prompt_language_phrase="Tamil",
        title_style_hint="Short Tamil news ticker headline in Tamil script. No transliteration.",
    ),
    "kn": LanguageConfig(
        code="kn", name_english="Kannada", name_native="ಕನ್ನಡ", script="Kannada",
        font_primary   = _p("NotoSansKannada-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansKannada-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="ಕೈಸರ್ ಸುದ್ದಿಯನ್ನು ಅನುಸರಿಸಿ",
        news_search_seed=["Kannada news today", "ಕನ್ನಡ ಸುದ್ದಿ", "Bangalore news"],
        prompt_language_phrase="Kannada",
        title_style_hint="Short Kannada news ticker headline in Kannada script. No transliteration.",
    ),
    "ml": LanguageConfig(
        code="ml", name_english="Malayalam", name_native="മലയാളം", script="Malayalam",
        font_primary   = _p("NotoSansMalayalam-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansMalayalam-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="കൈസർ ന്യൂസ് പിന്തുടരുക",
        news_search_seed=["Malayalam news today", "മലയാളം വാർത്ത", "Kerala news"],
        prompt_language_phrase="Malayalam",
        title_style_hint="Short Malayalam news ticker headline in Malayalam script.",
    ),
    "bn": LanguageConfig(
        code="bn", name_english="Bengali", name_native="বাংলা", script="Bengali",
        font_primary   = _p("NotoSansBengali-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansBengali-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="কাইজার নিউজ অনুসরণ করুন",
        news_search_seed=["Bengali news today", "বাংলা খবর", "Kolkata news"],
        prompt_language_phrase="Bengali",
        title_style_hint="Short Bengali news ticker headline in Bengali script.",
    ),
    "mr": LanguageConfig(
        code="mr", name_english="Marathi", name_native="मराठी", script="Devanagari",
        font_primary   = _p("NotoSansDevanagari-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansDevanagari-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="कैझर न्यूजला फॉलो करा",
        news_search_seed=["Marathi news today", "मराठी बातम्या", "Maharashtra news"],
        prompt_language_phrase="Marathi",
        title_style_hint="Short Marathi news ticker headline in Devanagari script.",
    ),
    "gu": LanguageConfig(
        code="gu", name_english="Gujarati", name_native="ગુજરાતી", script="Gujarati",
        font_primary   = _p("NotoSansGujarati-Bold.ttf") or _FALLBACK_BOLD,
        font_secondary = _p("NotoSansGujarati-Regular.ttf") or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="કૈસર ન્યૂઝ ફોલો કરો",
        news_search_seed=["Gujarati news today", "ગુજરાતી સમાચાર", "Gujarat news"],
        prompt_language_phrase="Gujarati",
        title_style_hint="Short Gujarati news ticker headline in Gujarati script.",
    ),
    "en": LanguageConfig(
        code="en", name_english="English", name_native="English", script="Latin",
        font_primary   = _LATIN_BOLD or _FALLBACK_BOLD,
        font_secondary = _LATIN_REG or _FALLBACK_REG,
        font_fallback  = _LATIN_BOLD,
        follow_bar_text="FOLLOW KAIZER NEWS",
        news_search_seed=["breaking news today", "world news", "top stories"],
        prompt_language_phrase="English",
        title_style_hint="Short, punchy English news ticker headline. All caps if it fits.",
    ),
}

DEFAULT_LANG = "te"


def get(code: Optional[str]) -> LanguageConfig:
    """Resolve a language code with graceful fallback to Telugu."""
    if not code:
        return LANGUAGES[DEFAULT_LANG]
    return LANGUAGES.get(code.lower().strip()) or LANGUAGES[DEFAULT_LANG]


def list_options() -> List[Dict[str, str]]:
    """Payload for the frontend language picker."""
    return [
        {"code": c.code, "english": c.name_english, "native": c.name_native, "script": c.script}
        for c in LANGUAGES.values()
    ]


def get_font(code: str, weight: str = "bold") -> str:
    """Return absolute font path for a language + weight."""
    cfg = get(code)
    if weight == "bold":
        return cfg.font_primary
    return cfg.font_secondary
