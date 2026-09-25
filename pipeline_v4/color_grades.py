"""Color-grade registry — the 23 broadcast looks, as ffmpeg chains.

Each grade is a vetted, NVENC-safe ffmpeg video-filter fragment (only
standard filters: eq / colorbalance / colorchannelmixer / curves / hue /
colorhold / pseudocolor). Style packs, formulas, the trailer engine and
the editor all reference grades BY ID, so a look is defined once and
used everywhere.

Drop-in LUT override: put a `<key>.cube` file in
``KaizerBackend/assets/luts/`` and ``get_grade_vf(key)`` returns a
``lut3d`` chain instead of the procedural fragment — purchased looks
upgrade the whole system with zero code changes.

Every chain is validated by rendering a real frame in
tests/test_color_grades.py — this is the test class that catches
"option does not exist on this ffmpeg" bugs before a paid render does.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Grade:
    key: str
    label: str
    chain: str          # ffmpeg -vf fragment
    used_for: str


_G: tuple[Grade, ...] = (
    Grade("newsroom_neutral", "Newsroom Neutral",
          "eq=contrast=1.10:saturation=1.12,unsharp=5:5:0.5:5:5:0.0",
          "Default news, interviews — clean whites, broadcast safe"),
    Grade("breaking_red_alert", "Breaking Red Alert",
          "eq=contrast=1.22:saturation=0.85,colorbalance=rs=0.10:rm=0.14",
          "Breaking news, alerts — desaturated with red channel push"),
    Grade("cinematic_teal_orange", "Cinematic Teal & Orange",
          "colorbalance=bs=0.12:bm=0.04:rh=0.10:bh=-0.10,eq=contrast=1.12:saturation=1.08",
          "Human interest, documentaries — warm skin, cool shadows"),
    Grade("bleach_bypass", "Bleach Bypass",
          "eq=saturation=0.35:contrast=1.30",
          "Crime, war, serious — silver, gritty, low saturation"),
    Grade("high_contrast_noir", "High Contrast Noir",
          "hue=s=0,eq=contrast=1.45:gamma=0.90",
          "Investigations, night — deep blacks, bright whites"),
    Grade("warm_golden_hour", "Warm Golden Hour",
          "colorbalance=rs=0.10:rm=0.10:bs=-0.06,eq=brightness=0.03:saturation=1.10",
          "Hopeful endings, culture — amber warmth, soft shadows"),
    Grade("cold_blue_steel", "Cold Blue Steel",
          "colorbalance=bs=0.14:bm=0.10,eq=saturation=0.85:contrast=1.10",
          "Tech, finance, medical — clinical blue-gray"),
    Grade("desaturated_documentary", "Desaturated Documentary",
          "eq=saturation=0.65:contrast=1.08",
          "Tragedy, serious news — muted, respectful"),
    Grade("vibrant_pop", "Vibrant Pop",
          "eq=saturation=1.50:contrast=1.18",
          "Entertainment, sports, viral — punchy and saturated"),
    Grade("sepia_archive", "Sepia Archive",
          "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:0",
          "Historical context, flashbacks — vintage brown"),
    Grade("night_vision_green", "Night Vision Green",
          "hue=s=0,colorchannelmixer=0:0:0:0:.8:1:.2:0:0:0:0:0,noise=alls=8:allf=t",
          "Military, security, surveillance — mono green + grain"),
    Grade("thermal_camera", "Thermal Camera",
          "format=gray,pseudocolor=preset=inferno",
          "Search operations, tech demos — false-color heat map"),
    Grade("drone_flat", "Drone Flat",
          "eq=contrast=0.92:saturation=0.90:brightness=0.02",
          "Aerial footage, landscapes — flat log-style profile"),
    Grade("instagram_vintage", "Instagram Vintage",
          "curves=preset=vintage,eq=saturation=0.95",
          "Social media, youth culture — faded warm nostalgia"),
    Grade("neon_cyberpunk", "Neon Cyberpunk",
          "colorbalance=rs=-0.08:bs=0.16:rm=0.10:bm=0.06,eq=saturation=1.35:contrast=1.20",
          "Crypto, hacking, futuristic — magenta-cyan neon"),
    Grade("monochrome", "Monochrome",
          "hue=s=0,eq=contrast=1.15",
          "Memorial, serious, artistic — classic black & white"),
    Grade("monochrome_accent_red", "Monochrome + Red Accent",
          "colorhold=color=red:similarity=0.35:blend=0.0",
          "Danger, urgency — B&W with only red kept"),
    Grade("monochrome_accent_gold", "Monochrome + Gold Accent",
          "colorhold=color=orange:similarity=0.30:blend=0.1",
          "Exclusive, premium, awards — B&W with gold kept"),
    Grade("faded_film", "Faded Film",
          "eq=saturation=0.80:contrast=0.95:brightness=0.04,noise=alls=5:allf=t",
          "Memory, nostalgia, personal — washed soft grain"),
    Grade("blockbuster_orange_teal", "Blockbuster Orange Teal",
          "colorbalance=bs=0.20:bm=0.08:rh=0.16:bh=-0.16,eq=contrast=1.22:saturation=1.15",
          "Movie trailers, epic stories — aggressive orange-teal"),
    Grade("newsprint", "Newsprint",
          "hue=s=0,eq=contrast=1.50:gamma=1.05",
          "Print journalism, investigations — stark B&W"),
    Grade("police_interrogation", "Police Interrogation",
          "colorbalance=gm=0.10:gs=0.06,eq=contrast=1.25:gamma=0.92:saturation=0.75",
          "Interrogation, crime scene — harsh greenish overhead"),
    Grade("sunset_graded", "Sunset Graded",
          "colorbalance=rs=0.12:rm=0.10:bh=-0.10:bs=0.06,eq=saturation=1.15:brightness=0.02",
          "Nature, end of day, reflective — warm orange, purple shadow"),
)

GRADES: dict[str, Grade] = {g.key: g for g in _G}
DEFAULT_GRADE = "newsroom_neutral"


def _luts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "luts"


def get_grade_vf(key: str) -> str:
    """The -vf fragment for a grade id. A drop-in ``assets/luts/<key>.cube``
    wins over the procedural chain; unknown ids fall back to the neutral
    newsroom look (a wrong look must never kill a render)."""
    k = (key or "").strip().lower()
    cube = _luts_dir() / f"{k}.cube"
    if k and cube.is_file():
        # lut3d wants forward slashes + escaped colon on Windows paths.
        p = str(cube).replace("\\", "/").replace(":", "\\:")
        return f"lut3d=file='{p}'"
    return GRADES.get(k, GRADES[DEFAULT_GRADE]).chain


def grade_catalog() -> list[dict]:
    """[{key,label,used_for,source}] for pickers + the admin features tab."""
    out = []
    for g in _G:
        cube = _luts_dir() / f"{g.key}.cube"
        out.append({"key": g.key, "label": g.label, "used_for": g.used_for,
                    "source": "lut" if cube.is_file() else "procedural"})
    return out
