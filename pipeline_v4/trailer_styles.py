"""Trailer style packs — 30 content categories, each with its OWN look
and sound (spec: "for horror its own sound effect, for crime its own…").

A pack bundles everything category-specific about a trailer:
  * color grade + frame effects (tint, grain, sharpen, flash-cut entrance)
  * its transition palette (subset of the 54-transition catalog below)
  * pacing (crossfade speed)
  * title-card colors + sub-line
  * SOUND DESIGN parameters — the synth recipes in trailer.py are
    parameterized, so horror gets a 45 Hz long-decay boom over a dark
    drone bed while comedy gets a bright pop and no bed. All generated
    with ffmpeg lavfi: zero licensing.

The planner auto-classifies the content into one of these keys (the
operator can force one); unknown keys fall back to "news".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── The full transition catalog (ffmpeg xfade names, all vetted) ─────
TRANSITION_CATALOG: tuple[str, ...] = (
    "fade", "fadeblack", "fadewhite", "fadegrays", "distance", "dissolve",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "wipetl", "wipetr", "wipebl", "wipebr",
    "slideleft", "slideright", "slideup", "slidedown",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circlecrop", "rectcrop", "circleclose", "circleopen",
    "horzclose", "horzopen", "vertclose", "vertopen",
    "diagbl", "diagbr", "diagtl", "diagtr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "pixelize", "radial", "hblur", "zoomin",
    "squeezev", "squeezeh",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
    "hlwind", "hrwind", "vuwind", "vdwind",
)   # 56 built-in xfade names

# ── Advanced transitions — xfade CUSTOM expressions ──────────────────
# P runs 1→0 (1 = all first clip A, 0 = all second clip B). A/B are the
# current plane's pixels at X,Y; a0..a3/b0..b3 sample arbitrary coords.
# GOTCHA (verified on host): st()/ld() return garbage inside xfade
# custom exprs — coordinates must be INLINED into every sample call.
# Each transition is render-tested on the host like every other chain.


def _samp(src: str, xs: str, ys: str) -> str:
    """Plane-dispatched arbitrary-coordinate sample of input a or b."""
    return (f"if(eq(PLANE,0),{src}0({xs},{ys}),"
            f"if(eq(PLANE,1),{src}1({xs},{ys}),"
            f"if(eq(PLANE,2),{src}2({xs},{ys}),{src}3({xs},{ys}))))")

EXTRA_TRANSITIONS: dict[str, str] = {
    "blinds_h": "if(lt(mod(Y,H/12),(H/12)*(1-P)),B,A)",
    "blinds_v": "if(lt(mod(X,W/16),(W/16)*(1-P)),B,A)",
    "checkerboard": ("if(gt(1-P,0.25+0.5*mod(floor(X/(W/8))"
                     "+floor(Y/(H/4.5)),2)),B,A)"),
    "diag_soft_tl": ("A*(1-clip(((1-P)*1.4-((X/W+Y/H)/2))*6+0.5,0,1))"
                     "+B*clip(((1-P)*1.4-((X/W+Y/H)/2))*6+0.5,0,1)"),
    "diag_soft_br": ("A*(1-clip(((1-P)*1.4-(((W-X)/W+(H-Y)/H)/2))*6+0.5,0,1))"
                     "+B*clip(((1-P)*1.4-(((W-X)/W+(H-Y)/H)/2))*6+0.5,0,1)"),
    "barn_door_open": "if(lt(abs(X-W/2),(W/2)*(1-P)),B,A)",
    "iris_diamond": ("if(lt(abs(X-W/2)/(W/2)+abs(Y-H/2)/(H/2),"
                     "2*(1-P)),B,A)"),
    "pixel_dissolve": ("if(lt(mod(floor(X/16)*13+floor(Y/16)*7,29)/29,"
                       "1-P),B,A)"),
    "noise_dissolve": "if(lt(mod(X*31+Y*17,97)/97,1-P),B,A)",
    "quad_flip": ("if(gt(1-P,0.2+0.15*(floor(2*X/W)+2*floor(2*Y/H))),"
                  "B,A)"),
    "clock_sweep": "if(lt((atan2(Y-H/2,X-W/2)+PI)/(2*PI),1-P),B,A)",
    "melt_dissolve": "if(gt((1-P)*1.15,A/255),B,A)",
    # sampled-coordinate transitions (coords inlined per the st/ld gotcha)
    "ripple_wave": (
        _samp("a", "floor(clip(X+10*(1-P)*sin(Y/15+(1-P)*15),0,W-1))", "Y")
        + "*P+B*(1-P)"),
    "zoom_punch_in": (
        _samp("a", "floor(clip((X-W/2)/(1+0.6*(1-P))+W/2,0,W-1))",
              "floor(clip((Y-H/2)/(1+0.6*(1-P))+H/2,0,H-1))")
        + "*P+B*(1-P)"),
    "zoom_punch_out": (
        _samp("b", "floor(clip((X-W/2)/(1+0.6*P)+W/2,0,W-1))",
              "floor(clip((Y-H/2)/(1+0.6*P)+H/2,0,H-1))")
        + "*(1-P)+A*P"),
    "whip_left": (
        "if(lt(X+W*(1-P),W),"
        + _samp("a", "floor(mod(X+W*(1-P),W))", "Y") + ","
        + _samp("b", "floor(mod(X+W*(1-P),W))", "Y") + ")"),
    "whip_right": (
        "if(gt(X-W*(1-P),0),"
        + _samp("a", "floor(mod(X-W*(1-P)+W,W))", "Y") + ","
        + _samp("b", "floor(mod(X-W*(1-P)+W,W))", "Y") + ")"),
    "stretch_h": (
        _samp("a", "floor(clip((X-W/2)/max(P,0.05)+W/2,0,W-1))", "Y")
        + "*P+B*(1-P)"),
    "stretch_v": (
        _samp("a", "X", "floor(clip((Y-H/2)/max(P,0.05)+H/2,0,H-1))")
        + "*P+B*(1-P)"),
    "glitch_slices": (
        _samp("a", "floor(clip(X+(mod(floor(Y/(H/14))*37,11)-5)"
                   "*10*(1-P),0,W-1))", "Y")
        + "*P+B*(1-P)"),
}

TRANSITION_CATALOG = TRANSITION_CATALOG + tuple(sorted(EXTRA_TRANSITIONS))
# 56 built-ins + 20 custom = 76 total


def xfade_arg(name: str) -> str:
    """The xfade parameter string for a transition id — builtin name or
    a custom expression. Unknown → plain fade (never dies on a name)."""
    n = (name or "").strip().lower()
    if n in EXTRA_TRANSITIONS:
        return f"transition=custom:expr='{EXTRA_TRANSITIONS[n]}'"
    if n in TRANSITION_CATALOG:
        return f"transition={n}"
    return "transition=fade"


@dataclass(frozen=True)
class TrailerStyle:
    key: str
    label: str
    # ── frame treatment ──
    grade: str                     # eq=... fragment (always present)
    extra_vf: str = ""             # tint / grain / sharpen etc, appended
    flash_in: str = ""             # "white" | "black" | "" — flash-cut clip entrance
    bars: bool = True              # cinematic bars on 16:9
    # ── motion ──
    transitions: tuple = ("fadeblack", "wipeleft", "circleopen")
    pace: float = 0.25             # crossfade seconds (lower = harder cuts)
    # ── cards ──
    card_bg: tuple = (8, 8, 10)
    card_accent: tuple = (193, 18, 18)
    card_sub: str = "FULL VIDEO OUT NOW"
    # ── sound design (synth params) ──
    hit_freq: int = 65             # boom fundamental Hz
    hit_decay: float = 0.5         # boom tail seconds
    whoosh_band: tuple = (500, 5000)   # highpass, lowpass Hz
    riser_lp: int = 2400           # riser darkness (lower = darker)
    bed: Optional[str] = None      # (src_lavfi, af) mood bed under the whole cut
    bed_gain: float = 0.16


def _bed(src: str, af: str) -> tuple:
    return (src, af)


_DARK_DRONE = _bed("sine=frequency=55:duration={d},sine=frequency=58:duration={d}",
                   "lowpass=f=300,tremolo=f=0.4:d=0.6")
_TENSION_PAD = _bed("anoisesrc=color=brown:duration={d}:amplitude=0.5",
                    "lowpass=f=220,tremolo=f=0.8:d=0.4")
_WARM_PAD = _bed("sine=frequency=110:duration={d},sine=frequency=165:duration={d}",
                 "lowpass=f=500,tremolo=f=0.25:d=0.3")
_BRIGHT_PAD = _bed("sine=frequency=220:duration={d},sine=frequency=277:duration={d}",
                   "lowpass=f=900,tremolo=f=0.3:d=0.25")

# Reusable frame-effect fragments (all static, NVENC-safe filters).
_GRAIN = "noise=alls=6:allf=t"
_HEAVY_GRAIN = "noise=alls=11:allf=t"
# colorbalance's real options are per-channel shadows/midtones/highlights
# (rs/gs/bs, rm/gm/bm, rh/gh/bh) — "ms"/"hs" do NOT exist (caught by the
# 9:16 horror render smoke; ffmpeg exits "Option not found").
_COLD = "colorbalance=bs=0.18:bm=0.10:bh=0.04:rs=-0.06"
_WARM = "colorbalance=rs=0.12:rm=0.06:bs=-0.08"
_SEPIA = ("colorchannelmixer="
          ".393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:0")
_CRISP = "unsharp=5:5:0.8:5:5:0.0"
_DESAT_DARK = "eq=contrast=1.18:saturation=0.55:gamma=0.92"


def _mk(key, label, **kw) -> TrailerStyle:
    return TrailerStyle(key=key, label=label, **kw)


_PACKS: tuple[TrailerStyle, ...] = (
    # ── news family ──
    _mk("news", "News bulletin",
        grade="eq=contrast=1.12:saturation=1.30", extra_vf=_CRISP,
        transitions=("fadeblack", "wipeleft", "circleopen", "diag_soft_tl", "barn_door_open", "slideleft"),
        card_sub="FULL BULLETIN OUT NOW"),
    _mk("breaking_news", "Breaking news",
        grade="eq=contrast=1.20:saturation=1.35", extra_vf=_CRISP, flash_in="white",
        transitions=("fadewhite", "hlslice", "whip_left", "zoom_punch_in", "rectcrop", "hrslice"),
        pace=0.18, hit_freq=75, card_sub="BREAKING — WATCH NOW"),
    _mk("politics", "Politics",
        grade="eq=contrast=1.14:saturation=1.18", extra_vf=_CRISP,
        transitions=("fadeblack", "smoothleft", "barn_door_open", "diag_soft_br", "horzopen"),
        card_accent=(200, 120, 20), card_sub="THE FULL STORY"),
    _mk("crime", "Crime / investigation",
        grade=_DESAT_DARK, extra_vf=f"{_COLD},{_GRAIN}", flash_in="black",
        transitions=("fadeblack", "glitch_slices", "noise_dissolve", "melt_dissolve", "rectcrop", "vertclose"),
        pace=0.20, hit_freq=58, hit_decay=0.7, whoosh_band=(300, 3000),
        riser_lp=1500, bed=_TENSION_PAD, card_bg=(4, 4, 6),
        card_sub="THE INVESTIGATION"),
    _mk("horror", "Horror story",
        grade="eq=contrast=1.25:saturation=0.45:gamma=0.85",
        extra_vf=f"{_COLD},{_HEAVY_GRAIN}", flash_in="black",
        transitions=("fadeblack", "melt_dissolve", "noise_dissolve", "ripple_wave", "dissolve", "hblur"),
        pace=0.35, hit_freq=45, hit_decay=1.1, whoosh_band=(150, 1800),
        riser_lp=900, bed=_DARK_DRONE, bed_gain=0.22,
        card_bg=(2, 2, 3), card_accent=(120, 8, 8), card_sub="DARE TO WATCH"),
    _mk("thriller", "Thriller / suspense",
        grade="eq=contrast=1.20:saturation=0.75:gamma=0.9",
        extra_vf=f"{_COLD},{_GRAIN}", flash_in="black",
        transitions=("fadeblack", "glitch_slices", "iris_diamond", "distance", "circleclose"),
        pace=0.22, hit_freq=52, hit_decay=0.8, riser_lp=1200,
        bed=_TENSION_PAD, card_bg=(3, 3, 5), card_sub="EVERY SECOND COUNTS"),
    _mk("action", "Action / high energy",
        grade="eq=contrast=1.22:saturation=1.45", extra_vf=_CRISP, flash_in="white",
        transitions=("fadewhite", "whip_left", "whip_right", "zoom_punch_in", "hlwind", "hrslice"),
        pace=0.15, hit_freq=80, whoosh_band=(800, 8000),
        card_sub="FULL ACTION OUT NOW"),
    _mk("sports", "Sports",
        grade="eq=contrast=1.18:saturation=1.42", extra_vf=_CRISP, flash_in="white",
        transitions=("fadewhite", "whip_right", "zoom_punch_in", "slideleft", "coverleft"),
        pace=0.18, hit_freq=78, card_accent=(20, 160, 60),
        card_sub="MATCH HIGHLIGHTS INSIDE"),
    _mk("cricket", "Cricket",
        grade="eq=contrast=1.18:saturation=1.40", extra_vf=_CRISP, flash_in="white",
        transitions=("fadewhite", "whip_left", "zoom_punch_out", "zoomin", "coverright"),
        pace=0.18, hit_freq=78, card_accent=(20, 120, 200),
        card_sub="FULL INNINGS OUT NOW"),
    _mk("tech", "Technology",
        grade="eq=contrast=1.15:saturation=1.15", extra_vf=f"{_COLD},{_CRISP}",
        transitions=("pixel_dissolve", "glitch_slices", "blinds_v", "hlslice", "fade"),
        pace=0.22, whoosh_band=(1000, 9000), card_accent=(0, 170, 255),
        card_sub="THE FUTURE, EXPLAINED"),
    _mk("business", "Business",
        grade="eq=contrast=1.12:saturation=1.10", extra_vf=_CRISP,
        transitions=("fade", "blinds_h", "diag_soft_tl", "smoothright", "coverup"),
        card_accent=(20, 90, 200), card_sub="THE FULL ANALYSIS"),
    _mk("finance", "Finance / markets",
        grade="eq=contrast=1.16:saturation=1.12", extra_vf=_CRISP,
        transitions=("fade", "blinds_v", "checkerboard", "vuslice", "smoothup"),
        pace=0.2, card_accent=(20, 160, 60), card_sub="NUMBERS DON'T LIE"),
    _mk("health", "Health / medical",
        grade="eq=contrast=1.08:saturation=1.12:brightness=0.02",
        transitions=("fade", "smoothleft", "circleopen", "horzopen"),
        pace=0.3, card_accent=(0, 160, 140), card_sub="YOUR HEALTH MATTERS"),
    _mk("education", "Education / explainer",
        grade="eq=contrast=1.10:saturation=1.15",
        transitions=("fade", "blinds_h", "quad_flip", "wiperight", "vertopen"),
        pace=0.3, card_accent=(240, 180, 20), card_sub="LEARN THE FULL STORY"),
    _mk("devotional", "Devotional / spiritual",
        grade="eq=contrast=1.08:saturation=1.25:brightness=0.03", extra_vf=_WARM,
        transitions=("fade", "clock_sweep", "diag_soft_tl", "dissolve", "circleopen"),
        pace=0.45, hit_freq=110, hit_decay=1.4, whoosh_band=(300, 2500),
        bed=_WARM_PAD, bed_gain=0.2, card_bg=(20, 10, 4),
        card_accent=(230, 150, 20), card_sub="A JOURNEY OF FAITH"),
    _mk("mythology", "Mythology / epics",
        grade="eq=contrast=1.15:saturation=1.20:gamma=0.95", extra_vf=_WARM,
        transitions=("fadeblack", "clock_sweep", "ripple_wave", "dissolve", "radial"),
        pace=0.4, hit_freq=60, hit_decay=1.0, bed=_WARM_PAD,
        card_bg=(15, 8, 3), card_accent=(212, 175, 55), card_sub="THE LEGEND RETURNS"),
    _mk("history", "History",
        grade="eq=contrast=1.10:saturation=0.85", extra_vf=f"{_SEPIA},{_GRAIN}",
        transitions=("fadeblack", "clock_sweep", "melt_dissolve", "dissolve", "fadegrays"),
        pace=0.4, hit_freq=58, bed=_WARM_PAD, bed_gain=0.12,
        card_bg=(12, 9, 6), card_accent=(180, 140, 80), card_sub="HISTORY REMEMBERS"),
    _mk("documentary", "Documentary",
        grade="eq=contrast=1.10:saturation=0.95", extra_vf=_GRAIN,
        transitions=("fade", "dissolve", "fadeblack", "smoothright"),
        pace=0.45, hit_freq=60, card_sub="THE UNTOLD STORY"),
    _mk("travel", "Travel / places",
        grade="eq=contrast=1.12:saturation=1.40:brightness=0.02",
        transitions=("smoothleft", "ripple_wave", "diag_soft_br", "circleopen", "zoomin"),
        pace=0.3, whoosh_band=(700, 7000), card_accent=(0, 170, 200),
        card_sub="PACK YOUR BAGS"),
    _mk("food", "Food / cooking",
        grade="eq=contrast=1.12:saturation=1.45", extra_vf=_WARM,
        transitions=("fade", "circleopen", "smoothdown", "wipetl"),
        pace=0.3, hit_freq=90, card_accent=(230, 90, 20), card_sub="TASTE THE FULL RECIPE"),
    _mk("comedy", "Comedy / fun",
        grade="eq=contrast=1.12:saturation=1.40:brightness=0.03",
        transitions=("circlecrop", "zoom_punch_out", "quad_flip", "slideup", "zoomin"),
        pace=0.2, hit_freq=200, hit_decay=0.25, whoosh_band=(900, 9000),
        card_accent=(255, 200, 0), card_sub="LAUGHS GUARANTEED"),
    _mk("romance", "Romance",
        grade="eq=contrast=1.06:saturation=1.20:brightness=0.04", extra_vf=_WARM,
        transitions=("dissolve", "fade", "smoothup", "circleopen"),
        pace=0.5, hit_freq=100, hit_decay=1.2, bed=_WARM_PAD,
        card_bg=(20, 6, 10), card_accent=(220, 60, 110), card_sub="A LOVE STORY"),
    _mk("drama", "Drama / emotional",
        grade="eq=contrast=1.12:saturation=1.05:gamma=0.96", extra_vf=_GRAIN,
        transitions=("fadeblack", "dissolve", "fade", "distance"),
        pace=0.45, hit_freq=55, hit_decay=1.0, bed=_TENSION_PAD, bed_gain=0.1,
        card_sub="EVERY EMOTION, FELT"),
    _mk("music", "Music / performance",
        grade="eq=contrast=1.16:saturation=1.35", flash_in="white",
        transitions=("fadewhite", "ripple_wave", "zoom_punch_in", "stretch_h", "radial"),
        pace=0.18, hit_freq=85, whoosh_band=(800, 9000),
        card_accent=(180, 60, 220), card_sub="TURN IT UP"),
    _mk("festival", "Festival / celebration",
        grade="eq=contrast=1.12:saturation=1.50:brightness=0.03", extra_vf=_WARM,
        transitions=("fadewhite", "clock_sweep", "checkerboard", "circleopen", "zoomin"),
        pace=0.25, hit_freq=95, bed=_BRIGHT_PAD, bed_gain=0.12,
        card_bg=(22, 8, 2), card_accent=(255, 160, 0), card_sub="CELEBRATE WITH US"),
    _mk("motivational", "Motivational",
        grade="eq=contrast=1.16:saturation=1.20", extra_vf=_CRISP,
        transitions=("fadeblack", "smoothup", "zoomin", "fade"),
        pace=0.35, hit_freq=70, riser_lp=3000, bed=_BRIGHT_PAD, bed_gain=0.1,
        card_accent=(240, 180, 20), card_sub="YOUR MOMENT STARTS NOW"),
    _mk("celebrity", "Celebrity / entertainment",
        grade="eq=contrast=1.14:saturation=1.35", flash_in="white",
        transitions=("fadewhite", "stretch_h", "iris_diamond", "slideleft", "circlecrop"),
        pace=0.2, hit_freq=85, card_accent=(220, 40, 130), card_sub="THE INSIDE SCOOP"),
    _mk("movie_review", "Movie review / cinema",
        grade="eq=contrast=1.16:saturation=1.25:gamma=0.95", extra_vf=_GRAIN,
        transitions=("fadeblack", "iris_diamond", "melt_dissolve", "circleclose", "dissolve"),
        pace=0.3, hit_freq=60, bed=_TENSION_PAD, bed_gain=0.1,
        card_bg=(6, 6, 8), card_accent=(230, 180, 40), card_sub="VERDICT INSIDE"),
    _mk("gaming", "Gaming",
        grade="eq=contrast=1.20:saturation=1.45", extra_vf=_CRISP, flash_in="white",
        transitions=("glitch_slices", "pixel_dissolve", "whip_right", "zoomin", "hlslice"),
        pace=0.16, hit_freq=90, whoosh_band=(900, 10000),
        card_bg=(5, 5, 12), card_accent=(90, 220, 60), card_sub="GAME ON"),
    _mk("kids", "Kids / family",
        grade="eq=contrast=1.10:saturation=1.45:brightness=0.05",
        transitions=("circlecrop", "quad_flip", "checkerboard", "slideup", "circleopen"),
        pace=0.3, hit_freq=240, hit_decay=0.2, whoosh_band=(1000, 9000),
        bars=False, card_bg=(10, 16, 30), card_accent=(80, 190, 255),
        card_sub="FUN FOR EVERYONE"),
    _mk("weather", "Weather / disaster",
        grade="eq=contrast=1.16:saturation=1.10", extra_vf=_COLD,
        transitions=("fadeblack", "ripple_wave", "stretch_v", "vdwind", "wipedown"),
        pace=0.22, hit_freq=55, hit_decay=0.8, whoosh_band=(200, 2500),
        riser_lp=1300, bed=_TENSION_PAD, bed_gain=0.12,
        card_accent=(60, 140, 230), card_sub="STAY ALERT — FULL REPORT"),
)

STYLES: dict[str, TrailerStyle] = {p.key: p for p in _PACKS}
DEFAULT_STYLE = "news"


# ── Story-category taxonomy (spec §4): ~90 subcategories → packs ─────
# The spec's full editorial taxonomy resolves onto the 31 packs, so a
# classifier (or the API) may speak in precise editorial terms
# (MURDER_HOMICIDE, HURRICANE_CYCLONE, STOCK_MARKET_CRASH…) and always
# land on the right look + sound. Aliases are case-insensitive.
SUBCATEGORY_ALIASES: dict[str, str] = {
    # BREAKING_URGENT
    "breaking_news": "breaking_news", "developing_story": "breaking_news",
    "live_coverage": "breaking_news", "alert_warning": "breaking_news",
    "exclusive_scoop": "breaking_news",
    # CRIME_JUSTICE
    "murder_homicide": "crime", "theft_robbery": "crime",
    "assault_violence": "crime", "fraud_scam": "crime",
    "cybercrime": "tech", "drug_bust": "crime", "court_trial": "crime",
    "prison_jail": "crime", "missing_person": "crime",
    "organized_crime": "crime",
    # POLITICS_GOVERNMENT
    "election_voting": "politics", "policy_legislation": "politics",
    "protest_rally": "politics", "scandal_corruption": "politics",
    "international_summit": "politics", "press_conference": "politics",
    "campaign_trail": "politics",
    # FINANCE_BUSINESS
    "stock_market_crash": "finance", "bull_market_rally": "finance",
    "corporate_earnings": "business", "bankruptcy_collapse": "finance",
    "crypto_blockchain": "tech", "real_estate": "business",
    "startup_funding": "business", "unemployment_jobs": "business",
    "trade_tariffs": "finance",
    # GLOBAL_AFFAIRS
    "war_conflict": "thriller", "diplomacy_treaty": "politics",
    "refugee_crisis": "documentary", "natural_disaster_overseas": "weather",
    "terrorism": "thriller", "border_migration": "documentary",
    "human_rights": "documentary",
    # WEATHER_NATURAL_DISASTERS
    "hurricane_cyclone": "weather", "earthquake": "weather",
    "flood": "weather", "wildfire": "weather", "tornado": "weather",
    "heatwave": "weather", "blizzard": "weather", "tsunami": "weather",
    "drought": "weather", "space_weather": "tech",
    # HEALTH_MEDICAL
    "pandemic_outbreak": "health", "medical_breakthrough": "health",
    "hospital_er": "health", "mental_health": "health",
    "drug_pharma": "health", "fitness_wellness": "health",
    "food_safety": "food", "medical_research": "health",
    # ENTERTAINMENT_CULTURE
    "celebrity_news": "celebrity", "movie_tv_premiere": "movie_review",
    "music_album_drop": "music", "award_shows": "celebrity",
    "art_exhibition": "documentary", "book_literature": "history",
    "fashion_week": "celebrity", "viral_social_media": "comedy",
    # SPORTS
    "match_highlights": "sports", "transfer_trade": "sports",
    "injury_scandal": "sports", "championship_finals": "sports",
    "olympics": "sports", "esports": "gaming",
    "underdog_story": "motivational", "doping_cheating": "crime",
    # SCIENCE_TECHNOLOGY
    "space_nasa": "tech", "ai_robotics": "tech", "climate_science": "documentary",
    "archaeology_discovery": "history", "physics_quantum": "tech",
    "automotive_ev": "tech", "aviation": "tech",
    # EDUCATION_SOCIETY
    "school_university": "education", "exam_results": "education",
    "teacher_story": "education", "literacy_ngo": "education",
    "policy_change": "politics",
    # AGRICULTURE_RURAL
    "crop_harvest": "documentary", "farmer_crisis": "drama",
    "agri_tech": "tech", "food_supply_chain": "business",
    # INFRASTRUCTURE_INDUSTRY
    "construction_project": "business", "accident_collapse": "breaking_news",
    "energy_oil_gas": "business", "transport_traffic": "news",
    # INTERVIEW_PODCAST_TALKSHOW
    "political_interview": "politics", "celebrity_interview": "celebrity",
    "expert_academic": "education", "investigative_interview": "crime",
    "podcast_studio": "documentary", "panel_discussion": "news",
    "fireside_chat": "romance",
    # INVESTIGATIVE_DOCUMENTARY
    "deep_dive_expose": "thriller", "undercover": "crime",
    "data_investigation": "tech", "true_crime": "crime",
    # GENERIC_NEWS
    "local_news": "news", "community_event": "festival",
    "human_interest": "drama",
}


def resolve_category(key: str) -> str:
    """Pack key OR spec subcategory → the pack key. Unknown → default."""
    k = (key or "").strip().lower()
    if k in STYLES:
        return k
    return SUBCATEGORY_ALIASES.get(k, DEFAULT_STYLE)


def get_style(key: str) -> TrailerStyle:
    return STYLES[resolve_category(key)]


def style_catalog() -> list[dict]:
    """[{key, label}] for UI pickers."""
    return [{"key": p.key, "label": p.label} for p in _PACKS]


# ── Trailer STRUCTURES — the 10 spec assembly architectures ──────────
# A structure is HOW the teaser is built (ordering, cards, stingers),
# orthogonal to the style pack (look + sound). trailer.apply_structure
# turns one of these into a concrete assembly plan.

TRAILER_STRUCTURES: dict[str, dict] = {
    "classic": dict(
        label="Classic tease",
        used_for="Hook card → best moments in story order → end card",
        order="chrono", hook_first=True),
    "cold_open": dict(
        label="Cold open",
        used_for="The hardest moment SLAMS first, title card after",
        order="chrono", hook_first=False, cold_open=True),
    "crescendo": dict(
        label="Crescendo",
        used_for="Moments sorted soft→hard; the finale hits hardest",
        order="punch_asc", hook_first=True),
    "rapid_montage": dict(
        label="Rapid montage",
        used_for="No opening card, tighter cuts, pure adrenaline",
        order="punch_asc", hook_first=None, pace_mult=0.6),
    "quote_led": dict(
        label="Quote-led",
        used_for="The most quotable line opens, then the tease",
        order="chrono", hook_first=False, cold_open=True,
        cold_pick="overlay"),
    "flash_forward": dict(
        label="Flash-forward stinger",
        used_for="3 blink-fast glimpses, then the title, then the tease",
        order="chrono", hook_first=True, stinger=3),
    "countdown_led": dict(
        label="Countdown",
        used_for="Number cards count into the top moments",
        order="punch_asc", hook_first=True, number_cards=3),
    "bookend": dict(
        label="Bookend reprise",
        used_for="Opening beat returns for a blink before the end card",
        order="chrono", hook_first=True, reprise=True),
    "two_act": dict(
        label="Two acts",
        used_for="Mid-card break splits the tease into setup / payoff",
        order="chrono", hook_first=True, mid_card="అసలు ట్విస్ట్ ముందుంది"),
    "question_led": dict(
        label="Question-led",
        used_for="The hook question holds longer; answers stay unseen",
        order="chrono", hook_first=True, hook_hold=3.0),
}


def structure_catalog() -> list[dict]:
    return [{"key": k, "label": v["label"], "used_for": v["used_for"]}
            for k, v in TRAILER_STRUCTURES.items()]


# Visually-similar BUILTIN for every custom transition — the stitch
# retry ladder degrades custom→builtin (still a REAL visible move),
# never straight to an invisible plain fade.
CUSTOM_FALLBACK: dict[str, str] = {
    "blinds_h": "hlslice", "blinds_v": "vuslice",
    "checkerboard": "pixelize", "diag_soft_tl": "diagtl",
    "diag_soft_br": "diagbr", "barn_door_open": "horzopen",
    "iris_diamond": "circleopen", "pixel_dissolve": "pixelize",
    "noise_dissolve": "dissolve", "quad_flip": "rectcrop",
    "clock_sweep": "radial", "melt_dissolve": "dissolve",
    "ripple_wave": "smoothleft", "zoom_punch_in": "zoomin",
    "zoom_punch_out": "zoomin", "whip_left": "slideleft",
    "whip_right": "slideright", "stretch_h": "squeezeh",
    "stretch_v": "squeezev", "glitch_slices": "hlslice",
}


def builtin_equivalent(name: str) -> str:
    """A safe builtin with a similar motion for any transition id."""
    n = (name or "").strip().lower()
    if n in CUSTOM_FALLBACK:
        return CUSTOM_FALLBACK[n]
    return n if n in TRANSITION_CATALOG and n not in EXTRA_TRANSITIONS else "fadeblack"


def compose_pack(spec: dict, *, name: str = "My style") -> TrailerStyle:
    """USER-MIXED style pack: LOOK from one built-in pack, MOTION from
    another, SOUND from a third, CARDS from a fourth. Any missing /
    invalid component falls back to the news pack — a bad spec can
    never break a render."""
    def _p(key):
        return STYLES.get(str(spec.get(key, "") or "").strip().lower(),
                          STYLES[DEFAULT_STYLE])
    look, motion, sound, cards = (_p("look"), _p("motion"),
                                  _p("sound"), _p("cards"))
    return TrailerStyle(
        key="user_mix",
        label=(name or "My style")[:60],
        grade=look.grade, extra_vf=look.extra_vf,
        flash_in=look.flash_in, bars=look.bars,
        transitions=motion.transitions, pace=motion.pace,
        card_bg=cards.card_bg, card_accent=cards.card_accent,
        card_sub=cards.card_sub,
        hit_freq=sound.hit_freq, hit_decay=sound.hit_decay,
        whoosh_band=sound.whoosh_band, riser_lp=sound.riser_lp,
        bed=sound.bed, bed_gain=sound.bed_gain,
    )
