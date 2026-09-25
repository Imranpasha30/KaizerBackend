"""Layout library — 105 designed screen layouts + 15 PiP variants.

NOT wireframe filler (operator: "don't build blindly — it should
properly contain all types of things required for a video and be highly
designed"). Every layout is a complete broadcast screen design:

  * zones in canvas percentages — kind, position, size, z-order —
    covering everything a real video needs: primary media, secondary
    media, headline strap, caption area, ticker, logo slot, watermark
    slot, panels, CTAs;
  * every layout is VALIDATED (tests): at least one media zone, at
    least one text zone, a branding slot, all zones inside the canvas,
    and text zones never buried under media (z-order);
  * a designed preview renderer (dark stage, per-kind accent colors,
    labels, safe-margin guides) powers the Admin Features tab;
  * the geometry vocabulary (pct rects) matches CanvasLayout /
    layout_safety, so the Director and formulas can request layouts by
    id and the compose layer can honour them.

Families: news (16:9), talk/podcast (16:9), shorts (9:16), square
(1:1), specials (16:9), and the PiP variant set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# zone kinds → (accent RGB, label shown in previews)
ZONE_STYLE = {
    "bg_video":  ((45, 60, 95), "BG VIDEO"),
    "bg_image":  ((70, 60, 45), "BG IMAGE"),
    "video":     ((66, 135, 245), "VIDEO"),
    "video_b":   ((40, 180, 220), "VIDEO B"),
    "image":     ((235, 160, 40), "IMAGE"),
    "carousel":  ((235, 120, 40), "IMAGES"),
    "chart":     ((0, 190, 160), "CHART"),
    "map":       ((0, 170, 110), "MAP"),
    "headline":  ((220, 45, 45), "HEADLINE"),
    "subtext":   ((200, 90, 90), "SUBTEXT"),
    "caption":   ((90, 200, 90), "CAPTIONS"),
    "ticker":    ((240, 210, 50), "TICKER"),
    "logo":      ((212, 175, 55), "LOGO"),
    "watermark": ((150, 150, 160), "WM"),
    "panel":     ((160, 90, 220), "PANEL"),
    "cta":       ((250, 100, 180), "CTA"),
    "pip":       ((120, 200, 255), "PIP"),
    "strap":     ((220, 45, 45), "STRAP"),
    "waveform":  ((120, 220, 200), "WAVEFORM"),
    "score":     ((60, 200, 120), "SCORE"),
}

MEDIA_KINDS = {"video", "video_b", "image", "carousel", "chart", "map",
               "waveform", "pip", "bg_video", "bg_image"}
TEXT_KINDS = {"headline", "subtext", "caption", "ticker", "panel", "cta",
              "strap", "score"}
BRAND_KINDS = {"logo", "watermark"}


@dataclass(frozen=True)
class Zone:
    kind: str
    x: float          # pct of canvas width  (0-100)
    y: float          # pct of canvas height (0-100)
    w: float
    h: float
    z: int = 0        # draw order: higher = on top
    label: str = ""   # optional custom preview label


@dataclass(frozen=True)
class ScreenLayout:
    key: str
    label: str
    used_for: str
    aspect: str                    # "16:9" | "9:16" | "1:1"
    family: str
    zones: tuple = ()


def Z(kind, x, y, w, h, z=0, label=""):
    return Zone(kind, float(x), float(y), float(w), float(h), z, label)


# Standard furniture (16:9): logo TR, watermark BL, ticker bottom strip,
# headline strap above the ticker. Shared so the whole library is
# consistent — a broadcast identity, not 105 random screens.
_L16 = [Z("logo", 90.5, 3.5, 6.5, 10, 9), Z("watermark", 2, 90, 10, 6, 9)]
_TICK16 = Z("ticker", 0, 94.5, 100, 5.5, 8)
_STRAP16 = Z("strap", 0, 84.5, 100, 10, 8)
# 9:16 furniture: logo top-center-right, captions center-low (thumb-safe).
_L9 = [Z("logo", 78, 2.5, 16, 6, 9), Z("watermark", 4, 92.5, 18, 4, 9)]
_CAP9 = Z("caption", 8, 62, 84, 14, 8)


def _mk(key, label, used_for, aspect, family, *zones) -> ScreenLayout:
    return ScreenLayout(key, label, used_for, aspect, family, tuple(zones))


_LAYOUTS: list[ScreenLayout] = []


def _add(*a):
    _LAYOUTS.append(_mk(*a))


# ═══ FAMILY 1 — NEWS 16:9 (20) ═══════════════════════════════════════
_N = "news"
_add("news_bulletin_right", "Bulletin — image rail right",
     "The classic Kaizer bulletin: big video + image rail", "16:9", _N,
     Z("video", 1.5, 4.5, 64, 74), Z("carousel", 67.5, 4.5, 31, 74),
     _STRAP16, _TICK16, *_L16)
_add("news_bulletin_left", "Bulletin — image rail left",
     "Mirrored bulletin for variety across stories", "16:9", _N,
     Z("carousel", 1.5, 4.5, 31, 74), Z("video", 34.5, 4.5, 64, 74),
     _STRAP16, _TICK16, *_L16)
_add("news_fullscreen_anchor", "Full-screen anchor",
     "Anchor-to-camera, no side furniture", "16:9", _N,
     Z("video", 0, 0, 100, 84.5), _STRAP16, _TICK16, *_L16)
_add("news_lbar_right", "L-bar — content right",
     "Sponsor/aux rail on an L; video keeps playing", "16:9", _N,
     Z("video", 0, 0, 74, 74), Z("panel", 75.5, 2, 23, 70),
     Z("panel", 0, 75.5, 74, 8, 0, "AUX STRIP"), _STRAP16, _TICK16, *_L16)
_add("news_lbar_left", "L-bar — content left",
     "Mirrored L-bar", "16:9", _N,
     Z("panel", 1.5, 2, 23, 70), Z("video", 26, 0, 74, 74),
     Z("panel", 26, 75.5, 74, 8, 0, "AUX STRIP"), _STRAP16, _TICK16, *_L16)
_add("news_ots_right", "Over-the-shoulder right",
     "Anchor + story graphic box over the shoulder", "16:9", _N,
     Z("video", 0, 0, 100, 84.5), Z("image", 62, 8, 32, 40, 2, "OTS BOX"),
     _STRAP16, _TICK16, *_L16)
_add("news_ots_left", "Over-the-shoulder left",
     "Mirrored OTS", "16:9", _N,
     Z("video", 0, 0, 100, 84.5), Z("image", 6, 8, 32, 40, 2, "OTS BOX"),
     _STRAP16, _TICK16, *_L16)
_add("news_double_box", "Double box (anchor + guest)",
     "Two-way interview split", "16:9", _N,
     Z("video", 2, 8, 47, 62), Z("video_b", 51, 8, 47, 62),
     Z("subtext", 2, 71, 96, 8, 2, "NAME TAGS"), _STRAP16, _TICK16, *_L16)
_add("news_triple_box", "Triple box (panel)",
     "Anchor + two guests", "16:9", _N,
     Z("video", 2, 10, 32, 54), Z("video_b", 34.5, 10, 32, 54),
     Z("video_b", 67, 10, 31, 54, 0, "GUEST 2"),
     Z("subtext", 2, 66, 96, 8, 2, "NAME TAGS"), _STRAP16, _TICK16, *_L16)
_add("news_big_graphic", "Big graphic + presenter inset",
     "Data story: chart leads, presenter small", "16:9", _N,
     Z("chart", 1.5, 4.5, 70, 78), Z("pip", 74, 52, 24, 30),
     _STRAP16, _TICK16, *_L16)
_add("news_data_dashboard", "Data dashboard",
     "Chart + numbers panel + presenter strip", "16:9", _N,
     Z("chart", 1.5, 4.5, 55, 56), Z("panel", 58.5, 4.5, 40, 56, 0, "NUMBERS"),
     Z("video", 1.5, 62.5, 97, 20, 0, "PRESENTER STRIP"),
     _STRAP16, _TICK16, *_L16)
_add("news_map_focus", "Map focus",
     "Where it happened: map leads + reporter inset", "16:9", _N,
     Z("map", 0, 0, 100, 84.5), Z("pip", 72, 8, 24, 30),
     _STRAP16, _TICK16, *_L16)
_add("news_breaking_full", "Breaking takeover",
     "Red-alert breaking look: video + double strap", "16:9", _N,
     Z("video", 0, 0, 100, 74), Z("headline", 0, 74.5, 100, 10, 8, "BREAKING BANNER"),
     _STRAP16, _TICK16, *_L16)
_add("news_ticker_stack", "Double ticker stack",
     "Election/results nights: two info belts", "16:9", _N,
     Z("video", 0, 0, 100, 78), Z("ticker", 0, 78.5, 100, 5, 8, "RESULTS BELT"),
     _STRAP16, _TICK16, *_L16)
_add("news_magazine", "Magazine hero",
     "Feature stories: hero image + side headlines", "16:9", _N,
     Z("image", 1.5, 4.5, 60, 78), Z("panel", 63.5, 4.5, 35, 50, 0, "HEADLINES"),
     Z("cta", 63.5, 57, 35, 12), Z("subtext", 63.5, 71, 35, 11),
     _TICK16, *_L16)
_add("news_quote_focus", "Quote focus",
     "Statement stories: quote card leads, speaker inset", "16:9", _N,
     Z("panel", 8, 10, 60, 56, 0, "QUOTE CARD"), Z("pip", 71, 14, 24, 32),
     Z("subtext", 8, 68, 60, 10, 2, "ATTRIBUTION"), _STRAP16, _TICK16, *_L16)
_add("news_split_50", "Split 50/50",
     "Two developing stories at once", "16:9", _N,
     Z("video", 0, 0, 49.6, 84.5), Z("video_b", 50.4, 0, 49.6, 84.5),
     _STRAP16, _TICK16, *_L16)
_add("news_split_60_40", "Split 60/40",
     "Lead story + secondary visual", "16:9", _N,
     Z("video", 0, 0, 60, 84.5), Z("image", 60.8, 0, 39.2, 84.5),
     _STRAP16, _TICK16, *_L16)
_add("news_split_70_30", "Split 70/30",
     "Video + tall info rail", "16:9", _N,
     Z("video", 0, 0, 70, 84.5), Z("panel", 70.8, 0, 29.2, 84.5, 0, "INFO RAIL"),
     _STRAP16, _TICK16, *_L16)
_add("news_courtroom", "Courtroom / hearing",
     "Sketch/photo zone + case facts (cameras banned)", "16:9", _N,
     Z("image", 1.5, 4.5, 58, 78, 0, "SKETCH / PHOTO"),
     Z("panel", 61.5, 4.5, 37, 52, 0, "CASE FACTS"),
     Z("subtext", 61.5, 59, 37, 12, 0, "NEXT HEARING"),
     _STRAP16, _TICK16, *_L16)

# ═══ FAMILY 2 — TALK / PODCAST 16:9 (14) ═════════════════════════════
_P = "podcast"
_add("pod_grid_2", "Podcast 2-up", "Host + guest, equal", "16:9", _P,
     Z("video", 1.5, 6, 48, 76), Z("video_b", 50.5, 6, 48, 76),
     Z("subtext", 1.5, 84, 97, 7, 2, "NAME TAGS"), *_L16)
_add("pod_grid_3", "Podcast 3-up", "Host + 2 guests", "16:9", _P,
     Z("video", 1.5, 8, 32, 60), Z("video_b", 34.5, 8, 32, 60),
     Z("video_b", 67, 8, 31.5, 60, 0, "GUEST 2"),
     Z("caption", 10, 72, 80, 14), *_L16)
_add("pod_grid_4", "Podcast 2×2", "Four-person panel", "16:9", _P,
     Z("video", 1.5, 3, 48, 45), Z("video_b", 50.5, 3, 48, 45),
     Z("video_b", 1.5, 49.5, 48, 45, 0, "SPK 3"),
     Z("video_b", 50.5, 49.5, 48, 45, 0, "SPK 4"),
     Z("caption", 25, 95, 50, 4.5, 2), *_L16)
_add("pod_host_focus", "Host focus + guest strip",
     "One leads, others in a rail", "16:9", _P,
     Z("video", 1.5, 4, 70, 80), Z("video_b", 73, 4, 25.5, 25, 0, "GUEST 1"),
     Z("video_b", 73, 31, 25.5, 25, 0, "GUEST 2"),
     Z("video_b", 73, 58, 25.5, 25, 0, "GUEST 3"),
     Z("caption", 1.5, 86, 70, 9, 2), *_L16)
_add("pod_speaker_spotlight", "Speaker spotlight",
     "Active speaker big (auto-switch), thumbnails below", "16:9", _P,
     Z("video", 10, 3, 80, 66), Z("video_b", 22, 71, 13, 18, 0, "SPK"),
     Z("video_b", 37, 71, 13, 18, 0, "SPK"), Z("video_b", 52, 71, 13, 18, 0, "SPK"),
     Z("video_b", 67, 71, 13, 18, 0, "SPK"), Z("caption", 10, 90, 80, 8), *_L16)
_add("pod_waveform_audio", "Audio-only waveform",
     "No camera: art + live waveform + captions", "16:9", _P,
     Z("image", 34, 8, 32, 40, 0, "COVER ART"), Z("waveform", 15, 52, 70, 18),
     Z("caption", 10, 74, 80, 14), *_L16)
_add("pod_audiogram", "Audiogram card",
     "Clip promo: quote + waveform + big title", "16:9", _P,
     Z("headline", 8, 12, 55, 22), Z("panel", 8, 38, 55, 34, 0, "QUOTE"),
     Z("image", 68, 12, 26, 46, 0, "GUEST PHOTO"), Z("waveform", 8, 76, 86, 12),
     *_L16)
_add("pod_interview_ots", "Interview OTS pair",
     "Film-style alternating over-shoulder", "16:9", _P,
     Z("video", 0, 0, 100, 100), Z("pip", 70, 8, 26, 32, 2, "REVERSE ANGLE"),
     Z("caption", 15, 84, 70, 10, 3), *_L16)
_add("pod_roundtable_5", "Roundtable 5",
     "Big host center + 4 corners", "16:9", _P,
     Z("video", 30, 26, 40, 48), Z("video_b", 1.5, 3, 26, 38, 0, "SPK 1"),
     Z("video_b", 72.5, 3, 26, 38, 0, "SPK 2"),
     Z("video_b", 1.5, 57, 26, 38, 0, "SPK 3"),
     Z("video_b", 72.5, 57, 26, 38, 0, "SPK 4"),
     Z("subtext", 30, 76, 40, 6, 2, "NAME TAG"), *_L16)
_add("pod_screen_share", "Host + screen share",
     "Demos, chart walkthroughs, teaching", "16:9", _P,
     Z("chart", 1.5, 4, 68, 80, 0, "SCREEN"), Z("video", 71.5, 52, 27, 32),
     Z("caption", 1.5, 86, 68, 9), *_L16)
_add("pod_reaction_stack", "Reaction stack",
     "Clip on top, reactor below", "16:9", _P,
     Z("video_b", 15, 3, 70, 46, 0, "CLIP"), Z("video", 15, 51, 70, 42, 0, "REACTOR"),
     Z("caption", 15, 94, 70, 5, 2), *_L16)
_add("pod_chapters_rail", "Chapters rail",
     "Long episodes: content + chapter list", "16:9", _P,
     Z("video", 1.5, 4, 72, 88), Z("panel", 75, 4, 23.5, 62, 0, "CHAPTERS"),
     Z("cta", 75, 68, 23.5, 12, 0, "SUBSCRIBE"), *_L16)
_add("pod_quote_reel", "Quote reel",
     "Best-of lines: speaker + giant pull-quote", "16:9", _P,
     Z("video", 1.5, 4, 44, 88), Z("panel", 48, 14, 50, 48, 0, "PULL QUOTE"),
     Z("subtext", 48, 64, 50, 10, 0, "WHO SAID IT"), *_L16)
_add("pod_guest_intro", "Guest intro card",
     "Lower-half bio card while the guest talks", "16:9", _P,
     Z("video", 0, 0, 100, 100), Z("panel", 6, 62, 55, 26, 2, "BIO CARD"),
     Z("caption", 6, 90, 55, 8, 2), *_L16)

# ═══ FAMILY 3 — SHORTS 9:16 (30) ═════════════════════════════════════
_S = "shorts"
_add("short_torn_card", "Torn card", "Video / headline / image bands",
     "9:16", _S, Z("video", 0, 0, 100, 44), Z("headline", 4, 45, 92, 14),
     Z("image", 0, 60, 100, 30), _CAP9, *_L9)
_add("short_clean_card", "Clean card", "Minimal: video + one line",
     "9:16", _S, Z("video", 0, 0, 100, 58), Z("headline", 6, 60, 88, 12),
     _CAP9, *_L9)
_add("short_split_frame", "Split frame", "Video top / hero image bottom",
     "9:16", _S, Z("video", 0, 0, 100, 49), Z("image", 0, 51, 100, 49),
     Z("headline", 6, 44, 88, 12, 3), *_L9)
_add("short_follow_bar", "Follow bar", "Video + headline + follow CTA",
     "9:16", _S, Z("video", 0, 0, 100, 56), Z("headline", 6, 58, 88, 12),
     Z("cta", 20, 84, 60, 8), _CAP9, *_L9)
_add("short_dual_video", "Dual video", "Two cams stacked + title",
     "9:16", _S, Z("video", 0, 0, 100, 44), Z("headline", 4, 45, 92, 10),
     Z("video_b", 0, 56, 100, 44), *_L9)
_add("short_karaoke_center", "Karaoke captions",
     "Full video + word-by-word center captions", "9:16", _S,
     Z("video", 0, 0, 100, 100), Z("caption", 8, 58, 84, 18, 3, "KARAOKE"),
     *_L9)
_add("short_headline_top", "Headline top",
     "Hook line first, video under it", "9:16", _S,
     Z("headline", 4, 4, 92, 16), Z("video", 0, 22, 100, 66), _CAP9, *_L9)
_add("short_list_bottom", "Video + list",
     "Video top, animated list below", "9:16", _S,
     Z("video", 0, 0, 100, 52), Z("panel", 6, 55, 88, 34, 0, "LIST REVEAL"),
     *_L9)
_add("short_quote_card", "Quote short",
     "Speaker + big quote card", "9:16", _S,
     Z("video", 0, 0, 100, 46), Z("panel", 6, 50, 88, 28, 0, "QUOTE"),
     Z("subtext", 6, 80, 88, 8), *_L9)
_add("short_poll", "Poll short",
     "Question + live poll bar", "9:16", _S,
     Z("video", 0, 0, 100, 55), Z("headline", 6, 57, 88, 12, 0, "QUESTION"),
     Z("panel", 6, 72, 88, 12, 0, "POLL BAR"), *_L9)
_add("short_tweet_react", "Post reaction",
     "Viral post card + reaction video", "9:16", _S,
     Z("panel", 5, 4, 90, 30, 0, "VIRAL POST"), Z("video", 0, 38, 100, 52),
     _CAP9, *_L9)
_add("short_before_after", "Before / after",
     "Comparison split with labels", "9:16", _S,
     Z("image", 0, 0, 100, 46, 0, "BEFORE"), Z("image", 0, 54, 100, 46, 0, "AFTER"),
     Z("headline", 20, 46.5, 60, 7, 3, "VS"), *_L9)
_add("short_countdown", "Countdown short",
     "Number cards between clips", "9:16", _S,
     Z("video", 0, 0, 100, 74), Z("headline", 25, 30, 50, 22, 3, "BIG NUMBER"),
     Z("subtext", 10, 78, 80, 8), *_L9)
_add("short_big_stat", "Big stat short",
     "One giant number sells the story", "9:16", _S,
     Z("video", 0, 0, 100, 100), Z("headline", 8, 24, 84, 24, 3, "GIANT STAT"),
     Z("subtext", 14, 50, 72, 8, 3), _CAP9, *_L9)
_add("short_checklist", "Checklist short",
     "Steps tick off over footage", "9:16", _S,
     Z("video", 0, 0, 100, 100), Z("panel", 8, 20, 84, 38, 2, "CHECKLIST"),
     _CAP9, *_L9)
_add("short_pip_comment", "PiP commentary",
     "Main clip + commentator bubble", "9:16", _S,
     Z("video", 0, 0, 100, 100), Z("pip", 62, 66, 34, 20, 3, "COMMENTATOR"),
     Z("caption", 8, 88, 84, 9, 3), *_L9)
_add("short_gameplay_cam", "Gameplay + facecam",
     "Screen top, player cam bottom", "9:16", _S,
     Z("video_b", 0, 0, 100, 62, 0, "GAMEPLAY"), Z("video", 0, 64, 100, 36, 0, "FACECAM"),
     Z("caption", 8, 55, 84, 8, 3), *_L9)
_add("short_product", "Product showcase",
     "Hero product + price + CTA", "9:16", _S,
     Z("image", 10, 6, 80, 48, 0, "PRODUCT"), Z("headline", 8, 57, 84, 10),
     Z("panel", 20, 70, 60, 9, 0, "PRICE"), Z("cta", 22, 82, 56, 8), *_L9)
_add("short_news_brief", "News brief vertical",
     "Mini bulletin: strap + ticker in 9:16", "9:16", _S,
     Z("video", 0, 0, 100, 72), Z("strap", 0, 73, 100, 10),
     Z("ticker", 0, 94, 100, 6), Z("caption", 8, 84, 84, 8), *_L9)
_add("short_meme_frame", "Meme frame",
     "White card, top/bottom text, clip center", "9:16", _S,
     Z("headline", 4, 3, 92, 12, 2, "TOP TEXT"), Z("video", 0, 18, 100, 62),
     Z("headline", 4, 83, 92, 12, 2, "BOTTOM TEXT"), *_L9)
_add("short_interview_stack", "Interview stack",
     "Q on card, A on camera", "9:16", _S,
     Z("panel", 5, 5, 90, 20, 0, "QUESTION"), Z("video", 0, 28, 100, 62),
     _CAP9, *_L9)
_add("short_bilingual", "Bilingual captions",
     "Two caption belts (Telugu + English)", "9:16", _S,
     Z("video", 0, 0, 100, 100), Z("caption", 8, 56, 84, 10, 3, "తెలుగు"),
     Z("caption", 8, 68, 84, 10, 3, "ENGLISH"), *_L9)
_add("short_cinema_sub", "Cinema crop",
     "Letterboxed film look + subtitle", "9:16", _S,
     Z("video", 0, 20, 100, 56), Z("caption", 10, 80, 80, 9),
     Z("headline", 8, 6, 84, 10), *_L9)
_add("short_photo_story", "Photo story",
     "Ken-Burns photos + narration captions", "9:16", _S,
     Z("carousel", 0, 0, 100, 78), Z("caption", 8, 80, 84, 12), *_L9)
_add("short_recipe", "Recipe steps",
     "Cooking: video + step chip + ingredients", "9:16", _S,
     Z("video", 0, 0, 100, 66), Z("panel", 6, 68, 56, 24, 0, "INGREDIENTS"),
     Z("headline", 66, 68, 28, 10, 0, "STEP 2/5"), *_L9)
_add("short_workout", "Workout timer",
     "Exercise + rep counter + timer ring", "9:16", _S,
     Z("video", 0, 0, 100, 74), Z("panel", 8, 76, 40, 14, 0, "TIMER"),
     Z("headline", 52, 76, 40, 14, 0, "REPS"), *_L9)
_add("short_quiz", "Quiz reveal",
     "Question, options, delayed answer", "9:16", _S,
     Z("video", 0, 0, 100, 40), Z("headline", 6, 42, 88, 12, 0, "QUESTION"),
     Z("panel", 10, 57, 80, 30, 0, "OPTIONS A-D"), *_L9)
_add("short_location_tour", "Location tour",
     "Place video + map inset + name chip", "9:16", _S,
     Z("video", 0, 0, 100, 100), Z("map", 62, 6, 34, 18, 3, "MAP"),
     Z("headline", 6, 80, 66, 9, 3, "PLACE NAME"), _CAP9, *_L9)
_add("short_testimonial", "Testimonial",
     "Customer speaks + star strip + brand", "9:16", _S,
     Z("video", 0, 0, 100, 70), Z("panel", 14, 73, 72, 9, 0, "★★★★★"),
     Z("subtext", 10, 84, 80, 7), *_L9)
_add("short_devotional", "Devotional short",
     "Warm ornament margins + deity image + aarti text", "9:16", _S,
     Z("image", 6, 8, 88, 56, 0, "DEITY / TEMPLE"),
     Z("caption", 10, 68, 80, 14, 0, "SLOKA / AARTI"),
     Z("cta", 24, 86, 52, 7, 0, "SUBSCRIBE 🙏"), *_L9)

# ═══ FAMILY 4 — SQUARE 1:1 (8) ═══════════════════════════════════════
_Q = "square"
_add("sq_bulletin", "Square bulletin", "Feed-native news card", "1:1", _Q,
     Z("video", 0, 0, 100, 70), Z("strap", 0, 71, 100, 12),
     Z("ticker", 0, 93.5, 100, 6.5), Z("logo", 86, 4, 10, 8, 9),
     Z("watermark", 4, 86, 14, 5, 9))
_add("sq_quote", "Square quote", "Statement graphic for feeds", "1:1", _Q,
     Z("panel", 8, 12, 84, 50, 0, "QUOTE"), Z("image", 8, 66, 22, 22, 0, "FACE"),
     Z("subtext", 34, 70, 58, 12), Z("logo", 86, 4, 10, 8, 9))
_add("sq_stat", "Square stat", "One number + context", "1:1", _Q,
     Z("headline", 12, 18, 76, 30, 0, "GIANT NUMBER"),
     Z("subtext", 16, 52, 68, 12), Z("image", 0, 68, 100, 32, 0, "CONTEXT STRIP"),
     Z("logo", 86, 4, 10, 8, 9))
_add("sq_carousel", "Square carousel", "Swipe-style photo story", "1:1", _Q,
     Z("carousel", 0, 0, 100, 82), Z("caption", 6, 84, 88, 10),
     Z("logo", 86, 4, 10, 8, 9))
_add("sq_split", "Square split", "Two visuals side by side", "1:1", _Q,
     Z("video", 0, 0, 49.5, 84), Z("image", 50.5, 0, 49.5, 84),
     Z("headline", 6, 86, 88, 10), Z("logo", 86, 4, 10, 8, 9))
_add("sq_podcast", "Square podcast clip", "Feed clip: speaker + captions",
     "1:1", _Q, Z("video", 0, 0, 100, 72), Z("caption", 8, 74, 84, 14),
     Z("waveform", 8, 90, 84, 6), Z("logo", 86, 4, 10, 8, 9))
_add("sq_announcement", "Square announcement", "Event/launch card", "1:1", _Q,
     Z("headline", 8, 10, 84, 20), Z("image", 20, 34, 60, 38),
     Z("panel", 14, 76, 72, 12, 0, "DATE · VENUE"), Z("logo", 86, 4, 10, 8, 9))
_add("sq_poll", "Square poll", "Feed poll with bar", "1:1", _Q,
     Z("headline", 8, 12, 84, 18, 0, "QUESTION"), Z("video", 0, 34, 100, 40),
     Z("panel", 10, 78, 80, 12, 0, "POLL BAR"), Z("logo", 86, 4, 10, 8, 9))

# ═══ FAMILY 5 — SPECIALS 16:9 (18) ═══════════════════════════════════
_X = "special"
_add("sp_trailer_frame", "Trailer frame", "Cinematic bars + centered title zone",
     "16:9", _X, Z("video", 0, 11, 100, 78), Z("headline", 20, 42, 60, 16, 3, "TITLE CARD"),
     Z("watermark", 4, 90, 10, 5, 9), Z("logo", 90.5, 3.5, 6.5, 10, 9))
_add("sp_documentary", "Documentary",
     "Full-bleed footage + lower context caption", "16:9", _X,
     Z("video", 0, 0, 100, 100), Z("caption", 8, 82, 60, 9, 2),
     Z("subtext", 8, 92, 40, 5, 2, "LOCATION · YEAR"), *_L16)
_add("sp_sports_board", "Sports scoreboard",
     "Match layout: play + score bug + stats rail", "16:9", _X,
     Z("video", 0, 0, 100, 84.5), Z("score", 2, 3, 34, 8, 3),
     Z("panel", 74, 4, 24, 44, 2, "STATS"), _TICK16, Z("logo", 90.5, 90, 6.5, 8, 9),
     Z("watermark", 2, 90, 10, 5, 9))
_add("sp_weather_board", "Weather dashboard",
     "City grid + radar map + advisory strap", "16:9", _X,
     Z("map", 1.5, 4.5, 58, 78, 0, "RADAR"), Z("panel", 61.5, 4.5, 37, 52, 0, "CITY GRID"),
     Z("panel", 61.5, 59, 37, 12, 0, "ADVISORY"), _STRAP16, _TICK16, *_L16)
_add("sp_market_board", "Market dashboard",
     "Indices rail + chart + anchor inset", "16:9", _X,
     Z("chart", 1.5, 4.5, 62, 66), Z("panel", 65.5, 4.5, 33, 66, 0, "INDICES"),
     Z("pip", 78, 54, 20, 26, 2, "ANCHOR"), _STRAP16, _TICK16, *_L16)
_add("sp_election_board", "Election results",
     "Seat tally + map + leads belt", "16:9", _X,
     Z("panel", 1.5, 4.5, 44, 60, 0, "SEAT TALLY"), Z("map", 47.5, 4.5, 51, 60),
     Z("ticker", 0, 66, 100, 6, 3, "LEADS BELT"), _STRAP16, _TICK16, *_L16)
_add("sp_live_blog", "Live blog feed",
     "Video + scrolling updates feed", "16:9", _X,
     Z("video", 1.5, 4.5, 62, 78), Z("panel", 65.5, 4.5, 33, 78, 0, "LIVE UPDATES"),
     _TICK16, *_L16)
_add("sp_timeline", "Timeline",
     "How it unfolded: media + horizontal timeline", "16:9", _X,
     Z("video", 1.5, 4.5, 97, 58), Z("panel", 1.5, 65, 97, 16, 0, "TIMELINE"),
     _STRAP16, _TICK16, *_L16)
_add("sp_versus", "Versus / comparison",
     "Head-to-head: two subjects + spec table", "16:9", _X,
     Z("image", 1.5, 4.5, 30, 56, 0, "SIDE A"), Z("image", 68.5, 4.5, 30, 56, 0, "SIDE B"),
     Z("panel", 33, 4.5, 34, 56, 0, "VS TABLE"), Z("headline", 33, 62, 34, 10, 2, "VERDICT"),
     _TICK16, *_L16)
_add("sp_top5_board", "Top-5 board",
     "Countdown list + preview window", "16:9", _X,
     Z("panel", 1.5, 4.5, 40, 78, 0, "TOP-5 LIST"), Z("video", 43.5, 4.5, 55, 78),
     _STRAP16, _TICK16, *_L16)
_add("sp_awards", "Awards / felicitation",
     "Stage wide + winner card + sponsors strip", "16:9", _X,
     Z("video", 0, 0, 100, 74), Z("panel", 6, 58, 40, 20, 2, "WINNER CARD"),
     Z("panel", 0, 75, 100, 8, 0, "SPONSORS"), _TICK16, *_L16)
_add("sp_devotional_frame", "Devotional frame",
     "Warm margins + program schedule rail", "16:9", _X,
     Z("video", 4, 6, 62, 76), Z("panel", 68, 6, 28, 60, 0, "SCHEDULE"),
     Z("caption", 4, 84, 62, 8, 0, "SLOKA"), _TICK16, *_L16)
_add("sp_kids_frame", "Kids bright frame",
     "Rounded playful margins + big captions", "16:9", _X,
     Z("video", 3, 5, 70, 76), Z("panel", 75, 5, 22, 40, 0, "CHARACTER"),
     Z("caption", 10, 84, 80, 11), Z("logo", 90.5, 88, 6.5, 9, 9),
     Z("watermark", 2, 90, 10, 5, 9))
_add("sp_gaming_hud", "Gaming HUD",
     "Gameplay + facecam + chat rail + stats", "16:9", _X,
     Z("video_b", 0, 0, 74, 84.5, 0, "GAMEPLAY"), Z("video", 75.5, 3, 23, 30, 0, "FACECAM"),
     Z("panel", 75.5, 35, 23, 48, 0, "CHAT"), _TICK16, *_L16)
_add("sp_classroom", "Explainer board",
     "Teacher + whiteboard + key points", "16:9", _X,
     Z("chart", 1.5, 4.5, 64, 78, 0, "BOARD"), Z("video", 67.5, 4.5, 31, 44),
     Z("panel", 67.5, 50, 31, 32, 0, "KEY POINTS"), _TICK16, *_L16)
_add("sp_tech_demo", "Tech demo",
     "Device close-up + spec panel + presenter", "16:9", _X,
     Z("video", 1.5, 4.5, 58, 78, 0, "DEVICE"), Z("panel", 61.5, 4.5, 37, 46, 0, "SPECS"),
     Z("pip", 74, 54, 24, 28, 0, "PRESENTER"), _STRAP16, _TICK16, *_L16)
_add("sp_health_board", "Health advisory",
     "Doctor + symptoms panel + helpline strap", "16:9", _X,
     Z("video", 1.5, 4.5, 58, 78, 0, "DOCTOR"), Z("panel", 61.5, 4.5, 37, 56, 0, "SYMPTOMS"),
     Z("panel", 61.5, 62.5, 37, 10, 0, "HELPLINE"), _STRAP16, _TICK16, *_L16)
_add("sp_festival", "Festival greeting",
     "Celebration footage + ornament title + wishes belt", "16:9", _X,
     Z("video", 0, 0, 100, 84.5), Z("headline", 22, 30, 56, 18, 3, "FESTIVAL TITLE"),
     Z("caption", 20, 52, 60, 8, 3, "WISHES"), _TICK16, *_L16)

# ═══ FAMILY 6 — PiP VARIANTS (15) ════════════════════════════════════
_PIP = "pip"
for key, label, used, px, py, pw, ph, extra in [
    ("pip_tr_small", "PiP top-right small", "Default reporter inset", 74, 6, 22, 26, None),
    ("pip_tl_small", "PiP top-left small", "When TR clashes with the bug", 4, 6, 22, 26, None),
    ("pip_br_small", "PiP bottom-right small", "Above the strap, right", 74, 52, 22, 26, None),
    ("pip_bl_small", "PiP bottom-left small", "Above the watermark, left", 4, 52, 22, 26, None),
    ("pip_tr_medium", "PiP top-right medium", "Two-way where the guest matters", 64, 6, 32, 38, None),
    ("pip_circle_tr", "Circle PiP top-right", "Podcast-style round inset", 76, 6, 20, 30, "CIRCLE"),
    ("pip_circle_bl", "Circle PiP bottom-left", "Round inset, low-left", 4, 50, 20, 30, "CIRCLE"),
    ("pip_framed", "Framed PiP", "White-border classic inset", 72, 8, 24, 30, "WHITE FRAME"),
    ("pip_floating", "Floating shadow PiP", "Soft-shadow modern inset", 70, 10, 26, 32, "SHADOW"),
    ("pip_side_8020", "Side-by-side 80/20", "Main + persistent side strip", 80.5, 0, 19.5, 84.5, "FULL-HEIGHT"),
    ("pip_lower_band", "Letterbox PiP band", "Inset inside a bottom band", 66, 66, 30, 18, "IN BAND"),
    ("pip_with_nametag", "PiP + name tag", "Inset with speaker tag under it", 72, 8, 24, 30, "NAME TAG"),
    ("pip_double", "Double PiP", "Two insets (reporter + expert)", 72, 6, 24, 26, "TWO INSETS"),
    ("pip_full_rail", "Full-height right rail", "Tall inset rail (vertical phone clip)", 76, 0, 24, 84.5, "9:16 CLIP"),
    ("pip_speaker_follow", "Speaker-follow PiP", "Auto-jumps to the active speaker", 72, 8, 24, 30, "AUTO"),
]:
    zones = [Z("video", 0, 0, 100, 84.5), Z("pip", px, py, pw, ph, 3,
               extra or "")]
    if key == "pip_double":
        zones.append(Z("pip", 72, 36, 24, 26, 3, "INSET 2"))
    if key == "pip_with_nametag":
        zones.append(Z("subtext", 72, 39, 24, 6, 3, "NAME"))
    _add(key, label, used, "16:9", _PIP, *zones, _STRAP16, _TICK16, *_L16)

# ═══ FAMILY 7 — STUDIO / BACKGROUND-DRIVEN 16:9 (15) ═════════════════
# Operator requirement: layouts must support background video, background
# image and image carousels as first-class citizens. The bg zone sits at
# z=-1 (drawn first); framed content floats on top — this is exactly how
# the real V4 studio-bg compose works (bg_video_path + framed tiles).
_B = "studio"
_add("studio_bg_classic", "Studio bg — classic bulletin",
     "Looping studio bg behind the framed video + image rail", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1), Z("video", 1.5, 4.5, 64, 74),
     Z("carousel", 67.5, 4.5, 31, 74), _STRAP16, _TICK16, *_L16)
_add("studio_bg_center", "Studio bg — centered card",
     "One floating video card on a live set", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1), Z("video", 15, 8, 70, 70),
     _STRAP16, _TICK16, *_L16)
_add("studio_bg_dual", "Studio bg — dual cards",
     "Two floating cards (debate) on a live set", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1), Z("video", 4, 10, 44, 62),
     Z("video_b", 52, 10, 44, 62), _STRAP16, _TICK16, *_L16)
_add("studio_greenroom", "Green-room presenter",
     "Presenter cutout over a virtual set + topic panel", "16:9", _B,
     Z("bg_image", 0, 0, 100, 100, -1, "VIRTUAL SET"),
     Z("video", 55, 12, 42, 72, 0, "PRESENTER"),
     Z("panel", 4, 12, 46, 56, 0, "TOPIC BOARD"), _STRAP16, _TICK16, *_L16)
_add("studio_bg_carousel", "Studio bg — carousel front",
     "Full-width image carousel floating on the set", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1), Z("carousel", 6, 8, 88, 66),
     Z("caption", 20, 76, 60, 7, 2), _TICK16, *_L16)
_add("bg_image_window", "Image bg — video window",
     "Static branded backdrop + video window", "16:9", _B,
     Z("bg_image", 0, 0, 100, 100, -1, "BACKDROP"),
     Z("video", 26, 8, 48, 62), Z("headline", 20, 72, 60, 10, 2),
     _TICK16, *_L16)
_add("bg_blur_echo", "Blur-echo background",
     "The video's own blurred echo fills the frame (vertical clip on 16:9)",
     "16:9", _B, Z("bg_video", 0, 0, 100, 100, -1, "BLURRED SELF"),
     Z("video", 32, 0, 36, 84.5, 0, "9:16 CLIP"), _STRAP16, _TICK16, *_L16)
_add("bg_gradient_card", "Gradient bg — headline card",
     "Brand gradient + big text + media chip", "16:9", _B,
     Z("bg_image", 0, 0, 100, 100, -1, "GRADIENT"),
     Z("headline", 8, 14, 52, 26), Z("subtext", 8, 44, 46, 12),
     Z("image", 62, 12, 34, 56), Z("cta", 8, 62, 30, 10), _TICK16, *_L16)
_add("virtual_set_desk", "Virtual desk",
     "Anchor desk strap + set bg + side screen", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1, "SET"),
     Z("video", 8, 8, 52, 66, 0, "ANCHOR"),
     Z("image", 64, 10, 32, 44, 0, "SIDE SCREEN"),
     Z("panel", 0, 74, 100, 10, 0, "DESK STRAP"), _TICK16, *_L16)
_add("bg_particles_title", "Particle bg — title open",
     "Show open: particles + title + episode chip", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1, "PARTICLES"),
     Z("headline", 18, 30, 64, 20), Z("subtext", 30, 54, 40, 8),
     Z("logo", 44, 12, 12, 14, 2), Z("watermark", 2, 90, 10, 5, 9),
     _TICK16)
_add("photo_wall", "Photo wall",
     "Collage backdrop of story images + main video", "16:9", _B,
     Z("bg_image", 0, 0, 100, 100, -1, "PHOTO COLLAGE"),
     Z("video", 22, 6, 56, 70), Z("caption", 26, 78, 48, 7, 2),
     _TICK16, *_L16)
_add("bg_pip_stack", "Bg + PiP stack",
     "Set bg + main + two stacked insets", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1), Z("video", 4, 6, 62, 74),
     Z("pip", 70, 6, 26, 34, 2, "INSET 1"), Z("pip", 70, 44, 26, 34, 2, "INSET 2"),
     _STRAP16, _TICK16, *_L16)
_add("bg_map_ambient", "Ambient map bg",
     "Soft map bg + story card + dateline", "16:9", _B,
     Z("bg_image", 0, 0, 100, 100, -1, "MAP WASH"),
     Z("video", 10, 8, 55, 64), Z("panel", 68, 8, 28, 46, 0, "STORY FACTS"),
     Z("subtext", 68, 56, 28, 8, 0, "DATELINE"), _STRAP16, _TICK16, *_L16)
_add("bg_texture_devotional", "Devotional texture bg",
     "Ornament texture + deity image + sloka belt", "16:9", _B,
     Z("bg_image", 0, 0, 100, 100, -1, "ORNAMENT"),
     Z("image", 30, 6, 40, 62, 0, "DEITY"),
     Z("caption", 18, 72, 64, 9, 2, "SLOKA"), _TICK16, *_L16)
_add("bg_city_timelapse", "City timelapse bg",
     "Business open: skyline bg + indices board", "16:9", _B,
     Z("bg_video", 0, 0, 100, 100, -1, "TIMELAPSE"),
     Z("panel", 8, 10, 50, 62, 0, "INDICES BOARD"),
     Z("video", 62, 14, 34, 44, 0, "ANCHOR"), _STRAP16, _TICK16, *_L16)

LAYOUTS: dict[str, ScreenLayout] = {l.key: l for l in _LAYOUTS}


def validate_layout(l: ScreenLayout) -> list[str]:
    """Design-rule violations ('' clean). The 'not built blindly' guard:
    every layout must be a COMPLETE screen for a real video."""
    errs = []
    kinds = {z.kind for z in l.zones}
    if not kinds & MEDIA_KINDS:
        errs.append("no media zone")
    if not kinds & TEXT_KINDS:
        errs.append("no text zone")
    if not kinds & BRAND_KINDS:
        errs.append("no branding (logo/watermark) slot")
    for z in l.zones:
        if z.kind not in ZONE_STYLE:
            errs.append(f"unknown zone kind {z.kind!r}")
        if not (0 <= z.x <= 100 and 0 <= z.y <= 100
                and 0 < z.w <= 100 and 0 < z.h <= 100
                and z.x + z.w <= 100.01 and z.y + z.h <= 100.01):
            errs.append(f"{z.kind} out of canvas ({z.x},{z.y},{z.w},{z.h})")
    # text readability: every text zone must sit at z >= any media zone
    # that covers its center (text is never buried under media)
    for t in l.zones:
        if t.kind not in TEXT_KINDS:
            continue
        cx, cy = t.x + t.w / 2, t.y + t.h / 2
        for m in l.zones:
            if m.kind in MEDIA_KINDS and m.x <= cx <= m.x + m.w \
                    and m.y <= cy <= m.y + m.h and m.z > t.z:
                errs.append(f"{t.kind} buried under {m.kind}")
    return errs


def layout_catalog() -> list[dict]:
    return [{"key": l.key, "label": l.label,
             "used_for": f"[{l.family} · {l.aspect}] {l.used_for}",
             "aspect": l.aspect, "family": l.family}
            for l in _LAYOUTS]


def render_layout_preview(key: str, out_path: str) -> Optional[str]:
    """Designed schematic: dark stage, per-kind accent fills, labels,
    safe-margin guides. One renderer for the whole library."""
    l = LAYOUTS.get(key)
    if not l:
        return None
    try:
        from PIL import Image, ImageDraw
        from pipeline_v4.overlays import _font
        if l.aspect == "9:16":
            W, H = 216, 384
        elif l.aspect == "1:1":
            W, H = 300, 300
        else:
            W, H = 480, 270
        img = Image.new("RGB", (W, H), (13, 14, 20))
        d = ImageDraw.Draw(img)
        # stage texture: subtle grid
        for gx in range(0, W, 24):
            d.line([(gx, 0), (gx, H)], fill=(19, 20, 28))
        for gy in range(0, H, 24):
            d.line([(0, gy), (W, gy)], fill=(19, 20, 28))
        f = _font(None, 12 if W < 400 else 13)
        for z in sorted(l.zones, key=lambda z: z.z):
            color, default_lbl = ZONE_STYLE[z.kind]
            x0, y0 = int(z.x / 100 * W), int(z.y / 100 * H)
            x1, y1 = int((z.x + z.w) / 100 * W), int((z.y + z.h) / 100 * H)
            fill = tuple(int(c * 0.28) for c in color)
            d.rounded_rectangle([x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)],
                                radius=4, fill=fill, outline=color, width=2)
            lbl = z.label or default_lbl
            if (x1 - x0) > 34 and (y1 - y0) > 15:
                d.text((x0 + 5, y0 + 3), lbl[:20], font=f,
                       fill=(235, 236, 240))
        # safe-margin guide
        d.rectangle([int(W * 0.02), int(H * 0.02),
                     int(W * 0.98), int(H * 0.98)],
                    outline=(70, 72, 84), width=1)
        img.save(out_path, "PNG")
        return out_path
    except Exception as exc:
        print(f"[v4/layout_library] preview {key} failed: {exc}", flush=True)
        return None


# ── RENDERABLE subset — layouts the V4 bulletin composer can honour
# TODAY via CanvasLayout percentage tiles (video tile + picture tile;
# strap/ticker/logo are standard furniture). These appear in the New
# Job template picker as real choices; the rest of the library is the
# Director/formula vocabulary until the composer grows more zones.

RENDERABLE: tuple[str, ...] = (
    "news_bulletin_right", "news_bulletin_left", "news_split_50",
    "news_split_60_40", "news_split_70_30", "news_ots_right",
    "news_ots_left",
)


def to_canvas_pcts(key: str) -> Optional[dict]:
    """Map a renderable layout's video + image/carousel zones onto the
    composer's CanvasLayout percentage fields. None if not renderable."""
    l = LAYOUTS.get((key or "").strip().lower())
    if not l or l.key not in RENDERABLE:
        return None
    video = next((z for z in l.zones if z.kind == "video"), None)
    pic = next((z for z in l.zones
                if z.kind in ("carousel", "image", "video_b", "panel")), None)
    if video is None or pic is None:
        return None
    return {
        "video_x_pct": video.x, "video_y_pct": video.y,
        "video_w_pct": video.w, "video_h_pct": video.h,
        "picture_x_pct": pic.x, "picture_y_pct": pic.y,
        "picture_w_pct": pic.w, "picture_h_pct": pic.h,
    }


# ── LIVE LAYOUTS (per-story) — the expressible subset ────────────────
# story_geometry() projects a designed layout onto everything the V4
# per-story composer can draw TODAY: a bg underlay (studio video /
# story-image wash / the clip's own blurred echo), ONE main video tile,
# and ONE picture surface (the image carousel — a side tile OR an inset).
# Text zones stay standard broadcast furniture (the strap / ticker /
# captions are rendered by their own engines, not from zone rects).
# Layouts needing zones the composer can't draw yet (charts, maps, a
# second live camera feed, text panels, big center title cards) resolve
# to None and stay OUT of the Director's vocabulary until the composer
# grows those zones — a pick must never render a blank box.

_PICTURE_KINDS = ("carousel", "image", "video_b", "panel")
_FURNITURE_KINDS = {"strap", "ticker", "logo", "watermark", "caption",
                    "subtext"}
_BLOCKING_KINDS = {"chart", "map", "waveform", "score", "cta"}


@dataclass(frozen=True)
class PipInset:
    x: float
    y: float
    w: float
    h: float
    style: str = ""     # library badge (WHITE FRAME / SHADOW / …), informational


@dataclass(frozen=True)
class StoryGeometry:
    """Everything the per-story composer needs to honour a designed
    layout: rects in canvas percentages (0-100), the same vocabulary as
    CanvasLayout, so the compose layer applies them like editor
    drag-to-move overrides."""
    key: str
    video: tuple                            # (x, y, w, h) — main clip tile
    picture: Optional[tuple] = None         # image-carousel tile rect
    picture_kind: str = ""                  # which zone kind became the tile
    pips: tuple = ()                        # PipInset image inset (<= 1)
    bg: str = "inherit"    # inherit | bg_video | bg_image | blur_self
    video_borderless: bool = False          # full-bleed video: no 3px frame


def story_geometry(key: str) -> Optional[StoryGeometry]:
    """Project a 16:9 designed layout onto the composer's per-story
    geometry. None = not expressible yet (stays out of the vocabulary)."""
    l = LAYOUTS.get((key or "").strip().lower())
    if not l or l.aspect != "16:9":
        return None
    if any(z.kind in _BLOCKING_KINDS for z in l.zones):
        return None
    videos = [z for z in l.zones if z.kind == "video"]
    if len(videos) != 1:
        return None
    video = videos[0]
    # A headline zone high in the frame is a center TITLE CARD (trailer /
    # festival opens) — needs a title engine; low in the frame it's just
    # the strap band the standard furniture already draws.
    if any(z.kind == "headline" and z.y < 70 for z in l.zones):
        return None
    pips = [z for z in l.zones if z.kind == "pip"]
    if len(pips) > 1:       # one picture surface exists today — a second
        return None         # inset would render as a hole; keep it out
    # circle insets need an alpha mask the composer doesn't cut yet
    if any((z.label or "").strip().upper() == "CIRCLE" for z in pips):
        return None
    picture_zones = [z for z in l.zones if z.kind in _PICTURE_KINDS]
    if len(picture_zones) > 1:      # only one picture tile exists today
        return None
    if pips and picture_zones:      # both would fight over the one surface
        return None
    bg_zones = [z for z in l.zones if z.kind in ("bg_video", "bg_image")]
    if len(bg_zones) > 1:
        return None
    bg = "inherit"
    if bg_zones:
        bz = bg_zones[0]
        bg = ("blur_self"
              if (bz.label or "").strip().upper() == "BLURRED SELF"
              else bz.kind)
    # Anything else must be plain furniture; an unaccounted kind means a
    # zone engine we don't have — refuse rather than render a hole.
    known = ({"video", "pip", "bg_video", "bg_image", "headline"}
             | set(_PICTURE_KINDS) | _FURNITURE_KINDS)
    if any(z.kind not in known for z in l.zones):
        return None
    pic = picture_zones[0] if picture_zones else None
    return StoryGeometry(
        key=l.key,
        video=(video.x, video.y, video.w, video.h),
        picture=((pic.x, pic.y, pic.w, pic.h) if pic else None),
        picture_kind=(pic.kind if pic else ""),
        pips=tuple(PipInset(z.x, z.y, z.w, z.h, (z.label or "").strip())
                   for z in pips),
        bg=bg,
        video_borderless=bool(video.w >= 99.0),
    )


RENDERABLE_V2: tuple[str, ...] = tuple(
    l.key for l in _LAYOUTS if story_geometry(l.key) is not None)


def allowed_for(*, has_images: bool = True,
                has_job_bg: bool = False) -> tuple[str, ...]:
    """The per-story layout vocabulary for THIS job's assets. No images →
    layouts that would show an empty picture tile / inset / image wash
    are out. A layout wanting a studio bg the job doesn't carry still
    qualifies — the composer falls back to the clip's blurred echo."""
    out = []
    for k in RENDERABLE_V2:
        g = story_geometry(k)
        if g is None:
            continue
        if not has_images and (g.picture or g.pips or g.bg == "bg_image"):
            continue
        out.append(k)
    return tuple(out)


# ── Layout → editable HTML template (the custom-template bridge) ─────
# Operator: "user must have COMPLETE editing (colors, everything) on the
# layouts we build, and both systems must work together." A designed
# layout can be FORKED into a private CustomTemplate: zones become the
# template contract's slots/elements, so it opens in the existing visual
# builder (move/resize/colors/fonts/layers) and renders through the
# custom-template pipeline. One editor, two entry points.

_HTML_ZONE_COLORS = {
    "headline": "var(--kaizer-brand)", "strap": "var(--kaizer-brand)",
    "ticker": "#f0d232", "cta": "var(--kaizer-accent)",
    "panel": "rgba(16,16,22,.82)", "caption": "rgba(0,0,0,.55)",
    "subtext": "rgba(16,16,22,.75)", "score": "rgba(10,40,24,.9)",
}


def fork_eligible(key: str) -> bool:
    l = LAYOUTS.get((key or "").strip().lower())
    return bool(l and any(z.kind in ("video", "video_b", "bg_video", "pip")
                          for z in l.zones))


def layout_to_html(key: str) -> Optional[str]:
    """Generate contract-valid, builder-editable HTML for a layout.
    Media zones → data-kaizer slots; text zones → styled editable divs.
    None if the layout has no media slot (contract would be invalid)."""
    l = LAYOUTS.get((key or "").strip().lower())
    if not l or not fork_eligible(key):
        return None
    W, H = (1080, 1920) if l.aspect == "9:16" else (
        (1080, 1080) if l.aspect == "1:1" else (1920, 1080))

    def px(z):
        return (round(z.x * W / 100), round(z.y * H / 100),
                max(2, round(z.w * W / 100)), max(2, round(z.h * H / 100)))

    parts, used_video = [], 0
    for z in sorted(l.zones, key=lambda z: z.z):
        x, y, w, h = px(z)
        base = (f"left:{x}px;top:{y}px;width:{w}px;height:{h}px;")
        lbl = z.label or ZONE_STYLE.get(z.kind, ((0, 0, 0), z.kind))[1]
        if z.kind in ("video", "video_b", "pip"):
            used_video += 1
            parts.append(
                f'<div class="kx-el" id="kx-video{used_video}" '
                f'data-kaizer="video" style="{base}'
                f'background:#05070c;border:2px solid rgba(255,255,255,.85);"></div>')
        elif z.kind == "bg_video":
            parts.append(
                f'<div class="kx-el" id="kx-bg" data-kaizer="background" '
                f'style="left:0;top:0;width:{W}px;height:{H}px;'
                f'background:#0a0c12;"></div>')
        elif z.kind in ("image", "carousel", "chart", "map", "bg_image"):
            parts.append(
                f'<div class="kx-el" data-kaizer="image" style="{base}'
                f'background:#151a24;border:2px solid rgba(255,255,255,.6);"></div>')
        elif z.kind == "logo":
            parts.append(
                f'<div class="kx-el" data-kaizer="logo" style="{base}"></div>')
        elif z.kind == "ticker":
            parts.append(
                f'<div class="kx-el" data-kaizer="ticker" style="{base}'
                f'background:{_HTML_ZONE_COLORS["ticker"]};color:#111;'
                f'font-weight:700;font-size:{max(18, h - 18)}px;'
                f'line-height:{h}px;padding:0 18px;white-space:nowrap;'
                f'overflow:hidden;">Breaking updates scroll here…</div>')
        elif z.kind in ("headline", "strap"):
            parts.append(
                f'<div class="kx-el" data-kaizer="headline" style="{base}'
                f'background:{_HTML_ZONE_COLORS[z.kind]};color:#fff;'
                f'font-weight:800;font-size:{max(22, min(64, h - 24))}px;'
                f'padding:{max(6, (h - min(64, h - 24)) // 3)}px 24px;'
                f'overflow:hidden;">Your headline goes here</div>')
        elif z.kind in ("panel", "caption", "subtext", "cta", "score",
                        "waveform", "watermark"):
            bg = _HTML_ZONE_COLORS.get(z.kind, "rgba(16,16,22,.8)")
            fs = max(16, min(40, h // 3))
            parts.append(
                f'<div class="kx-el" style="{base}background:{bg};'
                f'color:var(--kaizer-text);font-size:{fs}px;font-weight:600;'
                f'padding:12px 16px;border-radius:10px;overflow:hidden;">'
                f'{lbl}</div>')
    body = "\n  ".join(parts)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="kaizer:canvas" content="{W}x{H}">
<title>{l.label}</title>
<style>
  :root{{--kaizer-brand:#c11212;--kaizer-accent:#3ad1c8;--kaizer-text:#ffffff;
        --kaizer-bg:#0b0d12;--kaizer-font:'Segoe UI',Roboto,Arial,sans-serif;}}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  html,body{{width:{W}px;height:{H}px;overflow:hidden;}}
  body{{position:relative;background:var(--kaizer-bg);color:var(--kaizer-text);
       font-family:var(--kaizer-font);}}
  #kx-stage{{position:absolute;inset:0;width:{W}px;height:{H}px;}}
  .kx-el{{position:absolute;box-sizing:border-box;}}
</style></head>
<body><div id="kx-stage">
  {body}
</div></body></html>"""
