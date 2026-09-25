"""Canvas JSON schema — the source of truth for V4 layout and timing.

This shape is:
  - Produced by the orchestrator after Step 1's trim plan
  - Stored on disk at ``<job.output_dir>/canvas.json``
  - Read by the editor for the per-story edit UI
  - Written back by the editor when the user replaces images, edits
    titles, drags duration sliders, or reorders things
  - Consumed by the canvas_engine in Step 2 to render the final video

Design constraint: every field that affects RENDERING must be in this
file. Anything not in canvas.json cannot influence the output. That
makes the editor's contract unambiguous — change a field, see the
change after re-render. No hidden state.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─── Image layer ────────────────────────────────────────────────────

class CanvasImage(BaseModel):
    """One image displayed in the picture panel for a duration."""
    src: str = Field(..., description="Filename of the image inside <bdir>/_pool/. We store filename only; canvas_engine resolves to absolute path.")
    t_start: float = Field(..., ge=0.0, description="Seconds from the start of the parent story-clip when this image becomes visible.")
    t_end: float = Field(..., gt=0.0, description="Seconds from the start of the parent story-clip when this image is replaced.")
    source: Literal["claude", "openai", "user", "manual"] = "claude"
    label: Optional[str] = Field(None, description="Operator-facing tag like 'Modi PC' for the editor UI.")
    # ─── Transition effect ──────────────────────────────────────────
    # Drives the in/out animation when this image appears / disappears.
    # The renderer reads `effect` + `effect_duration` and applies the
    # matching ffmpeg filter. "cut" = hard switch (legacy behaviour).
    # Default `fade` gives a smooth alpha crossfade — what the orchestrator
    # picks for every auto-populated image so the user gets a polished
    # carousel without touching anything.
    effect: Literal["cut", "fade", "slide_left", "slide_right", "zoom_in"] = "fade"
    effect_duration: float = Field(0.4, ge=0.0, le=2.0, description="Seconds the in/out animation takes. 0 = instant.")
    # ─── Framing inside the picture panel ──────────────────────────
    # cover  = scale-to-fill the panel and crop the excess (uploaded
    #          photos that are taller / wider than the panel get
    #          cropped on whichever axis overflows).
    # contain = scale-to-fit and pad with bg_color (no crop, may show
    #          letterbox bands).
    fit: Literal["cover", "contain"] = "cover"
    # When fit=cover, these decide WHICH part of the source shows.
    # 50/50 = centered (what V1 and the old renderer did). 0/0 = top-left
    # of the source pinned to top-left of the panel; 100/100 = bottom-right.
    # Driven by the 9-point focal-grid editor in the UI.
    offset_x_pct: float = Field(50.0, ge=0.0, le=100.0)
    offset_y_pct: float = Field(50.0, ge=0.0, le=100.0)
    # ─── Image↔speech sync (Phase-1 engine) ─────────────────────────
    # All Optional-with-default so canvases from before this feature
    # validate unchanged and render byte-identically.
    #
    # content = the timing AI may (re)place this image on the spoken
    #           words; pinned = the operator fixed t_start/t_end by hand
    #           and every re-sync must keep them verbatim.
    timing_mode: Literal["content", "pinned"] = "content"
    # Timing AI's certainty (0-1) that this image's label matches the
    # words in its window. Below the confidence gate the window became a
    # gap instead (main video shows). None = timed by the old heuristic.
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    # How central this moment is to the story (0-1). Drives spotlight
    # auto-selection; operator can override via `spotlight`.
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Spotlight override: None = auto (engine decides from importance),
    # "off" = never spotlight, "fullscreen" = pop full-screen for the
    # window, "pip" = picture-in-picture inset.
    spotlight: Optional[Literal["off", "fullscreen", "pip"]] = None
    # The anchor words the timing AI matched (joined text) — shown in the
    # editor so the operator sees WHY the image sits at this moment.
    matched_text: Optional[str] = None
    # ─── Reference-video cutaway (Phase 3) ──────────────────────────
    # media_kind="video" turns this entry into a full-screen VIDEO
    # cutaway (B-roll): `src` is an .mp4/.webm in _pool, played full-
    # screen for [t_start, t_end] (honours spotlight="fullscreen").
    # Back-compat: default "image" → every existing canvas is unchanged
    # and renders byte-identically.
    media_kind: Literal["image", "video"] = "image"
    # Where in the source clip the cutaway starts (seconds). 0 = head.
    video_trim_start: float = Field(0.0, ge=0.0)
    # Cutaway audio: "mute" = keep only the anchor narration; "duck" =
    # lower the anchor and play the clip's own sound during the window;
    # "auto" = the AI Director decides per moment. (duck/auto = Phase 3d.)
    audio_mode: Literal["mute", "duck", "auto"] = "mute"


# ─── Text overlays ──────────────────────────────────────────────────

class CanvasTextBlock(BaseModel):
    """A static text layer rendered as a PNG via PIL, then overlaid by
    ffmpeg with timed visibility. Used for the lower-third strap, the
    ticker, the story headline, etc."""
    kind: Literal["lower_third", "ticker", "headline", "watermark", "custom"]
    text: str = Field(..., description="The text shown. Telugu / Hindi / English allowed; PIL renderer picks the right font.")
    t_start: float = Field(0.0, ge=0.0)
    t_end: Optional[float] = Field(None, description="None = visible for the whole story-clip duration.")
    # Optional override of the default layout for this block. None →
    # the canvas_engine uses kind-specific defaults so most stories
    # don't need to set anything here.
    x_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Horizontal position as % of canvas width.")
    y_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    w_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    font_size_pct: Optional[float] = Field(None, gt=0.0, description="Font size as % of canvas height.")
    fg_color: Optional[str] = Field(None, description="Hex e.g. '#FFFFFF'")
    bg_color: Optional[str] = Field(None, description="Hex e.g. '#C10000' — None means transparent.")
    # Ticker-only knobs (kind == "ticker"). None → renderer defaults
    # (gold-bordered navy strip scrolling at ~200 px/s). The editor's live
    # ticker controls write these so the rendered MP4 matches the preview.
    ticker_speed: Optional[float] = Field(None, gt=0.0, le=120.0, description="Ticker only: seconds for one full scroll loop (lower = faster).")
    ticker_color: Optional[str] = Field(None, description="Ticker only: hex background color e.g. '#FFD400'. None = default navy/gold.")


# ─── Broadcast graphics (overlays / HUD) ────────────────────────────

class CanvasOverlay(BaseModel):
    """An AI-Director-placed (or user-added) broadcast graphic from the
    overlays REGISTRY — a banner, bug, stamp, locator, HUD, etc. — shown at a
    timed window over this story. Persisted in canvas.json so the editor can
    change its TEXT, timing, and which graphic, then re-render. ``fields``
    overrides the graphic's text kwargs (e.g. {"text": "BREAKING NEWS",
    "sub": "..."}); empty values fall back to the graphic's built-in defaults."""
    id: str = Field(..., description="overlays REGISTRY id, e.g. 'breaking_news_banner'.")
    t: float = Field(0.0, ge=0.0, description="Seconds into the story when it appears.")
    dur: Optional[float] = Field(None, gt=0.0, description="Seconds on screen (None = ~4.5s default).")
    fields: dict[str, str] = Field(default_factory=dict, description="Per-instance text overrides for the graphic's kwargs (text / sub / city / title / label…). Empty = REGISTRY defaults.")


class CanvasLayoutMoment(BaseModel):
    """One mid-story layout window: at ``t`` (story-relative seconds)
    the screen switches to ``layout`` for ``dur`` seconds, entering and
    leaving with ``transition``. The render sanitizes windows (edge
    clamps, min duration, overlap resolution) — a bad moment degrades
    to nothing, never to a broken frame."""
    layout: str
    t: float = Field(0.0, ge=0.0)
    dur: float = Field(4.0, gt=0.0)
    transition: Literal["push", "cut"] = "push"


# ─── Per-story clip on the timeline ─────────────────────────────────

class CanvasStory(BaseModel):
    """One news story on the canvas timeline. Corresponds to one chunk
    of the trimmed video — the talking-head segment for THIS story —
    with its own text overlays and image carousel."""
    story_index: int = Field(..., ge=0)
    video_t_start: float = Field(..., ge=0.0, description="Where in trimmed.mp4 this story's footage starts.")
    video_t_end: float = Field(..., gt=0.0, description="Where in trimmed.mp4 this story's footage ends.")
    title_native: str = ""
    title_english: str = ""
    summary: str = ""
    images: list[CanvasImage] = Field(default_factory=list, description="Carousel images shown over this story's duration. Times are RELATIVE to story start (0 = first frame of this story).")
    text_blocks: list[CanvasTextBlock] = Field(default_factory=list, description="lower-third / headline / ticker overlays for this story.")
    # When the editor enables this we let Claude re-pick image timings
    # on the next re-render. Defaults to using whatever timings are in
    # `images`.
    claude_decided_timings: bool = True
    # Polish A — name-strap lower-third: while an image is on screen, a
    # small strap shows its subject label ("who/what you're looking at").
    # Default False so every pre-engine canvas renders byte-identically;
    # the orchestrator turns it on for stories the timing engine placed
    # (their labels are meaningful subjects, not filename stems).
    name_strap: bool = False
    # Broadcast graphics (banners / bugs / stamps / HUD) shown over this
    # story. Seeded by the AI Director at render time (id + timing + default
    # text) and PERSISTED here so the editor can change the text / timing /
    # which graphic and re-render. Empty on legacy canvases → the renderer
    # falls back to the live Director plan (byte-identical).
    overlays: list[CanvasOverlay] = Field(default_factory=list)
    # LIVE LAYOUTS: the designed screen layout for THIS story (a
    # layout_library key, e.g. "news_ots_right"). Seeded by the AI
    # Director, persisted so editor re-renders honour it. "" = the job's
    # default layout — every pre-layout canvas renders byte-identically.
    # Resolved fail-soft at render via layout_library.story_geometry().
    layout_key: str = ""
    # LIVE LAYOUTS: mid-story layout switches — timed windows where the
    # screen changes to a DIFFERENT (video-centric) designed layout,
    # then returns: the "live broadcast direction" feel. Video-only:
    # narration audio is untouched (the branch overlays in the same
    # graph — no time is consumed). Image-importance moments use the
    # existing per-image spotlight="fullscreen" instead.
    layout_moments: list[CanvasLayoutMoment] = Field(default_factory=list)


# ─── Top-level canvas (one per output format) ───────────────────────

class CanvasLayout(BaseModel):
    """Static layout describing where the video and panels sit on the
    canvas. Same shape for bulletin (16:9) and short (9:16); only the
    numbers differ. Operator can override per-job for advanced cases."""
    width: int = Field(1920, gt=0)
    height: int = Field(1080, gt=0)
    bg_color: str = Field("#000000")
    # Where the trimmed video sits on the canvas. Pct of canvas.
    video_x_pct: float = 0.0
    video_y_pct: float = 0.0
    video_w_pct: float = 100.0
    video_h_pct: float = 100.0
    # Where the picture panel sits (image carousel layer).
    picture_x_pct: float = 70.0
    picture_y_pct: float = 8.0
    picture_w_pct: float = 26.0
    picture_h_pct: float = 40.0
    # Brand chrome.
    brand_logo_path: Optional[str] = None       # filesystem path; None = none
    brand_logo_x_pct: float = 92.0
    brand_logo_y_pct: float = 4.0
    brand_logo_w_pct: float = 6.0
    # Tile-frame border colour. v1_bridge draws a 3px border around the video +
    # picture tiles; default None → white (legacy). Set to the canvas bg colour
    # (e.g. "#000000") for a full-bleed look — used by the audio-first
    # fullscreen-image layout so the 3px frame is invisible.
    tile_border_color: Optional[str] = None
    # ─── Background video ────────────────────────────────────────────
    # Replaces the dead-black bg_color with a looping video — gives the
    # finished bulletin a TV-news studio feel instead of a flat backdrop.
    # Value format:
    #   None              → bg_color (current behaviour, unchanged)
    #   "sample:NAME.mp4" → bundled demo from public/video/
    #   "asset:<id>"      → a UserAsset the operator uploaded
    # The renderer resolves these to a local file path.
    bg_video_path: Optional[str] = None
    # 0.0 = muted (default; matches "studio bg has no diegetic audio").
    # 1.0 = full mix. Mixed against the trimmed video's audio track.
    bg_video_volume: float = Field(0.0, ge=0.0, le=1.0)
    # When > 0, the bg video plays FULL-SCREEN with its own audio at 1.0
    # for this many seconds as a "leader" / intro reel, then the
    # bulletin layout fades in on top and the bg audio drops to
    # bg_video_volume. 0 = no intro (current behaviour). Capped at 30s
    # so a typo can't produce a 5-minute intro.
    bg_intro_seconds: float = Field(0.0, ge=0.0, le=30.0)
    # Lower-third entrance easing (native motion toolkit). None = the
    # legacy linear slide (pre-motion canvases render byte-identically);
    # the orchestrator stamps "out_cubic" on NEW canvases for the
    # broadcast-grade decelerating entrance.
    lt_ease: Optional[Literal["out_cubic"]] = None
    # THEME PACKS: the visual skin of the built-in render ("" = classic,
    # byte-identical legacy). Resolved fail-soft via theme_packs.get_theme:
    # backdrop plate + tile frame colour + ticker colourway. Themes never
    # move placements — layouts/moments/animations run inside them.
    theme: str = ""


# ─── V1 short-template edit config ──────────────────────────────────
# Mirrors the per-clip controls V1's Editor.jsx exposes for shorts:
# headline text, font/size/colour, image, section %, torn-card style,
# follow-bar params. Every renderable knob the V4 editor needs lives
# here — same "canvas.json is the source of truth" rule the rest of
# this schema follows.

class ShortCardStyle(BaseModel):
    """Torn-card geometry sliders (V1 parity)."""
    seed: int = 7
    edge: int = 9
    jag: int = 60
    overlap: int = 20
    vsid: int = 35
    vcor: int = 72
    vwid: int = 74
    bgr0: int = 193
    bgr1: int = 128


class ShortFollowParams(BaseModel):
    """Follow-bar layout text + colours (V1 parity)."""
    follow_text: str = "FOLLOW KAIZER X TELUGU"
    follow_text_color: str = "#FFFFFF"
    bg_color: str = "#1A0A2E"
    text_color: str = "#FFFF00"


class ShortSectionPct(BaseModel):
    """Three vertical bands of the torn-card layout (V1 parity)."""
    video: float = 0.4619
    text:  float = 0.1691
    image: float = 0.3690


class ShortConfig(BaseModel):
    """Per-short editor settings. Drives :func:`v1_bridge.render_short`."""
    # Built-in: torn_card | clean_card | split_frame | follow_bar. Also accepts a
    # developer-uploaded template as the string "custom:<id>" (services/custom_templates).
    layout: str = "torn_card"
    # Headline text. None = use the parent story's title_native.
    text: Optional[str] = None
    # Font / colour / size (V1 controls).
    font_file: str = "NotoSansTelugu-Bold.ttf"
    font_size: Optional[int] = None         # None lets the V1 composer auto-size
    text_color: str = "#FFFFFF"
    # Which pool image to use as the short's hero image. Filename only;
    # resolves to <out_dir>/_pool/<filename>.
    image_filename: Optional[str] = None
    # Layout-specific blocks.
    section_pct: ShortSectionPct = Field(default_factory=ShortSectionPct)
    card_style:  ShortCardStyle  = Field(default_factory=ShortCardStyle)
    follow_params: ShortFollowParams = Field(default_factory=ShortFollowParams)
    # Display-only selection label: why this segment was auto-picked as a
    # short + its priority rank (1 = highest; candidates are kept lead-first).
    # Surfaced in JobDetail so the operator understands the shorts plan.
    # Persisted in canvas.json so it survives re-renders.
    priority: Optional[int] = None
    why_selected: Optional[str] = None


class CanvasSEO(BaseModel):
    """YouTube SEO metadata for one canvas (short or bulletin). Mirrors
    V2's ``clip.seo`` JSON shape so the publish flow can consume it
    directly. Editable via the V4 editor before YouTube upload."""
    title: str = ""
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    hook: str = ""
    thumbnail_text: str = ""
    metadata: dict = Field(default_factory=dict)
    language: str = ""
    model: str = ""
    edited_by_user: bool = False
    # Competitor style reference (a kind:styles Channel) this copy was
    # written in the voice of, or None for the default voice. Lets the
    # editor show "in X's style" and re-pick it on the next regenerate.
    style_source_id: Optional[int] = None


class Canvas(BaseModel):
    """One V4 output specification — either the bulletin or one short."""
    kind: Literal["bulletin", "short"]
    output_filename: str = Field(..., description="e.g. 'bulletin.mp4' or 'short_01.mp4'")
    layout: CanvasLayout
    stories: list[CanvasStory] = Field(default_factory=list, description="Bulletin has N stories. A short usually has 1.")
    # Path to the trimmed video this canvas consumes.
    trimmed_video_path: str = Field(..., description="Absolute path to the Step-1 output that feeds this canvas.")
    # Editor knobs for shorts (None on the bulletin canvas).
    short_config: Optional[ShortConfig] = None
    # SEO metadata generated post-canvas, editable before upload.
    seo: Optional[CanvasSEO] = None
    # Story-to-story transition at the stitch (bulletin only). None =
    # legacy hard cut (pre-motion canvases render identically); the
    # orchestrator stamps "fade" on NEW multi-story bulletins. The
    # renderer validates against its safe xfade set and falls back to
    # a hard cut on any failure.
    story_transition: Optional[str] = None


class V4JobCanvas(BaseModel):
    """The full canvas.json for a V4 job. One bulletin + N shorts."""
    job_id: int
    language: str = "te"
    bulletin: Canvas
    shorts: list[Canvas] = Field(default_factory=list)
    # Operator's render-output choice at job time:
    #   "both" (full video + shorts), "full-only" (bulletin, no shorts),
    #   "shorts-only" (shorts/reels, no bulletin). Carried here for
    #   display + audit; the actual gating happens in the orchestrator.
    output_format: str = "both"
    # Full-form effects mode picked at job time ("auto"/"rich"/"off").
    # "" = pre-effects canvas → renders exactly as before. Carried on the
    # doc so EDITOR re-renders (in-process, no runner env) honour it.
    effects_mode: str = ""
    # Bookkeeping
    schema_version: int = 1
    # Step 1 output paths (kept for editor's "rebuild bulletin only"
    # path — Step 1 is sunk cost, only Step 2 re-runs).
    trimmed_bulletin_path: str
    trimmed_shorts_paths: list[str] = Field(default_factory=list)
