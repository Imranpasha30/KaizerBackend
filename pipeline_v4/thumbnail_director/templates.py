"""Per-style plan templates — what each ScenePlan field should contain
for a given thumbnail style.

The PLANNER reads the template entry for the style being generated and
passes the entry into Gemini's system prompt so Pass 1 produces a
plan tuned to that style's editorial register.

Each entry describes:
  * focal_rule  — what should dominate the frame for this style
  * scene_rule  — what the scene_location should look like
  * action_rule — how to phrase the key_action
  * mood_rule   — what emotional register to target
  * avoid_rule  — known drift patterns to actively counter
  * safe_zone   — where the external text overlay will land
                  (matches what pipeline_v4.thumbnail_styles enforces
                   in its OUTPUT_RULES_BLOCK + thumbnail_ai's overlay
                   geometry — so the planner reserves the same area
                   the Pillow overlay paints into)

Adding a new style: add an entry here AND register the style in
pipeline_v4.thumbnail_styles. The catalog drives the picker; this
table drives the planner.

Product team owns this file — editorial direction lives here.
"""
from __future__ import annotations


STYLE_PLAN_TEMPLATES: dict[str, dict] = {
    # ── Style 1: Symbolic ──────────────────────────────────────────
    "symbolic": {
        "focal_rule": (
            "ONE concrete Indian symbol that maps directly to the news. "
            "Picks: Indian Khaki police uniform sleeves with handcuffs, "
            "Indian High Court facade (red sandstone), ₹500 / ₹2000 rupee "
            "bundles, Ashoka pillar government emblem, BSE building. "
            "DO NOT pick a generic American equivalent (US police uniform, "
            "US courthouse, US dollar bills)."
        ),
        "scene_rule": (
            "Specific Indian institutional setting drawn from the actual "
            "story (e.g. \"Telangana High Court steps\" not \"a courthouse\")."
        ),
        "action_rule": (
            "Static or near-static — the symbol IS the action. Phrase as "
            "a state, not a verb (e.g. \"evidence laid out on a wooden "
            "desk\", \"handcuffs hovering above a stack of rupee notes\")."
        ),
        "mood_rule": (
            "Investigative gravity — invites the viewer to want to know "
            "what happened. Use the tone palette's atmosphere."
        ),
        "avoid_rule": (
            "Do NOT include any human face. Do NOT use American visual "
            "vocabulary. Do NOT make the symbol generic stock — it must "
            "tie directly to the specific story details."
        ),
        "safe_zone": "bottom 28% strap, full-width — for the text overlay",
    },

    # ── Style 2: Reporter-led ──────────────────────────────────────
    "reporter_led": {
        "focal_rule": (
            "The reference person (the channel's anchor), framed "
            "head-and-shoulders on the LEFT half. Identity must be "
            "preserved EXACTLY from the reference photo — gender, age, "
            "skin tone, facial bone structure, hair, beard / clean-shaven, "
            "glasses or not, clothing style."
        ),
        "scene_rule": (
            "Indian newsroom set — massive LED video wall on the right "
            "half showing graphics related to the SPECIFIC story (a "
            "faded courthouse, a stock chart, a Parliament dome). NOT "
            "a one-camera American CNN desk."
        ),
        "action_rule": (
            "Anchor is mid-speech or about to speak, slight forward lean "
            "for authority. Confident expression matching story tone."
        ),
        "mood_rule": (
            "Authoritative live-broadcast energy. Use the tone palette to "
            "tint the LED backdrop and rim light."
        ),
        "avoid_rule": (
            "DO NOT change the reference person's gender, age, skin tone, "
            "or facial geometry. DO NOT replace them with a stock news "
            "anchor. DO NOT add other people to the frame."
        ),
        "safe_zone": "right half (40-50% of frame) — for the text overlay",
    },

    # ── Style 3: Subject photo ─────────────────────────────────────
    "subject_photo": {
        "focal_rule": (
            "The reference person (politician / accused / celebrity), "
            "face dominating 60% of the frame, slightly off-centre. "
            "Identity preserved EXACTLY from the reference photo."
        ),
        "scene_rule": (
            "Slightly blurred Indian-context backdrop drawn from the "
            "specific story (Indian High Court for a court ruling, "
            "Parliament dome for a political win, hospital ward for "
            "health story, podium for a press conference)."
        ),
        "action_rule": (
            "Subject's micro-expression matches the story tone: concerned "
            "for crime accusations, triumphant for victories, stunned for "
            "revelations, sombre for tragedies. Identity unchanged — only "
            "micro-expression flexes."
        ),
        "mood_rule": (
            "Tells you their emotional state at a glance. Apply the tone "
            "palette as colour grade on the whole image."
        ),
        "avoid_rule": (
            "DO NOT alter the reference person's identity. DO NOT replace "
            "them with a generic politician / celebrity. DO NOT add other "
            "people."
        ),
        "safe_zone": "bottom 30% strap, full-width — for the text overlay",
    },

    # ── Style 4: Scene photograph ──────────────────────────────────
    "scene_photograph": {
        "focal_rule": (
            "The story's central event at its peak moment — not a static "
            "wide shot. Examples by story type: chaotic Telangana street "
            "protest with placards, rescue crew at a collapsed building, "
            "fire fighters spraying water, flooded paddy field, vault "
            "being opened with a forensic team. Draw from the specific "
            "details in the user-provided story."
        ),
        "scene_rule": (
            "Real-photo Indian street / institutional setting with the "
            "telltale Indian details: hand-painted hoardings, auto-"
            "rickshaws, two-wheelers, dust haze, monsoon puddles, "
            "electric pole wires."
        ),
        "action_rule": (
            "Verb-driven — capture the action in mid-motion. Use specific "
            "verbs from the story (\"protesters surging forward\", "
            "\"smoke billowing from a window\", \"rescue worker pulling "
            "someone clear\")."
        ),
        "mood_rule": (
            "PTI / Reuters photojournalism — natural unpolished lighting, "
            "35mm grain, dramatic environmental shadows. Apply the tone "
            "palette as a colour grade."
        ),
        "avoid_rule": (
            "DO NOT include identifiable real public figures — use "
            "back-of-head shots, silhouettes, or out-of-focus crowd "
            "faces when humans appear. DO NOT use Western street "
            "furniture (US fire trucks, US police cars)."
        ),
        "safe_zone": (
            "bottom third strap — for the text overlay; keep that band "
            "darker and visually quieter so white-on-tone text reads"
        ),
    },

    # ── Style 5: Split-screen ──────────────────────────────────────
    "split_screen": {
        "focal_rule": (
            "Two reference people in a hard vertical split. FIRST "
            "reference on the LEFT half, SECOND reference on the RIGHT "
            "half. Both identities preserved EXACTLY from their "
            "respective references."
        ),
        "scene_rule": (
            "Each side's backdrop tints to the opposing political colour "
            "(saffron / blue) OR before-after states. NOT a handshake "
            "scene — this is confrontational."
        ),
        "action_rule": (
            "Each subject angled slightly INWARD toward the centre line "
            "(suggests confrontation). Slight tilt, intense expressions."
        ),
        "mood_rule": (
            "Aggressive political-clash energy. Tone palette drives the "
            "two background gradients."
        ),
        "avoid_rule": (
            "DO NOT swap the two subjects between halves. DO NOT make "
            "them look alike. DO NOT replace either with a generic "
            "stand-in. DO NOT pose them shaking hands."
        ),
        "safe_zone": (
            "centre band (~18% vertical) — for the VS / shout text overlay"
        ),
    },

    # ── Style 6: Live / Breaking ───────────────────────────────────
    "live_breaking": {
        "focal_rule": (
            "Minimal background — dominated by a deep red wash with a "
            "small symbolic backdrop element (broadcast tower silhouette, "
            "spinning earth, news desk profile) at 20-30% opacity."
        ),
        "scene_rule": (
            "Abstract broadcast space — not a literal newsroom. The red "
            "is the scene."
        ),
        "action_rule": (
            "Static plate. The LIVE / BREAKING badge in a top corner is "
            "the only graphic element."
        ),
        "mood_rule": (
            "URGENT live-broadcast intercept. Scan-line texture, pulsing "
            "red glow."
        ),
        "avoid_rule": (
            "DO NOT use this style for sports recaps, weather forecasts, "
            "or any non-urgent piece. DO NOT add people. DO NOT make the "
            "BREAKING badge larger than the corner can comfortably hold."
        ),
        "safe_zone": (
            "centre 70% — for HUGE shout text overlay (yellow on red)"
        ),
    },

    # ── Style 7: Text-dominant ─────────────────────────────────────
    "text_dominant": {
        "focal_rule": (
            "Minimal visual — one low-opacity (20-30%) Indian-context "
            "silhouette related to the specific story details "
            "(Parliament dome, ₹ symbol, temple Gopuram)."
        ),
        "scene_rule": (
            "Solid background using the tone palette colours — the "
            "background IS the scene."
        ),
        "action_rule": "Static plate.",
        "mood_rule": (
            "Maximalist typography energy. The silhouette only sets the "
            "mood; the text is the focal point."
        ),
        "avoid_rule": (
            "DO NOT include people. DO NOT add multiple symbols — one "
            "silhouette, minimal."
        ),
        "safe_zone": (
            "centre 70% — for HUGE multi-line shout text overlay"
        ),
    },

    # ── Style 8: Festival / Religion ───────────────────────────────
    "festival_religion": {
        "focal_rule": (
            "Devotional imagery drawn from the specific festival — "
            "Tirumala temple Gopuram, Ram Mandir spire, rows of glowing "
            "diyas, marigold-festooned temple entrance, crescent moon "
            "over a mosque silhouette, Indian church facade with star "
            "lanterns. Pick what matches the story."
        ),
        "scene_rule": (
            "South Indian / North Indian / cross-faith religious setting "
            "with authentic Indian architectural detail (Gopuram steps, "
            "kalyana mantapa, temple flagpoles, mosque minarets)."
        ),
        "action_rule": (
            "Devotees in soft motion (praying, raising hands, lighting "
            "lamps) OR static iconic shot. Use specific festival actions."
        ),
        "mood_rule": (
            "Soft devotional warmth — NEVER newsroom urgency. Soft "
            "golden-hour light, glowing diyas as practical light sources, "
            "slight bloom on highlights."
        ),
        "avoid_rule": (
            "DO NOT use harsh red/yellow newsroom palette. DO NOT use "
            "the LIVE / BREAKING badge. DO NOT make it feel like a "
            "crime scene — this is celebratory."
        ),
        "safe_zone": (
            "bottom 28% strap — for the text overlay (gold on crimson)"
        ),
    },

    # ── Style 9: Sports action ─────────────────────────────────────
    "sports_action": {
        "focal_rule": (
            "Frozen peak-action moment from the specific sport: batsman "
            "mid-stroke with ball blurred in flight, striker mid-volley "
            "with grass + dirt spraying, kabaddi defender mid-tackle, "
            "athlete mid-leap. Pick from the story specifics."
        ),
        "scene_rule": (
            "Authentic Indian sports venue: Wankhede / Eden Gardens / "
            "Uppal stadium floodlights, ISL stadium, Olympic track, "
            "kabaddi mat. Specific venue if named in the story."
        ),
        "action_rule": (
            "High shutter-speed verb: \"connecting with foot\", "
            "\"mid-dive\", \"mid-stroke\". Frozen motion with motion blur "
            "ONLY on the moving object (ball, bat tip)."
        ),
        "mood_rule": (
            "Aggressive sports-photography energy. High saturation, neon "
            "team-colour highlights, stadium floodlight flares."
        ),
        "avoid_rule": (
            "DO NOT pose a static portrait — this is action. DO NOT "
            "include identifiable real players (use silhouettes if "
            "needed). DO NOT use newsroom red/yellow — sports wants "
            "team colours."
        ),
        "safe_zone": (
            "bottom third — for the text overlay (slanted / italic to "
            "imply speed)"
        ),
    },

    # ── Style 10: Entertainment / Tollywood ───────────────────────
    "entertainment_tollywood": {
        "focal_rule": (
            "Movie poster vibe — subject in dramatic 3/4 profile with "
            "rim light, glossy film-still composition. OR a massive "
            "vertical \"₹ XXX cr\" box-office number stacked in glossy "
            "3D. OR a vibrant on-set dance pose silhouette."
        ),
        "scene_rule": (
            "Studio / movie-set / red-carpet / paparazzi background. "
            "Heavy bokeh — out-of-focus paparazzi flashes, RGB stage "
            "lights, or movie-set practicals."
        ),
        "action_rule": (
            "Movie-magazine glamour pose OR frozen dance step. Glossy "
            "retouched skin tones."
        ),
        "mood_rule": (
            "High-end film-magazine energy. Neon palette (hot pink, "
            "electric cyan, deep purple). NOT the standard news red/"
            "yellow."
        ),
        "avoid_rule": (
            "DO NOT use the LIVE / BREAKING badge. DO NOT make it look "
            "like a news report — this is entertainment glamour. DO NOT "
            "use newsroom palette."
        ),
        "safe_zone": (
            "bottom third — for the text overlay (movie-title-card "
            "styling, neon-cyan on deep purple)"
        ),
    },
}


def get_template(style: str) -> dict:
    """Resolve a style key to its plan template. Falls back to the
    symbolic template for unknown styles so a stale frontend can never
    starve the planner."""
    return STYLE_PLAN_TEMPLATES.get(style) or STYLE_PLAN_TEMPLATES["symbolic"]


__all__ = ["STYLE_PLAN_TEMPLATES", "get_template"]
