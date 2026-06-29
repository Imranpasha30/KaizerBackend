"""V4 thumbnail style catalog — Indian news channel aesthetic.

Each entry maps a UI-selectable style key to:
  * a human-readable label (shown in the picker)
  * a "best for" description (one-liner the operator sees on hover)
  * how many reference images the style consumes (0/1/2)
  * which kind of reference is expected (person face / scene / two faces)
  * a tailored system prompt that the prompt-writer Gemini call uses
    instead of the generic ``_PROMPT_WRITER_SYSTEM``

The catalog is the SINGLE source of truth — adding a new style means
appending one dataclass to ``STYLES`` and the picker auto-updates.

Audit-driven rewrite (2026-06-09):
  * Removed hard-coded "red / yellow" palette — replaced with a
    per-story tone palette resolved at prompt-build time so a
    festival story doesn't get the same colour wash as a crime
    story (operator feedback: "every thumbnail looks the same").
  * Localised the visual vocabulary — Indian khaki police uniforms,
    Indian High Court architecture, Rupee bundles (₹), Indian newsroom
    LED panels. Pure image-gen models default to American imagery
    without this nudge.
  * Added stronger face-preservation language for reference styles
    ("structural geometry, skin tone, hair") to stop identity drift.
  * Added 3 new styles for situations the original 7 forced into the
    wrong aesthetic: Festival/Religion, Sports Action, Entertainment.

System-prompt design rules (kept consistent across every entry):
  1. The LANGUAGE CONTRACT block from the legacy prompt is preserved
     verbatim — shout text MUST be native-script.
  2. The prompt explains where the reference image (if any) should go
     in the composition. Nano Banana respects the spatial instruction.
  3. Every prompt closes with the SAME output rules (single prompt
     string, 16:9, no logos, etc.) so the only thing that varies
     between styles is the editorial direction.
  4. Each prompt now references a TONE PALETTE that the prompt-writer
     fills in from the resolved tone (see ``resolve_tone_palette``).
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Tone × palette matrix ──────────────────────────────────────────
#
# Each entry describes the editorial tone of a story and the visual
# palette that should accompany it. Removes the "every thumbnail
# looks red-yellow" problem by tying colour to story content.
#
# Keys are the resolver's classification output — keep them stable
# because they're used across thumbnail_ai.py and the tests.

TONE_PALETTES: dict[str, dict] = {
    "crime": {
        "palette":     "gritty greens, stark blacks, harsh police-light blues/reds",
        "atmosphere":  "shadows, high contrast, CCTV-style grain, oppressive mood",
        "label":       "crime / scam / arrest",
    },
    "political_clash": {
        "palette":     "deep saffron / orange (one side) vs. cool blue (other side)",
        "atmosphere":  "hard newsroom lighting, sharp angular shadows, confrontational",
        "label":       "political clash / debate",
    },
    "political_victory": {
        "palette":     "bright gold, pure white, saffron accents",
        "atmosphere":  "triumphant, glowing, cinematic sunlight, rim-lit hero shot",
        "label":       "political victory / government announcement",
    },
    "tragedy": {
        "palette":     "desaturated slate, cold grays, dusty ash tones",
        "atmosphere":  "somber, muted, dusty, heavy vignette, low contrast",
        "label":       "tragedy / disaster / death toll",
    },
    "finance": {
        "palette":     "crisp navy blue, silver, bright green (bull) or red (bear)",
        "atmosphere":  "clean, corporate, glossy, neon data lines, futuristic chart overlays",
        "label":       "finance / business / markets",
    },
    "festival": {
        "palette":     "warm marigold yellows, deep crimsons, golden saffron",
        "atmosphere":  "soft golden hour, glowing diyas, vibrant flower garlands, devotional warmth",
        "label":       "festival / religious / cultural",
    },
    "entertainment": {
        "palette":     "hot pinks, electric cyan, deep purples, neon glow",
        "atmosphere":  "studio lighting, bokeh, paparazzi flashes, glossy magazine finish",
        "label":       "entertainment / film / celebrity",
    },
    "breaking": {
        "palette":     "deep blood-red wash, blinding yellow accents",
        "atmosphere":  "scan-line texture, pulsing red glow, urgent broadcast feel",
        "label":       "breaking news / urgent live",
    },
    # Catch-all when no other tone fits. Less saturated than the
    # legacy red-yellow default so generic stories don't shout.
    "general": {
        "palette":     "muted red, slate blue, off-white",
        "atmosphere":  "balanced newsroom lighting, light vignette",
        "label":       "general news",
    },
}


# Keyword heuristic for tone classification. Order matters — earlier
# rules win when a story matches multiple buckets. Tuned for
# Telugu / Hindi / English news coverage. Free + instant — upgrade to
# a Gemini classification call later if the heuristic ever drifts.

_TONE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("breaking",          ["breaking", "live update", "just in", "developing", "alert"]),
    # Finance MUST come before tragedy because "Sensex crashes" /
    # "market crash" matches the substring "crash" in tragedy
    # keywords without this priority. Real tragedy stories like
    # "Krishna river flood" still land in tragedy because they
    # carry zero finance vocabulary.
    ("finance",           ["sensex", "nifty", "stock", "stocks", "market", "ipo",
                            "rupee falls", "rupee strengthens", "inflation", "gst",
                            "budget", "rbi", "interest rate", "gdp", "data centre",
                            "investment", "crore", "merger", "acquisition",
                            "bse", "nse", "earnings", "quarterly results"]),
    ("tragedy",           ["dead", "killed", "death toll", "tragedy", "accident",
                            "disaster", "flood", "earthquake", "cyclone", "crash",
                            "collapse", "fire", "blast", "explosion"]),
    ("crime",             ["scam", "fraud", "arrest", "arrested", "raid", "chargesheet",
                            "accused", "cbi", "ed enforcement", "police", "encounter",
                            "kidnap", "rape", "murder", "heist", "looted", "smuggling",
                            "corruption", "bribery", "absconding"]),
    ("political_victory", ["wins", "won", "victory", "swept", "landslide", "majority",
                            "sworn in", "oath", "approved", "passes bill", "passes the bill",
                            "launches", "inaugurates", "foundation stone"]),
    ("political_clash",   ["vs", "versus", "attacks", "slams", "hits out", "clash",
                            "row", "debate", "lashes out", "war of words", "opposition",
                            "protest march", "no-confidence"]),
    ("festival",          ["festival", "diwali", "deepavali", "dussehra", "navaratri",
                            "navratri", "ugadi", "sankranti", "pongal", "vinayaka",
                            "ganesh chaturthi", "tirumala", "tirupati", "ayodhya",
                            "ram mandir", "temple", "puja", "darshan", "brahmotsavam",
                            "ramzan", "ramadan", "eid", "christmas"]),
    ("entertainment",     ["movie", "film", "trailer", "teaser", "release", "box office",
                            "actor", "actress", "tollywood", "bollywood", "kollywood",
                            "rrr", "pushpa", "salaar", "first look", "song launch",
                            "audio launch", "celebrity"]),
    # Note: finance keywords moved ABOVE tragedy to win the
    # "Sensex crashes" → finance race. See the comment up top.
]


def resolve_tone_palette(*, title: str = "", summary: str = "",
                          transcript: str = "") -> str:
    """Classify a story into one of TONE_PALETTES by keyword scan.

    Picks the FIRST matching bucket in priority order so "breaking" wins
    over "crime" when both appear ("BREAKING: CBI raids Hyderabad
    Rs 100cr scam" → breaking). Falls back to ``"general"`` when no
    bucket matches — gives a muted palette instead of a shouty default.

    The classifier sees title + summary + transcript concatenated and
    lower-cased. Transcript is included because Claude's summary
    sometimes drops the keyword the story actually hinges on.
    """
    haystack = " ".join((title or "", summary or "", transcript or "")).lower()
    if not haystack.strip():
        return "general"
    for tone_key, keywords in _TONE_KEYWORDS:
        for kw in keywords:
            if kw in haystack:
                return tone_key
    return "general"


# ── Common blocks reused inside every style's system prompt ─────────

# The LANGUAGE CONTRACT block is identical across every style. Keeping
# it here means a future tweak (e.g. adding Marathi guidance) edits one
# constant instead of every prompt.
LANGUAGE_CONTRACT_BLOCK = """\
# ⚠ TEXT IS OVERLAID EXTERNALLY — DO NOT ASK THE IMAGE MODEL FOR TEXT
The clip's LANGUAGE is specified in the user message (full name +
script) so you understand the editorial context, BUT you must NOT
ask the image generator to write any text inside the image.

Why: image generation models cannot reliably render Telugu / Hindi /
Tamil / Devanagari ligatures — they hallucinate strokes. Our pipeline
solves this hybrid: AI produces a TEXT-LESS plate, Python's Pillow
overlays the actual native-script text using Noto Sans fonts with
guaranteed-correct typography.

Your prompt MUST contain the literal phrase:
  "Do NOT render any text, captions, headlines, words, letters, or
   typography of any language inside the image. Leave the designated
   safe zone visually quiet so an external text overlay can be added
   later."

The language metadata in the user message tells you the editorial
TONE only — e.g., a Telugu news story has different visual cadence
than an English business update. Use the language as cultural
context (clothing, architecture, signage), never as text to render.
"""

INDIAN_CONTEXT_BLOCK = """\
# 🇮🇳 INDIAN VISUAL VOCABULARY (anti-Westernisation rule)
Image-gen models default to American imagery unless you tell them
otherwise. Every prompt MUST explicitly call out the Indian visual
equivalent when relevant:
- Courts → Indian High Court / Supreme Court architecture (red
  sandstone, columns, scales of justice). NOT a US courthouse.
- Police → Indian Khaki uniform with peaked cap, NOT US Blue.
- Currency → bundled Indian Rupee notes (₹500, ₹2000) with the
  Mahatma Gandhi watermark. NOT US greenbacks.
- Parliament → Indian Parliament dome (circular Sansad Bhavan).
- Court documents → English-Hindi bilingual letterheads with
  government emblem (Ashoka pillar lions).
- Newsrooms → giant LED video wall backdrops, NOT a one-camera
  desk. Telugu/Hindi news = saturated graphics + multiple screens.
- Streets → Indian street furniture (autos, two-wheelers, dust,
  hand-painted hoardings) NOT clean American suburbs.
"""

OUTPUT_RULES_BLOCK = """\
# Output rules
1. Output a single prompt string, no preamble, no JSON, no bullet points.
2. 16:9 aspect ratio.
3. Sharp focus, cinematic lighting, slight vignette.
4. No watermarks, no channel branding, no logos.
5. Use the TONE PALETTE provided in the user message — DO NOT default
   to red/yellow unless the tone palette explicitly says so.
6. Under 300 words.

# ⚠ TEXT-LESS PLATE RULE (CRITICAL FOR INDIC SCRIPT ACCURACY)
The image-gen model CANNOT reliably render Telugu / Hindi / Tamil
ligatures — they come out garbled. We solve this hybrid: the AI
produces a TEXT-LESS background plate, and a downstream Pillow pass
overlays the native-script text using the correct .ttf font.

Your prompt MUST tell the image model:
  - "Do NOT render any text, captions, headlines, words, or
     typography inside the image."
  - "Leave a clean visual SAFE ZONE for an external text overlay."
  - Where the safe zone sits depends on the style:
       * symbolic / scene_photograph / festival → bottom 35% darker
         and visually quieter so a white-on-dark text strap reads.
       * subject_photo / reporter_led → the half opposite the face
         (40-50% of frame) kept visually simple.
       * split_screen → the centre band (10-15% vertical) reserved.
       * live_breaking / text_dominant → the centre 60-80% kept as
         a solid colour wash for HUGE text.
       * sports_action / entertainment_tollywood → the bottom
         third reserved for a title-card strap.
  - "The safe zone must be visually quieter — solid gradient or
     blurred element — so externally-overlaid text reads crisply."

You DO NOT need to write or describe the actual text — just reserve
the space. The downstream Pillow pass renders the SEO hook text
using Noto Sans Telugu / Devanagari / etc. with perfect typography.

Output: ONE prompt string. Nothing else.
"""


@dataclass(frozen=True)
class ThumbnailStyle:
    key: str
    label: str
    description: str       # "Best for X" — surfaced as a hover hint
    needs_reference: int   # 0 | 1 | 2 — how many reference images the style consumes
    reference_kind: str    # "" | "person" | "scene" | "two_persons" | "before_after"
    system_prompt: str     # fully assembled prompt-writer system instruction


# Each style is built once at import time. ``system_prompt`` is the
# whole instruction Gemini sees — pre-stitched from the common blocks
# above so the style-specific editorial guidance is the only thing
# that varies.

STYLES: dict[str, ThumbnailStyle] = {}


def _register(s: ThumbnailStyle) -> None:
    STYLES[s.key] = s


# ── 1. Symbolic — Indian visual vocabulary, no real faces ───────────

_register(ThumbnailStyle(
    key="symbolic",
    label="Symbolic icons",
    description="Generic Indian news symbols (Indian Khaki police, ₹ bundles, Indian courthouse). Best for crime, scam, fraud, political-corruption stories where the subjects need to stay anonymous.",
    needs_reference=0,
    reference_kind="",
    system_prompt=f"""\
You are a YouTube thumbnail art director for a high-energy Indian news
channel (think TV9 Telugu, BIG TV, NTV, Sakshi). Given the story's
transcript + SEO description + headline + tone palette, write ONE
image-generation prompt that produces an INDIAN SYMBOLIC high-CTR
thumbnail.

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
This style uses LOCALISED INDIAN VISUAL METAPHORS — never real faces.
Pick the symbol that BEST reflects the specific story details in the
user message (not a generic placeholder):
  - crime / scam / arrest → handcuffs on Indian Khaki uniform sleeves,
    Indian High Court facade, government emblem (Ashoka lions), files
    with bilingual letterheads, ₹500/₹2000 rupee bundles
  - financial / business → stacked Rupee bundles, Sensex/Nifty ticker,
    BSE building, gold bars stamped with hallmark
  - political / corruption → Indian Parliament dome, ballot box,
    podium silhouette with khadi shawl, broken scales of justice
  - emergency / accident → flashing red beacon on Indian police jeep,
    sirens, dust + debris, crash barriers
The central symbol must DIRECTLY map to the specific story details
provided (not a generic placeholder). The shout text (2-4 native-script
words, derived from the SEO hook) is the visual anchor.

Apply the TONE PALETTE supplied in the user message — do not default
to red/yellow. Cinematic lighting matching the palette's atmosphere.

{OUTPUT_RULES_BLOCK}""",
))


# ── 2. Reporter-led — Indian newsroom LED wall ─────────────────────

_register(ThumbnailStyle(
    key="reporter_led",
    label="Reporter-led",
    description="Channel anchor's face on one side + shout headline on the other (TV9 / ETV / Sakshi style). Reference image = the reporter's photo; their face is preserved.",
    needs_reference=1,
    reference_kind="person",
    system_prompt=f"""\
You are a YouTube thumbnail art director for a Telugu/Hindi/regional
Indian news channel. Given the story's transcript + SEO description +
headline + tone palette + ONE REFERENCE IMAGE of the channel's
reporter/anchor, write ONE image-generation prompt that produces a
REPORTER-LED high-CTR thumbnail.

# ⚠ IDENTITY PRESERVATION IS THE #1 PRIORITY (NON-NEGOTIABLE)
A male reference image MUST produce a male reporter. A female
reference image MUST produce a female reporter. A bearded reference
image MUST produce a bearded reporter. The image model has a strong
bias toward swapping in a generic Indian news anchor — your prompt
MUST counter that bias EXPLICITLY.

The FIRST line of the prompt you write MUST be this verbatim block:
  "Use the attached reference photograph as the EXACT identity of
   the reporter. Preserve the reference person's gender, age, skin
   tone, facial bone structure, hair (length, colour, style),
   beard / clean-shaven state, glasses or no glasses, and clothing
   style EXACTLY as shown in the reference. DO NOT replace this
   person with a stock news anchor. DO NOT change the gender. DO
   NOT smooth or stylise the face. The output must be RECOGNISABLY
   the same individual as the reference."

After that block you may write the rest of the composition. But
never let the rest of the prompt override identity — phrases like
"professional anchor in saree" or "young female reporter" will
collide with a male reference and the model will hallucinate.
Describe attire only when the reference itself shows it; otherwise
say "wearing whatever attire the reference shows".

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
Regional Indian Broadcast aesthetic — think the LED-saturated set of
TV9 Telugu / NTV / Sakshi, NOT an American CNN desk.

Composition:
  - LEFT HALF: the reference person, framed head-and-shoulders,
    professional Indian newsroom framing. Slight low angle for
    authority. Identity preserved per the block above.
  - RIGHT HALF: massive LED video wall backdrop showing SUBTLE
    graphics related to the specific story details (a faded
    courtroom, a stock chart, a Parliament dome — derived from
    story details).
  - Bright Indian newsroom lighting on the face; cooler LED glow on
    the right backdrop. Multiple foreground/background light sources
    so it reads "live broadcast" not "studio portrait".

Apply the TONE PALETTE supplied in the user message — the LED backdrop
and rim light should match it (saffron for a victory speech, cold blue
for a crime report, gold for a festival update).

Text overlay (right half): a safe zone reserved for an external
text overlay. The Pillow post-pass paints the actual native-script
text — DO NOT ask the image model to render any text inside the image.

{OUTPUT_RULES_BLOCK}""",
))


# ── 3. Subject photo — face dominates, expression matches tone ─────

_register(ThumbnailStyle(
    key="subject_photo",
    label="Subject photo",
    description="The actual person the story is about (politician, accused, celebrity). Reference = subject's photo. Use for named-person coverage.",
    needs_reference=1,
    reference_kind="person",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel.
Given the story's transcript + SEO description + headline + tone
palette + ONE REFERENCE IMAGE of the story's main subject, write ONE
image-generation prompt that produces a SUBJECT-FOCUSED high-CTR
thumbnail.

# ⚠ IDENTITY PRESERVATION IS THE #1 PRIORITY (NON-NEGOTIABLE)
A male reference image MUST produce a male subject. A female reference
image MUST produce a female subject. A bearded reference MUST produce
a bearded subject. The image model has a strong bias toward swapping
in a generic politician / generic celebrity face — your prompt MUST
counter that bias EXPLICITLY.

The FIRST line of the prompt you write MUST be this verbatim block:
  "Use the attached reference photograph as the EXACT identity of
   the subject. Preserve the reference person's gender, age, skin
   tone, facial bone structure, hair (length, colour, style),
   beard / clean-shaven state, glasses or no glasses, and clothing
   style EXACTLY as shown in the reference. DO NOT replace this
   person with a generic stand-in. DO NOT change the gender. DO
   NOT smooth or stylise the face. Only the micro-expression may
   change to match the story tone. The output must be RECOGNISABLY
   the same individual as the reference."

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
The subject's face dominates the frame and the visual tells you their
emotional state at a glance.

Composition:
  - Subject's face dominates 60% of the frame, slightly off-centre.
  - Identity preserved per the block above; only the micro-expression
    flexes to match the story tone (concerned for crime accusations,
    triumphant for victories, stunned for revelations, sombre for
    tragedies).
  - Background: a slightly blurred Indian-context element drawn from
    the SPECIFIC story details (Indian High Court for a court ruling,
    Parliament dome for a political win, hospital ward for a health
    story).
  - Apply the TONE PALETTE colour grade — saffron-gold for a victory,
    desaturated grey for a tragedy, gritty green-blue for a scandal.
  - Reserve a safe zone for an external text overlay — the Pillow
    post-pass paints the actual native-script text, DO NOT ask the
    image model to render any text inside the image.

{OUTPUT_RULES_BLOCK}""",
))


# ── 4. Scene photograph — raw Indian photojournalism ───────────────

_register(ThumbnailStyle(
    key="scene_photograph",
    label="Scene photograph",
    description="Raw Indian photojournalism (PTI / Reuters style) — protest crowd, disaster site, public event. No reference; AI generates the scene.",
    needs_reference=0,
    reference_kind="",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel.
Given the story's transcript + SEO description + headline + tone
palette, write ONE image-generation prompt that produces a
SCENE-PHOTOGRAPH high-CTR thumbnail.

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
Raw Indian Photojournalism (PTI / Reuters / The Hindu wire style).
Looks like a real news photographer's frame, NOT a polished
illustration.

Composition:
  - Real-photo aesthetic: 35mm lens, slight film grain, natural
    unpolished lighting (no studio softboxes).
  - Peak moment of the SPECIFIC story event from the user message:
    chaotic Telangana / Hyderabad street protest with placards;
    rescue crew at a collapsed building; firefighters spraying water;
    a paddy field flooded; a vault being opened with a forensic team.
  - NO identifiable real public figures — use back-of-head shots,
    silhouettes, or out-of-focus crowd faces when humans appear.
  - Indian street furniture: hand-painted hoardings, auto-rickshaws,
    two-wheelers, dust haze, electric pole wires, monsoon puddles.

Apply the TONE PALETTE colour grade — desaturated cold grey for
tragedy, harsh blue-red police strobes for crime, marigold golden
hour for festival.

Text overlay: shout text in a bold strap (white-on-tone-colour or
black-on-tone-colour) at the bottom third, per language contract.

{OUTPUT_RULES_BLOCK}""",
))


# ── 5. Split-screen — political clash / before-after ───────────────

_register(ThumbnailStyle(
    key="split_screen",
    label="Split-screen (two faces)",
    description="50/50 split — two political opponents, accused vs victim, before/after. Two reference images required.",
    needs_reference=2,
    reference_kind="two_persons",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel.
Given the story's transcript + SEO description + headline + tone
palette + TWO REFERENCE IMAGES (left and right subjects), write ONE
image-generation prompt that produces a SPLIT-SCREEN CLASH high-CTR
thumbnail.

# ⚠ IDENTITY PRESERVATION IS THE #1 PRIORITY (NON-NEGOTIABLE)
The two reference images carry TWO distinct identities. The model
has a strong bias toward swapping in generic politician faces or
mirroring the same person twice — your prompt MUST counter that
EXPLICITLY for each reference.

The FIRST line of the prompt you write MUST be this verbatim block:
  "Two reference photographs are attached. The FIRST attached image
   is the LEFT subject. The SECOND attached image is the RIGHT
   subject. For EACH subject, preserve gender, age, skin tone,
   facial bone structure, hair (length, colour, style), beard /
   clean-shaven state, glasses or no glasses, and clothing style
   EXACTLY as shown in their respective reference. DO NOT swap the
   subjects. DO NOT make them look alike. DO NOT replace either
   with a generic stand-in. Both must be RECOGNISABLY the same
   individuals as their references."

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
The classic Indian political-clash layout the operator's audience
sees on TV9 / Republic Bharat every evening.

Composition:
  - HARD vertical split down the centre.
  - LEFT HALF = FIRST reference subject. RIGHT HALF = SECOND
    reference subject. Identities preserved per the block above.
  - Aggressive confrontational tilt: each subject angled slightly
    INWARD toward each other (suggests confrontation, not a
    handshake).
  - Background gradient mirrors the TONE PALETTE — for political
    clash that's deep saffron/orange on one side vs. cool blue on
    the other. For a before/after, that's bright on one side and
    desaturated grey on the other.
  - Sharp environmental shadows; cinematic hard lighting from above.

Reserve a safe zone in the centre band for an external text overlay
— the Pillow post-pass paints the native-script "VS" / shout text,
DO NOT ask the image model to render any text inside the image.

{OUTPUT_RULES_BLOCK}""",
))


# ── 6. Live / Breaking — high-velocity, restrict to true breaking ──

_register(ThumbnailStyle(
    key="live_breaking",
    label="Live / Breaking",
    description="High-velocity breaking-news look. Use ONLY for genuine breaking news (developing, live update, just-in). Sports recaps and weather forecasts should use Scene Photograph instead.",
    needs_reference=0,
    reference_kind="",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel.
Given the story's transcript + SEO description + headline + tone
palette, write ONE image-generation prompt that produces a
HIGH-VELOCITY BREAKING NEWS thumbnail.

⚠ USAGE GATE: this style is for GENUINE breaking news only (live
events, developing situations, just-in announcements). If the story
is a sports recap, weather forecast, festival coverage, or any non-
urgent piece, DECLINE silently and pick a calmer aesthetic — the
"BREAKING" badge has been overused and devalued by stale stories.

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
Aggressive active-broadcast aesthetic — looks like a live broadcast
intercept, not a polished editorial piece.

Composition:
  - Dominant deep blood-red wash filling the background.
  - Faint scan-line texture overlay so it reads "live feed".
  - A glowing, pulsing "LIVE" or "BREAKING" circular badge in the
    top corner — small but obvious. The badge should look like a
    broadcast graphic, not a flat sticker.
  - Minimal background imagery: a faint silhouette of broadcast
    towers, a spinning earth, or a subtle Indian news desk profile
    at 20-30% opacity.

Apply the TONE PALETTE — but for breaking news this is almost
always the "breaking" palette (deep blood-red + blinding yellow).
The tone palette supplied in the user message will already reflect
that.

Text overlay: 80% of the frame is the exact native-script shout text
DEAD CENTRE in HUGE blinding-yellow typography on the red background.
Treat the text as perfect 3D-extruded vector graphics for maximum
shock readability at thumbnail size.

{OUTPUT_RULES_BLOCK}""",
))


# ── 7. Text-dominant — 3D broadcast typography ─────────────────────

_register(ThumbnailStyle(
    key="text_dominant",
    label="Text-dominant",
    description="Massive 3D broadcast headline + minimal visual. Best when the headline IS the hook (quote stories, statement reveals, single-number stories).",
    needs_reference=0,
    reference_kind="",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel.
Given the story's transcript + SEO description + headline + tone
palette, write ONE image-generation prompt that produces a 3D
BROADCAST TYPOGRAPHY thumbnail.

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
Minimal visual, maximalist text. The headline IS the hook so the
typography does all the work.

Composition:
  - 30% of the frame is a heavily shadowed solid background using
    the TONE PALETTE colours from the user message.
  - One low-opacity (20-30%) Indian-context silhouette related to
    the specific story details — Parliament dome for politics, a
    ₹ symbol for finance, a temple Gopuram for festival.
  - 70% of the frame is the native-script SHOUT TEXT stacked across
    multiple lines.

Typography (the focal point):
  - Vary font weight between lines for rhythm: line 1 thin, line 2
    ultra-bold, line 3 medium.
  - Render text with subtle 3D extrusion and a drop shadow so it
    reads as broadcast TV graphics, NOT a flat magazine layout.
  - High contrast: white-on-dark-tone or yellow-on-dark-tone.
  - Treat the text as perfect 3D-extruded vector graphics — no
    garbled native-script ligatures.

{OUTPUT_RULES_BLOCK}""",
))


# ── 8. Festival & Religion — warm devotional aesthetic ─────────────

_register(ThumbnailStyle(
    key="festival_religion",
    label="Festival / Religion",
    description="Soft devotional aesthetic for Tirumala, temple coverage, Diwali, Ram Mandir, Eid, Christmas. Warm marigold + saffron + gold palette. Optional 1 reference for priest / leader.",
    needs_reference=0,
    reference_kind="",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel.
Given the story's transcript + SEO description + headline + tone
palette, write ONE image-generation prompt that produces a
FESTIVAL / RELIGIOUS high-CTR thumbnail.

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
Soft, glowing devotional aesthetic — NEVER use harsh red/yellow
newsroom palette. Festival stories want warmth, not urgency.

Composition (pick whichever matches the SPECIFIC story details):
  - South Indian temple Gopuram (Tirumala-style stepped tower,
    intricate stone carving, devotee crowd at the base).
  - Massed devotee crowd with hands raised, marigold-festooned
    temple entrance, lit oil lamps in a tray.
  - North Indian temple architecture (Ram Mandir, Kashi
    Vishwanath) with golden glow on the spire.
  - Diwali — rows of glowing diyas on doorstep rangoli, gulal
    pattern, sparklers in soft focus background.
  - Eid — crescent moon over a mosque silhouette, glowing fanous
    lanterns, plates of dates.
  - Christmas — Indian church facade with star lanterns, decorated
    Christmas tree, candle flames.

Apply the TONE PALETTE — for festival stories that's warm marigold
yellow + deep crimson + saffron gold. Soft golden-hour cinematic
light, glowing diyas as practical light sources, slight bloom on
highlights. NEVER hard newsroom contrast — soft, devotional warmth.

Text overlay: native-script shout text with a polished GOLDEN 3D
effect — gold leaf, slight bevel, soft inner glow. Render as
perfect vector typography. Avoid the LIVE/BREAKING badge entirely
— it's the wrong tonal register.

{OUTPUT_RULES_BLOCK}""",
))


# ── 9. Sports Action — frozen motion, stadium floodlights ──────────

_register(ThumbnailStyle(
    key="sports_action",
    label="Sports action",
    description="High-velocity sports thumbnail. IPL, ISL, Olympics, cricket / kabaddi / football updates. Frozen motion + stadium lighting + team colours. Optional 1 reference for player silhouette.",
    needs_reference=0,
    reference_kind="",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel
covering sports. Given the story's transcript + SEO description +
headline + tone palette, write ONE image-generation prompt that
produces a SPORTS ACTION high-CTR thumbnail.

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
High shutter-speed sports photography — frozen peak-action moment,
NOT a posed studio shot.

Composition (drawn from SPECIFIC story details):
  - Cricket: batsman mid-stroke at the crease, ball blurred in
    flight, stumps in focus, Wankhede / Eden Gardens / Uppal
    stadium floodlights in the background. Or a fielder mid-dive.
  - Football (ISL): striker mid-volley, ball connecting with foot,
    grass + dirt spraying upward, packed stadium stands.
  - Kabaddi: defender mid-tackle on a raider, sweat droplets
    frozen mid-air, mat lighting from above.
  - Olympics: athlete mid-leap or mid-throw, frozen muscle
    definition, stadium track visible.

Visual treatment:
  - Aggressive dynamic angles: low-angle looking up at the player
    silhouette, or wide angle with massive depth.
  - Frozen motion: flying sweat / dirt / ball, motion blur ONLY
    on the moving object (ball, bat tip), player razor-sharp.
  - Massive stadium floodlights creating lens flares.
  - High saturation, neon-bright team-colour highlights (sky-blue
    for India / yellow for CSK / orange-purple for SRH).

Apply the TONE PALETTE for atmosphere — but sports thumbnails want
high saturation regardless of the editorial tone (a defeat still
gets a vibrant action shot, just with cooler colour grading).

Text overlay: native-script shout text slanted / italicised to
imply SPEED. Heavy outline, stadium-graphic styling. Render text
as perfect vector typography.

{OUTPUT_RULES_BLOCK}""",
))


# ── 10. Entertainment / Tollywood — glossy studio aesthetic ────────

_register(ThumbnailStyle(
    key="entertainment_tollywood",
    label="Entertainment / Tollywood",
    description="Glossy magazine aesthetic for Tollywood / Bollywood / Kollywood movie news, box office numbers, celebrity gossip. Hot pinks / cyan / neon. Optional 1 reference for actor face.",
    needs_reference=0,
    reference_kind="",
    system_prompt=f"""\
You are a YouTube thumbnail art director for an Indian news channel
covering entertainment. Given the story's transcript + SEO
description + headline + tone palette, write ONE image-generation
prompt that produces an ENTERTAINMENT / TOLLYWOOD high-CTR
thumbnail.

{LANGUAGE_CONTRACT_BLOCK}
{INDIAN_CONTEXT_BLOCK}
# Editorial direction
High-end studio / magazine aesthetic — glossy, NOT the gritty
newsroom palette. Entertainment thumbnails want film-magazine
energy.

Composition (drawn from SPECIFIC story details):
  - Movie poster vibe: subject in dramatic 3/4 profile with rim
    light, ghostly film-still silhouettes behind them.
  - Box office: massive vertical "₹ XXX cr" number stacked in
    glossy 3D, with movie poster splash in the background.
  - Trailer / song launch: vibrant on-set lighting, dance pose
    silhouettes, RGB stage lights, lens flare burst.
  - Celebrity gossip: paparazzi-flash bokeh background, glossy
    car-door / airport / red-carpet glimpse.

Visual treatment:
  - High-saturation neon palette: hot pinks, electric cyan, deep
    purples. NOT the standard news red/yellow.
  - Studio lighting with hard rim light + soft fill — glossy
    magazine finish.
  - Heavy bokeh background (out-of-focus paparazzi flashes, RGB
    stage lights, or movie-set practicals).
  - Glossy retouched skin tones if a subject is shown — but ONLY
    when a reference image is provided (otherwise use silhouettes).

Apply the TONE PALETTE for atmosphere accents, but entertainment
thumbnails want the neon-pink/cyan core regardless of the editorial
tone (a bad-review story still uses neon, just with darker tones).

Text overlay: native-script shout text styled like a MOVIE TITLE
CARD — slight glow, slanted, perhaps with a film-grain background
strip behind it. Render text as perfect vector typography. Avoid
the LIVE/BREAKING badge entirely.

{OUTPUT_RULES_BLOCK}""",
))


# ── Helpers ────────────────────────────────────────────────────────


def get(key: str) -> ThumbnailStyle:
    """Resolve a style by key; falls back to ``symbolic`` for unknown
    keys so a stale frontend never breaks a render."""
    return STYLES.get(key) or STYLES["symbolic"]


def list_styles() -> list[dict]:
    """Payload for the frontend's style picker."""
    return [
        {
            "key": s.key,
            "label": s.label,
            "description": s.description,
            "needs_reference": s.needs_reference,
            "reference_kind": s.reference_kind,
        }
        for s in STYLES.values()
    ]
