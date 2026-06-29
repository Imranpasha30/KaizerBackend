"""Few-shot examples bank — exemplar ScenePlans + matching prompts.

Each entry pairs a (style, news context) tuple with a hand-crafted
example ScenePlan and the Nano Banana prompt that should result.
Showing Gemini concrete examples of the shape we want — rather than
just describing it — pulls the output quality up dramatically.

Curated for Telugu / Hindi / regional Indian news (TV9, NTV, BIG TV,
ABN Andhra Jyothy, Sakshi, ETV Bharat, Republic Bharat).

How to add an example (content / curator team):
  1. Pick a real story you've published thumbnails for
  2. Reverse-engineer the ScenePlan that should have been generated
  3. Write the Nano Banana prompt that you'd want from that plan
  4. Append a new dict to EXAMPLES for the matching style
  5. The planner samples up to N entries per style at random for the
     few-shot pool — order doesn't matter, count does
"""
from __future__ import annotations


# Each entry has:
#   style:       which style this exemplifies
#   input:       the news context as the planner would see it
#   plan:        the desired ScenePlan (as a dict — schema in schemas.py)
#   prompt:      the desired Nano Banana prompt for the plan
EXAMPLES: list[dict] = [
    # ── symbolic (crime/scam) ──────────────────────────────────────
    {
        "style":  "symbolic",
        "input":  "Hyderabad land scam: ₹500 crore in fake registration "
                  "documents, ED raids 5 properties in Banjara Hills.",
        "plan": {
            "primary_subject":  "Stacked bundles of ₹500 Indian Rupee notes wrapped in paper bands",
            "scene_location":   "Indian government-office desk with bilingual letterheads visible underneath",
            "key_action":       "Files being laid out beside the cash bundles — investigation in progress",
            "supporting_props": [
                "an Ashoka pillar government emblem stamp on a file cover",
                "evidence bag tags with serial numbers",
                "a wooden gavel partially visible",
                "stack of red ribbon-tied folders",
            ],
            "lighting":         "harsh overhead office fluorescent, deep cast shadows under the cash",
            "color_grade":      "gritty green-cyan office tone with stark blacks under the cash bundles",
            "framing":          "medium close-up, slight high angle looking down on the desk",
            "emotional_hook":   "the eye-watering scale of cash makes you want to know who took it",
            "what_to_avoid":    "no US dollar bills, no US courthouse, no American police badge, no human faces",
            "safe_zone":        "bottom 28% strap — visually quieter for text overlay",
            "text_for_overlay": "షాకింగ్! 500 కోట్ల కుంభకోణం",
        },
        "prompt": (
            "Indian forensic investigation desk photograph, 16:9 cinematic. "
            "Centre frame: tightly stacked bundles of ₹500 Indian Rupee notes "
            "wrapped in paper bands, slightly tilted into the foreground. "
            "Beside the cash: red ribbon-tied government folders, an evidence "
            "bag with a serial-number tag, a wooden gavel partially out of "
            "focus, an Ashoka pillar government emblem stamped on the topmost "
            "folder. Bilingual English-Hindi government letterheads peek out "
            "from beneath. Harsh overhead office fluorescent lighting, deep "
            "cast shadows under each bundle. Gritty green-cyan colour grade "
            "with stark black shadows. Medium close-up, slight high angle "
            "looking down at the desk. NO human faces. NO American visual "
            "vocabulary (no US dollar bills, no US badge, no US courthouse). "
            "NO logos, NO watermarks, NO text rendered inside the image — "
            "the native-script shout text is added externally. Leave the "
            "bottom 28% of the frame visually quieter as a safe zone for "
            "an external text overlay."
        ),
    },

    # ── reporter_led (anchor with reference) ───────────────────────
    {
        "style":  "reporter_led",
        "input":  "Telangana CM Revanth Reddy announces ₹2 lakh crore "
                  "railway investment plan in Hyderabad.",
        "plan": {
            "primary_subject":  "The reference photograph reporter (identity preserved exactly), framed head-and-shoulders, slight forward lean",
            "scene_location":   "Indian newsroom set with a massive curved LED video wall covering the right two-thirds of the background",
            "key_action":       "Reporter mid-speech, confident slight upward gaze, hand gesture implied just out of frame",
            "supporting_props": [
                "LED video wall showing a faded composite of Hyderabad railway map + a Parliament-dome silhouette",
                "subtle Indian newsroom anchor desk edge at the very bottom",
                "small overhead boom microphone hint in upper-left corner",
            ],
            "lighting":         "bright Indian newsroom key light from above-left on the reporter's face, cool LED glow on the right backdrop",
            "color_grade":      "bright gold and pure white with saffron rim light — political-victory tone palette",
            "framing":          "medium close-up of head-and-shoulders, reporter occupies LEFT 50% of frame",
            "emotional_hook":   "live-broadcast authority — the reporter is delivering news that matters",
            "what_to_avoid":    "do not change the reference person's gender / face / age / hair; do not use a US CNN-style anchor desk; do not add other people",
            "safe_zone":        "right half (40-50% of frame) — for the text overlay",
            "text_for_overlay": "2 లక్షల కోట్లు! రైల్వే ప్రణాళిక",
        },
        "prompt": (
            "Use the attached reference photograph as the EXACT identity of "
            "the reporter. Preserve gender, age, skin tone, facial bone "
            "structure, hair, beard / clean-shaven state, glasses, and "
            "clothing EXACTLY as shown — DO NOT change the gender, DO NOT "
            "replace with a stock anchor, the output must be recognisably "
            "the same individual. "
            "16:9 Indian regional news broadcast photograph. LEFT 50% of the "
            "frame: the reference reporter framed head-and-shoulders, slight "
            "forward lean, confident upward gaze, mid-speech. RIGHT 50%: a "
            "massive curved LED video wall showing a faded composite of a "
            "Hyderabad railway map overlaid with a Parliament-dome silhouette, "
            "kept visually quieter as a safe zone for an external text "
            "overlay. Bright Indian newsroom key light from above-left on "
            "the reporter's face; cool LED glow on the right backdrop. A "
            "small overhead boom microphone hint in the upper-left corner. "
            "Indian anchor desk edge barely visible at the very bottom. "
            "Bright gold and pure white with saffron rim light — political-"
            "victory tone palette. NO logos, NO watermarks, NO text rendered "
            "inside the image."
        ),
    },

    # ── scene_photograph (tragedy) ─────────────────────────────────
    {
        "style":  "scene_photograph",
        "input":  "Krishna river floods Vijayawada — 12 dead, Andhra CM "
                  "tours disaster zone, helicopter rescues ongoing.",
        "plan": {
            "primary_subject":  "Indian rescue boat with NDRF personnel pulling a soaked elderly woman aboard from chest-deep brown floodwater",
            "scene_location":   "Submerged Vijayawada residential street — only the upper floors of small two-storey houses + a corner Telugu hand-painted hoarding visible above water",
            "key_action":       "Rescue worker mid-pull, water spraying off the woman's saree as she's lifted",
            "supporting_props": [
                "orange NDRF jackets",
                "partly submerged auto-rickshaw with only its yellow top visible",
                "a hand-painted Telugu shop hoarding peeking above water",
                "monsoon-grey sky with a hint of helicopter blur upper-right",
            ],
            "lighting":         "overcast diffuse monsoon daylight, cold flat top light, no hard shadows",
            "color_grade":      "desaturated slate-grey and cold cyan with hints of NDRF orange — tragedy tone palette",
            "framing":          "medium-wide eye-level photojournalism shot, 35mm grain, slight motion blur on the water spray",
            "emotional_hook":   "the rescue itself — viewer wants to know if the woman survived and how many more are stranded",
            "what_to_avoid":    "no identifiable real public figures, no US fire trucks, no clean studio lighting, no polished illustration look",
            "safe_zone":        "bottom third — darker / quieter for white-on-red shout text strap",
            "text_for_overlay": "విషాదం! 12 మంది మృతి",
        },
        "prompt": (
            "Raw Indian photojournalism in the PTI / Reuters wire style, "
            "16:9, 35mm lens with visible grain, no studio polish. "
            "Submerged Vijayawada residential street: only the upper floors "
            "of small two-storey houses and the corner of a Telugu hand-"
            "painted shop hoarding rise above the brown chest-deep "
            "floodwater. Centre frame: an Indian rescue boat with three "
            "orange-jacketed NDRF personnel pulling a soaked elderly woman "
            "aboard from the water; water spraying off her saree, captured "
            "mid-motion with light motion blur on the spray. The corner of "
            "a partly submerged auto-rickshaw (only its yellow top visible) "
            "in the foreground. A monsoon-grey overcast sky with the "
            "faintest helicopter blur in the upper-right corner. Overcast "
            "diffuse daylight from above, cold flat top light, no hard "
            "shadows. Desaturated slate-grey and cold cyan colour grade "
            "with hints of NDRF orange — tragedy tone palette. Medium-wide "
            "eye-level photojournalism framing. NO identifiable real public "
            "figures (use silhouettes or out-of-focus crowd faces if more "
            "humans appear). NO US fire trucks, NO US-style emergency "
            "vehicles. NO logos, NO watermarks, NO text rendered inside "
            "the image. Leave the bottom third visually quieter and slightly "
            "darker as a safe zone for an external text overlay."
        ),
    },

    # ── festival_religion ──────────────────────────────────────────
    {
        "style":  "festival_religion",
        "input":  "Tirumala Brahmotsavam day 3 — record 1.5 lakh devotees, "
                  "Sri Venkateswara procession on Garudavahanam.",
        "plan": {
            "primary_subject":  "Sri Venkateswara deity on the golden Garudavahanam (eagle mount) procession vehicle, draped in marigold-and-tulasi garlands",
            "scene_location":   "Tirumala temple street at dusk — the stone Gopuram tower partially visible behind, devotee crowd pressing inward",
            "key_action":       "Procession mid-motion, garlands swaying, devotees raising hands and folded palms",
            "supporting_props": [
                "thousands of glowing oil-lamp diyas along the temple street",
                "marigold flower garlands strung overhead",
                "temple flagpoles with saffron pennants",
                "a faint hint of camphor smoke and incense haze",
            ],
            "lighting":         "warm golden-hour dusk with practical light from thousands of diyas creating warm specular highlights",
            "color_grade":      "marigold orange, deep crimson, saffron gold — festival tone palette with soft bloom on highlights",
            "framing":          "wide three-quarter angle, slight low angle looking up toward the deity for reverence",
            "emotional_hook":   "the devotional spectacle invites the viewer into the experience — feels like being there",
            "what_to_avoid":    "no harsh newsroom red/yellow palette; no BREAKING badge; no crime-scene mood; no identifiable real faces — devotees are a massed silhouette",
            "safe_zone":        "bottom 28% strap — slightly darker for gold-on-crimson text overlay",
            "text_for_overlay": "తిరుమల! 1.5 లక్షల భక్తులు",
        },
        "prompt": (
            "Devotional Indian temple-procession cinematic photograph, 16:9. "
            "Centre frame slightly elevated: the Sri Venkateswara deity on "
            "the golden Garudavahanam (eagle-mount) procession vehicle, "
            "draped in thick marigold-and-tulasi garlands, captured mid-"
            "motion with the garlands gently swaying. The Tirumala temple "
            "Gopuram tower rises behind in the haze; the procession is "
            "moving down a stone temple street lined with thousands of "
            "glowing oil-lamp diyas creating warm specular highlights. "
            "Devotees in the foreground (back-of-head silhouettes only — "
            "no identifiable faces) raise hands and folded palms. Marigold "
            "garlands strung overhead, saffron pennants on temple flagpoles, "
            "a faint hint of camphor smoke and incense haze. Warm golden-"
            "hour dusk light blended with the diya glow. Marigold orange, "
            "deep crimson, and saffron gold colour palette — festival tone, "
            "with soft bloom on the highlights. Wide three-quarter angle, "
            "slight low angle looking up toward the deity for reverence. "
            "NO BREAKING badge, NO harsh newsroom red/yellow palette, NO "
            "logos, NO watermarks, NO text rendered inside the image. "
            "Leave the bottom 28% visually quieter as a safe zone for an "
            "external text overlay."
        ),
    },
]


def examples_for_style(style: str, max_count: int = 2) -> list[dict]:
    """Return up to ``max_count`` few-shot examples for ``style``.
    The planner uses these as in-context priming.

    When no exemplars exist for the requested style, returns the most
    style-adjacent ones we have — better to anchor on a near-match
    than to send Gemini no examples at all.
    """
    own = [e for e in EXAMPLES if e["style"] == style][:max_count]
    if own:
        return own
    # Fallbacks by "shape":
    #   subject_photo / reporter_led / split_screen → reporter_led
    #   text_dominant / live_breaking               → symbolic
    #   sports_action / entertainment_tollywood     → scene_photograph
    fallback_chain = {
        "subject_photo":           ["reporter_led", "symbolic"],
        "reporter_led":            ["symbolic"],
        "split_screen":            ["reporter_led", "scene_photograph"],
        "text_dominant":           ["symbolic"],
        "live_breaking":           ["symbolic"],
        "sports_action":           ["scene_photograph"],
        "entertainment_tollywood": ["scene_photograph"],
        "festival_religion":       ["scene_photograph", "symbolic"],
    }.get(style, ["symbolic"])
    for alt in fallback_chain:
        matches = [e for e in EXAMPLES if e["style"] == alt][:max_count]
        if matches:
            return matches
    return EXAMPLES[:max_count]


__all__ = ["EXAMPLES", "examples_for_style"]
