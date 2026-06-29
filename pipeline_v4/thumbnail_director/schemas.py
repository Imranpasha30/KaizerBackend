"""ScenePlan — the structured contract between Pass 1 (planner) and
Pass 2 (prompter).

Pass 1 produces a ScenePlan. Pass 2 reads a ScenePlan and writes a
Nano Banana prompt from it. By making the plan a stable schema, the
two teams can evolve their halves independently — a planner-team
change to *what* gets decided doesn't break the prompter-team's
prompt-engineering pass, and vice versa.

The schema mirrors what a human art director would write on a
storyboard whiteboard: WHO is in the frame, WHERE it's happening,
WHAT they're doing, HOW it's lit, WHY the viewer should click.

Field reference:
  primary_subject   — what dominates the frame (the central focal point)
  scene_location    — where the action is set (specific Indian context)
  key_action        — what is literally occurring (verb phrase)
  supporting_props  — 3-5 specific objects that ground the scene
  lighting          — source / direction / time-of-day
  color_grade       — colour palette (derived from tone palette)
  framing           — wide / medium / close-up + angle
  emotional_hook    — what makes the viewer eager to click
  what_to_avoid     — common drift patterns the prompter must counter
  safe_zone         — where the external text overlay will sit
                      (so the AI plate leaves that area visually quiet)
  text_for_overlay  — the native-script shout text (informational —
                      we paint it via Pillow, not via the image model)

Pass 1 fills every field. Empty strings are valid only when a field is
genuinely not applicable (e.g. ``key_action="static portrait"`` for a
text-dominant style). Use ``ScenePlan.is_complete()`` to verify a
returned plan has the minimum signal needed for Pass 2 to succeed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ScenePlan:
    """Structured editorial plan for a single thumbnail."""
    primary_subject:  str = ""
    scene_location:   str = ""
    key_action:       str = ""
    supporting_props: list[str] = field(default_factory=list)
    lighting:         str = ""
    color_grade:      str = ""
    framing:          str = ""
    emotional_hook:   str = ""
    what_to_avoid:    str = ""
    safe_zone:        str = ""
    text_for_overlay: str = ""

    # Metadata — useful for logging + downstream debugging. Not asked
    # of the LLM, populated by the planner from inputs.
    style:            str = ""
    language:         str = ""
    has_references:   int = 0

    def is_complete(self) -> bool:
        """True iff the plan has enough signal for Pass 2 to produce a
        usable prompt. The bare minimum is: a primary subject, some
        location/scene context, and one of (action / hook) to drive
        composition. Other fields are nice-to-have."""
        has_subject = bool((self.primary_subject or "").strip())
        has_context = bool((self.scene_location or "").strip())
        has_intent  = bool((self.key_action or "").strip()
                            or (self.emotional_hook or "").strip())
        return has_subject and has_context and has_intent

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prose(self) -> str:
        """Last-resort fallback when Pass 2 fails: serialise the plan
        into a single paragraph that Nano Banana can still parse.
        Order matches what an image-gen model attends to most heavily:
        subject → location → action → props → lighting → mood.
        """
        bits: list[str] = []
        if self.primary_subject:
            bits.append(self.primary_subject)
        if self.scene_location:
            bits.append(f"in {self.scene_location}")
        if self.key_action:
            bits.append(self.key_action)
        if self.supporting_props:
            bits.append("featuring " + ", ".join(self.supporting_props[:5]))
        if self.lighting:
            bits.append(f"lit with {self.lighting}")
        if self.color_grade:
            bits.append(f"colour graded as {self.color_grade}")
        if self.framing:
            bits.append(self.framing)
        if self.emotional_hook:
            bits.append(f"to convey {self.emotional_hook}")
        if self.safe_zone:
            bits.append(
                f"Reserve {self.safe_zone} as a visually quiet safe zone "
                "for an external text overlay."
            )
        if self.what_to_avoid:
            bits.append(f"Avoid: {self.what_to_avoid}")
        bits.append(
            "Photographic Indian news-broadcast aesthetic, 16:9, sharp focus, "
            "cinematic lighting, no logos, no watermarks."
        )
        bits.append(
            "Do NOT render any text, captions, headlines, words, or "
            "typography inside the image — the native-script shout text "
            "is added externally."
        )
        return " ".join(bits)


# JSON schema fragment — used by planner.py to enforce Gemini's
# structured output. Keeping it here so schemas.py is the single
# source of truth for the contract shape.
SCENEPLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "primary_subject":  {"type": "string"},
        "scene_location":   {"type": "string"},
        "key_action":       {"type": "string"},
        "supporting_props": {"type": "array", "items": {"type": "string"}},
        "lighting":         {"type": "string"},
        "color_grade":      {"type": "string"},
        "framing":          {"type": "string"},
        "emotional_hook":   {"type": "string"},
        "what_to_avoid":    {"type": "string"},
        "safe_zone":        {"type": "string"},
        "text_for_overlay": {"type": "string"},
    },
    "required": [
        "primary_subject",
        "scene_location",
        "key_action",
        "supporting_props",
        "lighting",
        "color_grade",
        "framing",
        "emotional_hook",
        "safe_zone",
    ],
}


__all__ = ["ScenePlan", "SCENEPLAN_JSON_SCHEMA"]
