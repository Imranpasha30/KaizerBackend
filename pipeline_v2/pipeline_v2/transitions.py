"""Item 104 -- Transition catalog for V2 bulletin stitching.

Defines the user-selectable list of inter-clip transitions exposed
on the V2 wizard. ``smart_cut`` is the default and the only one
guaranteed to be rendered with no fallback. The other six entries
are reserved slots: the UI shows them with a "Coming soon" badge
and selecting them falls back to ``smart_cut`` at render time with
a structured log warning (NEVER silent -- the operator must see
that their selection didn't take effect).

Design rationale:
  - We ship the SELECTION plumbing now (Job.transition_style column,
    event payload field, Stage4Render constructor kwarg, NewJob.jsx
    dropdown) so the user can pin a future transition choice on a
    job today. The actual ffmpeg pipelines for the 6 non-default
    transitions land in subsequent items as each one is requested.
  - The catalog is the single source of truth: UI dropdown labels,
    backend validation, and Stage 4 dispatch all import from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Transition:
    """One row in the transition catalog.

    Fields:
      - ``name``:        Stable machine key. Stored in Job.transition_style.
      - ``display_name``: Operator-facing dropdown label.
      - ``description``: Short helper text shown under the dropdown.
      - ``duration_s``:  Overlap window between the outgoing and
        incoming clip. 0.0 == hard cut (no overlap, no re-encode).
      - ``implemented``: True if a real ffmpeg pipeline exists today;
        False entries fall back to ``smart_cut`` at render time.
    """

    name: str
    display_name: str
    description: str
    duration_s: float
    implemented: bool


# ---- The 7 catalog entries ------------------------------------------

SMART_CUT = Transition(
    name="smart_cut",
    display_name="Smart Cut",
    description=(
        "80ms audio crossfade at every splice; video uses hard cuts "
        "(imperceptible at 30fps). Smooths splice-point audio "
        "spikes without a visible video transition. Default for "
        "news bulletins (item 108/111)."
    ),
    duration_s=0.08,
    implemented=True,
)

CROSSFADE = Transition(
    name="crossfade",
    display_name="Crossfade",
    description=(
        "500ms audio crossfade at every splice; video uses hard "
        "cuts. Pronounced audio blend suited to feature-style "
        "bulletins (item 108/111)."
    ),
    duration_s=0.5,
    implemented=True,
)

FADE_TO_BLACK = Transition(
    name="fade_to_black",
    display_name="Fade to Black",
    description=(
        "Fade out to black + fade in (~0.6s total). "
        "Cinematic, suits long-form stories."
    ),
    duration_s=0.6,
    implemented=False,
)

DIP_TO_WHITE = Transition(
    name="dip_to_white",
    display_name="Dip to White",
    description=(
        "Fade out to white + fade in (~0.6s total). "
        "Bright editorial feel."
    ),
    duration_s=0.6,
    implemented=False,
)

SLIDE_LEFT = Transition(
    name="slide_left",
    display_name="Slide Left",
    description=(
        "Outgoing clip slides off to the left as the incoming "
        "slides in from the right (~0.4s)."
    ),
    duration_s=0.4,
    implemented=False,
)

WIPE_RIGHT = Transition(
    name="wipe_right",
    display_name="Wipe Right",
    description=(
        "Vertical wipe revealing the incoming clip from left to "
        "right (~0.4s)."
    ),
    duration_s=0.4,
    implemented=False,
)

DISSOLVE = Transition(
    name="dissolve",
    display_name="Dissolve",
    description=(
        "Longer alpha dissolve (~1.0s). Soft, documentary-style "
        "blend between clips."
    ),
    duration_s=1.0,
    implemented=False,
)


# Catalog dict + ordered list (the UI dropdown renders in this order).
TRANSITIONS_ORDERED: tuple[Transition, ...] = (
    SMART_CUT,
    CROSSFADE,
    FADE_TO_BLACK,
    DIP_TO_WHITE,
    SLIDE_LEFT,
    WIPE_RIGHT,
    DISSOLVE,
)

TRANSITIONS: dict[str, Transition] = {t.name: t for t in TRANSITIONS_ORDERED}

DEFAULT_TRANSITION_NAME: str = "smart_cut"


def get_transition(name: Optional[str]) -> Transition:
    """Lookup a transition by name. Falls back to the default.

    Falling back is the right choice for renderer-side calls: an
    unknown / blank ``Job.transition_style`` (legacy rows pre-item-104
    have NULL) must not crash the render. Callers that need strict
    validation use ``is_valid_transition`` before storing the value.
    """
    if not name:
        return TRANSITIONS[DEFAULT_TRANSITION_NAME]
    return TRANSITIONS.get(name, TRANSITIONS[DEFAULT_TRANSITION_NAME])


def is_valid_transition(name: str) -> bool:
    """Strict catalog membership check. Used by the create-job
    endpoint to reject typos before they hit the DB."""
    return name in TRANSITIONS


def resolve_for_render(name: Optional[str]) -> Transition:
    """Renderer-side resolver: falls back to ``smart_cut`` for any
    non-implemented selection AND for unknown/blank names.

    Distinct from ``get_transition`` only when the named transition
    exists in the catalog but ``implemented=False`` -- callers can
    use this to get a guaranteed-renderable transition without
    losing the operator's original choice (which stays on the Job
    row for UI display + future re-render).
    """
    sel = get_transition(name)
    if sel.implemented:
        return sel
    return TRANSITIONS[DEFAULT_TRANSITION_NAME]


def overlap_for_render(name: Optional[str]) -> tuple[float, float]:
    """Return ``(audio_overlap_s, video_overlap_s)`` for a transition.

    Used by Stage 4 to dispatch into the crossfade stitcher.

    Item 111 update: video overlap is now ALWAYS 0.0 (hard cut at
    every splice). ffmpeg's xfade filter does not chain reliably
    for 20+ video transitions; the audio acrossfade chain works
    fine. The 3-pass stitcher concat-demuxes video losslessly and
    only acrossfades audio. Hard video cuts at 80ms boundaries are
    visually imperceptible.

      - smart_cut -> (0.08, 0.0)    80ms audio, hard-cut video
      - crossfade -> (0.50, 0.0)    half-second audio blend
      - any other implemented entry: audio = ``Transition.duration_s``,
        video = 0.0
      - non-implemented (resolves to smart_cut) -> (0.08, 0.0)

    Returns ``(0.0, 0.0)`` only when the resolver returns a
    Transition with ``duration_s == 0`` (a "true hard cut" entry,
    none in the catalog today but reserved for future fast-cut
    variants).
    """
    sel = resolve_for_render(name)
    if sel.name == "smart_cut":
        return 0.08, 0.0
    if sel.name == "crossfade":
        return 0.50, 0.0
    if sel.duration_s <= 0:
        return 0.0, 0.0
    return float(sel.duration_s), 0.0
