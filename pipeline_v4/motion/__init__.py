"""Native motion toolkit — Kaizer's own animation engine.

Ported verbatim from kaizer-fx (kaizer_fx/compositor/anim.py), which
reimplemented the whole motion language — 24 easing curves, physics
springs, interpolate(), directional motion blur — from motion-design
first principles in pure Python/NumPy. 100%% self-owned: no Remotion,
no third-party motion framework, nothing to license or attribute
(see kaizer-fx THIRD_PARTY.md, "Rendering engine — 100%% native").

Consumers: the eased lower-third entrance and the story-transition
stitcher today; kinetic typography, animated straps/charts and camera
moves as the effect families migrate in.
"""
from pipeline_v4.motion.anim import (   # noqa: F401
    EASINGS, clamp01, ease, interpolate, lerp, motion_blur, spring,
)
