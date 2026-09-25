"""Native motion toolkit — the animation engine behind flagship-grade effects.

Easing curves, physics springs, value interpolation, and motion blur, all
reimplemented from motion-design first principles in pure Python/NumPy. This is
the shared motion language every effect family uses so the whole engine moves
with one consistent, premium feel. No third-party motion library — this is ours.

Design goals:
  * `interpolate(...)` — map an input range to an output range with optional
    easing + clamping (the workhorse for keyframed motion).
  * `spring(...)` — a damped-harmonic spring (mass/damping/stiffness) for natural,
    overshooting motion (entrances, pops, snaps).
  * a rich `EASINGS` set (quad..quint, expo, circ, back, elastic, bounce) in
    in/out/in-out variants.
  * `motion_blur(...)` — directional accumulation blur for fast moves.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────
def clamp01(t: float) -> float:
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ── Easing curves: take normalized t in [0,1], return eased value in [0,1] ──────
def linear(t: float) -> float:
    return t


def _in_out(fin: Callable[[float], float]) -> Callable[[float], float]:
    """Build an in-out easing from an ease-in curve."""
    def eased(t: float) -> float:
        if t < 0.5:
            return fin(2 * t) / 2
        return 1 - fin(2 * (1 - t)) / 2
    return eased


def _out(fin: Callable[[float], float]) -> Callable[[float], float]:
    return lambda t: 1 - fin(1 - t)


# Power eases (ease-in family) → derive out / in-out.
_in_quad = lambda t: t * t
_in_cubic = lambda t: t ** 3
_in_quart = lambda t: t ** 4
_in_quint = lambda t: t ** 5
_in_expo = lambda t: 0.0 if t == 0 else math.pow(2, 10 * (t - 1))
_in_circ = lambda t: 1 - math.sqrt(max(0.0, 1 - t * t))


def _in_back(t: float, s: float = 1.70158) -> float:
    return t * t * ((s + 1) * t - s)


def _out_elastic(t: float) -> float:
    if t in (0.0, 1.0):
        return t
    p = 0.3
    return math.pow(2, -10 * t) * math.sin((t - p / 4) * (2 * math.pi) / p) + 1


def _out_bounce(t: float) -> float:
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


EASINGS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_in": _in_cubic, "ease_out": _out(_in_cubic), "ease_in_out": _in_out(_in_cubic),
    "in_quad": _in_quad, "out_quad": _out(_in_quad), "in_out_quad": _in_out(_in_quad),
    "in_cubic": _in_cubic, "out_cubic": _out(_in_cubic), "in_out_cubic": _in_out(_in_cubic),
    "in_quart": _in_quart, "out_quart": _out(_in_quart), "in_out_quart": _in_out(_in_quart),
    "in_quint": _in_quint, "out_quint": _out(_in_quint), "in_out_quint": _in_out(_in_quint),
    "in_expo": _in_expo, "out_expo": _out(_in_expo), "in_out_expo": _in_out(_in_expo),
    "in_circ": _in_circ, "out_circ": _out(_in_circ), "in_out_circ": _in_out(_in_circ),
    "in_back": _in_back, "out_back": _out(_in_back), "in_out_back": _in_out(_in_back),
    "out_elastic": _out_elastic, "out_bounce": _out_bounce,
}


def ease(name: str, t: float) -> float:
    """Apply a named easing to a normalized t (clamped to [0,1])."""
    return EASINGS.get(name, EASINGS["ease_out"])(clamp01(t))


# ── interpolate: map input range → output range with easing + clamp ─────────────
def interpolate(
    x: float,
    in_range: Sequence[float],
    out_range: Sequence[float],
    *,
    easing: str | Callable[[float], float] = "linear",
    clamp: bool = True,
) -> float:
    """Piecewise-linear map of ``x`` from ``in_range`` to ``out_range`` with an
    easing applied within each segment. Mirrors the familiar interpolate() motion
    primitive. ``in_range`` must be monotonically increasing."""
    if len(in_range) != len(out_range) or len(in_range) < 2:
        raise ValueError("in_range/out_range must be equal length >= 2")
    fn = easing if callable(easing) else EASINGS.get(easing, linear)
    if x <= in_range[0]:
        if clamp:
            return float(out_range[0])
    if x >= in_range[-1]:
        if clamp:
            return float(out_range[-1])
    # find segment
    for i in range(len(in_range) - 1):
        a, b = in_range[i], in_range[i + 1]
        if (a <= x <= b) or (i == len(in_range) - 2):
            span = (b - a) or 1e-9
            t = (x - a) / span
            return float(lerp(out_range[i], out_range[i + 1], fn(clamp01(t)) if clamp else fn(t)))
    return float(out_range[-1])


# ── spring: damped harmonic oscillator (natural, overshooting motion) ───────────
@lru_cache(maxsize=4096)
def _spring_value(frame: int, fps: int, mass: float, damping: float,
                  stiffness: float, velocity: float) -> float:
    """Position in [~0..1] of a unit-target damped spring at ``frame`` (cached).
    Semi-implicit Euler integration from rest → target 1.0."""
    if frame <= 0:
        return 0.0
    dt = 1.0 / max(1, fps)
    x, v = 0.0, velocity
    steps = frame
    # sub-step for stability with stiff springs
    sub = 8
    h = dt / sub
    for _ in range(steps * sub):
        force = stiffness * (1.0 - x) - damping * v
        v += (force / mass) * h
        x += v * h
    return x


def spring(
    frame: float,
    fps: int = 30,
    *,
    mass: float = 1.0,
    damping: float = 12.0,
    stiffness: float = 140.0,
    velocity: float = 0.0,
    from_: float = 0.0,
    to: float = 1.0,
    delay: int = 0,
) -> float:
    """A physics spring value at ``frame`` mapped from ``from_`` → ``to``.
    Defaults give a snappy, slightly-overshooting entrance. Deterministic + cached."""
    f = int(round(frame)) - int(delay)
    if f <= 0:
        return float(from_)
    unit = _spring_value(f, int(fps), float(mass), float(damping), float(stiffness), float(velocity))
    return float(from_ + (to - from_) * unit)


# ── motion blur: directional accumulation (fast moves) ──────────────────────────
def motion_blur(frame_rgba: np.ndarray, dx: float, dy: float, *, samples: int = 8,
                decay: float = 1.0) -> np.ndarray:
    """Directional accumulation blur: average ``samples`` shifted copies along
    (dx, dy) pixels. Cheap, GPU-free, good enough to sell fast camera/transition
    moves. ``frame_rgba`` is HxWx4 uint8; returns the same shape."""
    if samples <= 1 or (abs(dx) < 0.5 and abs(dy) < 0.5):
        return frame_rgba
    h, w = frame_rgba.shape[:2]
    acc = np.zeros((h, w, frame_rgba.shape[2]), dtype=np.float32)
    weight = 0.0
    for i in range(samples):
        f = i / max(1, samples - 1)        # 0..1 along the trail
        ox, oy = int(round(-dx * f)), int(round(-dy * f))
        shifted = np.roll(frame_rgba, (oy, ox), axis=(0, 1)).astype(np.float32)
        wgt = decay ** i
        acc += shifted * wgt
        weight += wgt
    return (acc / max(weight, 1e-6)).clip(0, 255).astype(np.uint8)
