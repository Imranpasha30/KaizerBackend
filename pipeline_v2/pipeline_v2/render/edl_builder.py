"""Item 117 Phase 1 -- EDL (Edit Decision List) builder.

The item-117 architecture replaces V2's cut + compose + stitch chain
with a SINGLE ffmpeg invocation that decodes the mezzanine once and
emits all raw timeline outputs (bulletin_raw.mp4 + per-short raw
files) in parallel. This module is the PURE function that converts
the upstream cut decisions into the ``-filter_complex`` graph string
+ the per-output map labels.

Architecture diagram:

    mezzanine.mp4
        |
        v
    [ffmpeg single decode]
        |
        +--> [filter_complex]
        |      bulletin: trim+atrim per cut -> concat -> [bv_out],[ba_out]
        |      shorts:   trim+atrim per cut -> [svNN],[saNN]
        |
        +--> bulletin_raw.mp4         (-map [bv_out],[ba_out])
        +--> short_01_raw.mp4         (-map [sv00],[sa00])
        +--> short_02_raw.mp4         (-map [sv01],[sa01])
        +--> ...

This module is impure-free: NO ffmpeg invocation, NO file I/O. Tests
exercise its full surface without mocks.

Invariants enforced:

  * All boundaries are snapped to the video frame grid (default
    ``1/30s``). Snapping uses bank-round (Python's ``round`` on
    ``t / grid``), then multiplied back.
  * Cuts whose snapped duration is <= 0 are dropped (logged via the
    returned ``dropped`` list so the caller can surface a warning).
  * Either the bulletin or the shorts plan may be empty; both empty
    raises ``ValueError`` (no work to do).
  * Single-cut bulletin: trim+atrim runs but the concat node is
    SKIPPED (ffmpeg's ``concat=n=1`` is wasteful and on older builds
    can error). The single trim's labels become the output labels.

Diagnostic test (item 117 architecture validation) used this exact
shape on Job 51's 28-bulletin + 8-shorts plan and measured A/V
delta -0.01ms (vs the legacy -695.8ms cut-step cumulative drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


# Production default: V2's mezzanine is 30fps CFR.
DEFAULT_SNAP_GRID_S: float = 1.0 / 30.0


@dataclass(frozen=True)
class OutputSpec:
    """One output of the multi-output ffmpeg call.

    ``role`` is "bulletin" (always exactly one, when present) or
    "short" (zero or more, one per shorts_cut after dropping
    zero-length entries).

    ``v_label`` / ``a_label`` are filter-graph node names ready to
    drop into ``-map "[label]"`` arguments. Bulletin labels come from
    the concat output (or the single trim when N=1); short labels are
    the per-cut trim outputs.

    ``duration_s`` is the EXPECTED output duration computed from the
    snapped cuts -- callers can assert ffprobe's measured duration
    against this within tolerance for the post-extract verification.

    ``source_cuts`` is the list of ``(start_s, end_s)`` ranges (already
    snapped) that contribute to this output. Bulletin keeps N entries
    (one per kept range); short keeps exactly one entry.
    """

    role: str
    index: int               # bulletin always 0; shorts are 1-based by call order
    v_label: str
    a_label: str
    duration_s: float
    source_cuts: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DroppedCut:
    """A cut that was dropped during EDL construction (zero/negative
    duration after snapping)."""

    role: str
    original: tuple[float, float]
    snapped: tuple[float, float]
    reason: str


@dataclass(frozen=True)
class EDL:
    """Output of ``build_extraction_edl``: everything the caller needs
    to assemble the ffmpeg command. The ``filter_complex`` string is
    ready to pass to ``-filter_complex``; the ``outputs`` list is in
    the natural ffmpeg output order (bulletin first when present,
    then shorts in input order)."""

    filter_complex: str
    outputs: tuple[OutputSpec, ...]
    dropped: tuple[DroppedCut, ...] = field(default_factory=tuple)


def snap_to_grid(t: float, grid_s: float = DEFAULT_SNAP_GRID_S) -> float:
    """Round ``t`` to the nearest video frame boundary.

    Symmetric, branchless. ``round(t / grid) * grid`` is the canonical
    expression but Python's bank-round on integer-valued ``t / grid``
    is deterministic + matches the cut step's existing helper.
    """
    return round(t / grid_s) * grid_s


def _normalize_cuts(
    cuts: Iterable[tuple[float, float]],
    grid_s: float,
    role: str,
) -> tuple[list[tuple[float, float]], list[DroppedCut]]:
    """Snap every (start, end) to the grid and drop degenerate
    entries. Returns ``(kept, dropped)``."""
    kept: list[tuple[float, float]] = []
    dropped: list[DroppedCut] = []
    for raw_start, raw_end in cuts:
        s = snap_to_grid(float(raw_start), grid_s)
        e = snap_to_grid(float(raw_end), grid_s)
        if e <= s:
            dropped.append(DroppedCut(
                role=role,
                original=(float(raw_start), float(raw_end)),
                snapped=(s, e),
                reason="non-positive duration after snap",
            ))
            continue
        kept.append((s, e))
    return kept, dropped


def _fmt(t: float) -> str:
    """Format a timestamp for ffmpeg with microsecond precision so
    float-format truncation can't shift frame inclusion."""
    return f"{t:.6f}"


def build_extraction_edl(
    bulletin_cuts: Iterable[tuple[float, float]],
    shorts_cuts: Iterable[tuple[float, float]] = (),
    *,
    snap_grid_s: float = DEFAULT_SNAP_GRID_S,
) -> EDL:
    """Convert cut decisions into a filter_complex graph + map specs.

    Parameters
    ----------
    bulletin_cuts
        Iterable of ``(start_s, end_s)`` ranges. Order is preserved
        in the concat -- the caller must hand them in playback order.
        Empty bulletin is allowed (-> no bulletin output).
    shorts_cuts
        Iterable of ``(start_s, end_s)`` ranges, one per desired short.
        Empty is allowed (-> no short outputs).
    snap_grid_s
        Frame grid for boundary snapping. Default is 1/30 s (V2's
        mezzanine fps).

    Returns
    -------
    EDL
        ``filter_complex`` string + ordered list of OutputSpec.

    Raises
    ------
    ValueError
        Both bulletin_cuts and shorts_cuts are empty (no work).
    """
    bulletin_cuts = list(bulletin_cuts)
    shorts_cuts = list(shorts_cuts)
    if not bulletin_cuts and not shorts_cuts:
        raise ValueError(
            "build_extraction_edl: nothing to extract -- both "
            "bulletin_cuts and shorts_cuts are empty."
        )

    bulletin_kept, bulletin_dropped = _normalize_cuts(
        bulletin_cuts, snap_grid_s, "bulletin",
    )
    shorts_kept, shorts_dropped = _normalize_cuts(
        shorts_cuts, snap_grid_s, "short",
    )

    parts: list[str] = []
    outputs: list[OutputSpec] = []

    # --- Bulletin ---
    if bulletin_kept:
        for i, (s, e) in enumerate(bulletin_kept):
            parts.append(
                f"[0:v]trim=start={_fmt(s)}:end={_fmt(e)},"
                f"setpts=PTS-STARTPTS[bv{i:02d}]"
            )
            parts.append(
                f"[0:a]atrim=start={_fmt(s)}:end={_fmt(e)},"
                f"asetpts=PTS-STARTPTS[ba{i:02d}]"
            )
        if len(bulletin_kept) == 1:
            # N=1: skip concat node; the single trim outputs ARE the bulletin.
            v_label = "bv00"
            a_label = "ba00"
        else:
            bv_in = "".join(f"[bv{i:02d}]" for i in range(len(bulletin_kept)))
            ba_in = "".join(f"[ba{i:02d}]" for i in range(len(bulletin_kept)))
            parts.append(
                f"{bv_in}concat=n={len(bulletin_kept)}:v=1:a=0[bv_out]"
            )
            parts.append(
                f"{ba_in}concat=n={len(bulletin_kept)}:v=0:a=1[ba_out]"
            )
            v_label = "bv_out"
            a_label = "ba_out"
        bul_dur = sum(e - s for s, e in bulletin_kept)
        outputs.append(OutputSpec(
            role="bulletin",
            index=0,
            v_label=v_label,
            a_label=a_label,
            duration_s=bul_dur,
            source_cuts=tuple(bulletin_kept),
        ))

    # --- Shorts ---
    for i, (s, e) in enumerate(shorts_kept):
        v_lab = f"sv{i:02d}"
        a_lab = f"sa{i:02d}"
        parts.append(
            f"[0:v]trim=start={_fmt(s)}:end={_fmt(e)},"
            f"setpts=PTS-STARTPTS[{v_lab}]"
        )
        parts.append(
            f"[0:a]atrim=start={_fmt(s)}:end={_fmt(e)},"
            f"asetpts=PTS-STARTPTS[{a_lab}]"
        )
        outputs.append(OutputSpec(
            role="short",
            index=i + 1,
            v_label=v_lab,
            a_label=a_lab,
            duration_s=e - s,
            source_cuts=((s, e),),
        ))

    if not outputs:
        # Everything was zero-length after snapping. Surface as a value
        # error (the caller should not invoke ffmpeg with no outputs).
        all_dropped = bulletin_dropped + shorts_dropped
        raise ValueError(
            f"build_extraction_edl: all {len(all_dropped)} cuts had "
            f"non-positive duration after snapping; nothing to extract."
        )

    return EDL(
        filter_complex=";".join(parts),
        outputs=tuple(outputs),
        dropped=tuple(bulletin_dropped + shorts_dropped),
    )
