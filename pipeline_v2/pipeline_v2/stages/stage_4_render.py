"""Stage 4 -- Render (port of V1 pipeline_core.pipeline rendering).

V2 owns the orchestration + JobOutput->V1-dict conversion. V1 owns
the actual FFmpeg / Pillow work. Per Step 9 D-9.1: REUSE V1's render
primitives verbatim. Byte-identical output is the architectural
contract that lets V2 reuse V1's editor UI without modification.

V1 PRIMITIVES IMPORTED (any signature change in V1 here breaks V2 --
keep this list current):

  pipeline_core.pipeline.cut_video_clips         (Step 9.1)
  pipeline_core.pipeline.resolve_image_plan      (Step 9.2)
  pipeline_core.pipeline.compose_clip            (Step 9.2: torn_card layout)
  pipeline_core.pipeline.compose_split_frame     (Step 9.2: split_frame layout)
  pipeline_core.pipeline.compose_clip_clean_card (Step 9.2: clean_card layout)
  pipeline_core.pipeline.compose_follow_bar      (Step 9.2: follow_bar layout)
  pipeline_core.pipeline.overlay_image_plan      (Step 9.3 bulletin overlay)
  pipeline_core.pipeline._maybe_upload_final     (optional, R2 upload)
  pipeline_core.pipeline.FFMPEG_BIN              (Step 9.2 thumbnails)

  pipeline_core.longform_compose.compose_bulletin_story  (Step 9.3)
  pipeline_core.longform_compose.compose_pip_story       (Step 9.3 with PiP)
  pipeline_core.longform_compose.render_ticker           (Step 9.3 ticker)
  pipeline_core.longform_compose.render_channel_bug      (Step 9.3 bug)
  pipeline_core.longform_compose.make_sidebar_placeholder (Step 9.3 fallback)
  pipeline_core.longform_compose.pick_pip_source         (Step 9.3 PiP)
  pipeline_core.longform_compose.StoryMeta               (Step 9.3 dataclass)

  pipeline_core.image_carousel.build_sidebar_carousel    (Step 9.3 carousel)
  pipeline_core.image_carousel.build_fullscreen_takeover (Step 9.3 takeover)

  pipeline_core.bulletin_stitcher.stitch_bulletin        (Step 9.3 final concat)
  pipeline_core.bulletin_stitcher.BulletinStitchError    (Step 9.3 stitch exc)

  pipeline_core.compose_deps.is_fresh / mark_built       (Step 9.3 cache)

  pipeline_core.hw_accel.ACTIVE_ENCODER          (encoder selection, indirect)
  pipeline_core.hw_accel.h264_args               (encoder args builder, indirect)

KaizerBackend/ must be on sys.path for these imports to resolve. The
test path is set up by ``pipeline_v2/conftest.py``; production runs
via the dispatcher (Step 10) which adds the path explicitly.

Decisions implemented in this module (cross-reference to Step 9 D-list):

  D-9.1  -- reuse V1 verbatim (import, do not reimplement)
  D-9.2  -- hybrid: Stage4Render class + 3 pure converter functions
  D-9.3  -- output_dir is caller-supplied (constructor arg)
  D-9.5  -- image_pool is instance state, shared across shorts+bulletin
  D-9.7  -- 50% per-clip-failure guardrail (Step 9.2)
  D-9.8  -- two sub-renders, sequential (Step 9.3 / 9.4)
  D-9.10 -- frame_layout caller-supplied, default ``torn_card``
  D-9.11 -- mirror V1 filename conventions, leave intermediates
  D-9.13 -- three pure converter helpers (this file)
  D-9.14 -- skip ChatGPT title step (V2 metadata.shorts_headline_native is
            Pydantic-required, never empty)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---- V1 primitive imports -------------------------------------------
#
# These imports may fail at module-load time if KaizerBackend/ is not
# on sys.path. We provide a clear error message rather than letting
# the import error propagate ambiguously.
def _ensure_kaizer_backend_on_path() -> None:
    """Add KaizerBackend/ to sys.path if missing.

    Mirrors what ``conftest.py`` does for tests + what each script in
    ``pipeline_v2/scripts/`` does for ad-hoc runs. The production
    dispatcher (Step 10) sets the path before importing Stage 4, so
    this is a defensive fallback.
    """
    here = Path(__file__).resolve()
    # .../KaizerBackend/pipeline_v2/pipeline_v2/stages/stage_4_render.py
    # parents[3] = KaizerBackend/
    kaizer_backend = here.parents[3]
    s = str(kaizer_backend)
    if s not in sys.path:
        sys.path.insert(0, s)


_ensure_kaizer_backend_on_path()

# Imports MUST come after path setup. These are checked at module load
# so an environment without V1 fails fast with a clear traceback.
from pipeline_core.pipeline import cut_video_clips as _v1_cut_video_clips
from pipeline_core.pipeline import resolve_image_plan as _v1_resolve_image_plan
from pipeline_core.pipeline import compose_clip as _v1_compose_clip
from pipeline_core.pipeline import compose_clip_clean_card as _v1_compose_clean_card
from pipeline_core.pipeline import compose_follow_bar as _v1_compose_follow_bar
from pipeline_core.pipeline import compose_split_frame as _v1_compose_split_frame
from pipeline_core.pipeline import overlay_image_plan as _v1_overlay_image_plan
from pipeline_core.pipeline import FFMPEG_BIN as _v1_FFMPEG_BIN
from pipeline_core.longform_compose import (
    compose_bulletin_story as _v1_compose_bulletin_story,
    compose_pip_story as _v1_compose_pip_story,
    render_ticker as _v1_render_ticker,
    render_channel_bug as _v1_render_channel_bug,
    make_sidebar_placeholder as _v1_make_sidebar_placeholder,
    pick_pip_source as _v1_pick_pip_source,
    StoryMeta as _V1StoryMeta,
)
from pipeline_core.image_carousel import (
    build_sidebar_carousel as _v1_build_sidebar_carousel,
    build_fullscreen_takeover as _v1_build_fullscreen_takeover,
)
from pipeline_core.bulletin_stitcher import (
    stitch_bulletin as _v1_stitch_bulletin,
    BulletinStitchError as _V1BulletinStitchError,
)
from pipeline_core import compose_deps as _v1_compose_deps
import subprocess
import languages as _v1_languages

from pipeline_v2.editor_meta_adapter import (
    ClipRenderArtifacts,
    build_v1_bulletin_editor_meta,
    build_v1_shorts_editor_meta,
)
from pipeline_v2.models import (
    Entity,
    EntityType,
    FullVideoCut,
    ImagePlan,
    ImagePlanEntry,
    JobOutput,
    Metadata,
    ShortsCut,
    SkippedSegment,
    Word,
)
from pipeline_v2.stages.stage_4_image_source import ImageSourcer

logger = logging.getLogger("pipeline_v2.stage_4")


# ====================================================================== #
# Defaults                                                                #
# ====================================================================== #

DEFAULT_FRAME_LAYOUT = "torn_card"      # D-9.10: V1's most common layout
DEFAULT_PLATFORM = "full_video_shorts_v2"
DEFAULT_DROP_RATIO_THRESHOLD = 0.5      # D-9.7: mirror D-7.10's guardrail


# ====================================================================== #
# PermanentRenderError (Step 10.3, mirrors PermanentSTTError pattern)    #
# ====================================================================== #


class PermanentRenderError(Exception):
    """Raised by Stage 4 when render failure is unlikely to succeed
    on retry. Orchestrator translates to Inngest's
    ``NonRetriableError`` so Inngest skips retries.

    Conditions providers SHOULD raise this for (slug prefix in message):

      * ``ffmpeg_not_found``: FFmpeg binary not in PATH or wrong version
      * ``disk_full``: ENOSPC on output dir write attempt
      * ``source_video_corrupt``: FFprobe fails to parse input video
      * ``encoder_unavailable_no_fallback``: all encoders fail
        (NVENC, QSV, AMF, libx264)

    Conditions providers MUST NOT raise this for (retry normally):

      * Individual clip compose failures (handled by D-9.7 50% guardrail)
      * R2 upload failures (transient)
      * GPU OOM on one clip (transient -- may succeed with cleaner state)
      * Filesystem permission errors (could be transient mount issue)

    Reference: backlog item 50 (Inngest 0.5.18 has no per-step retries
    so we use NonRetriableError for permanent-failure early-exit). A
    12-min render that fails systematically (disk full, FFmpeg
    missing) without this class costs 24 min of wasted Inngest retry.
    """


def _classify_render_error(exc: BaseException) -> Optional[str]:
    """Map an exception to a PermanentRenderError slug, or None.

    Returns a slug string (e.g. "ffmpeg_not_found: ...") when the
    exception is a documented permanent condition; returns None for
    everything else (caller treats as transient, Inngest will retry).

    Pure classification helper: no side effects, safe to call from
    inside an except clause.
    """
    msg = str(exc)
    msg_lower = msg.lower()

    # ffmpeg_not_found: FileNotFoundError mentioning ffmpeg
    if isinstance(exc, FileNotFoundError):
        if "ffmpeg" in msg_lower or "ffprobe" in msg_lower:
            return f"ffmpeg_not_found: {exc}"

    # disk_full: ENOSPC (errno 28 on Linux, 39 on Windows for "disk full")
    if isinstance(exc, OSError):
        errno = getattr(exc, "errno", None)
        # ENOSPC=28 (linux/macos); on Windows it's ERROR_DISK_FULL=112
        if errno in (28, 112):
            return f"disk_full: {exc}"
        if "no space left" in msg_lower or "disk full" in msg_lower:
            return f"disk_full: {exc}"

    # source_video_corrupt: ffprobe/ffmpeg with corrupt-stream signals
    if "ffprobe" in msg_lower and (
        "could not parse" in msg_lower
        or "invalid data" in msg_lower
        or "moov atom not found" in msg_lower
    ):
        return f"source_video_corrupt: {exc}"

    # encoder_unavailable_no_fallback: hw_accel module reports all
    # encoders failed
    if "no h264 encoder available" in msg_lower or (
        "encoder" in msg_lower and "unavailable" in msg_lower
    ):
        return f"encoder_unavailable_no_fallback: {exc}"

    return None


@dataclass(frozen=True)
class RenderResult:
    """Top-level render output.

    The caller (Step 10 Inngest dispatcher) gets back this tuple-like
    carrier with paths to every artifact Stage 4 produced. Both
    editor_meta.json files are written to disk; the paths here are
    for telemetry / R2 upload bookkeeping.
    """
    # Paths to the two editor_meta.json files Stage 4 writes.
    # Either may be None when the corresponding pass didn't run
    # (e.g. job_output has zero shorts_cuts -> shorts pass skipped).
    shorts_editor_meta_path:   Optional[str]
    bulletin_editor_meta_path: Optional[str]

    # Composed shorts artifacts (one per produced short, post-drop).
    # Empty list if the shorts pass didn't run or produced nothing.
    composed_shorts: list[dict]

    # Bulletin assembly result (the dict returned by render_bulletin),
    # or None if the bulletin pass didn't run.
    bulletin: Optional[dict]


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #


def _ffprobe_audio_duration_s(path: str) -> float:
    """Backlog item 100: return the AUDIO stream duration of *path*
    in seconds (float). Returns 0.0 when path has no audio stream
    OR when ffprobe fails (best-effort -- the invariant check uses
    this and tolerates probe failures via its outer try/except).
    """
    import subprocess as _sp
    try:
        r = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        out = (r.stdout or "").strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def _ffprobe_video_duration_s(path: str) -> float:
    """Backlog item 100: same as ``_ffprobe_audio_duration_s`` but for
    the VIDEO stream. Used to size takeover transitions which have
    no audio of their own."""
    import subprocess as _sp
    try:
        r = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        out = (r.stdout or "").strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


# --- Item 102: extracted gating helpers for test coverage --------------
#
# Item 100 introduced three guardrails as inline expressions in
# _render_impl and render_bulletin. Item 102 extracts each one to a
# pure function so the behaviour can be unit-tested without spinning
# up the full render pipeline. Behaviour is preserved 1:1; the
# extraction is purely a testability refactor.


# Single literal source of truth for which video_types benefit from
# a picture-in-picture inset. SOLO (talking-head news monologue) is
# deliberately absent: V1's pick_pip_source would just pull another
# segment of the same anchor.
PIP_ALLOWED_VIDEO_TYPES: frozenset[str] = frozenset({
    "INTERVIEW", "PRESS_CONFERENCE", "PANEL", "MIXED",
})

# Minimum distinct FullVideoCuts before adaptive takeovers may fire.
# Below this floor the bulletin is one narrative thread split into
# spliced sub-cuts; takeovers between sub-cuts produce ~6.5s of dead
# air per boundary (item 100's 132-second bloat bug).
TAKEOVER_MIN_DISTINCT_CUTS: int = 3

# A/V invariant: bulletin audio duration may differ from the
# expected sum (narration audio + takeover video) by at most this
# many seconds. Larger deltas indicate an unintended source of
# video-without-audio (intro/outro padding, stray ffmpeg filter,
# new pre-stitch insertion).
AV_INVARIANT_TOLERANCE_S: float = 1.0


def _compute_takeovers_enabled(
    use_takeovers: bool, full_video_cuts: list,
) -> bool:
    """Item 102: adaptive takeover gate.

    Takeovers are enabled only when the operator opted in AND
    Stage 2 produced enough distinct stories to warrant TV-style
    inter-story transitions. Single-story monologues split by
    splicing must concat seamlessly -- inserting takeovers between
    sub-cuts of one narrative thread adds dead air between what
    the listener perceives as one continuous statement.
    """
    return bool(use_takeovers) and len(full_video_cuts) >= TAKEOVER_MIN_DISTINCT_CUTS


def _compute_pip_enabled(use_pip: bool, video_type: str) -> bool:
    """Item 102: PiP gate.

    PiP shows ``next story's first 8s`` as a B-roll inset. For
    multi-source video types (INTERVIEW / PRESS_CONFERENCE / PANEL /
    MIXED) the next story usually means a different speaker or
    angle -- useful inset. For SOLO it would just be the same
    anchor talking-head in the corner.
    """
    return bool(use_pip) and (video_type or "") in PIP_ALLOWED_VIDEO_TYPES


def _should_insert_takeover_between(
    this_parent: Optional[int], next_parent: Optional[int],
) -> bool:
    """Item 102: splice-group-aware boundary check.

    A takeover transition only belongs between two clips that come
    from DIFFERENT FullVideoCut parents. Two adjacent clips sharing
    the same ``parent_v2_index`` are spliced sub-cuts of one parent
    -- the audience perceives them as one statement, so a 6.5s
    branded transition between them would be the dead-air bug.

    ``None`` for ``next_parent`` means there is no next clip (the
    current clip is the last) -- no takeover after the final story.
    """
    if next_parent is None:
        return False
    return this_parent != next_parent


def _validate_av_invariant(
    actual_audio_s: float,
    composed_narration_s: float,
    takeover_video_s: float,
    tolerance_s: float = AV_INVARIANT_TOLERANCE_S,
) -> None:
    """Item 102: hard A/V invariant.

    Bulletin audio duration MUST equal ``narration + takeover_video``
    within ``tolerance_s`` seconds. Larger deltas indicate an
    unintended source of video-without-audio bloating the bulletin
    (a new intro/outro insertion, stray ffmpeg filter, etc).
    Raises ``RuntimeError`` on violation so the operator sees the
    regression at render time rather than via a support ticket.
    """
    expected = composed_narration_s + takeover_video_s
    delta = actual_audio_s - expected
    if abs(delta) > tolerance_s:
        raise RuntimeError(
            f"A/V invariant violated: bulletin audio "
            f"{actual_audio_s:.2f}s != narration "
            f"{composed_narration_s:.2f}s + transitions "
            f"{takeover_video_s:.2f}s (delta {delta:+.2f}s, "
            f"tolerance +/-{tolerance_s:.2f}s). Some component "
            f"added fake audio padding. Check takeovers, intro, "
            f"outro, or any new pre-stitch insertion."
        )


def _format_mmss_mmm(seconds: float) -> str:
    """Format float-second to MM:SS.mmm.

    Same algorithm as ``editor_meta_adapter._format_mmss_mmm`` --
    duplicated here so Stage 4 doesn't take a dependency on the
    adapter module (the dependency goes the other way: Stage 4
    CALLS the adapter at the end). Handles 59.9999 -> "01:00.000"
    carry-forward correctly.
    """
    total_ms = round(seconds * 1000)
    total_secs = total_ms // 1000
    ms = total_ms % 1000
    minutes = total_secs // 60
    secs = total_secs % 60
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


# ====================================================================== #
# Pure converter functions (D-9.13)                                       #
# ====================================================================== #


def shorts_cuts_to_v1_clip_dicts(
    shorts_cuts: list[ShortsCut],
    metadata: Metadata,
) -> list[dict]:
    """V2 ShortsCut list -> V1-compatible clip dict list.

    V1's render functions read ``clip["start"]`` / ``clip["end"]`` as
    MM:SS.mmm strings, plus ``clip["summary"]`` / ``["mood"]`` /
    ``["importance"]`` for editor_meta enrichment. Order is preserved
    (Stage 4 sorts upstream).

    Side-effect contract: V1's ``cut_video_clips`` MUTATES each clip
    dict to add ``raw_path`` and ``duration_sec``. Callers should
    treat the returned dicts as mutable carriers, NOT as
    snapshot-then-discard data.
    """
    return [
        {
            "start":      _format_mmss_mmm(c.start_sec),
            "end":        _format_mmss_mmm(c.end_sec),
            # Raw float seconds alongside the MM:SS.mmm strings. V1
            # helpers read ``start`` / ``end`` strings; V2's per-clip
            # image selection in compose_shorts (Step 12.2a re-run #4)
            # reads ``start_sec`` / ``end_sec`` floats so it can
            # overlap-match against ``source_show_at_sec`` from
            # resolved image_plan entries.
            "start_sec":  c.start_sec,
            "end_sec":    c.end_sec,
            "summary":    c.hook,         # D-8.9: hook is the per-clip summary
            "mood":       "",             # V2 doesn't track per-clip mood
            "importance": c.importance,
            "video_type": metadata.video_type,
            # Bookkeeping: original V2 ShortsCut.index, so post-cut
            # we can sort+pair back to the source after cut_video_clips
            # potentially drops clips (e.g. zero-duration).
            "v2_index":   c.index,
        }
        for c in shorts_cuts
    ]


def splice_cuts_minus_skipped(
    full_video_cuts: list[FullVideoCut],
    skipped_segments: list[SkippedSegment],
) -> tuple[list[FullVideoCut], list[int]]:
    """Backlog item 97: subtract SkippedSegment time-ranges from each
    FullVideoCut, producing a flat list of sub-cuts that EXCLUDE the
    retake / filler / dead-air spans Stage 2 identified.

    Item 100: also returns a parallel list of ``parent_v2_index`` (one
    per emitted sub-cut). The renderer uses this to decide whether to
    insert an inter-story TV-style takeover transition: takeovers
    only fire BETWEEN different parents (true story boundaries), not
    BETWEEN spliced sub-cuts of the same FullVideoCut (which together
    form one continuous narrative thread). Without this, item 97's
    splice fix accidentally created 21 takeovers between 22 sub-cuts
    of a single story = ~132s of dead-air padding (Bug 1).

    Per the Stage 2 prompt's design (stage_2_prompt.md, rule #6):
    ``full_video_cuts`` may span across skipped regions; the RENDERER
    is responsible for splicing the skipped span out.

    Returns ``(expanded_cuts, parent_v2_indexes)``. If no
    skipped_segments overlap, returns the input unchanged + each cut's
    own index as parent (identity).
    """
    if not skipped_segments:
        return list(full_video_cuts), [c.index for c in full_video_cuts]

    skipped_sorted = sorted(skipped_segments, key=lambda s: s.start_sec)
    out: list[FullVideoCut] = []
    parents: list[int] = []
    new_idx = 0
    for cut in full_video_cuts:
        relevant = [
            s for s in skipped_sorted
            if s.end_sec > cut.start_sec and s.start_sec < cut.end_sec
        ]
        cursor = cut.start_sec
        cursor_word = cut.start_word_idx
        for sk in relevant:
            sk_start = max(sk.start_sec, cut.start_sec)
            sk_end = min(sk.end_sec, cut.end_sec)
            sk_word_start = max(sk.start_word_idx, cut.start_word_idx)
            sk_word_end = min(sk.end_word_idx, cut.end_word_idx)
            if sk_start > cursor:
                out.append(FullVideoCut(
                    index=new_idx,
                    start_word_idx=cursor_word,
                    end_word_idx=max(sk_word_start - 1, cursor_word),
                    start_sec=cursor,
                    end_sec=sk_start,
                    importance=cut.importance,
                ))
                parents.append(cut.index)
                new_idx += 1
            cursor = sk_end
            cursor_word = sk_word_end + 1
        if cursor < cut.end_sec:
            out.append(FullVideoCut(
                index=new_idx,
                start_word_idx=min(cursor_word, cut.end_word_idx),
                end_word_idx=cut.end_word_idx,
                start_sec=cursor,
                end_sec=cut.end_sec,
                importance=cut.importance,
            ))
            parents.append(cut.index)
            new_idx += 1
    return out, parents


# --- Item 103: micro-fragment collapse ---------------------------------

MICRO_FRAGMENT_THRESHOLD_S: float = 1.5


def collapse_micro_fragments(
    sub_cuts: list[FullVideoCut],
    parent_v2_indexes: list[int],
    threshold_s: float = MICRO_FRAGMENT_THRESHOLD_S,
) -> tuple[list[FullVideoCut], list[int]]:
    """Drop sub-cuts shorter than ``threshold_s`` (default 1.5s).

    After ``splice_cuts_minus_skipped`` runs, the bulletin can
    contain sub-cuts spanning < 1.5s -- e.g. a 0.3s sliver of
    on-air content sandwiched between two consecutive retake
    skipped_segments. These micro-fragments cause:

      1. Perceptible chop in the rendered bulletin (each fragment
         is a hard cut point with no smoothing).
      2. Wasted ffmpeg seek work on near-zero-length segments.
      3. Operator-visible clutter in the editor timeline.

    Policy: **drop, never merge**.

    Merging is tempting but unsafe: every adjacent pair of sub-cuts
    within one parent is, by construction, separated by a skipped
    span (otherwise they would have been one sub-cut). Merging
    would extend a sub-cut's time range across the skipped span,
    re-including the very retake content the splice was meant to
    remove.

    Per-parent safety: if dropping leaves a parent (one
    FullVideoCut) with ZERO sub-cuts, keep its LONGEST sub-cut so
    the story doesn't disappear entirely from the bulletin.

    Renumbers ``FullVideoCut.index`` to stay contiguous on the
    output; ``parent_v2_indexes`` is filtered in parallel.
    """
    if not sub_cuts:
        return [], []
    if len(sub_cuts) != len(parent_v2_indexes):
        raise ValueError(
            f"collapse_micro_fragments: length mismatch -- "
            f"{len(sub_cuts)} sub_cuts vs "
            f"{len(parent_v2_indexes)} parent_v2_indexes."
        )

    # Group sub-cut positions by parent_v2_index for per-parent safety.
    per_parent: dict[int, list[int]] = {}
    for i, p in enumerate(parent_v2_indexes):
        per_parent.setdefault(p, []).append(i)

    keep_set: set[int] = set()
    for parent_id, idxs in per_parent.items():
        kept_above = [
            i for i in idxs
            if (sub_cuts[i].end_sec - sub_cuts[i].start_sec) >= threshold_s
        ]
        if kept_above:
            keep_set.update(kept_above)
        else:
            # Every sub-cut from this parent is below threshold --
            # keep the longest so the story survives.
            longest = max(
                idxs,
                key=lambda i: sub_cuts[i].end_sec - sub_cuts[i].start_sec,
            )
            keep_set.add(longest)

    # Emit in original chronological order; renumber index field.
    out_cuts: list[FullVideoCut] = []
    out_parents: list[int] = []
    for orig_i, src in enumerate(sub_cuts):
        if orig_i not in keep_set:
            continue
        out_cuts.append(FullVideoCut(
            index=len(out_cuts),
            start_word_idx=src.start_word_idx,
            end_word_idx=src.end_word_idx,
            start_sec=src.start_sec,
            end_sec=src.end_sec,
            importance=src.importance,
        ))
        out_parents.append(parent_v2_indexes[orig_i])
    return out_cuts, out_parents


# --- Item 106: defensive guardrails (monotonic, repeat-word, precision)

CUT_PRECISION_DECIMALS: int = 3   # 3 decimals = 1ms (ffmpeg's tick)


def assert_cuts_monotonic(cuts: list[FullVideoCut]) -> None:
    """Item 106 / Bug B: verify cuts are sorted by ``start_sec`` and
    do not overlap each other.

    Raises ``ValueError`` with a descriptive message naming the
    offending pair (index + start/end times) so the operator can
    locate the upstream bug. Empty / single-element lists are
    trivially monotonic.

    Called at the END of the transform chain (splice + silence +
    micro-fragments) right before the renderer hands cuts to ffmpeg.
    A non-monotonic list would render time-traveled / duplicated
    content -- failing loudly here saves a confusing visual bug
    later.
    """
    for i in range(1, len(cuts)):
        prev, curr = cuts[i - 1], cuts[i]
        if curr.start_sec < prev.end_sec:
            raise ValueError(
                f"Cuts not monotonic: cut[{i - 1}] ends at "
                f"{prev.end_sec:.3f}s but cut[{i}] starts at "
                f"{curr.start_sec:.3f}s (overlap of "
                f"{prev.end_sec - curr.start_sec:.3f}s). Cut indexes: "
                f"prev.index={prev.index} curr.index={curr.index}."
            )


def collapse_repeated_words(
    words: list[Word],
    *,
    case_insensitive: bool = True,
    strip_punctuation: bool = True,
) -> list[Word]:
    """Item 106 / Bug A: collapse consecutive identical words.

    A common Stage 2 / Deepgram artefact: the same word appears
    twice in a row ("the the", "ఈరోజు ఈరోజు") -- usually a stutter
    that wasn't large enough to register as a hesitation segment.
    Renders as awkward repetition in the bulletin audio.

    Compares each word to its predecessor after optional lowercasing
    + trailing-punctuation strip. When they match, the SECOND copy
    is dropped (the earlier word's start_sec is kept; if the dropped
    word extended the time range, the kept word's end_sec is updated
    to the dropped word's end_sec so the audio span is preserved).

    Non-adjacent duplicates (e.g. ``"the cat the dog"``) are left
    alone -- only consecutive matches collapse.
    """
    if not words:
        return []
    out: list[Word] = []

    def _norm(w: str) -> str:
        s = w.lower() if case_insensitive else w
        if strip_punctuation:
            # Strip a single trailing punctuation mark if present.
            # Devanagari "।" and Telugu sentence-final marks are
            # treated the same as ASCII ".,!?;:".
            s = s.rstrip(".,!?;:।.")
        return s

    for w in words:
        if out and _norm(out[-1].w) == _norm(w.w):
            # Extend the kept word's range to swallow the duplicate
            # (preserves the audio span; e.g. if the duplicate was
            # 0.5s long, the kept word's end_sec moves forward by
            # that amount).
            kept = out[-1]
            out[-1] = Word(
                w=kept.w,
                s=kept.s,
                e=max(kept.e, w.e),
                speaker=kept.speaker,
                confidence=kept.confidence,
            )
            continue
        out.append(w)
    return out


def round_cut_precision(
    cuts: list[FullVideoCut],
    decimals: int = CUT_PRECISION_DECIMALS,
) -> list[FullVideoCut]:
    """Item 106 / Bug C: round ``start_sec`` / ``end_sec`` to a
    consistent precision (default 3 decimals = 1ms).

    Float accumulation across splice + silence + micro-fragment
    transforms can produce 4.999999...s instead of 5.000s. ffmpeg
    accepts the long-form value but downstream consumers (editor
    metadata JSON, manifest writers) render the long form back to
    the operator -- noisy and hard to compare against the spec.

    Raises ``ValueError`` if rounding produces a zero-length or
    negative-duration cut (defensive: the input chain shouldn't
    contain such cuts but if it does we surface immediately rather
    than silently rendering a broken segment).
    """
    out: list[FullVideoCut] = []
    for cut in cuts:
        new_start = round(float(cut.start_sec), decimals)
        new_end = round(float(cut.end_sec), decimals)
        if new_end <= new_start:
            raise ValueError(
                f"round_cut_precision: cut index={cut.index} has "
                f"zero/negative duration after rounding to {decimals} "
                f"decimals: start={new_start} end={new_end} "
                f"(pre-round: {cut.start_sec} -> {cut.end_sec})."
            )
        out.append(FullVideoCut(
            index=cut.index,
            start_word_idx=cut.start_word_idx,
            end_word_idx=cut.end_word_idx,
            start_sec=new_start,
            end_sec=new_end,
            importance=cut.importance,
        ))
    return out


# --- Item 105: silence trimming ----------------------------------------

SILENCE_TRIM_THRESHOLD_S: float = 1.5


def detect_silence_trims(
    words: list[Word],
    threshold_s: float = SILENCE_TRIM_THRESHOLD_S,
) -> list[tuple[float, float]]:
    """Find inter-word gaps longer than ``threshold_s`` in the source
    word array. Returns a list of ``(silence_start_sec, silence_end_sec)``
    tuples chronologically sorted.

    Silence is defined as the time between ``words[i].e`` and
    ``words[i+1].s`` when that gap exceeds ``threshold_s``.

    Pure word-time arithmetic -- deterministic, no LLM call. The
    threshold default (1.5s) matches the operator decision behind
    item 103's micro-fragment threshold. Set to 0 (or negative) to
    disable detection.
    """
    if not words or threshold_s <= 0 or len(words) < 2:
        return []
    out: list[tuple[float, float]] = []
    for prev, curr in zip(words[:-1], words[1:]):
        gap = float(curr.s) - float(prev.e)
        if gap > threshold_s:
            out.append((float(prev.e), float(curr.s)))
    return out


def apply_silence_trims_to_cuts(
    cuts: list[FullVideoCut],
    parent_v2_indexes: list[int],
    silence_trims: list[tuple[float, float]],
) -> tuple[list[FullVideoCut], list[int]]:
    """Subtract silence time ranges from each cut's span.

    For each silence range that falls inside a cut, the cut is split
    into two halves: ``[cut.start, silence_start]`` and
    ``[silence_end, cut.end]``. Both halves inherit the cut's
    ``parent_v2_index``. Cuts unaffected by any silence pass through
    unchanged.

    Renumbers ``FullVideoCut.index`` to stay contiguous on output;
    ``parent_v2_indexes`` is filtered/extended in parallel so every
    output sub-cut has its parent.

    Word-index bounds on new halves: ``start_word_idx`` / ``end_word_idx``
    are PRESERVED from the source cut (the boundaries are time-only;
    word indices on the source cut were already approximate after
    splice_cuts_minus_skipped). Downstream renderers cut on ``start_sec`` /
    ``end_sec`` so word-idx drift here is a telemetry concern only.

    Runs AFTER ``splice_cuts_minus_skipped`` (cuts already sub-cut by
    editorial drops) and BEFORE ``collapse_micro_fragments`` (so any
    silence-induced fragment < 1.5s gets cleaned up by item 103).
    """
    if not silence_trims or not cuts:
        return list(cuts), list(parent_v2_indexes)
    if len(cuts) != len(parent_v2_indexes):
        raise ValueError(
            f"apply_silence_trims_to_cuts: length mismatch -- "
            f"{len(cuts)} cuts vs {len(parent_v2_indexes)} parent_v2_indexes."
        )

    trims_sorted = sorted(silence_trims, key=lambda t: t[0])
    out_cuts: list[FullVideoCut] = []
    out_parents: list[int] = []
    for cut, parent in zip(cuts, parent_v2_indexes):
        # Silences inside this cut's time span (strict overlap).
        relevant = [
            t for t in trims_sorted
            if t[1] > cut.start_sec and t[0] < cut.end_sec
        ]
        if not relevant:
            out_cuts.append(FullVideoCut(
                index=len(out_cuts),
                start_word_idx=cut.start_word_idx,
                end_word_idx=cut.end_word_idx,
                start_sec=cut.start_sec,
                end_sec=cut.end_sec,
                importance=cut.importance,
            ))
            out_parents.append(parent)
            continue
        cursor = cut.start_sec
        for (sil_start, sil_end) in relevant:
            seg_start = max(sil_start, cut.start_sec)
            seg_end = min(sil_end, cut.end_sec)
            if seg_start > cursor:
                out_cuts.append(FullVideoCut(
                    index=len(out_cuts),
                    start_word_idx=cut.start_word_idx,
                    end_word_idx=cut.end_word_idx,
                    start_sec=cursor,
                    end_sec=seg_start,
                    importance=cut.importance,
                ))
                out_parents.append(parent)
            cursor = max(cursor, seg_end)
        if cursor < cut.end_sec:
            out_cuts.append(FullVideoCut(
                index=len(out_cuts),
                start_word_idx=cut.start_word_idx,
                end_word_idx=cut.end_word_idx,
                start_sec=cursor,
                end_sec=cut.end_sec,
                importance=cut.importance,
            ))
            out_parents.append(parent)
    return out_cuts, out_parents


def full_video_cuts_to_v1_clip_dicts(
    full_video_cuts: list[FullVideoCut],
    metadata: Metadata,
    parent_v2_indexes: Optional[list[int]] = None,
) -> list[dict]:
    """V2 FullVideoCut list -> V1-compatible clip dict list.

    Same shape as the shorts converter but with FullVideoCut's
    importance + no per-cut hook (FullVideoCut doesn't carry one; the
    bulletin uses the overall headline). ``summary`` is empty string
    for compatibility with V1's bulletin path which doesn't read
    per-clip summaries.

    Item 100: when ``parent_v2_indexes`` is provided (one int per
    cut), each dict gets a ``parent_v2_index`` field used by the
    bulletin compositor to decide whether to insert a takeover
    transition between this clip and the next (takeovers only fire
    between DIFFERENT parents). When omitted, ``parent_v2_index``
    defaults to the cut's own index (each cut is its own parent).
    """
    if parent_v2_indexes is None:
        parent_v2_indexes = [c.index for c in full_video_cuts]
    return [
        {
            "start":            _format_mmss_mmm(c.start_sec),
            "end":              _format_mmss_mmm(c.end_sec),
            "start_sec":        c.start_sec,   # see shorts converter for rationale
            "end_sec":          c.end_sec,
            "summary":          "",
            "mood":             "",
            "importance":       c.importance,
            "video_type":       metadata.video_type,
            "v2_index":         c.index,
            "parent_v2_index":  parent_v2_indexes[i],
        }
        for i, c in enumerate(full_video_cuts)
    ]


def synth_id_map_for_image_plan(image_plan: ImagePlan) -> dict[str, str]:
    """Build ``{entity_name: synth_id}`` for an ImagePlan, in
    first-appearance order.

    Synthesised IDs (``img_000``, ``img_001``, ...) are the V2->V1
    boundary contract: V1's ``resolve_image_plan`` reads
    ``entry["id"]`` and uses it as the pool_manifest lookup key. V2's
    Entity model uses ``canonical_name`` as identity (locked in Step
    6 D-12); rather than polluting V2's schema with a V1-compat ID
    field, we synthesise here at the boundary. Two image_plan entries
    that reference the same entity share the same synth_id so they
    resolve to ONE image file -- mirroring V1's "one image per unique
    id, reused across moments" pattern.
    """
    out: dict[str, str] = {}
    for e in image_plan.entries:
        if e.entity_name in out:
            continue
        out[e.entity_name] = f"img_{len(out):03d}"
    return out


def image_plan_to_v1_dict(
    image_plan: ImagePlan,
    entities: list[Entity],
) -> list[dict]:
    """V2 ImagePlanEntry list -> V1-compatible image_plan entry list.

    V1's ``resolve_image_plan`` reads:
      - id (synthesised at this boundary -- see synth_id_map_for_image_plan)
      - entity_name (matched against canonical entities)
      - description (image-search seed)
      - clip_index (which cut this overlay belongs to)
      - show_at (timestamp string "MM:SS.mmm")
      - duration (float seconds)

    V2's ``ImagePlanEntry`` has all fields except ``id`` directly
    (Entity uses canonical_name as identity per Step 6 D-12). We
    synthesise the V1-required id here so V1's pool_manifest lookup
    works without touching V2's schema.

    The only field-shape conversions:
      - ``show_at_sec`` (float) -> ``show_at`` (MM:SS.mmm string)
      - ``duration_sec`` -> ``duration``

    ``entities`` is unused in the conversion itself but kept in the
    signature to leave room for future entity-lookup logic (e.g.
    enriching with native_name when ImagePlanEntry doesn't have one).
    """
    synth_id = synth_id_map_for_image_plan(image_plan)
    return [
        {
            "id":                 synth_id[e.entity_name],
            "entity_name":        e.entity_name,
            "entity_name_native": e.entity_name_native,
            "description":        e.description,
            "clip_index":         e.clip_index,
            "show_at":            _format_mmss_mmm(e.show_at_sec),
            "duration":           e.duration_sec,
        }
        for e in image_plan.entries
    ]


# ====================================================================== #
# Stage4Render class                                                      #
# ====================================================================== #


@dataclass
class Stage4Render:
    """Orchestrator for the V2 render pipeline.

    Carries caller-supplied runtime state (paths, frame_layout,
    platform, video source) and shared instance state (image_pool
    per D-9.5). Stateless helper functions live at module level.

    Mutability note: this class is NOT frozen; ``image_pool`` is
    populated by Step 9.2's resolve_image_plan wiring and reused
    across shorts + bulletin passes.

    Bulletin feature toggles
    ------------------------
    The three ``use_*`` flags default to **True**, matching V1
    production behavior so V2 output looks visibly equivalent to V1.
    Each flag exists so ops can toggle a specific feature off if it
    surfaces issues in production -- the per-feature OFF path is
    just "skip the V1 helper call", no new code.

      - ``use_sidebar_carousel``: build Ken-Burns sidebar video per
        story (V1's build_sidebar_carousel). When False, falls back
        to make_sidebar_placeholder static image.
      - ``use_pip``: include a picture-in-picture overlay from
        another clip (V1's compose_pip_story). When False, uses the
        non-PiP compose_bulletin_story unconditionally.
      - ``use_takeovers``: insert full-screen transition clips
        between stories (V1's build_fullscreen_takeover). When
        False, stories concat directly with no transition.
    """

    output_dir: Path
    video_path: Path
    preset: dict
    frame_layout: str = DEFAULT_FRAME_LAYOUT
    platform: str = DEFAULT_PLATFORM
    drop_ratio_threshold: float = DEFAULT_DROP_RATIO_THRESHOLD

    # Bulletin feature toggles. Item 100 changes the defaults so V2
    # honours the "bulletin video duration must equal narration audio
    # duration" invariant out of the box:
    #
    #   use_takeovers default flipped True -> False. The previous
    #     default created ~6.5s of dead-air video between every story
    #     segment. After item 97's splice fix produces N sub-cuts of a
    #     single story, that meant N-1 takeovers per story = up to
    #     +130s of fake silent audio. _render_impl now also adaptively
    #     enables takeovers ONLY when Stage 2 produced >= 3 distinct
    #     FullVideoCuts (TV-style multi-story bulletins).
    #
    #   use_pip stays True -- but _render_impl gates the actual PiP
    #     source pick on metadata.video_type so SOLO talking-head
    #     videos don't end up with a corner inset of the same anchor.
    #
    #   use_sidebar_carousel stays True (the entity-image carousel is
    #     a pure overlay; doesn't extend bulletin duration).
    use_sidebar_carousel: bool = True
    use_pip: bool = True
    use_takeovers: bool = False

    # Item 103: drop sub-cuts shorter than this threshold after the
    # splice_cuts_minus_skipped pass. 1.5s is the default per
    # operator decision -- short enough to keep an intentional
    # one-word reaction, long enough to drop a 0.3s sliver between
    # consecutive retakes that otherwise produces visible chop in
    # the bulletin. Set to 0.0 to disable micro-fragment dropping.
    micro_fragment_threshold_s: float = MICRO_FRAGMENT_THRESHOLD_S

    # Item 104: operator's bulletin transition selection. One of the
    # catalog names in pipeline_v2.transitions. Stored verbatim so
    # the UI can echo the operator's original choice; the renderer
    # falls back to "smart_cut" via resolve_for_render when the
    # selected entry is not yet implemented (one structured warning
    # logged per render so the operator sees the fallback).
    transition_style: str = "smart_cut"

    # Item 105: subtract long inter-word silences (> threshold) from
    # each kept cut's time range. Pure word-time arithmetic, runs
    # after splice_cuts_minus_skipped + before collapse_micro_fragments.
    # Set to 0.0 to disable silence trimming.
    silence_trim_threshold_s: float = SILENCE_TRIM_THRESHOLD_S

    # Item 105: source Stage 1 word array used by silence detection.
    # The orchestrator populates this after constructing the renderer.
    # When None or empty, silence trimming is a no-op (the renderer
    # has no source-time references to compute gaps from).
    original_words: Optional[list] = None

    # Instance-level cache, populated by Step 9.2 image resolution.
    # Maps canonical entity_name -> absolute path of resolved image
    # file. Shared between shorts and bulletin passes so the same
    # entity isn't re-downloaded twice.
    image_pool: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        # Normalise paths to Path objects (callers may pass strings)
        self.output_dir = Path(self.output_dir)
        self.video_path = Path(self.video_path)
        # Ensure output_dir exists. We do NOT create the video_path
        # parent or do any I/O on the video file -- the source must
        # already exist by the time Stage 4 runs (it was produced
        # by Stage 0).
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 9.1: raw cut wiring -------------------------------------

    def cut_raw_shorts(
        self,
        shorts_cuts: list[ShortsCut],
        metadata: Metadata,
    ) -> list[dict]:
        """Cut the source video into per-shorts raw segments.

        Wires V1's ``cut_video_clips`` against converted V2 inputs.
        V1 mutates each clip dict in place to add ``raw_path`` (the
        absolute path to the produced ``raw_clip_NN.mp4``) and
        ``duration_sec``. Returns the mutated clip dict list.

        On idempotent re-runs (V1's --resume-dir cache), already-cut
        clips are reused without re-encoding (V1 checks size >100KB).

        Per D-9.7: clip dicts where the raw cut failed to produce a
        valid file will be missing ``raw_path``. This method returns
        ALL dicts (including failed ones); the >50% guardrail is
        applied at the compose stage (Step 9.2), not here.
        """
        v1_clips = shorts_cuts_to_v1_clip_dicts(shorts_cuts, metadata)
        # V1's signature: cut_video_clips(video_path: str, clips: list, output_dir: str)
        _v1_cut_video_clips(
            str(self.video_path),
            v1_clips,
            str(self.output_dir),
        )
        return v1_clips

    def cut_raw_bulletin_stories(
        self,
        full_video_cuts: list[FullVideoCut],
        metadata: Metadata,
        parent_v2_indexes: Optional[list[int]] = None,
    ) -> list[dict]:
        """Cut the source video into per-story raw segments for the
        bulletin pass.

        Same wiring as ``cut_raw_shorts`` but with full_video_cuts and
        a SEPARATE output directory (``bulletin_dir`` instead of
        ``output_dir``).

        ``parent_v2_indexes`` (item 100) is the parallel list returned
        by ``splice_cuts_minus_skipped`` -- it stamps each V1 clip
        dict with ``parent_v2_index`` so the compositor only inserts
        takeover transitions between DIFFERENT parent FullVideoCuts.

        Why bulletin_dir, not output_dir (Step 12.2a re-run #4 fix):
        V1's ``cut_video_clips`` is path-cache idempotent -- if
        ``<output_dir>/raw_clip_NN.mp4`` already exists at >100KB it
        SKIPS the cut and reuses the cached file
        (pipeline.py:1235). V2 calls cut_video_clips twice in the
        same render (once for shorts, once for bulletin); both
        passes ask V1 to write ``raw_clip_NN.mp4`` to the same
        output_dir, so the bulletin pass would silently reuse the
        shorts' raw_clip files. With shorts typically being 15-60s
        and the bulletin's full_video_cuts spanning the full
        source video, the cached-reuse produces a bulletin
        truncated to the first short's duration (the symptom that
        triggered this fix). Writing to ``bulletin_dir`` gives the
        bulletin pass a dedicated namespace so V1's cache check
        never collides with the shorts pass.
        """
        v1_clips = full_video_cuts_to_v1_clip_dicts(
            full_video_cuts, metadata,
            parent_v2_indexes=parent_v2_indexes,
        )
        _v1_cut_video_clips(
            str(self.video_path),
            v1_clips,
            str(self.bulletin_dir),
        )
        return v1_clips

    # ---- Step 9.2: image resolution + compose dispatcher -------------

    def resolve_images(
        self,
        image_plan: ImagePlan,
        entities: list[Entity],
        kept_clip_dicts: list[dict],
        full_metadata: Metadata,
        whisper_words: list = None,
        video_duration_sec: Optional[float] = None,
    ) -> list[dict]:
        """Resolve V2's image_plan to actual on-disk image files.

        Sequence (post-12.2a Bug #1/#2/#3 fixes + Bug #4 sourcing):

          1. Walk unique entity_names from image_plan.entries. For
             each not already in ``self.image_pool``, call
             ``ImageSourcer.source_for_entity`` -- applies the
             locked policy (PERSON -> search-only, never generate;
             non-PERSON -> search-first, generate as last resort).
             Successful resolutions land in ``self.image_pool``.

          2. Build the V1-shape ``image_plan`` dict list (with
             synthetic ``id`` keys per D-9.2's "V2->V1 boundary
             contract") + the V1-shape ``pool_manifest`` keyed by
             those same synthetic ids.

          3. Call V1's ``resolve_image_plan`` (NO ``output_dir``
             kwarg -- V1's signature is positional ``image_plan``
             + keyword ``pool_manifest, kept_clips, whisper_words,
             video_duration_sec``). V1 maps each entry onto the
             stitched bulletin timeline.

        Per D-9.5: ``image_pool`` is INSTANCE STATE. Same entity
        resolved across both shorts and bulletin passes downloads
        ONCE. The first pass populates the pool; subsequent passes
        short-circuit the sourcing step.

        Per D-9.4: ``whisper_words`` defaults to ``None`` -- Stage 3c
        already aligned timestamps against full_video_cuts boundaries.

        Backlog item 60: V1's own image_plan flow
        (``generate_image_pool_from_plan``) pure-generates for every
        entity including PERSON, violating the locked V2 policy. V2
        does NOT inherit that behaviour; the policy lives in
        ``ImageSourcer`` and runs ahead of V1's mapping function.
        """
        # ---- 1. Source images for entities not already in pool -----
        # Build a {canonical_name -> Entity} lookup for the sourcer.
        # Per the locked policy, entries whose entity_name isn't in
        # canonical_entities default to EntityType.OTHER (safe: the
        # non-PERSON branch tries search-first, generation last).
        entity_by_name: dict[str, Entity] = {
            e.canonical_name: e for e in entities
        }
        brief = (full_metadata.overall_summary or "")[:200]
        sourcer = ImageSourcer(language=full_metadata.language)
        sourcing_out_dir = self.output_dir / "v2_images"

        # Walk unique entity_names in first-appearance order so the
        # logs read in the same order as Stage 3c's image_plan.
        seen: set[str] = set()
        for entry in image_plan.entries:
            name = entry.entity_name
            if name in seen:
                continue
            seen.add(name)
            if name in self.image_pool:
                # Cross-pass cache hit (D-9.5).
                continue
            entity = entity_by_name.get(name)
            if entity is None:
                # Orphan entity_name (no matching canonical entity).
                # Stage 3c's post-validate already drops orphans, so
                # this is a defensive path. Build a typed OTHER stub
                # so the sourcer's policy branch still works.
                entity = Entity(
                    canonical_name=name,
                    native_name=entry.entity_name_native or name,
                    first_mention_word_idx=0,
                    type=EntityType.OTHER,
                    mentions=[0],
                )
            path = sourcer.source_for_entity(
                entity, brief=brief, out_dir=sourcing_out_dir,
            )
            if path is not None:
                self.image_pool[name] = path

        # ---- 2. Build V1-shape inputs -----------------------------
        v1_image_plan = image_plan_to_v1_dict(image_plan, entities)
        synth_id_by_name = synth_id_map_for_image_plan(image_plan)

        # V1's pool_manifest shape (per pipeline.py:1563-1571):
        #   { eid: {"id": eid, "topic_clue": clue, "path": render_path,
        #           "description": desc, ...} }
        # Keyed by entity ID (the synth_id we just generated). V1
        # reads ``pool_manifest[eid]["path"]`` so the ``path`` key is
        # the contract; the rest are diagnostic fields.
        pool_manifest: dict = {}
        # First-appearance description per entity (for diagnostics)
        first_desc: dict[str, str] = {}
        for entry in image_plan.entries:
            first_desc.setdefault(entry.entity_name, entry.description)
        for name, path in self.image_pool.items():
            synth_id = synth_id_by_name.get(name)
            if synth_id is None:
                # Pool has an entry for an entity not referenced by
                # this image_plan (e.g. seeded from a prior pass).
                # Skip -- V1 only needs synth_ids referenced by
                # the current v1_image_plan list.
                continue
            pool_manifest[synth_id] = {
                "id":          synth_id,
                "topic_clue":  name,
                "description": first_desc.get(name, ""),
                "path":        str(path),
            }

        # ---- 3. Call V1's resolve_image_plan ----------------------
        # V1's signature (pipeline.py:1582):
        #   resolve_image_plan(image_plan, pool_manifest, kept_clips,
        #                      whisper_words, video_duration_sec)
        # No ``output_dir`` kwarg -- V1 is a pure timeline mapper,
        # not an image producer.
        resolved = _v1_resolve_image_plan(
            v1_image_plan,
            pool_manifest=pool_manifest,
            kept_clips=kept_clip_dicts,
            whisper_words=whisper_words,        # None per D-9.4
            video_duration_sec=video_duration_sec,
        )

        # ---- 4. Defensive pool refresh from resolved --------------
        # The sourcing pre-step (1) already populated the pool with
        # everything V1 will resolve "ready". This loop is a safety
        # net: if V1 returns a ``ready`` entry referencing a path
        # we somehow didn't track, we still cache it. Harmless when
        # the pre-step has already done the work.
        for r in resolved:
            name = r.get("entity_name", "")
            img_path = r.get("image_path", "")
            status = r.get("status", "")
            if name and img_path and status == "ready":
                self.image_pool[name] = Path(img_path)

        logger.info(
            "stage_4: resolved %d image_plan entries (pool now has %d "
            "entities cached)",
            len(resolved), len(self.image_pool),
        )
        return resolved

    def _dispatch_compose(
        self,
        raw_path: str,
        image_path: str,
        card_text: str,
        out_path: str,
        lang_font_basename: str,
        lang_follow_text: str,
    ) -> dict:
        """Dispatch to the correct V1 compose function based on
        ``self.frame_layout``. Returns the per-clip param dict
        (card_params, split_params, follow_params) that V1 normally
        stores in the editor_meta clip entry.

        Calls into V1 verbatim with the exact arg bundle V1's
        ``_compose_one`` uses (pipeline.py:4494-4565). Any deviation
        from these defaults would break the byte-identical contract.
        """
        if self.frame_layout == "split_frame":
            _v1_compose_split_frame(
                raw_path, image_path, out_path, self.preset,
                platform=self.platform,
            )
            return {
                "card_params":  {"font_file": lang_font_basename},
                "split_params": {"bg_color": "#1a0a2e"},
                "follow_params": {},
            }
        if self.frame_layout == "clean_card":
            compose_meta = _v1_compose_clean_card(
                raw_path, image_path, card_text, out_path, self.preset,
                font_size=80,
                font_file=lang_font_basename,
                bg_color="#c10000",
                video_pct=0.50,
                headline_pct=0.18,
                image_h_pct=0.30,
                image_w_pct=0.80,
                image_border_px=14,
                image_border_color="#ffffff",
                platform=self.platform,
            )
            return {
                "card_params": {
                    "font_size": (compose_meta or {}).get("font_size", 80),
                    "font_file": (compose_meta or {}).get("font_file", lang_font_basename),
                    "bg_color":  "#c10000",
                    "image_border_px":    14,
                    "image_border_color": "#ffffff",
                },
                "split_params":  {},
                "follow_params": {},
            }
        if self.frame_layout == "follow_bar":
            velvet = {
                "c-top":"#2d0a4e","c-bot":"#1a0a2e","c-vdark":"#0a001a",
                "c-vlight":"#3d0060","patch-scale":80,"octaves":5,
                "contrast":107,"brightness":35,"warp":55,"warp-scale":65,
                "grain":14,"edge-dark":33,"c-dot":"#7b3fb8","dot-op":38,
                "dot-r":5,"dot-sp":18,"dot-rows":5,"dot-cols":5,
            }
            _v1_compose_follow_bar(
                raw_path, out_path, self.preset,
                title_text=card_text,
                font_file=lang_font_basename,
                text_color="#ffff00",
                bg_color="#1a0a2e",
                follow_text=lang_follow_text,
                velvet_style=velvet,
                platform=self.platform,
            )
            return {
                "card_params": {
                    "font_file": lang_font_basename,
                    "font_size": 60,
                    "text_color": "#ffff00",
                },
                "split_params":  {},
                "follow_params": {
                    "bg_color": "#1a0a2e",
                    "text_color": "#ffff00",
                    "follow_text": lang_follow_text,
                    "follow_text_color": "#ffffff",
                    "social_logos": [],
                    "velvet_style": velvet,
                },
            }
        # Default: torn_card (V1's most common layout per D-9.10)
        compose_meta = _v1_compose_clip(
            raw_path, image_path, card_text, out_path, self.preset,
            font_size=80,
            font_file=lang_font_basename,
            section_pct={"video": 0.4619, "text": 0.1691, "image": 0.3690},
            card_style={
                "card_c0": "#c10000", "card_c1": "#800000",
                "edge": 9, "jag": 60, "seed": 7,
                "vsid": 35, "vcor": 72, "vwid": 74, "overlap": 20,
            },
            platform=self.platform,
        )
        return {
            "card_params": {
                "font_size": (compose_meta or {}).get("font_size", 80),
                "font_file": (compose_meta or {}).get("font_file", lang_font_basename),
                "card_c0": "#c10000", "card_c1": "#800000",
                "edge": 9, "jag": 60, "seed": 7,
                "vsid": 35, "vcor": 72, "vwid": 74, "overlap": 20,
            },
            "split_params":  {},
            "follow_params": {},
        }

    def _generate_thumbnail(self, composed_path: str, thumb_path: str) -> bool:
        """Generate a one-frame thumbnail for a composed clip.

        Returns True on success, False on FFmpeg failure (per V1's
        defensive pattern at pipeline.py:4569-4575 -- thumbnail
        failure is non-fatal; the clip still ships, the thumb_path
        just ends up empty in editor_meta).
        """
        try:
            subprocess.run(
                [
                    _v1_FFMPEG_BIN, "-y", "-i", composed_path,
                    "-vframes", "1", "-q:v", "2", thumb_path,
                ],
                capture_output=True, check=True,
            )
            return True
        except Exception as exc:
            logger.warning(
                "stage_4: thumbnail generation failed for %s: %s",
                composed_path, exc,
            )
            return False

    def compose_shorts(
        self,
        clip_dicts: list[dict],
        metadata: Metadata,
        resolved_image_plan: list[dict],
    ) -> list[dict]:
        """Compose each shorts clip via V1's compose functions.

        Inputs:
          clip_dicts: post-raw-cut shorts clip dicts (have raw_path
            + duration_sec populated by V1.cut_video_clips).
          metadata: V2 Metadata (for headline, language).
          resolved_image_plan: V1.resolve_image_plan output -- maps
            each clip_index to a resolved image_path.

        Per-clip failure handling (D-9.7):
          - Each clip's compose runs in a try/except. On failure,
            log a structured warning and skip the clip (don't add to
            output).
          - After processing all clips, if (failures / total) >
            drop_ratio_threshold, raise RuntimeError -> Inngest
            retries the step.

        Returns:
          list of clip dicts (subset of input) with these additional
          keys populated by this method:
            clip_path:     absolute path to the composed clip_NN.mp4
            thumb_path:    absolute path to thumb_NN.jpg (or "")
            image_path:    absolute path to the news image used
            card_params, split_params, follow_params: render config
        """
        lang_cfg = _v1_languages.get(metadata.language.split("-", 1)[0])
        # Resolve the font basename (V1 stores the full path; we want
        # the basename for the editor_meta clip dict).
        lang_font_basename = os.path.basename(lang_cfg.font_primary) if lang_cfg.font_primary else ""
        lang_follow_text = lang_cfg.follow_bar_text

        # Per-shorts image assignment (Step 12.2a re-run #4 fix):
        # Build an overlay list of (source_show_at_sec, image_path) for
        # every "ready" entry in ``resolved_image_plan``. For each
        # shorts cut at [start_sec, end_sec], we pick the first overlay
        # whose ``source_show_at_sec`` falls inside that window. When
        # no overlay matches, we round-robin through ``self.image_pool``
        # by short index so each short still gets a DIFFERENT image.
        #
        # Why not key by ``clip_index`` like the previous implementation:
        # ``ImagePlanEntry.clip_index`` references the bulletin's
        # ``full_video_cuts[idx]``, NOT the shorts cut index. Reusing
        # that field for the shorts mapping meant every shorts cut
        # looked up bulletin-cut-0's image, collapsing every short to
        # the same picture. The show_at_sec overlap is the correct
        # cross-cut bridge because Stage 3c emits per-moment timestamps
        # in source-video time, which is the same coordinate space
        # the shorts cuts use.
        #
        # Defensive: ``source_show_at_sec`` is emitted by V1's
        # resolve_image_plan only for entries it successfully
        # timeline-mapped (pipeline.py:1635). Entries that fell into
        # an "image_missing" / "in_cut_span" branch never get this
        # field. We exclude them from overlap matching.
        overlays: list[tuple[float, str]] = []
        for r in resolved_image_plan:
            if r.get("status") != "ready":
                continue
            img_path = r.get("image_path", "")
            show_at = r.get("source_show_at_sec")
            if not img_path or show_at is None:
                continue
            try:
                overlays.append((float(show_at), img_path))
            except (TypeError, ValueError):
                continue

        pool_values = list(self.image_pool.values())

        card_text = metadata.shorts_headline_native or "KAIZER NEWS"
        composed: list[dict] = []
        failures: list[dict] = []
        total = len(clip_dicts)

        for i, clip in enumerate(clip_dicts):
            raw_path = clip.get("raw_path", "")
            if not raw_path or not os.path.exists(raw_path):
                failures.append({"i": i, "reason": "missing_raw_path"})
                logger.warning(
                    "stage_4: skipping clip %d -- raw_path missing or "
                    "file not found: %s", i, raw_path,
                )
                continue

            # ---- Per-clip image selection (overlap first, RR fallback) ----
            clip_start = float(clip.get("start_sec") or 0.0)
            clip_end = float(clip.get("end_sec") or 0.0)
            image_path = ""
            for show_at, img in overlays:
                if clip_start <= show_at <= clip_end:
                    image_path = img
                    break
            if not image_path and pool_values:
                # Round-robin fallback: cycle the resolved image pool
                # by short index so each short gets a different image
                # even when no entity overlay falls inside its window.
                image_path = str(pool_values[i % len(pool_values)])
            if not image_path:
                failures.append({"i": i, "reason": "no_image_available"})
                logger.warning(
                    "stage_4: skipping clip %d -- no image available "
                    "in resolved plan or pool", i,
                )
                continue

            out_path = str(self.output_dir / f"clip_{i+1:02d}.mp4")
            thumb_path = str(self.output_dir / f"thumb_{i+1:02d}.jpg")

            try:
                params = self._dispatch_compose(
                    raw_path, image_path, card_text, out_path,
                    lang_font_basename, lang_follow_text,
                )
            except Exception as exc:
                failures.append({"i": i, "reason": f"compose_failed: {exc!r}"})
                logger.warning(
                    "stage_4: compose failed for clip %d (%s): %s",
                    i, self.frame_layout, exc,
                )
                continue

            thumb_ok = self._generate_thumbnail(out_path, thumb_path)
            if not thumb_ok:
                thumb_path = ""

            # Enrich the clip dict with composed-output fields.
            clip["clip_path"]  = os.path.abspath(out_path)
            clip["thumb_path"] = os.path.abspath(thumb_path) if thumb_path else ""
            clip["image_path"] = os.path.abspath(image_path)
            clip.update(params)
            composed.append(clip)

        # Per D-9.7: >50% failure ratio -> raise so Inngest retries.
        if total > 0 and (len(failures) / total) > self.drop_ratio_threshold:
            raise RuntimeError(
                f"Stage 4 compose_shorts: {len(failures)}/{total} clips "
                f"failed "
                f"({len(failures)/total:.0%} > "
                f"{self.drop_ratio_threshold:.0%} threshold). Indicates "
                f"systemic render failure. Failures: {failures!r}. "
                f"Inngest will retry the step."
            )

        if failures:
            logger.info(
                "stage_4: composed %d/%d clips (skipped %d below the "
                "%d%% threshold)",
                len(composed), total, len(failures),
                int(self.drop_ratio_threshold * 100),
            )

        return composed

    # ---- Step 9.3: bulletin assembly (long-form pass) ----------------

    @property
    def bulletin_dir(self) -> Path:
        """Bulletin subdirectory of ``output_dir``. Created lazily on
        first access. V1's bulletin loop puts all assembly artifacts
        (composed_story_NN.mp4, _sidebar_NN.mp4, _ticker.png, _bug.png,
        takeover_NN.mp4, bulletin.mp4) under this dir.
        """
        d = self.output_dir / "bulletin"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _images_for_story(self, clip: dict) -> list[str]:
        """Pull image paths usable for a single bulletin story.

        For now we use the image_pool populated by ``resolve_images``
        (which feeds both shorts and bulletin passes per D-9.5).
        Returns up to 5 image paths -- the limit V1 uses for sidebar
        carousels (``imgs[:5]``).

        If we ever need per-story image generation distinct from the
        global pool (V1 has _gen_for_story for this), add a separate
        method; this default is correct for V2's design where Stage
        3c's image_plan already selected the right images.
        """
        return [str(p) for p in list(self.image_pool.values())[:5]]

    def render_bulletin(
        self,
        full_video_cuts: list[FullVideoCut],
        metadata: Metadata,
        entities: list[Entity],
        image_plan: ImagePlan,
        channel_name: str = "",
        logo_path: Optional[str] = None,
        parent_v2_indexes: Optional[list[int]] = None,
        takeovers_enabled: Optional[bool] = None,
        pip_enabled: Optional[bool] = None,
    ) -> dict:
        """Assemble the full-form bulletin video.

        Sequence (matches V1's run_pipeline bulletin loop at
        pipeline.py:4023-4316):

          1. Cut raw segments from source (V1.cut_video_clips)
          2. Resolve images into self.image_pool (V1.resolve_image_plan,
             reuses pool from prior shorts pass per D-9.5)
          3. Render shared overlays ONCE:
               - ticker      (V1.render_ticker)
               - channel bug (V1.render_channel_bug)
          4. Per story:
               - Build sidebar (V1.build_sidebar_carousel or
                 make_sidebar_placeholder)
               - Pick PiP source if use_pip (V1.pick_pip_source)
               - Compose story (V1.compose_pip_story or
                 compose_bulletin_story)
               - compose_deps content-aware cache check at each step
                 so a --resume-dir re-run skips already-built segments
               - Optional inter-story takeover (V1.build_fullscreen_takeover)
          5. Stitch story segments into bulletin.mp4 (V1.stitch_bulletin)
          6. Apply image_plan overlays -> bulletin_with_overlays.mp4
             (V1.overlay_image_plan)

        Returns a dict with keys: bulletin_path (carousel-only),
        overlay_path (with image_plan applied; same as bulletin_path
        if no overlays applied), duration_s, stories_rendered,
        stories_skipped, warnings.
        """
        lang_cfg = _v1_languages.get(metadata.language.split("-", 1)[0])
        bulletin_dir = self.bulletin_dir

        # Item 100: resolve effective takeovers / pip toggles. Callers
        # (i.e. _render_impl) can override via explicit args; if not
        # provided fall back to the instance defaults (False / True
        # respectively).
        effective_takeovers = (
            takeovers_enabled if takeovers_enabled is not None
            else self.use_takeovers
        )
        effective_pip = (
            pip_enabled if pip_enabled is not None
            else self.use_pip
        )

        # ---- 1. Raw cut for the bulletin pass --------------------
        clips = self.cut_raw_bulletin_stories(
            full_video_cuts, metadata,
            parent_v2_indexes=parent_v2_indexes,
        )

        # ---- 2. Image resolution (reuses image_pool per D-9.5) ----
        # Stage 3c's image_plan already targets the bulletin's
        # full_video_cuts -- pass them as kept_clips so V1's
        # resolve_image_plan can validate boundaries the same way it
        # did for shorts (the post-validate happened in Stage 3c
        # already, but V1's resolver does its own checks).
        resolved = self.resolve_images(
            image_plan, entities,
            kept_clip_dicts=clips,
            full_metadata=metadata,
        )

        # ---- 3. Shared overlays (rendered ONCE for the bulletin) --
        ticker_path = str(bulletin_dir / "_ticker.png")
        ticker_inputs = [lang_cfg.font_primary] if lang_cfg.font_primary else []
        ticker_headlines = list(metadata.bulletin_marquee_points) or ["KAIZER NEWS"]
        ticker_extra = {
            "headlines": ticker_headlines,
            "lang": lang_cfg.code,
        }
        if _v1_compose_deps.is_fresh(ticker_path, ticker_inputs, ticker_extra, min_size=1000):
            logger.info("stage_4: bulletin ticker cached (skipping)")
        else:
            try:
                _v1_render_ticker(
                    ticker_headlines, lang_cfg.code,
                    lang_cfg.font_primary, ticker_path,
                )
                _v1_compose_deps.mark_built(ticker_path, ticker_inputs, ticker_extra)
            except Exception as exc:
                logger.warning("stage_4: ticker render failed: %s", exc)
                ticker_path = ""

        # Backlog item 92: when channel_name is empty (the V2 default),
        # skip the channel bug entirely so the bulletin renders as a
        # CLEAN MASTER. V1's pattern is to overlay the destination
        # logo at upload/publish time per-destination
        # (youtube/logo_overlay.py); baking a fixed "KAIZER NEWS" pill
        # at render time both duplicates that step and forces the
        # same brand mark across every channel a clip is published to.
        if not channel_name:
            logger.info(
                "stage_4: channel bug skipped (clean master -- channel "
                "logo applied at upload time per destination)"
            )
            bug_path = None
        else:
            bug_path = str(bulletin_dir / "_bug.png")
            bug_inputs = [logo_path] if logo_path else []
            bug_extra = {"channel_name": channel_name}
            if _v1_compose_deps.is_fresh(bug_path, bug_inputs, bug_extra, min_size=500):
                logger.info("stage_4: bulletin channel bug cached (skipping)")
            else:
                try:
                    _v1_render_channel_bug(channel_name, logo_path, bug_path)
                    _v1_compose_deps.mark_built(bug_path, bug_inputs, bug_extra)
                except Exception as exc:
                    logger.warning("stage_4: channel bug render failed: %s", exc)
                    bug_path = None

        # ---- 4. Per-story compose -----------------------------------
        story_paths: list[str] = []
        failures: list[dict] = []
        total = len(clips)

        for i, clip in enumerate(clips):
            raw_path = clip.get("raw_path", "")
            if not raw_path or not os.path.isfile(raw_path):
                failures.append({"i": i, "reason": "missing_raw_path"})
                logger.warning(
                    "stage_4: bulletin story %d skipped -- raw_path "
                    "missing or empty: %s", i, raw_path,
                )
                continue

            story_dur_s = float(clip.get("duration_sec") or 60.0)
            imgs = self._images_for_story(clip)

            # ---- 4a. Sidebar (carousel or placeholder) -----------
            sidebar_path: Optional[str] = None
            sidebar_is_video = False
            if self.use_sidebar_carousel and len(imgs) >= 2:
                sidebar_video = str(bulletin_dir / f"_sidebar_{i:02d}.mp4")
                sidebar_inputs = list(imgs[:5])
                sidebar_extra = {"duration_s": round(story_dur_s, 3)}
                if _v1_compose_deps.is_fresh(sidebar_video, sidebar_inputs, sidebar_extra):
                    logger.info("stage_4: sidebar story %d cached", i)
                    sidebar_path = sidebar_video
                    sidebar_is_video = True
                else:
                    try:
                        _v1_build_sidebar_carousel(
                            imgs[:5], story_dur_s, sidebar_video,
                            work_dir=str(bulletin_dir / f"_sidebar_work_{i:02d}"),
                        )
                        _v1_compose_deps.mark_built(sidebar_video, sidebar_inputs, sidebar_extra)
                        sidebar_path = sidebar_video
                        sidebar_is_video = True
                    except Exception as exc:
                        logger.warning(
                            "stage_4: sidebar carousel story %d failed: %s",
                            i, exc,
                        )
                        sidebar_path = None
            if sidebar_path is None:
                # Fallback to static placeholder (V1's pattern, also
                # used when use_sidebar_carousel=False)
                sidebar_static = str(bulletin_dir / f"_sidebar_{i:02d}.png")
                try:
                    _v1_make_sidebar_placeholder(
                        imgs[0] if imgs else None,
                        sidebar_static,
                    )
                    sidebar_path = sidebar_static
                except Exception as exc:
                    logger.warning(
                        "stage_4: sidebar placeholder story %d failed: %s",
                        i, exc,
                    )
                    sidebar_path = None

            # ---- 4b. StoryMeta + PiP source picking --------------
            importance = int(clip.get("importance") or 5)
            kicker = "BREAKING" if importance >= 8 else "NEWS"
            story_meta = _V1StoryMeta(
                title=(metadata.shorts_headline_native or "KAIZER NEWS")[:200],
                kicker=kicker,
                language=lang_cfg.code,
                story_index=i,
                total_stories=total,
                importance=importance,
            )

            # Item 100: PiP source picker is gated on effective_pip
            # (which _render_impl sets to False for SOLO video_type).
            pip_src = _v1_pick_pip_source(clips, i) if effective_pip else None

            # ---- 4c. compose_deps fingerprint for this story -----
            composed_path = str(bulletin_dir / f"composed_story_{i:02d}.mp4")
            composed_inputs = [
                raw_path, sidebar_path, ticker_path, bug_path,
                lang_cfg.font_primary,
            ]
            if pip_src:
                composed_inputs.append(pip_src[0])
            composed_extra = {
                "title":       story_meta.title,
                "kicker":      story_meta.kicker,
                "language":    story_meta.language,
                "importance":  int(story_meta.importance),
                "story_index": int(story_meta.story_index),
                "total":       int(story_meta.total_stories),
                "use_pip":     bool(pip_src),
                "pip_start_s": round(float(pip_src[1]), 3) if pip_src else None,
                "pip_dur_s":   round(float(pip_src[2]), 3) if pip_src else None,
                "sidebar_is_video": bool(sidebar_is_video),
            }

            composed_ok = False
            if _v1_compose_deps.is_fresh(composed_path, composed_inputs, composed_extra):
                logger.info("stage_4: composed_story_%02d.mp4 cached", i)
                composed_ok = True
            else:
                try:
                    if pip_src and sidebar_path and ticker_path:
                        pip_clip, pip_start, pip_dur = pip_src
                        _v1_compose_pip_story(
                            raw_path, story_meta, composed_path,
                            pip_clip_path=pip_clip,
                            pip_start_s=pip_start,
                            pip_duration_s=pip_dur,
                            sidebar_path=sidebar_path,
                            ticker_path=ticker_path,
                            channel_bug_path=bug_path,
                            font_path=lang_cfg.font_primary,
                            sidebar_is_video=sidebar_is_video,
                            work_dir=str(bulletin_dir),
                        )
                        composed_ok = True
                    elif sidebar_path and ticker_path:
                        _v1_compose_bulletin_story(
                            raw_path, story_meta, composed_path,
                            sidebar_path=sidebar_path,
                            ticker_path=ticker_path,
                            channel_bug_path=bug_path,
                            font_path=lang_cfg.font_primary,
                            sidebar_is_video=sidebar_is_video,
                            work_dir=str(bulletin_dir),
                        )
                        composed_ok = True
                except Exception as exc:
                    logger.warning(
                        "stage_4: compose story %d failed: %s", i, exc,
                    )

            if composed_ok and os.path.isfile(composed_path):
                _v1_compose_deps.mark_built(composed_path, composed_inputs, composed_extra)
                story_paths.append(composed_path)
            else:
                # Fall back to raw slice so the story still ships
                # (V1's pattern: pipeline.py:4290).
                story_paths.append(raw_path)
                failures.append({"i": i, "reason": "compose_fallback_to_raw"})
                logger.warning(
                    "stage_4: bulletin story %d fell back to raw slice",
                    i,
                )

            # ---- 4d. Inter-story takeover (optional) -------------
            # Item 100: gate on effective_takeovers (caller controls
            # globally via _render_impl: only ON when Stage 2 produced
            # >=3 distinct FullVideoCuts). Additionally, within the
            # enabled case, ONLY insert a takeover when this story
            # belongs to a DIFFERENT parent_v2_index than the next
            # one -- so spliced sub-cuts of the same narrative thread
            # concat seamlessly (no dead-air silence between them).
            this_parent = clip.get("parent_v2_index", clip.get("v2_index"))
            next_parent = None
            if i + 1 < total:
                next_clip = clips[i + 1]
                next_parent = next_clip.get(
                    "parent_v2_index", next_clip.get("v2_index")
                )
            inter_story_boundary = _should_insert_takeover_between(
                this_parent, next_parent,
            )
            if (
                effective_takeovers
                and inter_story_boundary
                and 0 < i < total - 1
                and len(imgs) >= 2
            ):
                takeover_dur = max(4.0, min(8.0, story_dur_s * 0.10))
                takeover_path = str(bulletin_dir / f"takeover_{i:02d}.mp4")
                takeover_inputs = list(imgs[:4])
                takeover_extra = {"duration_s": round(takeover_dur, 3)}
                if _v1_compose_deps.is_fresh(takeover_path, takeover_inputs, takeover_extra):
                    logger.info("stage_4: takeover_%02d.mp4 cached", i)
                    story_paths.append(takeover_path)
                else:
                    try:
                        _v1_build_fullscreen_takeover(
                            imgs[:4], takeover_dur, takeover_path,
                            work_dir=str(bulletin_dir / f"_takeover_work_{i:02d}"),
                        )
                        _v1_compose_deps.mark_built(takeover_path, takeover_inputs, takeover_extra)
                        story_paths.append(takeover_path)
                    except Exception as exc:
                        logger.warning(
                            "stage_4: takeover %d failed (skipping): %s",
                            i, exc,
                        )

        # ---- 4.5. Per-story guardrail (D-9.7) ----------------------
        if total > 0 and (len(failures) / total) > self.drop_ratio_threshold:
            raise RuntimeError(
                f"Stage 4 render_bulletin: {len(failures)}/{total} "
                f"stories failed "
                f"({len(failures)/total:.0%} > "
                f"{self.drop_ratio_threshold:.0%} threshold). Indicates "
                f"systemic bulletin compose failure. Failures: "
                f"{failures!r}. Inngest will retry the step."
            )

        if not story_paths:
            raise RuntimeError(
                "Stage 4 render_bulletin: no story segments produced. "
                "Bulletin cannot be stitched. Inngest will retry."
            )

        # ---- 5. Stitch story segments into bulletin.mp4 ------------
        bulletin_out = str(bulletin_dir / "bulletin.mp4")
        try:
            stitch_result = _v1_stitch_bulletin(
                story_paths,
                bulletin_out,
                work_dir=str(bulletin_dir),
            )
        except _V1BulletinStitchError as exc:
            raise RuntimeError(
                f"Stage 4 render_bulletin: stitch_bulletin failed: "
                f"{exc!r}. Inngest will retry."
            ) from exc

        total_duration_s = float(getattr(stitch_result, "total_duration_s", 0.0) or 0.0)
        stories_rendered = int(getattr(stitch_result, "stories_rendered", len(story_paths)))
        stories_skipped = int(getattr(stitch_result, "stories_skipped", 0))
        stitch_warnings = list(getattr(stitch_result, "warnings", []))

        # ---- 6. Apply image_plan overlays --------------------------
        overlay_out = str(bulletin_dir / "bulletin_with_overlays.mp4")
        overlay_applied = False
        if image_plan.entries:
            ready_count = sum(1 for r in resolved if r.get("status") == "ready")
            if ready_count > 0:
                if (
                    os.path.exists(overlay_out)
                    and os.path.getsize(overlay_out) > 100_000
                ):
                    logger.info(
                        "stage_4: bulletin_with_overlays.mp4 cached"
                    )
                    overlay_applied = True
                else:
                    try:
                        _v1_overlay_image_plan(bulletin_out, resolved, overlay_out)
                        overlay_applied = os.path.exists(overlay_out) and os.path.getsize(overlay_out) > 100_000
                    except Exception as exc:
                        logger.warning(
                            "stage_4: overlay_image_plan failed "
                            "(keeping un-overlaid bulletin): %s", exc,
                        )

        return {
            "bulletin_path":     bulletin_out,
            "overlay_path":      overlay_out if overlay_applied else bulletin_out,
            "overlay_applied":   overlay_applied,
            "duration_s":        total_duration_s,
            "stories_rendered":  stories_rendered,
            "stories_skipped":   stories_skipped,
            "warnings":          stitch_warnings,
        }

    # ---- Step 9.4: top-level render() orchestrator ------------------

    def render(
        self,
        job_output: JobOutput,
        timestamp: str,
        title_english: str = "",
        channel_name: str = "",
        logo_path: Optional[str] = None,
        cancel_check: Optional[callable] = None,
        progress_cb: Optional[callable] = None,
    ) -> RenderResult:
        """End-to-end Stage 4: shorts pass + bulletin pass.

        Sequence:
          1. SHORTS pass:
             a. Cut raw shorts segments
             b. Resolve images (populates image_pool per D-9.5)
             c. Compose each short via _dispatch_compose
             d. Build shorts editor_meta.json via Step 8 adapter
             e. Write {output_dir}/editor_meta.json
          2. BULLETIN pass (image_pool reused from step 1):
             a. render_bulletin -- the full long-form assembly
             b. Build bulletin editor_meta.json via Step 8 adapter
             c. Write {output_dir}/bulletin/editor_meta.json

        Edge cases:
          - 0 shorts_cuts: skip shorts pass; shorts_editor_meta_path
            is None. Bulletin still runs.
          - 0 full_video_cuts: skip bulletin pass;
            bulletin_editor_meta_path is None. Shorts still runs.
          - Both empty: returns RenderResult with both paths None and
            empty composed_shorts -- unusual but legitimate.

        Errors propagate (Inngest retries at outer step). Per-clip /
        per-story 50% guardrails fire inside the sub-renders; this
        method does not add additional guardrails.

        Step 10.3 wrapper: classify exceptions via
        ``_classify_render_error`` and convert known-permanent
        conditions to ``PermanentRenderError`` so the orchestrator's
        Stage 4 handler can map to Inngest ``NonRetriableError``.
        Other exceptions propagate as-is for Inngest retry.

        ``cancel_check`` parameter (Step 12.3 Test 2 fix, backlog
        item 76): a callable invoked between every Stage 4 sub-phase
        (cut_raw_shorts / resolve_images / compose_shorts /
        render_bulletin). The callable should raise (typically
        ``NonRetriableError``) when the user has requested cancel,
        which propagates up through ``render()`` and out to the
        orchestrator's terminal catch -- which marks the Job as
        failed and short-circuits the run. Without this, a mid-
        Stage-4 cancel waited the full ~5 min for the stage to
        finish before the cooperative check at finalize fired.
        Defaults to ``None`` (no cancel checks) so existing unit
        tests + the V1-only call path are unaffected.
        """
        try:
            return self._render_impl(
                job_output, timestamp,
                title_english=title_english,
                channel_name=channel_name,
                logo_path=logo_path,
                cancel_check=cancel_check,
                progress_cb=progress_cb,
            )
        except PermanentRenderError:
            raise   # already classified; don't re-classify
        except Exception as exc:
            slug = _classify_render_error(exc)
            if slug is not None:
                raise PermanentRenderError(slug) from exc
            raise

    def _render_impl(
        self,
        job_output: JobOutput,
        timestamp: str,
        title_english: str = "",
        channel_name: str = "",
        logo_path: Optional[str] = None,
        cancel_check: Optional[callable] = None,
        progress_cb: Optional[callable] = None,
    ) -> RenderResult:
        """Internal: the real render() body. Wrapped by render()'s
        try/except to convert known-permanent failures into
        PermanentRenderError.

        ``cancel_check`` (when provided) is invoked between sub-phases
        so a user cancel mid-Stage-4 short-circuits within seconds
        rather than waiting for the whole stage to complete. The
        callable raises on cancel (typically NonRetriableError);
        propagation exits ``_render_impl`` and runs through render()'s
        classifier and out to the orchestrator's terminal catch.
        """
        shorts_cuts = job_output.shorts_cuts
        full_video_cuts = job_output.stage_two.full_video_cuts
        metadata = job_output.metadata
        entities = list(job_output.canonical_entities)
        image_plan = job_output.image_plan

        composed_shorts: list[dict] = []
        shorts_editor_meta_path: Optional[str] = None
        bulletin_result: Optional[dict] = None
        bulletin_editor_meta_path: Optional[str] = None

        # Tiny helper: invoke progress_cb defensively (callback errors
        # MUST NOT take down a render). Backlog item 88.
        def _p(msg: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(msg)
                except Exception as exc:
                    logger.warning(
                        "stage_4 progress_cb raised (ignored): %s", exc,
                    )

        # ---- 1. SHORTS pass ------------------------------------------
        if shorts_cuts:
            if cancel_check is not None:
                cancel_check()
            _p(f"Stage 5/7 cutting {len(shorts_cuts)} raw shorts segments")
            # Backlog item 98 -- emit one line per short cut so the V1
            # "Cutting clip N: HH:MM:SS -> HH:MM:SS (Xs)" verbosity is
            # preserved on V2 too.
            for sc in shorts_cuts:
                dur = sc.end_sec - sc.start_sec
                _p(
                    f"  Cutting short {sc.index + 1}/{len(shorts_cuts)}: "
                    f"{_format_mmss_mmm(sc.start_sec)} -> "
                    f"{_format_mmss_mmm(sc.end_sec)} ({dur:.1f}s)"
                )
            shorts_clips = self.cut_raw_shorts(shorts_cuts, metadata)
            _p(f"Stage 5/7 raw shorts cut ({len(shorts_clips)} clips on disk)")
            if cancel_check is not None:
                cancel_check()
            entity_count = len(entities)
            _p(
                f"Stage 6/7 sourcing images for {entity_count} entities: "
                + ", ".join(e.canonical_name for e in entities[:6])
                + (" …" if len(entities) > 6 else "")
            )
            resolved_shorts = self.resolve_images(
                image_plan, entities,
                kept_clip_dicts=shorts_clips,
                full_metadata=metadata,
            )
            # Surface what got sourced (entity_name -> path basename +
            # provider tag derived from the synth filename pattern).
            for ent_name, img_path in (self.image_pool or {}).items():
                try:
                    basename = Path(img_path).name
                    _p(f"  [image] {ent_name} -> {basename}")
                except Exception:
                    pass
            _p(f"Stage 6/7 images resolved ({len(self.image_pool)} sourced)")
            if cancel_check is not None:
                cancel_check()
            _p(f"Stage 6/7 composing {len(shorts_clips)} shorts with overlays")
            composed_shorts = self.compose_shorts(
                shorts_clips, metadata, resolved_shorts,
            )
            for cs in composed_shorts:
                clip_name = Path(cs.get("clip_path", "")).name or "<unknown>"
                v2_idx = cs.get("v2_index", "?")
                _p(f"  [short {v2_idx}] composed {clip_name}")
            _p(f"Stage 6/7 shorts composed ({len(composed_shorts)} produced)")

            # Build ClipRenderArtifacts list (one per produced short).
            # composed_shorts has clip_path / thumb_path / image_path
            # populated by compose_shorts. The shorts adapter needs
            # one ClipRenderArtifacts per ShortsCut in job_output --
            # if any clip got dropped by the guardrail, the artifact
            # count would mismatch.
            artifacts = [
                ClipRenderArtifacts(
                    clip_path=c.get("clip_path", ""),
                    raw_path=c.get("raw_path", ""),
                    thumb_path=c.get("thumb_path", ""),
                    image_path=c.get("image_path", ""),
                    storage_url=c.get("storage_url", ""),
                    storage_key=c.get("storage_key", ""),
                    storage_backend=c.get("storage_backend", ""),
                )
                for c in composed_shorts
            ]

            # If clip count mismatch (compose_shorts dropped some),
            # we'd violate the shorts adapter's contract. Re-shape
            # job_output's shorts_cuts list to match the kept clips.
            if len(artifacts) == len(shorts_cuts):
                shorts_job_for_adapter = job_output
            else:
                kept_indices = {c["v2_index"] for c in composed_shorts}
                kept_shorts = [c for c in shorts_cuts if c.index in kept_indices]
                # Rebuild with 0-based contiguous indices to satisfy
                # the adapter's D-8.12 contiguity guardrail.
                renumbered = [
                    ShortsCut(
                        index=i,
                        start_sec=c.start_sec, end_sec=c.end_sec,
                        hook=c.hook, importance=c.importance,
                    )
                    for i, c in enumerate(kept_shorts)
                ]
                shorts_job_for_adapter = job_output.model_copy(
                    update={"shorts_cuts": renumbered},
                )

            shorts_meta = build_v1_shorts_editor_meta(
                shorts_job_for_adapter,
                video_path=str(self.video_path),
                platform=self.platform,
                frame_layout=self.frame_layout,
                preset=self.preset,
                timestamp=timestamp,
                clip_artifacts=artifacts,
                title_english=title_english,
            )
            shorts_meta_path = self.output_dir / "editor_meta.json"
            shorts_meta_path.write_text(
                json.dumps(shorts_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shorts_editor_meta_path = str(shorts_meta_path)
            logger.info(
                "stage_4: wrote shorts editor_meta -> %s (%d clips)",
                shorts_editor_meta_path, len(composed_shorts),
            )

        # ---- 2. BULLETIN pass ----------------------------------------
        if full_video_cuts:
            if cancel_check is not None:
                cancel_check()
            # Backlog item 97: splice out the SkippedSegments before
            # handing cuts to the renderer. Stage 2's prompt
            # explicitly lets FullVideoCuts span over skipped regions
            # and expects the renderer to do the splice. Previously
            # we passed the raw cuts to V1's cut_video_clips, which
            # cut ONE contiguous segment per cut = bulletin matched
            # mezzanine duration with zero trim.
            skipped_segments = job_output.stage_two.skipped_segments
            spliced_cuts, parent_v2_indexes = splice_cuts_minus_skipped(
                full_video_cuts, skipped_segments,
            )
            original_kept_dur = sum(c.end_sec - c.start_sec for c in full_video_cuts)
            spliced_dur = sum(c.end_sec - c.start_sec for c in spliced_cuts)
            trimmed_s = max(0.0, original_kept_dur - spliced_dur)
            _p(
                f"Stage 6/7 rendering bulletin "
                f"({len(full_video_cuts)} cut span -> {len(spliced_cuts)} "
                f"keep sub-segments after splicing "
                f"{len(skipped_segments)} skipped; trimmed {trimmed_s:.1f}s)"
            )

            # Item 104: resolve operator's transition selection -> what
            # the renderer will actually apply. Today only smart_cut is
            # implemented; other catalog entries fall back to smart_cut
            # with a single structured progress line so the operator
            # sees their choice didn't take effect (no silent fallback).
            try:
                from pipeline_v2.transitions import (
                    resolve_for_render as _resolve_transition,
                    get_transition as _get_transition,
                )
                _selected = _get_transition(self.transition_style)
                _effective = _resolve_transition(self.transition_style)
                _p(
                    f"  [transition] selected={_selected.name!r} "
                    f"effective={_effective.name!r} "
                    f"(implemented={_selected.implemented})"
                )
                if _effective.name != _selected.name:
                    _p(
                        f"  [transition] NOTE: {_selected.name!r} is not "
                        f"yet implemented; falling back to "
                        f"{_effective.name!r}. Operator's choice is "
                        f"preserved on the Job row for UI display."
                    )
            except Exception as _trans_exc:
                logger.warning(
                    "stage_4: transition resolution failed (non-fatal, "
                    "falling back to smart_cut): %s", _trans_exc,
                )

            # Item 105: silence trim. Subtract long inter-word silences
            # detected in the Stage 1 word array from each kept cut.
            # No-op when no original_words were plumbed through OR
            # threshold is 0. Runs BEFORE collapse_micro_fragments so
            # any silence-induced micro-fragments get cleaned up by
            # item 103's drop pass.
            if (
                self.silence_trim_threshold_s > 0
                and self.original_words
                and spliced_cuts
            ):
                silence_trims = detect_silence_trims(
                    list(self.original_words),
                    threshold_s=self.silence_trim_threshold_s,
                )
                if silence_trims:
                    pre_silence_count = len(spliced_cuts)
                    pre_silence_dur = sum(
                        c.end_sec - c.start_sec for c in spliced_cuts
                    )
                    spliced_cuts, parent_v2_indexes = apply_silence_trims_to_cuts(
                        spliced_cuts, parent_v2_indexes, silence_trims,
                    )
                    post_silence_dur = sum(
                        c.end_sec - c.start_sec for c in spliced_cuts
                    )
                    _p(
                        f"  [silence-trim] removed "
                        f"{pre_silence_dur - post_silence_dur:.2f}s "
                        f"of silence > {self.silence_trim_threshold_s:.1f}s "
                        f"(cuts {pre_silence_count} -> "
                        f"{len(spliced_cuts)} after splits)"
                    )

            # Item 103: drop micro-fragments < threshold (default 1.5s)
            # so the bulletin doesn't include tiny slivers between
            # consecutive retakes that produce visible chop.
            if self.micro_fragment_threshold_s > 0 and spliced_cuts:
                pre_count = len(spliced_cuts)
                pre_dur = sum(c.end_sec - c.start_sec for c in spliced_cuts)
                spliced_cuts, parent_v2_indexes = collapse_micro_fragments(
                    spliced_cuts, parent_v2_indexes,
                    threshold_s=self.micro_fragment_threshold_s,
                )
                post_count = len(spliced_cuts)
                post_dur = sum(c.end_sec - c.start_sec for c in spliced_cuts)
                if post_count != pre_count:
                    _p(
                        f"  [micro-fragments] dropped "
                        f"{pre_count - post_count} sub-cut(s) "
                        f"< {self.micro_fragment_threshold_s:.1f}s "
                        f"(removed {pre_dur - post_dur:.2f}s)"
                    )

            # Item 106: defensive guardrails. Round to ms precision
            # (Bug C) then verify the chain produced a sorted,
            # non-overlapping cut list (Bug B). round_cut_precision
            # raises on zero/negative-duration cuts; assert_cuts_monotonic
            # raises on overlap. Both surface upstream regressions
            # at render time rather than via a broken bulletin.
            if spliced_cuts:
                spliced_cuts = round_cut_precision(
                    spliced_cuts, decimals=CUT_PRECISION_DECIMALS,
                )
                assert_cuts_monotonic(spliced_cuts)
            # Backlog item 98: list each sub-cut so the operator can see
            # exactly which spans got stitched together.
            for sc in spliced_cuts:
                dur = sc.end_sec - sc.start_sec
                _p(
                    f"  [bulletin sub {sc.index}] "
                    f"{_format_mmss_mmm(sc.start_sec)} -> "
                    f"{_format_mmss_mmm(sc.end_sec)} ({dur:.1f}s)"
                )

            # Item 100/102: adaptive takeovers + PiP gate via pure
            # helpers. Behaviour identical to the inline expressions
            # the old code used; extracted so both branches have
            # direct unit-test coverage (TestAdaptiveTakeoverGate /
            # TestPiPGate in test_stage_4_render.py).
            takeovers_enabled = _compute_takeovers_enabled(
                self.use_takeovers, full_video_cuts,
            )
            _p(
                f"  [takeovers] enabled={takeovers_enabled} "
                f"(use_takeovers={self.use_takeovers}, "
                f"full_video_cuts={len(full_video_cuts)} -- "
                f"need >={TAKEOVER_MIN_DISTINCT_CUTS} distinct stories)"
            )

            pip_enabled = _compute_pip_enabled(
                self.use_pip, metadata.video_type,
            )
            _p(
                f"  [pip] enabled={pip_enabled} "
                f"(use_pip={self.use_pip}, "
                f"video_type={metadata.video_type})"
            )

            bulletin_result = self.render_bulletin(
                full_video_cuts=spliced_cuts,
                metadata=metadata,
                entities=entities,
                image_plan=image_plan,
                channel_name=channel_name,
                logo_path=logo_path,
                parent_v2_indexes=parent_v2_indexes,
                takeovers_enabled=takeovers_enabled,
                pip_enabled=pip_enabled,
            )
            _p(
                f"Stage 6/7 bulletin assembled "
                f"({bulletin_result.get('duration_s', 0):.0f}s; "
                f"applying overlays now)"
            )

            # Backlog item 96: write bulletin/_generated_images.json so
            # routers/bulletin_images.py's editor "Images" panel can
            # surface the sourced images for the operator's
            # replace/recompose feature. V2 stores its entity images
            # in <out>/v2_images/<entity>__<source>_NN.jpg; the panel
            # scans bulletin/ for either a manifest OR the V1-shape
            # story_NN_assets/ / _job_pool/ layouts -- so without a
            # manifest pointing back at v2_images, the panel showed
            # "No bulletin images on disk yet."
            try:
                v2_images_dir = self.output_dir / "v2_images"
                if v2_images_dir.is_dir():
                    img_entries = []
                    for i, p in enumerate(sorted(v2_images_dir.iterdir()), start=1):
                        if not p.is_file():
                            continue
                        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                            continue
                        img_entries.append({
                            "path":        str(p.resolve()),
                            "filename":    f"news_{i:02d}{p.suffix.lower()}",
                            "story_index": 0,
                        })
                    if img_entries:
                        manifest_path = self.bulletin_dir / "_generated_images.json"
                        manifest_path.write_text(
                            json.dumps(img_entries, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        logger.info(
                            "stage_4: wrote bulletin image manifest -> %s "
                            "(%d entries)",
                            manifest_path, len(img_entries),
                        )
            except Exception as exc:
                logger.warning(
                    "stage_4: bulletin image manifest write failed (non-"
                    "fatal -- the Images panel will fall through to "
                    "filesystem-scan and may not see V2 entity images): %s",
                    exc,
                )

            # Item 100: A/V invariant guardrail. The bulletin's audio
            # duration MUST equal the sum of composed_story audio
            # durations PLUS the sum of takeover VIDEO durations
            # (takeovers have no audio of their own, so the concat
            # step pads them with silence -- that's intentional
            # padding when takeovers are enabled at story boundaries).
            # Any larger delta indicates an UNINTENDED source of
            # video-without-audio is bloating the bulletin -- a
            # future intro/outro insertion, a stray ffmpeg filter
            # that pads, etc. We raise a hard error so the operator
            # sees the regression immediately rather than discovering
            # it via a "bulletin is 25% too long" support ticket.
            try:
                bulletin_to_check = (
                    bulletin_result.get("overlay_path")
                    or bulletin_result.get("bulletin_path")
                )
                if bulletin_to_check and os.path.isfile(bulletin_to_check):
                    actual_audio = _ffprobe_audio_duration_s(bulletin_to_check)
                    composed_audio_total = sum(
                        _ffprobe_audio_duration_s(p) for p in story_paths
                        if os.path.isfile(p) and "takeover_" not in os.path.basename(p)
                    )
                    takeover_video_total = sum(
                        _ffprobe_video_duration_s(p) for p in story_paths
                        if os.path.isfile(p) and "takeover_" in os.path.basename(p)
                    )
                    expected = composed_audio_total + takeover_video_total
                    delta = actual_audio - expected
                    _p(
                        f"Stage 6/7 A/V invariant check: "
                        f"bulletin audio {actual_audio:.1f}s == "
                        f"narration {composed_audio_total:.1f}s + "
                        f"transitions {takeover_video_total:.1f}s "
                        f"(delta {delta:+.1f}s)"
                        + (" OK" if abs(delta) <= AV_INVARIANT_TOLERANCE_S else " FAIL")
                    )
                    # Item 102: delegate the threshold check to the
                    # pure helper so the violation branch has direct
                    # unit-test coverage.
                    _validate_av_invariant(
                        actual_audio_s=actual_audio,
                        composed_narration_s=composed_audio_total,
                        takeover_video_s=takeover_video_total,
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                # Probing failures shouldn't crash the render -- log and
                # continue. Only invariant violations (RuntimeError
                # above) are hard.
                logger.warning(
                    "stage_4: A/V invariant probe failed (non-fatal): %s",
                    exc,
                )

            bulletin_artifacts = ClipRenderArtifacts(
                clip_path=bulletin_result["overlay_path"],
                clip_path_overlay=bulletin_result["overlay_path"]
                                  if bulletin_result["overlay_applied"]
                                  else "",
                clip_path_carousel_only=bulletin_result["bulletin_path"]
                                        if bulletin_result["overlay_applied"]
                                        else "",
            )

            bulletin_meta = build_v1_bulletin_editor_meta(
                job_output,
                platform=self.platform,
                bulletin_artifacts=bulletin_artifacts,
                bulletin_duration_s=bulletin_result["duration_s"],
            )
            bulletin_meta_path = self.bulletin_dir / "editor_meta.json"
            bulletin_meta_path.write_text(
                json.dumps(bulletin_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            bulletin_editor_meta_path = str(bulletin_meta_path)
            logger.info(
                "stage_4: wrote bulletin editor_meta -> %s "
                "(%d stories, %.1fs)",
                bulletin_editor_meta_path,
                bulletin_result["stories_rendered"],
                bulletin_result["duration_s"],
            )

        return RenderResult(
            shorts_editor_meta_path=shorts_editor_meta_path,
            bulletin_editor_meta_path=bulletin_editor_meta_path,
            composed_shorts=composed_shorts,
            bulletin=bulletin_result,
        )
