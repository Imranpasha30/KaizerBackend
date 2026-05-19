"""V2 JobOutput -> V1 editor_meta.json adapter (Step 8).

V1's render path reads ``editor_meta.json`` from disk; V2 produces a
typed ``JobOutput`` Pydantic object. This module is the bridge:
**pure data transformation**, no I/O, no SDK, no side effects.

V1 emits TWO different editor_meta.json shapes (discovered while
mapping fields):

  - **Shorts shape** (used by ``youtube_short`` + ``instagram_reel``):
    multi-clip output with per-clip render config, language metadata,
    title/summary in 3 scripts, etc. Emitted by V1's pipeline.py
    around line 4645. Multi-entry ``clips[]``, one per short.

  - **Bulletin shape** (used by ``youtube_full``): single-clip
    output describing the assembled bulletin. Top-level keys are
    different (``render_mode``, ``stories``, ``skipped``,
    ``duration_s``). Emitted by V1's pipeline.py around line 4356.
    One-entry ``clips[]``.

V2's "Full Video + Shorts" platform produces BOTH outputs. The
adapter exposes two functions; the caller (Stage 9 renderer)
invokes both and writes two files.

Pure-function contract (D-8.14): no SDK clients, no env vars, no
filesystem I/O. Caller passes everything in. Tests are trivial
(input -> output diff).

V2-INTERNAL FIELDS INTENTIONALLY DROPPED (per D-8.11):

The following fields appear in ``JobOutput`` / its sub-models but
are NOT surfaced in ``editor_meta.json``. **Future maintainers:**
do not add these without an explicit user decision -- they were
deliberately omitted because V1's editor schema would either reject
them or treat them as garbage.

  - ``stage_two.skipped_segments`` -- intentionally dropped. Debug
    information about what Stage 2 removed; renderer doesn't need
    it because the clean_transcript / clip_boundaries already
    encode the kept ranges.
  - ``stage_two.retake_audit`` -- intentionally dropped. Operator
    debug field; not a render input.
  - ``stage_two.clean_transcript.clip_boundaries`` -- intentionally
    dropped. V2-internal index mapping (clean array <-> clip
    space). Renderer reads original ``full_video_cuts`` and clean
    words separately.
  - ``stage_two.clean_transcript.source_word_map`` -- intentionally
    dropped. V2-internal index mapping (clean array <-> original
    Stage 1 array). Only Stage 9 render code uses it, never the
    V1 editor.
  - ``canonical_entities`` (the full Entity objects) --
    intentionally dropped at the boundary. Only ``key_people`` (the
    English names) flow through, per D-8.8. Mention indices /
    types / native names are V2-internal.
  - ``image_plan`` (entire field) -- intentionally dropped per
    D-8.11. Renderer (Stage 9) consumes ``image_plan`` directly
    from ``JobOutput`` and applies overlays during render. Adding
    image_plan to editor_meta.json would break V1's editor schema
    (V1 handles overlays internally and only updates ``clip_path``
    to point at the overlaid version).
  - ``metadata.key_people_native`` -- intentionally dropped per
    D-8.8. V1's schema has only the English ``people`` field;
    native names still surface in ``title_native`` + per-clip
    ``text``.
  - ``metadata.key_locations`` -- intentionally dropped per D-8.7.
    V1's ``keywords`` field is a separate concept (always empty in
    V1 today); not a place to dump locations.
  - ``metadata.image_search_queries`` -- intentionally dropped.
    V1's render code generates image search queries from
    ``people`` + ``topics`` + entity context, not from a
    pre-baked list.
  - ``metadata.bulletin_marquee_points`` -- intentionally dropped
    here. Used by the V2 renderer to compose the bulletin's ticker
    overlay; lives in render config, not editor metadata.
  - ``metadata.total_speakers`` -- intentionally dropped. V1's
    editor_meta doesn't track this; V1's report.txt does, but
    that's a separate artifact.
  - ``stage_two.full_video_cuts`` (FOR THE SHORTS ADAPTER only)
    -- intentionally dropped. Shorts cuts are independent of
    bulletin cuts (per Stage 3a's prompt: shorts can come from
    anywhere in clean_transcript, including outside
    ``full_video_cuts``). The bulletin adapter DOES use
    ``full_video_cuts`` to count ``stories``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# V1's language config -- single source of truth for English/native
# name + script lookups (D-8.4). KaizerBackend/ must be on sys.path
# when this module is imported. Test path is set up by
# pipeline_v2/conftest.py; production scripts add the path
# explicitly (see scripts/step5_0_v1_regression.py for the pattern).
import languages as _v1_languages

from pipeline_v2.models import JobOutput


# ====================================================================== #
# ClipRenderArtifacts -- caller-supplied per-clip render outputs         #
# ====================================================================== #


@dataclass(frozen=True)
class ClipRenderArtifacts:
    """Per-clip rendering outputs that the adapter splices into the
    V1 clip dict. Stage 9 (renderer) constructs one per produced
    short and one for the bulletin (with ``raw_path=""`` since the
    bulletin doesn't have a separate raw path).

    All fields default to empty string so callers can pass partial
    artifacts (e.g. when storage upload is disabled).
    """
    clip_path: str
    raw_path: str = ""
    thumb_path: str = ""
    image_path: str = ""
    storage_url: str = ""
    storage_key: str = ""
    storage_backend: str = ""

    # Bulletin-only optional fields. Caller passes these to the
    # bulletin adapter when the overlay step has produced both a
    # carousel-only and an overlay version of bulletin.mp4 (V1
    # tracks both in the same clip dict).
    clip_path_overlay: str = ""
    clip_path_carousel_only: str = ""


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #


def _strip_lang_suffix(code: str) -> str:
    """Strip code-mix suffix from a BCP-47-style language tag.

    V2's Stage 3b emits ``language`` like "te-en" (Telugu with English
    code-mixing). V1's ``languages.get()`` expects the base ISO 639-1
    code. Strip the first hyphen-and-after.
    """
    return code.split("-", 1)[0].strip()


def _resolve_lang_cfg(code: str):
    """Look up a language config by code, with code-mix suffix
    stripping (D-8.4). Falls back to Telugu if code is unknown
    (matches V1's ``languages.get()`` semantics).
    """
    return _v1_languages.get(_strip_lang_suffix(code))


def _format_mmss_mmm(seconds: float) -> str:
    """Format a float-second timestamp as ``MM:SS.mmm``.

    Per D-8.5: ``MM:SS.mmm`` always (minutes can exceed 60, no
    HH:MM:SS switch). Carries forward correctly on edge cases like
    59.9999 -> "01:00.000" (avoids invalid "00:60.000" output).
    """
    total_ms = round(seconds * 1000)
    total_secs = total_ms // 1000
    ms = total_ms % 1000
    minutes = total_secs // 60
    secs = total_secs % 60
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _validate_shorts_indices_contiguous(shorts_cuts) -> None:
    """Enforce D-8.12 GUARDRAIL: ShortsCut.index values must form a
    contiguous 0-based sequence ``[0, 1, 2, ..., n-1]``.

    Stage 3a's prompt requires sequential indices, but the prompt is
    advisory; this validator turns prompt violations into a clear
    ValueError at the adapter boundary (same defensive pattern as
    Step 5.4's ``build_clean_transcript`` overlap check).

    Out-of-order indices (e.g. [2, 0, 1]) are FINE -- the adapter
    sorts before emitting. Gaps and non-zero-start are NOT fine and
    raise here.

    Empty shorts_cuts is fine (no indices to validate).
    """
    if not shorts_cuts:
        return
    indices = sorted(c.index for c in shorts_cuts)
    n = len(indices)
    expected = list(range(n))
    if indices == expected:
        return
    # First-element check: must be 0
    if indices[0] != 0:
        raise ValueError(
            f"ShortsCut.index expected to start at 0, got "
            f"{indices[0]}. Sorted indices: {indices}. Stage 3a "
            f"prompt requires sequential 0-based indices."
        )
    # Walk to find the first missing index. Since indices is
    # sorted and starts at 0, the first divergence reveals what's
    # missing (or duplicate).
    for i in range(n):
        if indices[i] != i:
            raise ValueError(
                f"ShortsCut.index has a gap: missing index {i}. "
                f"Sorted indices: {indices}. Stage 3a prompt requires "
                f"sequential 0-based indices."
            )
    # Unreachable: indices != expected but all i==indices[i] up to n.
    raise ValueError(
        f"ShortsCut.index sequence is non-contiguous. Sorted: "
        f"{indices}, expected: {expected}"
    )


# ====================================================================== #
# Shorts-shape adapter                                                    #
# ====================================================================== #


def build_v1_shorts_editor_meta(
    job_output: JobOutput,
    *,
    video_path: str,
    platform: str,
    frame_layout: str,
    preset: dict,
    timestamp: str,
    clip_artifacts: list[ClipRenderArtifacts],
    title_english: str = "",
    card_params: dict = None,
    split_params: dict = None,
    follow_params: dict = None,
) -> dict:
    """V2 ``JobOutput`` -> V1 ``editor_meta.json`` (shorts shape).

    Used for the multi-short output of the V2 "Full Video + Shorts"
    platform. The renderer (Stage 9) writes the returned dict to
    a path like ``output/.../shorts/editor_meta.json`` so V1's
    editor recognises it.

    All caller-supplied args are kwarg-only for forward-compat. The
    ``clip_artifacts`` list MUST be aligned with
    ``job_output.shorts_cuts`` by index AFTER sorting -- i.e.
    ``clip_artifacts[k]`` is the render output for the ShortsCut
    with ``index == k``.

    Raises:
      ValueError: ShortsCut.index sequence is non-contiguous or
        doesn't start at 0 (D-8.12).
      ValueError: ``len(clip_artifacts) != len(shorts_cuts)``.
    """
    shorts_cuts = job_output.shorts_cuts
    _validate_shorts_indices_contiguous(shorts_cuts)
    if len(clip_artifacts) != len(shorts_cuts):
        raise ValueError(
            f"clip_artifacts length {len(clip_artifacts)} does not "
            f"match shorts_cuts length {len(shorts_cuts)}. Caller "
            f"must supply one artifact per ShortsCut."
        )
    # Sort by index ASC (D-8.12). The artifacts list is index-aligned
    # to the SORTED cuts list per the caller contract above.
    sorted_cuts = sorted(shorts_cuts, key=lambda c: c.index)

    meta = job_output.metadata
    lang_cfg = _resolve_lang_cfg(meta.language)
    lang_code = lang_cfg.code

    title_native = meta.shorts_headline_native
    summary = meta.overall_summary
    summary_native = meta.overall_summary_native

    # Default mutable args (per Python best practice, never mutable
    # default arg values in the signature).
    card_params = card_params if card_params is not None else {}
    split_params = split_params if split_params is not None else {}
    follow_params = follow_params if follow_params is not None else {}

    clips_out = []
    for cut, art in zip(sorted_cuts, clip_artifacts):
        clips_out.append({
            "clip_path":      art.clip_path,
            "raw_path":       art.raw_path,
            "thumb_path":     art.thumb_path,
            "image_path":     art.image_path,
            # text: global headline burned on every short (D-8.x).
            "text":           title_native,
            "language":       lang_code,
            "title_native":   title_native,
            # title_telugu: legacy alias for title_native -- V1 emits
            # both for backwards-compat with older editor UI code.
            "title_telugu":   title_native,
            "title_english":  title_english,
            "start":          _format_mmss_mmm(cut.start_sec),
            "end":            _format_mmss_mmm(cut.end_sec),
            "duration":       round(cut.end_sec - cut.start_sec, 2),
            # Per D-8.9: hook is the per-clip summary.
            "summary":        cut.hook,
            # mood: V2 doesn't track per-clip mood; always empty.
            "mood":           "",
            "importance":     cut.importance,
            "video_type":     meta.video_type,
            "frame_type":     frame_layout,
            "card_params":    card_params,
            "split_params":   split_params,
            "follow_params":  follow_params,
            # V1 duplicates the top-level preset onto each clip dict
            # (legacy quirk; the editor reads from clip[].preset in
            # some code paths). Preserve byte-for-byte.
            "preset":         preset,
            "storage_url":    art.storage_url,
            "storage_key":    art.storage_key,
            "storage_backend": art.storage_backend,
        })

    return {
        "video_path":       video_path,
        "platform":         platform,
        "frame_layout":     frame_layout,
        "language":         lang_code,
        "language_english": lang_cfg.name_english,
        "language_native":  lang_cfg.name_native,
        "script":           lang_cfg.script,
        "preset":           preset,
        "video_type":       meta.video_type,
        "title_native":     title_native,
        # Legacy alias for title_native.
        "title_telugu":     title_native,
        "title_english":    title_english,
        "summary":          summary,
        "summary_native":   summary_native,
        # Legacy alias for summary_native.
        "summary_telugu":   summary_native,
        # D-8.8: only English names. key_people_native is dropped.
        "people":           list(meta.key_people),
        "topics":           list(meta.key_topics),
        # D-8.7: V1's keywords is always [] in produced files; pin
        # the empty-list emission and keep key_locations out.
        "keywords":         [],
        "clips":            clips_out,
        "created":          timestamp,
    }


# ====================================================================== #
# Bulletin-shape adapter                                                  #
# ====================================================================== #


def build_v1_bulletin_editor_meta(
    job_output: JobOutput,
    *,
    platform: str,
    bulletin_artifacts: ClipRenderArtifacts,
    bulletin_duration_s: float,
) -> dict:
    """V2 ``JobOutput`` -> V1 ``editor_meta.json`` (bulletin shape).

    Used for the single assembled bulletin output of the V2 "Full
    Video + Shorts" platform. The renderer (Stage 9) writes the
    returned dict to a path like
    ``output/.../bulletin/editor_meta.json``.

    Top-level shape differs from the shorts adapter: V1 emits
    ``render_mode=\"bulletin\"`` plus ``stories`` / ``skipped`` /
    ``duration_s`` instead of the language/title/summary fields.
    """
    meta = job_output.metadata
    lang_cfg = _resolve_lang_cfg(meta.language)
    lang_code = lang_cfg.code

    # text: truncated overall_summary_native (V1's truncation point
    # is 500 chars per pipeline.py:4362).
    bulletin_text = meta.overall_summary_native[:500]

    bulletin_clip = {
        "clip_path":      bulletin_artifacts.clip_path,
        "thumb_path":     bulletin_artifacts.thumb_path,
        "image_path":     bulletin_artifacts.image_path,
        "duration":       bulletin_duration_s,
        "frame_type":     "bulletin",
        "text":           bulletin_text,
        "sentiment":      "",
        # V1 puts ``people`` here as ``entities`` (different key name;
        # see pipeline.py:4364 ``"entities": people or []``).
        "entities":       list(meta.key_people),
        "card_params":    {},
        "section_pct":    {},
        "follow_params":  {},
        "storage_url":    bulletin_artifacts.storage_url,
        "storage_key":    bulletin_artifacts.storage_key,
        "storage_backend": bulletin_artifacts.storage_backend,
        # Optional bulletin-only fields. V1's overlay step at
        # pipeline.py:3597 produces both an overlay version and a
        # carousel-only version; both paths appear in the same dict.
        # Empty strings if Stage 9 hasn't run an overlay step.
        "clip_path_overlay":        bulletin_artifacts.clip_path_overlay,
        "clip_path_carousel_only":  bulletin_artifacts.clip_path_carousel_only,
    }

    return {
        "render_mode": "bulletin",
        "platform":    platform,
        "language":    lang_code,
        "stories":     len(job_output.stage_two.full_video_cuts),
        "skipped":     len(job_output.stage_two.skipped_segments),
        "duration_s":  bulletin_duration_s,
        "clips":       [bulletin_clip],
    }
