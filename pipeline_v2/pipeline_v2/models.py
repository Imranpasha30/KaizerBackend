"""Pydantic v2 schemas for every stage boundary.

Verbatim from the source-of-truth implementation plan's "Data Models"
section. Every LLM response and inter-stage payload is validated through
these models — no raw dicts cross stage boundaries.

Pydantic v2 syntax notes:
  - ``@field_validator("end_sec")`` defaults to ``mode="after"`` — the
    validator receives the already-coerced value and ``info.data``
    contains fields that were declared earlier in the model class.
  - ``Field(max_length=6)`` enforces list length at validation time
    (Pydantic v2 replacement for v1's ``max_items``).
  - PEP-604 union syntax (``int | None``) and PEP-585 generics
    (``list[Word]``, ``dict[int, tuple[int, int]]``) require Python 3.11+
    which the project already targets.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum


class Word(BaseModel):
    w: str
    s: float
    e: float
    speaker: int | None = None
    # Per-word confidence. Convention: [0.0, 1.0] but not validated --
    # providers expose this on different scales (Chirp 3 ~ [0, 1],
    # Whisper log-prob normalised to ~ [0, 1], some return None). Stage
    # 2 may use this to weight cut-boundary decisions.
    confidence: float | None = None


class WordLevelTranscript(BaseModel):
    """Backend-agnostic word-level transcript.

    Originally named ``DeepgramTranscript`` (the implementation plan's
    proposed backend), then briefly carried a faster-whisper-specific
    flavour. After Step 4's expansion to a multi-provider STT layer,
    the schema is the architectural contract: every provider (Chirp 3,
    Whisper-Groq, Deepgram, AssemblyAI, Sarvam, future) emits exactly
    this shape. ``provider`` records which one produced it.

    CRITICAL: ``words`` must contain word-level timestamps -- segment-
    or sentence-level outputs are an architectural violation. The
    dispatcher in ``pipeline_v2.stages.stt`` enforces a non-empty word
    list before returning.
    """

    words: list[Word]
    duration_sec: float
    detected_languages: list[str]
    # Registry key of the producing provider, e.g. "chirp3",
    # "whisper-groq". Set by the provider; the dispatcher validates
    # it matches the dispatched provider name.
    provider: str


class SkippedCategory(str, Enum):
    warm_up = "warm_up"
    retake = "retake"
    crew_talk = "crew_talk"
    hesitation = "hesitation"
    aside = "aside"
    self_correction = "self_correction"


class SkippedSegment(BaseModel):
    start_word_idx: int
    end_word_idx: int
    start_sec: float
    end_sec: float
    category: SkippedCategory
    reason: str


class FullVideoCut(BaseModel):
    index: int
    start_word_idx: int
    end_word_idx: int
    start_sec: float
    end_sec: float
    importance: int = Field(ge=1, le=10)


class CleanTranscript(BaseModel):
    words: list[Word]
    clip_boundaries: dict[int, tuple[int, int]]
    source_word_map: list[int]   # clean_idx -> original_idx


class StageTwoOutput(BaseModel):
    """Full Stage 2 return: LLM decisions + reconstructed clean transcript.

    .. note:: clip_boundaries vs full_video_cuts asymmetry

       Some entries in ``full_video_cuts`` may not have corresponding
       ``clean_transcript.clip_boundaries`` entries -- specifically,
       any ``FullVideoCut`` whose ``[start_word_idx, end_word_idx]``
       range is ENTIRELY covered by ``skipped_segments`` is dropped
       from ``clip_boundaries`` (with a structured warning) but
       remains in ``full_video_cuts`` for telemetry. This preserves
       the LLM's full intent without silently editing it.

       Renderer / downstream consumers MUST gate every cut with::

           bounds = stage_two.clean_transcript.clip_boundaries.get(cut.index)
           if bounds is None:
               continue   # cut was entirely skipped
           clean_start, clean_end = bounds
           ...

       Iterating ``full_video_cuts`` without this check will produce
       KeyError on the boundary lookup OR render empty clips.
    """

    full_video_cuts: list[FullVideoCut]
    skipped_segments: list[SkippedSegment]
    clean_transcript: CleanTranscript
    retake_audit: str


class Stage2Output(BaseModel):
    """LLM contract for Stage 2 (Continuity Editor) -- what Gemini emits.

    Passed as ``response_schema=`` to ``client.models.generate_content()``
    so Gemini structures its JSON output to round-trip through this
    schema. The FULL stage return shape (``StageTwoOutput`` above)
    is built from this PLUS a deterministically-reconstructed
    ``CleanTranscript`` (Step 5.4's job; the LLM never emits word-level
    transcript data -- it would burn output tokens and risk
    hallucination).

    Why ``retake_audit`` is OPTIONAL here even though Step 0's V1
    prompt makes it MANDATORY (the "HARD RULES" forbid omitting it):
      - We mirror the mandatory rule in the prompt body (Step 5.3)
      - But keep the Pydantic field nullable so that if the model
        somehow omits it under token pressure, ``response.parsed``
        still returns a valid Stage2Output instead of silently None
        (recall: google-genai 2.4.0 returns None on schema mismatch
        without raising; see Step 5 research notes)
      - The corrective-retry layer (Step 5.2) catches an empty/missing
        retake_audit and prompts for a re-emit.

    .. note:: full_video_cuts vs downstream clip_boundaries

       Every cut emitted here will appear in the final
       ``StageTwoOutput.full_video_cuts``. BUT not every cut will
       have a corresponding entry in
       ``StageTwoOutput.clean_transcript.clip_boundaries`` -- a cut
       whose entire word range overlaps a SkippedSegment is dropped
       from ``clip_boundaries`` during reconstruction (Step 5.4).
       Downstream renderers MUST gate via
       ``clip_boundaries.get(cut.index)`` before rendering. See
       ``StageTwoOutput`` docstring for the canonical iteration
       pattern.
    """

    full_video_cuts: list[FullVideoCut]
    skipped_segments: list[SkippedSegment]
    retake_audit: str | None = Field(
        default=None,
        description=(
            "One-sentence prose summary of what was skipped and why. "
            "Operator-debug field -- useful for spot-checking the "
            "Continuity Editor's reasoning without having to read every "
            "skipped_segments entry. The prompt forbids omitting it; "
            "this Pydantic field is optional only as a safety net "
            "against silent None responses."
        ),
    )


class EntityType(str, Enum):
    """Locked entity-type taxonomy for Stage 2.5.

    Mirrors the SkippedCategory pattern: 5 literal strings, no
    invented values at either prompt or schema level. The
    forbid-invention rule is enforced in the prompt body
    (``stage_2_5_prompt.md``) and at validation time here.

    Edge cases (mapped in the prompt's edge-case table):
      - "scheme name" / "policy name" -> EVENT
      - "law / act name" / "RTI" -> OTHER
      - "minister portfolio name" (e.g. "Finance Minister") -> ORG
      - "company brand" -> ORG
      - "city/state/country" -> PLACE
    """
    PERSON = "PERSON"
    ORG = "ORG"
    PLACE = "PLACE"
    EVENT = "EVENT"
    OTHER = "OTHER"


class Entity(BaseModel):
    """One canonicalized entity surfaced across the clean transcript.

    Naming note (per Step 6 D12): the class is ``Entity`` (identity --
    what it IS) rather than ``CanonicalEntity`` (process -- what's
    been done to it). The class doesn't know it's canonical; Stage
    2.5 makes it canonical. ``JobOutput.canonical_entities`` is then
    "a list of Entity objects (which happen to have been
    canonicalized)" -- the field name carries the process meaning,
    the type name carries the identity meaning.

    All ``mentions`` indices reference the CLEAN transcript word
    array, not the original Stage 1 word array. To map back to
    original word indices (and therefore original timestamps), use
    ``clean_transcript.source_word_map[mentions[i]]``.
    """
    canonical_name: str
    """English / Latin-script canonical form. e.g. 'Revanth Reddy'."""

    native_name: str
    """Native-script form. e.g. 'రేవంత్ రెడ్డి'. Per D4: required
    non-empty. For entities that have no native rendering (English-
    only acronyms like 'BRS', 'RTI'), set this equal to canonical_name
    -- never leave it blank."""

    first_mention_word_idx: int
    """Index into clean_transcript.words of this entity's first
    appearance. Always == min(mentions)."""

    type: EntityType

    mentions: list[int]
    """All clean-transcript word indices where this entity (or any
    of its aliases) appears. Ascending order. Length >= 1."""


class Stage2_5Output(BaseModel):
    """LLM contract for Stage 2.5 (Entity Canonicalizer) -- what
    Gemini Flash emits.

    Hard cap of 6 entities is enforced at THREE layers (per D3):

      1. Prompt body (``stage_2_5_prompt.md``'s HARD RULES section).
      2. Pydantic ``Field(max_length=6)`` -- compiles to JSON Schema
         ``maxItems`` which Gemini's structured-output engine sees.
         NOTE: empirically a SOFT hint on Flash (per google-genai
         issue #699), not a hard guarantee. Treat as belt-and-
         suspenders with layer 3.
      3. Post-validate truncation in
         ``Stage2_5EntityCanonicalizer._truncate_to_cap()``: if the
         LLM still returns 7+ entities, sort by mention count DESC
         (tiebreak: first_mention_word_idx ASC) and keep top 6, with
         a structured logger.warning() naming the dropped entities.

    The product reasoning behind the sort (D3-ADDITIONAL): the
    renderer uses entities for image overlays. Entities mentioned
    more times deserve more screen time. Late-introduced entities
    with one mention are unlikely to carry visual weight.
    """
    entities: list[Entity] = Field(max_length=6)


class ShortsCut(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    hook: str
    importance: int = Field(ge=1, le=10)

    @field_validator("end_sec")
    @classmethod
    def duration_range(cls, v, info):
        start = info.data.get("start_sec", 0)
        d = v - start
        if not (15 <= d <= 60):
            raise ValueError(f"shorts must be 15-60s, got {d}s")
        return v


class Stage3aOutput(BaseModel):
    """LLM contract for Stage 3a (Shorts Generator) -- what Gemini
    Flash emits.

    Pydantic-side count cap is min=3 / max=10 (loose enough for
    creative output, tight enough to catch obvious drift). The
    acceptance target is 5-10 shorts; the dispatcher does NOT
    truncate below this band -- prompt + temperature 0.7 produce
    enough variety that natural rejection of duplicates lands here
    anyway. If the LLM emits 11+, Pydantic's max_length=10 catches
    it (soft hint to Gemini; hard enforce at validation).
    """
    shorts_cuts: list[ShortsCut] = Field(min_length=3, max_length=10)


class Metadata(BaseModel):
    video_type: Literal["SOLO", "INTERVIEW", "PRESS_CONFERENCE", "PANEL", "MIXED"]
    language: str
    total_speakers: int
    overall_summary: str
    overall_summary_native: str
    shorts_headline_native: str
    bulletin_marquee_points: list[str]
    image_search_queries: list[str]
    key_people: list[str]
    key_people_native: list[str]
    key_topics: list[str]
    key_locations: list[str]


class ImagePlanEntry(BaseModel):
    """One image overlay scheduled inside a bulletin clip.

    Trimmed from the original 10-field shape (Step 7.1 / D-7.2) to
    drop the V1-abandoned fields (``id``, ``topic_clue``,
    ``search_query``, ``search_query_native``, ``reason``). Gemini
    burns output tokens on every field; trimming saves ~40% of
    Stage 3c's output bytes and removes the Instagram-Reel "?" id
    bug as a side effect (see post_v2_backlog.md).

    Entity reference (D-7.3): ``entity_name`` is a string that MUST
    match one of ``Stage2_5Output.entities[].canonical_name``. The
    Step 7.4 dispatcher post-validates: entries whose ``entity_name``
    has no matching canonical entity are dropped with a structured
    logger.warning() (never silent). ``entity_name_native`` carries
    the native-script form for on-screen overlay text.

    Boundary contract (D-7.10): ``show_at_sec + duration_sec`` MUST
    fall entirely inside the FullVideoCut at ``clip_index``. Stage
    3c's dispatcher post-validates this too; out-of-bounds entries
    are dropped with a warning. If MORE THAN 50% of entries are
    dropped (boundary OR orphan-entity reasons), the dispatcher
    raises -- that ratio indicates systemic prompt/model failure,
    not occasional outliers.
    """
    entity_name: str
    entity_name_native: str
    description: str
    clip_index: int
    show_at_sec: float
    duration_sec: float = Field(ge=2.0)


class ImagePlan(BaseModel):
    entries: list[ImagePlanEntry]


# --- Stage 0 (ingest) --------------------------------------------------
# Not part of the inter-LLM contract but useful for telemetry +
# downstream stages that need duration. Added here so all v2 typed
# payloads live in one module.


class Stage0Output(BaseModel):
    mezzanine_path: str
    audio_path: str
    duration_sec: float
    encoder_used: Literal["h264_nvenc", "libx264"]
    width: int | None = None
    height: int | None = None
    source_was_vfr: bool
    transcode_seconds: float
    audio_extract_seconds: float
    wall_seconds: float


# --- Stage 1 (transcribe) ----------------------------------------------


class Stage1Output(BaseModel):
    """Stage 1 output: word-level transcript + per-job STT metadata.

    Provider-agnostic. The ``stt_*`` fields make up the per-job audit /
    cost ledger that gets persisted to
    ``jobs/{job_id}/v2/stt_metadata.json`` (see Q5 in Step 4.0 scope
    confirmation). A global rollup ledger is out of scope for Step 4
    and lives in Step 12.5's audit-notes system.
    """

    transcript: WordLevelTranscript
    transcript_json_path: str | None = None
    metadata_json_path: str | None = None

    # ---- Per-job STT metadata (the audit/cost ledger fields) ----
    stt_provider: str                                # registry key, e.g. "chirp3"
    stt_audio_duration_sec: float                    # provider's reported duration
    stt_wall_seconds: float                          # dispatcher-measured wall clock
    stt_cost_usd: float                              # provider-computed cost
    stt_word_count: int
    stt_avg_confidence: float | None = None          # None if no provider per-word conf
    stt_language_detected: str
    stt_request_id: str                              # vendor's tracing ID

    # ---- Input echoes (handy for debugging without joining tables) ----
    stt_language_hint: str | None = None
    stt_brief: str | None = None
    stt_names: list[str] = Field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        """audio_duration / wall_seconds. Higher = faster than realtime."""
        if self.stt_wall_seconds <= 0:
            return float("inf")
        return self.stt_audio_duration_sec / self.stt_wall_seconds


class JobOutput(BaseModel):
    stage_two: StageTwoOutput
    canonical_entities: list[Entity]
    shorts_cuts: list[ShortsCut]
    metadata: Metadata
    image_plan: ImagePlan
