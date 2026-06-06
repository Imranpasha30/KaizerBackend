from pipeline_v2.models import FullVideoCut, SkippedSegment, Word
import logging

logger = logging.getLogger(__name__)

SILENCE_TRIM_THRESHOLD_S = 1.5
MICRO_FRAGMENT_THRESHOLD_S = 1.5
CUT_PRECISION_DECIMALS = 3

def detect_silence_trims(
    words: list[Word],
    threshold_s: float = SILENCE_TRIM_THRESHOLD_S,
) -> list[tuple[float, float]]:
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
    silence_trims: list[tuple[float, float]],
) -> list[FullVideoCut]:
    if not silence_trims or not cuts:
        return list(cuts)

    trims_sorted = sorted(silence_trims, key=lambda t: t[0])
    out_cuts: list[FullVideoCut] = []
    
    for cut in cuts:
        parent = cut.parent_v2_index if cut.parent_v2_index is not None else cut.index
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
                parent_v2_index=parent,
            ))
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
                    parent_v2_index=parent,
                ))
            cursor = max(cursor, seg_end)
            
        if cursor < cut.end_sec:
            out_cuts.append(FullVideoCut(
                index=len(out_cuts),
                start_word_idx=cut.start_word_idx,
                end_word_idx=cut.end_word_idx,
                start_sec=cursor,
                end_sec=cut.end_sec,
                importance=cut.importance,
                parent_v2_index=parent,
            ))
            
    return out_cuts

def splice_cuts_minus_skipped(
    cuts: list[FullVideoCut],
    skipped_segments: list[SkippedSegment],
) -> list[FullVideoCut]:
    if not skipped_segments or not cuts:
        return list(cuts)

    skips_sorted = sorted(skipped_segments, key=lambda s: s.start_sec)
    out_cuts: list[FullVideoCut] = []
    
    for cut in cuts:
        parent = cut.parent_v2_index if cut.parent_v2_index is not None else cut.index
        relevant = [
            s for s in skips_sorted
            if s.end_sec > cut.start_sec and s.start_sec < cut.end_sec
        ]
        if not relevant:
            out_cuts.append(FullVideoCut(
                index=len(out_cuts),
                start_word_idx=cut.start_word_idx,
                end_word_idx=cut.end_word_idx,
                start_sec=cut.start_sec,
                end_sec=cut.end_sec,
                importance=cut.importance,
                parent_v2_index=parent,
            ))
            continue
            
        cursor = cut.start_sec
        for skip in relevant:
            seg_start = max(skip.start_sec, cut.start_sec)
            seg_end = min(skip.end_sec, cut.end_sec)
            if seg_start > cursor:
                out_cuts.append(FullVideoCut(
                    index=len(out_cuts),
                    start_word_idx=cut.start_word_idx,
                    end_word_idx=cut.end_word_idx,
                    start_sec=cursor,
                    end_sec=seg_start,
                    importance=cut.importance,
                    parent_v2_index=parent,
                ))
            cursor = max(cursor, seg_end)
            
        if cursor < cut.end_sec:
            out_cuts.append(FullVideoCut(
                index=len(out_cuts),
                start_word_idx=cut.start_word_idx,
                end_word_idx=cut.end_word_idx,
                start_sec=cursor,
                end_sec=cut.end_sec,
                importance=cut.importance,
                parent_v2_index=parent,
            ))
            
    return out_cuts

def collapse_micro_fragments(
    cuts: list[FullVideoCut],
    threshold_s: float = MICRO_FRAGMENT_THRESHOLD_S,
) -> list[FullVideoCut]:
    if not cuts or threshold_s <= 0:
        return list(cuts)

    per_parent: dict[int, list[int]] = {}
    for i, cut in enumerate(cuts):
        p = cut.parent_v2_index if cut.parent_v2_index is not None else cut.index
        per_parent.setdefault(p, []).append(i)

    keep_set: set[int] = set()
    for parent_id, idxs in per_parent.items():
        kept_above = [
            i for i in idxs
            if (cuts[i].end_sec - cuts[i].start_sec) >= threshold_s
        ]
        if kept_above:
            keep_set.update(kept_above)
        else:
            longest = max(
                idxs,
                key=lambda i: cuts[i].end_sec - cuts[i].start_sec,
            )
            keep_set.add(longest)

    out_cuts: list[FullVideoCut] = []
    for orig_i, src in enumerate(cuts):
        if orig_i not in keep_set:
            continue
        p = src.parent_v2_index if src.parent_v2_index is not None else src.index
        out_cuts.append(FullVideoCut(
            index=len(out_cuts),
            start_word_idx=src.start_word_idx,
            end_word_idx=src.end_word_idx,
            start_sec=src.start_sec,
            end_sec=src.end_sec,
            importance=src.importance,
            parent_v2_index=p,
        ))
    return out_cuts


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