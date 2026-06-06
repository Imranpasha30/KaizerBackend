import pathlib
import re

# 1. Update cut_utils.py collapse_micro_fragments
p_cu = pathlib.Path('pipeline_v2/pipeline_v2/stages/cut_utils.py')
text_cu = p_cu.read_text(encoding='utf-8')

new_collapse = '''def collapse_micro_fragments(
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
    return out_cuts'''

text_cu = re.sub(r'def collapse_micro_fragments\(.*?return out_cuts', new_collapse, text_cu, flags=re.DOTALL)
p_cu.write_text(text_cu, encoding='utf-8')


# 2. Fix test_stage_4_render.py to not pass parents separately
p_t = pathlib.Path('pipeline_v2/tests/test_stage_4_render.py')
text_t = p_t.read_text(encoding='utf-8')

# Change _full_video_cut to accept parent
text_t = text_t.replace('def _full_video_cut(idx: int, start: float, end: float) -> FullVideoCut:', 'def _full_video_cut(idx: int, start: float, end: float, parent: int | None = None) -> FullVideoCut:')
text_t = text_t.replace('importance=8,', 'importance=8, parent_v2_index=parent,')

# In tests, replace the calls.
# e.g. out_cuts, out_parents = collapse_micro_fragments(cuts, parents)
# and out_cuts, out_parents = splice_cuts_minus_skipped(cuts, skips)
text_t = re.sub(r'out_cuts,\s*out_parents\s*=\s*(collapse_micro_fragments|splice_cuts_minus_skipped|apply_silence_trims_to_cuts)\s*\(\s*cuts,\s*parents(?:,\s*([^)]+))?\s*\)', r'# Assign parents directly\n        for c, p in zip(cuts, parents):\n            c.parent_v2_index = p\n        out_cuts = \1(cuts\2)', text_t)

text_t = re.sub(r'out_strict,\s*_\s*=\s*(collapse_micro_fragments)\s*\(\s*cuts,\s*parents,\s*threshold_s=1\.0\s*\)', r'for c, p in zip(cuts, parents):\n            c.parent_v2_index = p\n        out_strict = \1(cuts, threshold_s=1.0)', text_t)

text_t = text_t.replace('out_cuts, out_parents = collapse_micro_fragments([], [])', 'out_cuts = collapse_micro_fragments([])')

# Length mismatch test isn't needed anymore if they are part of the same object, so just pass or remove.
# We can replace the raise checking logic.
text_t = re.sub(r'with _pytest\.raises\(ValueError, match="length mismatch"\):.*?collapse_micro_fragments\(cuts, \[0\]\).*?(?=\n\s+def )', 'pass\n', text_t, flags=re.DOTALL)

text_t = text_t.replace('assert out_parents == ', 'assert [c.parent_v2_index for c in out_cuts] == ')
text_t = text_t.replace('self.assertEqual(out_parents,', 'self.assertEqual([c.parent_v2_index for c in out_cuts],')

p_t.write_text(text_t, encoding='utf-8')
print('Tests updated')
