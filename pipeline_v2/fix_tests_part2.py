import pathlib
import re

p = pathlib.Path('tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

# Fix out, parents = splice_cuts_minus_skipped(cuts, skipped)
text = re.sub(r'out,\s*parents\s*=\s*splice_cuts_minus_skipped\(cuts,\s*skipped\)', 'out = splice_cuts_minus_skipped(cuts, skipped)', text)

# Fix test_align_composed_audio_to_video_invokes_ffmpeg_with_atrim_apad not being fully removed
text = re.sub(r'    def test_align_composed_audio_to_video_invokes_ffmpeg_with_atrim_apad.*?return _Result\(\)\n\n.*?stage_4_render\._align_composed_audio_to_video\(str\(fake_input\)\).*?(?=\n\n    def test_|\Z)', '', text, flags=re.DOTALL)

# Fix out_loose, _ = collapse_micro_fragments(cuts, parents, threshold_s=0.45)
text = re.sub(r'out_loose,\s*_\s*=\s*collapse_micro_fragments\(cuts,\s*parents,\s*threshold_s=0\.45\)', 'for c, p in zip(cuts, parents): c.parent_v2_index = p\n        out_loose = collapse_micro_fragments(cuts, threshold_s=0.45)', text)

# Fix out_cuts = apply_silence_trims_to_cuts(cutssilence_trims,
text = re.sub(r'out_cuts\s*=\s*apply_silence_trims_to_cuts\(cutssilence_trims,.*?\)', 'out_cuts = apply_silence_trims_to_cuts(cuts, silence_trims)', text, flags=re.DOTALL)

# Replace all occurrences of out_something, out_parents = ... with out_something = ...
text = re.sub(r'(out_.*?),\s*out_parents\s*=\s*([a-zA-Z_0-9]+)\((.*?)\)', r'\1 = \2(\3)', text)

# Also test_multiple_cuts_each_with_their_own_skips had out, parents = ... which is fixed above.
text = re.sub(r'out,\s*out_parents\s*=\s*([a-zA-Z_0-9]+)\((.*?)\)', r'out = \1(\2)', text)

# Remove any remaining , _ =  unpacking for those functions
text = re.sub(r'(out.*?),\s*_\s*=\s*(collapse_micro_fragments|splice_cuts_minus_skipped|apply_silence_trims_to_cuts)\((.*?)\)', r'\1 = \2(\3)', text)

p.write_text(text, encoding='utf-8')
print('Tests partially fixed')
