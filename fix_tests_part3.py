import pathlib
import re

p = pathlib.Path('pipeline_v2/tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

# Remove leftover ssert parents == ...
text = re.sub(r'\s*assert parents == \[.*?\]\n', '\n', text)

# Fix out_disabled = collapse_micro_fragments(cuts, parents, threshold_s=0.0)
text = text.replace('out_disabled = collapse_micro_fragments(cuts, parents, threshold_s=0.0)', 'out_disabled = collapse_micro_fragments(cuts, threshold_s=0.0)')

# Fix out_identity = apply_silence_trims_to_cuts(cuts, parents, [])
text = text.replace('out_identity = apply_silence_trims_to_cuts(cuts, parents, [])', 'out_identity = apply_silence_trims_to_cuts(cuts, [])')

p.write_text(text, encoding='utf-8')
print('Fixed leftover test issues')
