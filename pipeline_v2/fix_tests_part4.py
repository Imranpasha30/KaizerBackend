import pathlib

p = pathlib.Path('tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

# Fix out, parents = splice_cuts_minus_skipped(cuts, [])
text = text.replace('out, parents = splice_cuts_minus_skipped(cuts, [])', 'out = splice_cuts_minus_skipped(cuts, [])')

p.write_text(text, encoding='utf-8')
