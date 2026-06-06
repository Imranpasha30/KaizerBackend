import pathlib

p = pathlib.Path('tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

# We need to replace imports of the 6 functions in the test file
imports_to_change = [
    'splice_cuts_minus_skipped',
    'detect_silence_trims',
    'apply_silence_trims_to_cuts',
    'collapse_micro_fragments',
    'round_cut_precision',
    'assert_cuts_monotonic',
    'CUT_PRECISION_DECIMALS',
    'SILENCE_TRIM_THRESHOLD_S'
]

# They are imported in various ways, for example:
# from pipeline_v2.stages.stage_4_render import assert_cuts_monotonic
# Let's just find and replace the import module path where those functions are imported.
# Actually, the simplest way is to replace 'from pipeline_v2.stages.stage_4_render import ' with 'from pipeline_v2.stages.cut_utils import ' for those specific functions.

import re

for func in imports_to_change:
    # A bit tricky since there might be multi-line imports.
    # Instead, let's just do a blanket regex:
    # We can just change all occurrences of those function names to be imported from cut_utils.
    # Actually, they are inside individual test methods!
    text = text.replace(f'from pipeline_v2.stages.stage_4_render import {func}', f'from pipeline_v2.stages.cut_utils import {func}')
    text = text.replace(f'from pipeline_v2.stages.stage_4_render import (\n            {func}', f'from pipeline_v2.stages.cut_utils import (\n            {func}')
    text = text.replace(f'from pipeline_v2.stages.stage_4_render import \\\n            {func}', f'from pipeline_v2.stages.cut_utils import \\\n            {func}')
    text = text.replace(f'from pipeline_v2.stages.stage_4_render import (\n            round_cut_precision, CUT_PRECISION_DECIMALS,\n        )', f'from pipeline_v2.stages.cut_utils import (\n            round_cut_precision, CUT_PRECISION_DECIMALS,\n        )')

p.write_text(text, encoding='utf-8')
