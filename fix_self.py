import pathlib

p = pathlib.Path('pipeline_v2/tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

# Fix duplicate self
text = text.replace('self, mock_run, \n        self', 'self, mock_run')
p.write_text(text, encoding='utf-8')
