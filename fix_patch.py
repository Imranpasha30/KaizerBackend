import pathlib

p = pathlib.Path('pipeline_v2/tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

# Change @mock.patch to @patch
text = text.replace('@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")', '@patch("pipeline_v2.single_pass_renderer.run_ffmpeg")')
p.write_text(text, encoding='utf-8')
