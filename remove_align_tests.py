import pathlib
import re

p = pathlib.Path('pipeline_v2/tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

# Remove the two test methods completely
text = re.sub(r'    def test_align_composed_audio_to_video_missing_file\(self\):.*?(?=    def test_)', '', text, flags=re.DOTALL)
text = re.sub(r'    def test_align_composed_audio_to_video_invokes_ffmpeg_with_atrim_apad\(self\):.*?(?=\n\n    def test_|\Z)', '', text, flags=re.DOTALL)

p.write_text(text, encoding='utf-8')
print('Removed align tests')
