import pathlib
import re
p = pathlib.Path('pipeline_v2/pipeline_v2/stages/stage_4_render.py')
text = p.read_text(encoding='utf-8')

# Search for the block calling render_bulletin
start = text.find('bulletin_result = self.render_bulletin(')
end = text.find('def full_video_cuts_to_v1_clip_dicts', start)
print(text[start:start+1000])
