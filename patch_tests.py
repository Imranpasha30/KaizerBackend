import pathlib
import re

p = pathlib.Path('pipeline_v2/tests/test_stage_4_render.py')
text = p.read_text(encoding='utf-8')

text = re.sub(r'def test_returned_result_carries_sub_render_outputs\(', 
    '''@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")\n    def test_returned_result_carries_sub_render_outputs(\n        self, mock_run, ''', text)
        
text = re.sub(r'def test_full_pipeline_writes_both_editor_metas\(', 
    '''@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")\n    def test_full_pipeline_writes_both_editor_metas(\n        self, mock_run, ''', text)
        
text = re.sub(r'def test_shorts_editor_meta_has_expected_structure\(', 
    '''@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")\n    def test_shorts_editor_meta_has_expected_structure(\n        self, mock_run, ''', text)
        
text = re.sub(r'def test_bulletin_editor_meta_has_expected_structure\(', 
    '''@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")\n    def test_bulletin_editor_meta_has_expected_structure(\n        self, mock_run, ''', text)
        
text = re.sub(r'def test_zero_shorts_skips_shorts_pass\(', 
    '''@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")\n    def test_zero_shorts_skips_shorts_pass(\n        self, mock_run, ''', text)
        
text = re.sub(r'def test_shorts_pass_runs_before_bulletin_pass\(', 
    '''@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")\n    def test_shorts_pass_runs_before_bulletin_pass(\n        self, mock_run, ''', text)
        
text = re.sub(r'def test_zero_full_video_cuts_skips_bulletin_pass\(', 
    '''@mock.patch("pipeline_v2.single_pass_renderer.run_ffmpeg")\n    def test_zero_full_video_cuts_skips_bulletin_pass(\n        self, mock_run, ''', text)

p.write_text(text, encoding='utf-8')
print('Tests patched successfully.')
