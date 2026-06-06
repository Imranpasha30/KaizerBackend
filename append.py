import pathlib
p = pathlib.Path('pipeline_v2/tests/test_stage_2_continuity.py')
text = p.read_text(encoding='utf-8')
text += '''
class TestSemanticGuard:
    @pytest.mark.asyncio
    async def test_semantic_guard_under_segmented(self, stage1_output, prompt_file):
        from pipeline_v2.models import Stage2Output, FullVideoCut
        stage1_output.stt_audio_duration_sec = 65.0
        decisions = Stage2Output(
            full_video_cuts=[
                FullVideoCut(index=1, start_word_idx=0, end_word_idx=10, start_sec=0.0, end_sec=5.0, importance=5),
                FullVideoCut(index=2, start_word_idx=11, end_word_idx=20, start_sec=10.0, end_sec=15.0, importance=5)
            ],
            skipped_segments=[],
            retake_audit="Under-segmented"
        )
        editor = Stage2ContinuityEditor(provider_name="gemini", prompt_path=prompt_file)
        class MockProvider:
            async def decide(self, s1, correction_note=""):
                return decisions
            @property
            def last_cost_usd(self): return 0.0
            @property
            def last_usage(self): return {}
        editor._provider = MockProvider()
        
        with pytest.raises(ValueError, match="Under-segmented bulletin: LLM returned 2 cuts for 65.0s audio"):
            await editor.transcribe_to_decisions(stage1_output)
'''
p.write_text(text, encoding='utf-8')
