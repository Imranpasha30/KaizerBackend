import pathlib
import re

p = pathlib.Path('pipeline_v2/pipeline_v2/stages/stage_4_render.py')
text = p.read_text(encoding='utf-8')

start_str = '        lang_cfg = _v1_languages.get(metadata.language.split("-", 1)[0])'

# Find start of the body
start_idx = text.find(start_str)

# Find the end of the method body (the return statement)
end_str = '        return {\n            "bulletin_path":     bulletin_out,'
end_idx = text.find(end_str, start_idx)

# We need to find the end of the return statement dict.
end_return = text.find('        }', end_idx) + 9

new_body = '''        import asyncio
        from pipeline_v2.single_pass_renderer import render_otio_timeline_single_pass
        from pipeline_v2.edl_builder import build_otio_timeline
        from pipeline_v2.models import StageTwoOutput, CleanTranscript

        bulletin_out = self.bulletin_dir / "bulletin.mp4"
        bulletin_out.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_stage2 = StageTwoOutput(
            full_video_cuts=full_video_cuts,
            skipped_segments=[],
            clean_transcript=CleanTranscript(words=[], clip_boundaries={}, source_word_map=[]),
            retake_audit="ok"
        )
        timeline = build_otio_timeline(dummy_stage2, str(self.video_path))
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("Stage4Render.render called from within an event loop, cannot run async ffmpeg.")
        else:
            asyncio.run(render_otio_timeline_single_pass(
                timeline=timeline,
                output_path=str(bulletin_out),
                progress_cb=progress_cb
            ))
        
        return {
            "bulletin_path": str(bulletin_out),
            "overlay_path": str(bulletin_out),
            "overlay_applied": False,
            "duration_s": sum(c.end_sec - c.start_sec for c in full_video_cuts),
            "stories_rendered": len(full_video_cuts),
            "stories_skipped": 0,
            "warnings": [],
            "editor_clip_artifacts": []
        }
'''

text = text[:start_idx] + new_body + text[end_return:]
p.write_text(text, encoding='utf-8')
print('Injected single_pass_renderer into render_bulletin')
