import pathlib
import re

p = pathlib.Path('pipeline_v2/pipeline_v2/stages/stage_4_render.py')
text = p.read_text(encoding='utf-8')

old_call = '''            bulletin_result = self.render_bulletin(
                full_video_cuts=spliced_cuts,
                metadata=metadata,
                entities=entities,
                image_plan=image_plan,
                channel_name=channel_name,
                logo_path=logo_path,
                takeovers_enabled=takeovers_enabled,
                pip_enabled=pip_enabled,
                progress_cb=_p,
            )'''

new_call = '''            import asyncio
            from pipeline_v2.single_pass_renderer import render_otio_timeline_single_pass
            from pipeline_v2.edl_builder import build_otio_timeline
            from pipeline_v2.models import StageTwoOutput, CleanTranscript
            
            bulletin_out = self.output_dir / "bulletin" / "bulletin.mp4"
            bulletin_out.parent.mkdir(parents=True, exist_ok=True)
            
            dummy_stage2 = StageTwoOutput(
                full_video_cuts=spliced_cuts,
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
                # If we're already in an event loop (e.g. some tests), use run_coroutine_threadsafe
                # Actually, running a nested event loop is tricky.
                # Let's just create a new loop if needed or run directly.
                raise RuntimeError("Stage4Render.render called from within an event loop, cannot run async ffmpeg.")
            else:
                asyncio.run(render_otio_timeline_single_pass(
                    timeline=timeline,
                    output_path=str(bulletin_out),
                    progress_cb=_p
                ))
            
            bulletin_result = {
                "bulletin_path": str(bulletin_out),
                "overlay_path": str(bulletin_out),
                "overlay_applied": False,
                "duration_s": sum(c.end_sec - c.start_sec for c in spliced_cuts),
                "stories_rendered": len(spliced_cuts),
                "stories_skipped": 0,
                "warnings": [],
            }'''

text = text.replace(old_call, new_call)
p.write_text(text, encoding='utf-8')
print('Injected single_pass_renderer into Stage4Render.render')
