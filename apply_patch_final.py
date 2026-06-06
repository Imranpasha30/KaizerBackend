import os

with open("pipeline_v2/pipeline_v2/stages/stage_4_render.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Disable _v2_extract_enabled to bypass _run_unified_raw_extract
content = content.replace(
    '        v = (_os.environ.get("KAIZER_USE_V2_RAW_EXTRACT") or "").strip().lower()\n        return v in ("1", "true", "yes", "on")',
    '        return False'
)

# 2. Disable cut_clips_frame_aligned in cut_raw_bulletin_stories
content = content.replace(
    '''        if self.use_frame_aligned_cut:
            cut_clips_frame_aligned(
                str(self.video_path),
                v1_clips,
                str(self.bulletin_dir),
                progress_cb=progress_cb,
            )
        else:
            _v1_cut_video_clips(
                str(self.video_path),
                v1_clips,
                str(self.bulletin_dir),
            )''',
    '''        pass'''
)

# 3. Inject single pass renderer into render_bulletin
start_marker = "# ---- 4. Per-story compose -----------------------------------"
end_marker = "# ---- 6. Apply image_plan overlays --------------------------"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Markers not found")
    exit(1)

new_compose = """# ---- 4. Per-story compose -----------------------------------
        from pipeline_v2.single_pass_renderer import render_bulletin_single_pass, StoryAssets
        import asyncio

        story_assets_list: list[StoryAssets] = []
        failures: list[dict] = []
        total = len(clips)

        for i, clip in enumerate(clips):
            story_dur_s = float(clip.get("duration_sec") or 60.0)
            imgs = self._images_for_story(clip)

            # ---- 4a. Sidebar (carousel or placeholder) -----------
            sidebar_path: Optional[str] = None
            sidebar_is_video = False
            if self.use_sidebar_carousel and len(imgs) >= 2:
                sidebar_video = str(bulletin_dir / f"_sidebar_{i:02d}.mp4")
                try:
                    _v1_build_sidebar_carousel(
                        imgs[:5], story_dur_s, sidebar_video,
                        work_dir=str(bulletin_dir / f"_sidebar_work_{i:02d}"),
                    )
                    sidebar_path = sidebar_video
                    sidebar_is_video = True
                except Exception as exc:
                    logger.warning("stage_4: sidebar carousel story %d failed: %s", i, exc)
            if not sidebar_path:
                sidebar_static = str(bulletin_dir / f"_sidebar_{i:02d}.png")
                try:
                    _v1_make_sidebar_placeholder(imgs[0] if imgs else None, sidebar_static)
                    sidebar_path = sidebar_static
                except Exception as exc:
                    logger.warning("stage_4: sidebar placeholder story %d failed: %s", i, exc)
                    sidebar_path = ""

            # ---- 4b. StoryMeta + PiP source picking --------------
            importance = int(clip.get("importance") or 5)
            kicker = "BREAKING" if importance >= 8 else "NEWS"
            story_meta = _V1StoryMeta(
                title=(metadata.shorts_headline_native or "KAIZER NEWS")[:200],
                kicker=kicker,
                language=lang_cfg.code,
                story_index=i,
                total_stories=total,
                importance=importance,
            )

            # Build lower-third natively like V1 did inside compose_bulletin_story
            from pipeline_core.longform_compose import render_lower_third as _v1_render_lower_third
            lt_path = str(bulletin_dir / f"_lt_{i:02d}.png")
            try:
                _, lt_w = _v1_render_lower_third(story_meta, lang_cfg.font_primary, lt_path)
            except Exception as e:
                logger.warning(f"stage_4: lower-third story {i} failed: {e}")
                lt_w = 1920 # Default

            story_assets_list.append(StoryAssets(
                sidebar_path=sidebar_path,
                sidebar_is_video=sidebar_is_video,
                lt_path=lt_path,
                lt_w=lt_w
            ))

        # ---- 4.5. Per-story guardrail (D-9.7) ----------------------
        if not story_assets_list:
            raise RuntimeError(
                "Stage 4 render_bulletin: no story segments produced. "
                "Bulletin cannot be stitched. Inngest will retry."
            )

        # ---- 5. Stitch story segments into bulletin.mp4 ------------
        # Using single-pass renderer to bypass intermediate composition and crossfade issues
        bulletin_out = str(bulletin_dir / "bulletin.mp4")
        
        try:
            asyncio.run(render_bulletin_single_pass(
                cuts=full_video_cuts,
                source_url=str(self.video_path),
                output_path=bulletin_out,
                story_assets=story_assets_list,
                ticker_path=ticker_path,
                bug_path=bug_path,
                progress_cb=progress_cb
            ))
        except Exception as exc:
            raise RuntimeError(f"Single pass renderer failed: {exc}")

        # V1 return pattern requires tracking durations.
        total_duration_s = sum((c.end_sec - c.start_sec) for c in full_video_cuts)
        stories_rendered = total - len(failures)
        stories_skipped = len(failures)
        stitch_warnings = [f.get("reason") for f in failures]

        """

content = content[:start_idx] + new_compose + content[end_idx:]

with open("pipeline_v2/pipeline_v2/stages/stage_4_render.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied cleanly")
