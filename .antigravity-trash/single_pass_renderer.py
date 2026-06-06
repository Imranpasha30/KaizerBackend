import asyncio
import logging
import os
from typing import Optional, List
from dataclasses import dataclass
from pipeline_v2.utils.ffmpeg_runner import run_ffmpeg
from pipeline_v2.models import FullVideoCut

logger = logging.getLogger("pipeline_v2.single_pass_renderer")

@dataclass
class StoryAssets:
    sidebar_path: str
    sidebar_is_video: bool
    lt_path: str
    lt_w: int

async def render_bulletin_single_pass(
    cuts: List[FullVideoCut],
    source_url: str,
    output_path: str,
    story_assets: List[StoryAssets],
    ticker_path: str,
    bug_path: Optional[str] = None,
    ticker_speed_px_s: float = 200.0,
    width: int = 1920,
    height: int = 1080,
    progress_cb=None
) -> None:
    """
    Renders an entire bulletin in a single ffmpeg invocation, applying
    trims, TV9-style layouts (sidebars, hstack, padding), and overlays 
    (lower-thirds, ticker, bug) for all stories concurrently using filter_complex.
    """
    if not cuts:
        logger.warning("No cuts found. Nothing to render.")
        return

    logger.info(f"Rendering {len(cuts)} clips in a single pass to {output_path}")

    args = ["-y", "-i", source_url]
    input_idx = 1

    # Add ticker
    ticker_idx = None
    if ticker_path and os.path.isfile(ticker_path):
        args.extend(["-loop", "1", "-i", ticker_path])
        ticker_idx = input_idx
        input_idx += 1

    # Add bug
    bug_idx = None
    if bug_path and os.path.isfile(bug_path):
        args.extend(["-loop", "1", "-i", bug_path])
        bug_idx = input_idx
        input_idx += 1

    # Add story assets
    story_input_maps = []
    for asset in story_assets:
        if asset.sidebar_is_video:
            args.extend(["-i", asset.sidebar_path])
        else:
            args.extend(["-loop", "1", "-i", asset.sidebar_path])
        sb_idx = input_idx
        input_idx += 1

        args.extend(["-loop", "1", "-i", asset.lt_path])
        lt_idx = input_idx
        input_idx += 1

        story_input_maps.append((sb_idx, lt_idx, asset.lt_w))

    filter_complex = []
    concat_inputs = ""

    main_w, main_h = 1280, 800
    side_w, side_h = 580, 800
    lt_h = 140
    bug_pad_x, bug_pad_y = 30, 30
    ticker_h = 50
    lt_y = height - lt_h - ticker_h
    ticker_y = height - ticker_h

    for i, (cut, (sb_idx, lt_idx, lt_w)) in enumerate(zip(cuts, story_input_maps)):
        duration_sec = cut.end_sec - cut.start_sec

        # 1. Audio/Video Trim & Sync
        # We explicitly set frame rate to 30 to avoid variable frame rate drift
        v_main = f"[v_main_{i}]"
        filter_complex.append(
            f"[0:v]trim=start={cut.start_sec:.3f}:duration={duration_sec:.3f},"
            f"setpts=PTS-STARTPTS,fps=30,"
            f"scale={main_w}:{main_h}:force_original_aspect_ratio=increase,"
            f"crop={main_w}:{main_h},setsar=1{v_main}"
        )

        a_main = f"[a_main_{i}]"
        filter_complex.append(
            f"[0:a]atrim=start={cut.start_sec:.3f}:duration={duration_sec:.3f},"
            f"asetpts=PTS-STARTPTS{a_main}"
        )

        # 2. Sidebar
        v_side = f"[v_side_{i}]"
        filter_complex.append(
            f"[{sb_idx}:v]scale={side_w}:{side_h}:force_original_aspect_ratio=increase,"
            f"crop={side_w}:{side_h},setsar=1{v_side}"
        )

        # 3. Stack & Pad
        v_top = f"[v_top_{i}]"
        # shortest=1 ensures the looped static image sidebars don't extend forever
        filter_complex.append(f"{v_main}{v_side}hstack=inputs=2:shortest=1{v_top}")

        v_stage = f"[v_stage_{i}]"
        filter_complex.append(f"{v_top}pad={width}:{height}:30:0:black{v_stage}")

        # 4. Lower Third
        if lt_w > width:
            lt_x_expr = (
                f"if(lt(t\\,0.4)\\,-w+w*t/0.4\\,"
                f"if(lt(t\\,2.0)\\,0\\,"
                f"max({width}-w\\,-((t-2.0)*60))))"
            )
        else:
            lt_x_expr = "if(lt(t\\,0.4)\\,-w+w*t/0.4\\,0)"

        v_lt = f"[v_lt_{i}]"
        filter_complex.append(
            f"{v_stage}[{lt_idx}:v]overlay="
            f"x='{lt_x_expr}':"
            f"y={lt_y}:format=auto:shortest=1{v_lt}"
        )

        # 5. Ticker
        v_tick = f"[v_tick_{i}]"
        filter_complex.append(
            f"{v_lt}[{ticker_idx}:v]overlay="
            f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':"
            f"y={ticker_y}:format=auto:shortest=1{v_tick}"
        )

        # 6. Bug
        v_out = f"[v_out_{i}]"
        if bug_idx is not None:
            filter_complex.append(
                f"{v_tick}[{bug_idx}:v]overlay="
                f"x=W-w-{bug_pad_x}:y={bug_pad_y}:format=auto:shortest=1{v_out}"
            )
        else:
            filter_complex.append(f"{v_tick}copy{v_out}")

        concat_inputs += f"{v_out}{a_main}"

    num_clips = len(cuts)
    filter_complex.append(f"{concat_inputs}concat=n={num_clips}:v=1:a=1[vout][aout]")

    filter_complex_str = ";".join(filter_complex)

    args.extend([
        "-filter_complex", filter_complex_str,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-fps_mode", "cfr",
        "-async", "1",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ])

    if progress_cb:
        progress_cb("Starting robust single-pass ffmpeg render...")

    await run_ffmpeg(args, log_label="single_pass_renderer")

    if progress_cb:
        progress_cb("Single-pass render completed successfully.")


