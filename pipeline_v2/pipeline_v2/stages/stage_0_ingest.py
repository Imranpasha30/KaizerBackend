"""Stage 0 -- Ingest.

ffprobe the source, pick an encoder (NVENC on production GPU boxes,
libx264 fallback elsewhere), then run transcode-to-mezzanine and
audio-extraction in parallel via ``asyncio.gather``.

Inputs:
  ``src_path``  -- absolute path to the source video on local disk.
  ``out_dir``   -- directory where ``mezzanine.mp4`` and ``source.mp3``
                   should be written.

Outputs (in ``out_dir``):
  ``mezzanine.mp4`` -- CFR 30fps H.264 + AAC 48kHz stereo
  ``source.mp3``    -- mp3 audio, 48kHz mono, 128k for Deepgram

The mezzanine is CFR-locked even when the source is VFR because every
downstream stage assumes source-time == mezzanine-time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from pipeline_v2.models import Stage0Output
from pipeline_v2.utils.ffmpeg_runner import (
    EncoderName,
    detect_encoder,
    extract_audio,
    transcode_to_mezzanine,
)
from pipeline_v2.utils.ffprobe import probe

logger = logging.getLogger("pipeline_v2.stage_0")


async def _timed(coro, label: str) -> float:
    """Await ``coro``, return its wall-clock duration in seconds.

    Errors propagate -- we don't swallow them; the retry layer needs
    them to surface.
    """
    start = time.perf_counter()
    try:
        await coro
    finally:
        # The 'finally' runs both for success and exception paths so the
        # duration log is always present. Re-raise (implicit) carries the
        # original exception up to gather().
        elapsed = time.perf_counter() - start
        logger.info("stage_0: %s finished in %.2fs", label, elapsed)
    return elapsed


async def run_stage_0(
    src_path: str,
    out_dir: str,
    *,
    encoder: Optional[EncoderName] = None,
) -> Stage0Output:
    """Run Stage 0 end-to-end.

    Args:
        src_path: Path to the input video file.
        out_dir: Directory to write ``mezzanine.mp4`` and ``source.mp3``
            into. Created if missing.
        encoder: Optionally pin the encoder name. When None (default),
            ``detect_encoder()`` chooses based on local hardware.

    Returns:
        Stage0Output Pydantic model with output paths + telemetry.

    Raises:
        FileNotFoundError: if the source video doesn't exist, or if
            ffmpeg / ffprobe aren't on PATH.
        RuntimeError: if ffprobe or ffmpeg exit non-zero.
    """
    src = Path(src_path)
    out = Path(out_dir)

    if not src.is_file():
        raise FileNotFoundError(f"stage_0: source video not found: {src}")

    out.mkdir(parents=True, exist_ok=True)

    # -- Probe ----------------------------------------------------------
    p = probe(str(src))
    logger.info(
        "stage_0: source=%s codec=%s fps=%s (nominal=%s, vfr=%s) "
        "%sx%s duration=%.2fs",
        src.name, p.video_codec, p.fps, p.nominal_fps, p.is_vfr,
        p.width, p.height, p.duration_sec,
    )

    if p.duration_sec <= 0:
        # ffprobe accepts the file but couldn't read a duration -- this
        # is a degenerate input (corrupt header, image, etc.). Surface
        # it now rather than letting downstream stages produce garbage.
        raise RuntimeError(
            f"stage_0: source has no readable duration "
            f"(ffprobe duration={p.duration_sec}). Path: {src}"
        )

    # -- Encoder choice -------------------------------------------------
    chosen = encoder or detect_encoder()

    mezz = out / "mezzanine.mp4"
    audio = out / "source.mp3"

    # -- Parallel transcode + audio extract -----------------------------
    wall_start = time.perf_counter()

    transcode_secs, audio_secs = await asyncio.gather(
        _timed(
            transcode_to_mezzanine(str(src), str(mezz), encoder=chosen),
            "transcode",
        ),
        _timed(
            extract_audio(str(src), str(audio)),
            "audio extract",
        ),
    )
    wall_secs = time.perf_counter() - wall_start

    logger.info(
        "stage_0: complete. encoder=%s wall=%.2fs (transcode=%.2fs, audio=%.2fs)",
        chosen, wall_secs, transcode_secs, audio_secs,
    )

    return Stage0Output(
        mezzanine_path=str(mezz),
        audio_path=str(audio),
        duration_sec=p.duration_sec,
        encoder_used=chosen,
        width=p.width,
        height=p.height,
        source_was_vfr=p.is_vfr,
        transcode_seconds=transcode_secs,
        audio_extract_seconds=audio_secs,
        wall_seconds=wall_secs,
    )
