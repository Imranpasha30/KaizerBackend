"""ffmpeg subprocess wrapper for the v2 ingest stage.

Provides:
  - ``list_encoders()`` : parse ``ffmpeg -encoders``.
  - ``nvenc_runtime_ok()`` : actually attempt a 0.1s NVENC encode to
    confirm the driver / GPU is available (not just that NVENC was
    compiled into ffmpeg).
  - ``detect_encoder()`` : returns ``"h264_nvenc"`` only if both the
    static check AND the runtime probe succeed; else ``"libx264"``.
  - ``transcode_to_mezzanine()`` and ``extract_audio()`` : async ffmpeg
    invocations with proper cancellation handling.

No silent excepts. ffmpeg non-zero exit codes are re-raised with the
last ~2000 chars of stderr attached for diagnosis.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Literal

logger = logging.getLogger("pipeline_v2.ffmpeg")

EncoderName = Literal["h264_nvenc", "libx264"]


# ---------------------------------------------------------------------- #
# Encoder discovery                                                      #
# ---------------------------------------------------------------------- #


def list_encoders() -> set[str]:
    """Return the set of encoder names compiled into the local ffmpeg.

    Raises:
        FileNotFoundError: if ffmpeg isn't on PATH.
        RuntimeError: if ``ffmpeg -encoders`` exits non-zero.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False, timeout=15,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg is not on PATH. Install it (e.g. winget install ffmpeg)."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg -encoders failed: {proc.stderr.strip()[-500:]}")

    encoders: set[str] = set()
    in_table = False
    for line in proc.stdout.splitlines():
        if line.strip().startswith("------"):
            in_table = True
            continue
        if not in_table or not line.strip():
            continue
        # Row format: ` V....D h264_nvenc  NVIDIA NVENC H.264 encoder ...`
        parts = line.split(None, 2)
        if len(parts) >= 2:
            encoders.add(parts[1])
    return encoders


def nvenc_runtime_ok(timeout_s: float = 15.0) -> bool:
    """Runtime-probe NVENC by attempting a 1-frame nullsrc encode.

    NVENC may be compiled into the ffmpeg binary but unavailable at
    runtime (no NVIDIA GPU, missing driver, exhausted session limit).
    This catches that case.

    Returns False rather than raising for any non-success outcome,
    including timeout — caller is expected to fall back to libx264.

    NB on frame size: 256x256 is large enough to clear EVERY modern
    NVENC generation's minimum-dimension requirement. Earlier versions
    of this probe used 64x64 which Blackwell-era cards (RTX 50-series)
    reject with "Frame Dimension less than the minimum supported value"
    -- the probe then returned False and Stage 0 fell back to libx264
    on machines that actually had a healthy NVENC. Backlog item 90.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
        "-c:v", "h264_nvenc", "-frames:v", "1",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            check=False, timeout=timeout_s,
        )
    except FileNotFoundError:
        # No ffmpeg at all — caller should not have reached here.
        return False
    except subprocess.TimeoutExpired:
        logger.warning("NVENC runtime probe timed out after %.1fs", timeout_s)
        return False
    return proc.returncode == 0


def detect_encoder() -> EncoderName:
    """Pick ``h264_nvenc`` if both compiled in and runtime-functional.

    Falls back to ``libx264`` in every other case, with a clear log line
    naming why.
    """
    encoders = list_encoders()
    if "h264_nvenc" not in encoders:
        logger.info(
            "encoder: libx264 (h264_nvenc not compiled into this ffmpeg)"
        )
        return "libx264"

    if nvenc_runtime_ok():
        logger.info("encoder: h264_nvenc (NVENC compiled in + runtime probe OK)")
        return "h264_nvenc"

    logger.info(
        "encoder: libx264 (h264_nvenc compiled in BUT runtime probe failed -- "
        "no NVIDIA GPU/driver). Software encoding will be slower; this is "
        "expected on machines without an NVIDIA GPU."
    )
    return "libx264"


# ---------------------------------------------------------------------- #
# Async ffmpeg invocations                                               #
# ---------------------------------------------------------------------- #


async def run_ffmpeg(args: list[str], *, log_label: str) -> None:
    """Run an ffmpeg command async, surface non-zero exit codes.

    Cancellation (asyncio.CancelledError) terminates the child process
    so Inngest job cancels don't leak ffmpeg zombies.
    """
    cmd = ["ffmpeg", "-hide_banner", "-y", "-v", "error"] + args
    logger.info("%s: %s", log_label, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await proc.communicate()
    except asyncio.CancelledError:
        # Clean termination so Inngest cancel doesn't orphan ffmpeg.
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            proc.kill()
        raise

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError(
            f"{log_label} failed (rc={proc.returncode}):\n{stderr_text}"
        )


async def transcode_to_mezzanine(
    src: str, dst: str, *, encoder: EncoderName,
) -> None:
    """Transcode ``src`` to a CFR 30fps mezzanine at ``dst``.

    Mezzanine spec (consistent across encoders):
      - Video: 30fps CFR (``-vsync cfr -r 30``). Critical: downstream
        timestamp math assumes source-time == mezzanine-time, which
        only holds with CFR.
      - Audio: AAC 48kHz stereo 192k.
      - Container: mp4 with ``+faststart`` so streamed reads work.

    Encoder-specific quality settings:
      - NVENC: ``-preset p4 -tune hq -rc vbr -cq 23`` (per the plan).
      - libx264: ``-preset medium -crf 23 -pix_fmt yuv420p`` (equivalent
        quality target, software encoding).
    """
    if encoder == "h264_nvenc":
        video_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "23",
        ]
    else:
        video_args = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]

    await run_ffmpeg(
        [
            "-i", src,
            *video_args,
            "-vsync", "cfr", "-r", "30",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
            "-movflags", "+faststart",
            dst,
        ],
        log_label=f"transcode ({encoder})",
    )


async def extract_audio(src: str, dst: str) -> None:
    """Extract ``src`` audio as 48kHz 128kbps mp3 for Deepgram."""
    await run_ffmpeg(
        [
            "-i", src,
            "-vn",
            "-c:a", "libmp3lame",
            "-ar", "48000",
            "-b:a", "128k",
            dst,
        ],
        log_label="audio extract",
    )
