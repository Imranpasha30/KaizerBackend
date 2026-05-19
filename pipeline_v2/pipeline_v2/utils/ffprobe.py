"""ffprobe wrapper.

Returns a typed ``VideoProbe`` for any video file. Failures (missing
ffprobe, malformed input, ffprobe non-zero exit) raise — no silent
swallowing.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("pipeline_v2.ffprobe")


@dataclass
class VideoProbe:
    """Structured ffprobe output. ``raw`` retains the full JSON for any
    downstream code that needs fields we didn't surface here."""

    duration_sec: float
    video_codec: Optional[str]
    audio_codec: Optional[str]
    fps: Optional[float]            # avg_frame_rate
    nominal_fps: Optional[float]    # r_frame_rate
    width: Optional[int]
    height: Optional[int]
    sample_rate: Optional[int]
    channels: Optional[int]
    nb_streams: int
    raw: dict[str, Any] = field(repr=False)

    @property
    def is_vfr(self) -> bool:
        """Heuristic VFR detection: avg and nominal framerate diverge.

        ffprobe reports avg_frame_rate = total_frames / duration. For a
        CFR file this equals r_frame_rate exactly. VFR sources drift.
        """
        if self.fps is None or self.nominal_fps is None:
            return False
        return abs(self.fps - self.nominal_fps) > 0.01


def _parse_fraction(value: Optional[str]) -> Optional[float]:
    """Parse a string like ``"30000/1001"`` to a float. None / "0/0" -> None."""
    if not value or value == "0/0":
        return None
    num, _, den = value.partition("/")
    try:
        d = int(den) if den else 1
        if d == 0:
            return None
        return float(num) / float(d)
    except ValueError:
        return None


def probe(path: str) -> VideoProbe:
    """Run ffprobe and return a structured VideoProbe.

    Raises:
        FileNotFoundError: if ffprobe isn't on PATH.
        RuntimeError: if ffprobe exits non-zero or returns unparseable JSON.
    """
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=60,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffprobe is not on PATH. Install ffmpeg (which ships ffprobe)."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (rc={proc.returncode}) for {path!r}: "
            f"{proc.stderr.strip()[-500:]}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ffprobe returned unparseable JSON for {path!r}: {exc}"
        ) from exc

    return _build_probe(data)


def _build_probe(data: dict[str, Any]) -> VideoProbe:
    """Construct VideoProbe from already-parsed ffprobe JSON."""
    fmt = data.get("format") or {}
    streams = data.get("streams") or []

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration_str = fmt.get("duration")
    try:
        duration_sec = float(duration_str) if duration_str is not None else 0.0
    except (TypeError, ValueError):
        duration_sec = 0.0

    sample_rate = None
    if audio and audio.get("sample_rate"):
        try:
            sample_rate = int(audio["sample_rate"])
        except (TypeError, ValueError):
            sample_rate = None

    return VideoProbe(
        duration_sec=duration_sec,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        fps=_parse_fraction(video.get("avg_frame_rate")) if video else None,
        nominal_fps=_parse_fraction(video.get("r_frame_rate")) if video else None,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        sample_rate=sample_rate,
        channels=audio.get("channels") if audio else None,
        nb_streams=len(streams),
        raw=data,
    )
