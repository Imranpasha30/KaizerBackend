"""Post-render QC gate for V4 outputs (Wave 4, item H).

Cheap ffprobe-based verification that a rendered artifact is actually
a playable video with intact audio, the right resolution, and the
expected duration — catching truncated / silent / wrongly-scaled
outputs BEFORE they get materialised as Clips or shipped to R2.

Skippable via ``KAIZER_V4_QC=0`` (default on).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# A real render — even a 5-second short — is comfortably above 100 KB.
# Anything smaller is a truncated / empty ffmpeg casualty.
_MIN_FILE_BYTES = 100 * 1024

_PROBE_TIMEOUT_S = 15


def qc_enabled() -> bool:
    """QC gate toggle — ``KAIZER_V4_QC=0`` disables, default enabled."""
    return (os.environ.get("KAIZER_V4_QC", "1") or "1").strip() != "0"


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def probe_media(path) -> dict:
    """ffprobe the file and return a summary dict.

    Keys: ``duration`` (container, float|None), ``n_video``, ``n_audio``,
    ``width``, ``height`` (first video stream), ``video_duration``,
    ``audio_duration`` (per-stream, float|None), ``streams`` (raw list).

    Raises ``RuntimeError`` when ffprobe fails / times out. 15s timeout.
    Stdout is captured as bytes (decoded manually) to dodge the known
    Windows hang in ``subprocess.run(capture_output=True, text=True)``.
    """
    cmd = [
        _ffprobe_bin(), "-v", "error",
        "-show_entries",
        "stream=index,codec_type,width,height,duration:format=duration",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out after {_PROBE_TIMEOUT_S}s: {path}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffprobe failed (rc={proc.returncode}): {tail}")

    raw = (proc.stdout or b"").decode("utf-8", errors="replace") or "{}"
    data = json.loads(raw)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    def _f(v) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    v0 = video[0] if video else {}
    a0 = audio[0] if audio else {}
    return {
        "duration": _f(fmt.get("duration")),
        "n_video": len(video),
        "n_audio": len(audio),
        "width": int(v0.get("width") or 0),
        "height": int(v0.get("height") or 0),
        "video_duration": _f(v0.get("duration")),
        "audio_duration": _f(a0.get("duration")),
        "streams": streams,
    }


def verify_render(
    path,
    *,
    expected_duration: Optional[float] = None,
    expected_w: int,
    expected_h: int,
    tolerance_s: float = 0.5,
    av_sync_tolerance_s: float = 0.5,
) -> list[str]:
    """Verify a rendered artifact. Returns a list of violations (empty = pass).

    Checks:
      - file exists and is > 100 KB
      - has >= 1 video stream AND >= 1 audio stream
      - first video stream resolution == expected_w x expected_h
      - duration: asymmetric TRUNCATION detection only (see below). The
        ``expected_duration`` is a PLAN estimate (sum of story spans),
        NOT an exact source length — the real render legitimately drifts
        (overlays / intro / fades make it longer; keyframe + encoder
        boundary alignment make it a touch shorter). Blocking a publish
        on a small drift silently dropped valid Full Videos and Shorts,
        so we now fail ONLY on genuine truncation: output shorter than
        the plan by more than ``max(tolerance_s, 3s, 25% of estimate)``.
        Longer-than-planned always passes.
      - |video_stream_duration - audio_stream_duration| <
        ``av_sync_tolerance_s`` (lip-sync invariant). When either
        per-stream duration is missing from the container we fall back
        to the container duration — i.e. PASS (mp4s normally carry
        per-stream durations; some muxers omit them).
    """
    violations: list[str] = []
    p = Path(path)
    if not p.is_file():
        return [f"file missing: {path}"]
    size = p.stat().st_size
    if size <= _MIN_FILE_BYTES:
        violations.append(f"file too small ({size} bytes <= {_MIN_FILE_BYTES})")

    try:
        info = probe_media(p)
    except Exception as exc:
        violations.append(f"ffprobe failed: {exc}")
        return violations

    if info["n_video"] < 1:
        violations.append("no video stream")
    if info["n_audio"] < 1:
        violations.append("no audio stream")
    if info["n_video"] >= 1 and (info["width"] != int(expected_w)
                                 or info["height"] != int(expected_h)):
        violations.append(
            f"resolution {info['width']}x{info['height']} != "
            f"expected {expected_w}x{expected_h}"
        )

    dur = info["duration"]
    if expected_duration is not None and expected_duration > 0:
        if dur is None:
            violations.append("container duration missing")
        else:
            exp = float(expected_duration)
            # Asymmetric: only a DRAMATIC shortfall is a corruption
            # signal (ffmpeg died mid-encode). A small under/over is the
            # normal gap between the plan estimate and the real render.
            shortfall = exp - dur
            truncation_floor = max(float(tolerance_s), 3.0, 0.25 * exp)
            if shortfall > truncation_floor:
                violations.append(
                    f"truncated: duration {dur:.2f}s is {shortfall:.2f}s "
                    f"shorter than the ~{exp:.2f}s plan "
                    f"(truncation floor {truncation_floor:.2f}s)"
                )

    vd, ad = info["video_duration"], info["audio_duration"]
    if vd is not None and ad is not None:
        drift = abs(vd - ad)
        if drift >= av_sync_tolerance_s:
            violations.append(
                f"A/V stream duration drift {drift:.3f}s >= "
                f"{av_sync_tolerance_s:.3f}s (video={vd:.3f}s audio={ad:.3f}s)"
            )
    # else: per-stream duration(s) missing -> container duration already
    # checked above; treat as pass per the QC contract.

    return violations
