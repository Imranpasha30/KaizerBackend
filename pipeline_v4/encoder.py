"""Video-encoder selector for the V4 pipeline.

Centralises the choice between NVIDIA NVENC and libx264 so every
ffmpeg invocation in the V4 path uses the same encoder + quality
target. Without this, the 11 hard-coded ``-c:v libx264 -preset medium
-crf 20`` blocks scattered through v1_bridge / canvas_engine /
trim_engine / watermark drift independently and any GPU rollout
touches a dozen call sites.

Selection precedence
--------------------
1. ``KAIZER_VIDEO_ENCODER`` env var
     - ``nvenc``    → force NVENC (errors at first ffmpeg run if it's
                      not in the local ffmpeg build)
     - ``libx264``  → force CPU encode (the legacy behaviour)
     - ``auto`` / unset → detect once at startup. Use NVENC if the
                          local ffmpeg lists ``h264_nvenc`` in its
                          encoders table, otherwise libx264.
2. Detection is cached for the process lifetime — we don't shell out
   per render.

Quality mapping
---------------
libx264 ``-crf N`` maps to NVENC ``-rc vbr -cq N+2 -b:v 0`` which is
the standard CRF-to-CQ rule of thumb (NVENC's CQ tends to encode ~2
points hotter at the same perceptual quality). ``-b:v 0`` disables
the bitrate ceiling so CQ is the only quality lever.

Preset hint
-----------
The libx264 preset name is passed through ``preset_hint`` and mapped
to the NVENC ``p1..p7`` scale:

    veryfast  → p1   (fastest, lowest quality)
    fast      → p3
    medium    → p4   (balanced — matches libx264 medium)
    slow      → p5
    slower    → p6

Callers should pass the libx264 preset name they would have used; the
helper normalises it for whichever backend is active.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import List


# Mirrors NVENC's preset ladder. Ordered slowest → fastest mirrors
# libx264's veryslow→ultrafast scale, but flipped: p1 is fastest in
# NVENC because lower presets do less analysis. We map by perceived
# quality, not by name.
_NVENC_PRESET_BY_LIBX264 = {
    "ultrafast": "p1",
    "superfast": "p1",
    "veryfast":  "p1",
    "faster":    "p2",
    "fast":      "p3",
    "medium":    "p4",
    "slow":      "p5",
    "slower":    "p6",
    "veryslow":  "p7",
}


class EncoderUnavailable(RuntimeError):
    """Raised when a FORCED encoder (e.g. NVENC) can't actually run, so the
    render fails LOUD instead of silently crawling on the CPU for minutes."""


def _encoder_intent() -> str:
    """Resolve the operator's intended encoder: ``nvenc`` | ``libx264`` | ``auto``.

    Honors BOTH knobs (read every call so tests can flip them):
      - ``KAIZER_VIDEO_ENCODER`` — the V4-specific selector. An explicit
        ``nvenc`` / ``libx264`` here WINS.
      - ``KAIZER_FORCE_ENCODER`` — the global ".env" force knob
        (``h264_nvenc`` / ``libx264``). Consulted when VIDEO_ENCODER is
        ``auto`` / unset. This is the "force NVENC, fail loud" lever, which
        the old code ignored entirely.
    """
    ve = (os.environ.get("KAIZER_VIDEO_ENCODER") or "").strip().lower()
    if ve == "nvenc":
        return "nvenc"
    if ve in ("libx264", "x264", "cpu"):
        return "libx264"
    # VIDEO_ENCODER is auto/unset → consult the explicit force knob.
    fe = (os.environ.get("KAIZER_FORCE_ENCODER") or "").strip().lower()
    if fe in ("h264_nvenc", "nvenc"):
        return "nvenc"
    if fe in ("libx264", "x264", "cpu"):
        return "libx264"
    return "auto"


@lru_cache(maxsize=1)
def _nvenc_available() -> bool:
    """True iff the local ffmpeg can ACTUALLY open an h264_nvenc session.

    Two gates (cached once per process):
      1. The build advertises ``h264_nvenc`` (``ffmpeg -encoders``).
      2. A real 1-frame test-encode opens an NVENC session and succeeds.
         This is what the old list-only check missed — a build can list
         NVENC while the GPU/driver refuses a session (exhausted sessions
         from leaked renders, driver fault, headless GPU). Without this,
         ``-c:v h264_nvenc`` would be selected and then either error or, in
         the auto path, the pipeline silently used CPU and crawled.

    The session test retries twice to ride out a transient contention blip.
    Escape hatch: ``KAIZER_NVENC_PROBE=list`` skips the session test (use the
    cheap list-only check) if the test-encode ever misbehaves on a host.
    """
    if not shutil.which("ffmpeg"):
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0 or "h264_nvenc" not in (proc.stdout or ""):
        return False

    if (os.environ.get("KAIZER_NVENC_PROBE") or "session").strip().lower() == "list":
        return True

    # Real session test — encode a tiny synthetic clip to /dev/null.
    # 256x256 stays safely above NVENC's minimum frame dimensions (a 64x64
    # probe falsely fails with "Frame Dimension less than the minimum
    # supported value" even when NVENC is perfectly healthy).
    test_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1:r=10",
        "-pix_fmt", "yuv420p", "-c:v", "h264_nvenc", "-f", "null", "-",
    ]
    for _ in range(2):
        try:
            p = subprocess.run(test_cmd, capture_output=True, text=True, timeout=20)
            if p.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass
    return False


def resolve_status() -> dict:
    """Resolve the encoder backend + an advisory, WITHOUT raising.

    Returns ``{backend, intent, nvenc_ok, level, message}`` where:
      - ``backend`` is what the render should actually use.
      - ``level`` is ``ok`` | ``warn`` | ``error``.
      - ``message`` is a human-readable advisory the orchestrator logs.

    Decision matrix:
      intent=libx264                 -> libx264 (ok)
      intent=nvenc  & nvenc usable   -> nvenc   (ok)
      intent=nvenc  & NOT usable     -> nvenc   (ERROR — caller should fail loud;
                                                 operator demanded GPU)
      intent=auto   & nvenc usable   -> nvenc   (ok)
      intent=auto   & NOT usable     -> libx264 (WARN — CPU is ~10x slower)
    """
    intent = _encoder_intent()
    if intent == "libx264":
        return {"backend": "libx264", "intent": intent, "nvenc_ok": False,
                "level": "ok", "message": "CPU encode (libx264) — encoder forced to CPU."}
    nvenc_ok = _nvenc_available()
    if intent == "nvenc":
        if nvenc_ok:
            return {"backend": "nvenc", "intent": intent, "nvenc_ok": True,
                    "level": "ok", "message": ""}
        return {
            "backend": "nvenc", "intent": intent, "nvenc_ok": False, "level": "error",
            "message": (
                "NVENC is force-selected (KAIZER_FORCE_ENCODER / "
                "KAIZER_VIDEO_ENCODER=nvenc) but this host can't open an "
                "h264_nvenc session right now (driver fault, or NVENC sessions "
                "exhausted by leaked/zombie renders). Refusing to silently "
                "fall back to CPU — a CPU render is ~10x slower (minutes vs "
                "seconds). Fix the GPU/driver (or restart to free leaked NVENC "
                "sessions), or set KAIZER_VIDEO_ENCODER=libx264 to allow CPU."
            ),
        }
    # auto
    if nvenc_ok:
        return {"backend": "nvenc", "intent": intent, "nvenc_ok": True,
                "level": "ok", "message": ""}
    return {
        "backend": "libx264", "intent": intent, "nvenc_ok": False, "level": "warn",
        "message": (
            "NVENC not usable on this host — falling back to CPU (libx264). "
            "Renders will be ~10x slower. Set KAIZER_FORCE_ENCODER=h264_nvenc "
            "to make this a hard failure instead of a slow fallback."
        ),
    }


def _selected_backend() -> str:
    """Return the active backend: ``"nvenc"`` or ``"libx264"``.

    Pure resolution (never raises) — the orchestrator enforces the loud-fail
    for the forced-NVENC-unavailable case via ``resolve_status()`` BEFORE the
    render starts. If that enforcement is bypassed, returning ``"nvenc"`` here
    means ffmpeg itself errors out loudly rather than CPU-crawling.
    """
    return resolve_status()["backend"]


def video_encoder_args(
    crf: int = 20,
    preset_hint: str = "medium",
) -> List[str]:
    """Return the ffmpeg args for the active video encoder.

    Drops straight in where call sites used to hard-code
    ``["-c:v", "libx264", "-preset", "medium", "-crf", "20"]``.
    Always emits the encoder + preset + quality knob; pixel format
    and faststart stay at the call site since they're orthogonal.

    Parameters
    ----------
    crf
        libx264 CRF value (0–51, lower = better). Mapped to ``cq+2``
        for NVENC because their scales differ by ~2 points perceptually.
    preset_hint
        libx264 preset name. Mapped to NVENC ``p1..p7`` via the table
        above. Unknown names fall back to ``p4`` (medium).
    """
    backend = _selected_backend()
    if backend == "nvenc":
        nv_preset = _NVENC_PRESET_BY_LIBX264.get(
            (preset_hint or "medium").lower(), "p4"
        )
        # CQ ~2 points hotter than CRF for similar perceived quality.
        # Clamp so we never ship absurd values when a caller passes
        # crf=0 (lossless libx264, which NVENC can't match anyway).
        cq = max(1, min(51, int(crf) + 2))
        return [
            "-c:v", "h264_nvenc",
            "-preset", nv_preset,
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ]
    # libx264 fallback — identical to the legacy hard-coded form
    return [
        "-c:v", "libx264",
        "-preset", (preset_hint or "medium"),
        "-crf", str(int(crf)),
    ]


def video_decoder_args() -> List[str]:
    """Input-side GPU decode args (Wave 4, item B).

    Returns ``["-hwaccel", "cuda"]`` when the NVENC backend is active,
    ``[]`` otherwise. IMPORTANT: ``-hwaccel`` is an INPUT option — the
    caller must place these args BEFORE the ``-i`` of the video input
    they should accelerate (V4 call sites put them before the FIRST
    ``-i``). We deliberately do NOT emit ``-hwaccel_output_format cuda``
    so decoded frames land back in system memory and the existing CPU
    filtergraphs (overlay/trim/concat/scale) keep working unchanged;
    ffmpeg also silently falls back to software decode for codecs the
    GPU can't handle, so this is safe-by-default.

    Override via ``KAIZER_VIDEO_DECODER``:
      - ``auto`` / unset → follow the encoder backend (cuda iff nvenc)
      - ``cuda``         → force GPU decode
      - ``cpu``          → force software decode (empty list)
    """
    forced = (os.environ.get("KAIZER_VIDEO_DECODER") or "auto").strip().lower()
    if forced in ("cpu", "none", "off", "soft", "software"):
        return []
    if forced == "cuda":
        return ["-hwaccel", "cuda"]
    # auto / anything else → tie to the encoder backend selection
    return ["-hwaccel", "cuda"] if _selected_backend() == "nvenc" else []


def active_backend_label() -> str:
    """One-word label used by the orchestrator's start-of-stage log
    so the user can tell at a glance whether a job is running on GPU
    or CPU. Returns ``"nvenc"`` or ``"libx264"``.
    """
    return _selected_backend()
