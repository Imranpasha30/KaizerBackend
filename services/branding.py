"""Branding service — Phase 2.D Branding agent.

Owns the publish-time overlay pass: takes a clean MasterVideo + a
ResolvedBrand and produces a per-channel branded artifact, cached in
R2 under ``branded/{master_video_id}/{brand_profile_version}.mp4``
(Decision 11, CONTRACTS §4.3).

This module is the **public surface** the Upload agent (E) calls into
from inside ``services/scheduler._run_job()``. It is synchronous so
the scheduler can ``await asyncio.to_thread(branding.process_upload_job, id)``
in Phase 2.E.

PUBLIC API
----------
  - ``brand_artifact_cache_key(master_video_id, version) -> str``
  - ``process_upload_job(upload_job_id) -> str``
  - ``cleanup_expired() -> int``
  - ``snapshot() -> dict``

CACHE KEY (Decision 11)
-----------------------
``branded/{master_video_id}/{brand_profile_version}.mp4`` — no date
prefix, no user prefix. The R2_KEY_PREFIX env var of the storage
provider gives us the env namespacing (e.g. ``local/branded/...``).

TTL (CONTRACTS §2)
------------------
``KAIZER_BRANDED_ARTIFACT_TTL_HOURS`` (default 24). Enforced by
``cleanup_expired()`` (an hourly cron the F-agent wires up — the
function itself lives here).

LEGACY-DURING-TRANSITION (Decision 12)
---------------------------------------
``KAIZER_CLEAN_MASTER=0`` is the default: existing render path still
bakes the logo. If we then overlay the logo again here, the result is
double-stamped. The MasterVideo row's ``clean_master`` boolean tells
us which world it came from:

  - ``clean_master=True``  → render produced a fully clean master;
    overlay both the logo (top-right bug) AND the text watermark.
  - ``clean_master=False`` → render already baked the logo; SKIP the
    logo overlay, apply only the text watermark. Logged as a warning
    so ops sees the transitional behaviour during cutover.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from database import SessionLocal
import models

from services.brand_resolver import ResolvedBrand, resolve_brand_profile

log = logging.getLogger("kaizer.branding")


# ─── Tunables (env-driven; never hardcoded) ────────────────────────────────

_DEFAULT_TTL_HOURS = 24


def _ttl_hours() -> int:
    raw = os.environ.get("KAIZER_BRANDED_ARTIFACT_TTL_HOURS", "").strip()
    if not raw:
        return _DEFAULT_TTL_HOURS
    try:
        n = int(raw)
        return max(1, n)
    except Exception:
        log.warning(
            "branding: KAIZER_BRANDED_ARTIFACT_TTL_HOURS=%r invalid; using %d",
            raw, _DEFAULT_TTL_HOURS,
        )
        return _DEFAULT_TTL_HOURS


def _clean_master_flag() -> bool:
    """True when the render pipeline is operating in clean-master mode
    (Decision 12). Default '0' during transition."""
    return os.environ.get("KAIZER_CLEAN_MASTER", "0").strip() == "1"


def _ffmpeg_bin() -> str:
    """Honour the same FFMPEG_BIN selection as pipeline_core."""
    try:
        from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
        if FFMPEG_BIN:
            return FFMPEG_BIN
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or "ffprobe"


# ─── Exceptions (stable codes the caller maps to UploadJob.last_error) ─────


class BrandingError(Exception):
    """Base class for branding service errors."""
    code: str = "branding_error"


class MasterVideoMissingError(BrandingError):
    code = "master_video_missing"


class MasterVideoNotReadyError(BrandingError):
    code = "master_video_not_ready"


class BrandResolveError(BrandingError):
    code = "brand_resolve_failed"


class FFmpegOverlayError(BrandingError):
    code = "ffmpeg_overlay_failed"

    def __init__(self, message: str, *, returncode: int, stderr_tail: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class StorageWriteError(BrandingError):
    code = "storage_write_failed"


# ─── Observability ─────────────────────────────────────────────────────────

_snapshot_lock = threading.Lock()
_snapshot: dict = {
    "in_flight": 0,
    "hits": 0,
    "misses": 0,
    "ffmpeg_invocations": 0,
    "last_cleanup_removed_count": 0,
    "last_cleanup_at": None,        # ISO-8601 UTC
    "last_ffmpeg_command": "",      # shell-safe rendering of the last filter graph
    "last_ffmpeg_duration_seconds": None,
}


def _bump(key: str, delta: int = 1) -> None:
    with _snapshot_lock:
        _snapshot[key] = int(_snapshot.get(key, 0)) + delta


def _set_snap(key: str, value) -> None:
    with _snapshot_lock:
        _snapshot[key] = value


def snapshot() -> dict:
    """Observability snapshot (used by /admin in Phase 2.G)."""
    with _snapshot_lock:
        return dict(_snapshot)


# ─── Cache key ────────────────────────────────────────────────────────────


def brand_artifact_cache_key(master_video_id: int, brand_profile_version: str) -> str:
    """Decision 11: ``branded/{master_video_id}/{brand_profile_version}.mp4``.

    The R2_KEY_PREFIX env var of the storage provider handles env
    namespacing (dev/prod). This function returns the unprefixed
    caller-facing key, which the provider's ``_k()`` helper will then
    prefix internally on upload/download.
    """
    if not brand_profile_version:
        raise ValueError("brand_artifact_cache_key: empty brand_profile_version")
    return f"branded/{int(master_video_id)}/{brand_profile_version}.mp4"


# ─── Cache check / cleanup (R2 ops via storage provider + boto3 fallback) ──


def _storage_provider():
    """Return the configured storage provider (R2 in prod, local in dev)."""
    from pipeline_core.storage import get_storage_provider
    return get_storage_provider()


def _cache_hit(cache_key: str) -> bool:
    """Does an artifact already exist at ``cache_key``?

    Uses ``provider.exists()`` which is supported on both LocalStorage
    and R2Storage in ``pipeline_core/storage.py`` (no new methods on
    the read-only contract).
    """
    try:
        return bool(_storage_provider().exists(cache_key))
    except Exception as exc:
        log.warning("branding: cache lookup failed for key=%r: %s", cache_key, exc)
        return False


# ─── Concurrency control: serialize concurrent cache-miss writers ─────────
#
# Without this, N concurrent jobs sharing the same brand_profile_version
# all see cache_miss=True because none has yet finished _persist_cache_key()'d,
# so each runs ffmpeg redundantly and races on the R2 write. We mirror the
# pg_advisory_xact_lock pattern from services/credits.py — Postgres in prod
# gets correct serialization, SQLite degrades to a process-local lock.


def _is_postgres(db: Session) -> bool:
    """Detect the active SQLAlchemy dialect so we can branch on locking
    primitives. Mirrors services/credits.py:_is_postgres."""
    try:
        return db.bind.dialect.name == "postgresql"  # type: ignore[union-attr]
    except Exception:
        return False


def _cache_key_to_lock_id(cache_key: str) -> int:
    """Stable signed int64 derived from the cache_key, suitable for
    ``pg_advisory_xact_lock(bigint)``. Negative values are fine —
    postgres advisory locks accept signed bigint."""
    digest = hashlib.sha256(cache_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


# Process-local fallback for SQLite (which has no advisory locks). Keyed
# by cache_key so two threads in the same process at least serialize. R2
# write contention across processes is impossible in dev (LocalStorage
# is filesystem-only and a single dev process).
_PROCESS_LOCAL_KEY_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCAL_KEY_LOCKS_GUARD = threading.Lock()


def _get_process_local_lock(cache_key: str) -> threading.Lock:
    with _PROCESS_LOCAL_KEY_LOCKS_GUARD:
        lk = _PROCESS_LOCAL_KEY_LOCKS.get(cache_key)
        if lk is None:
            lk = threading.Lock()
            _PROCESS_LOCAL_KEY_LOCKS[cache_key] = lk
        return lk


@contextlib.contextmanager
def _advisory_lock(db: Session, cache_key: str):
    """Acquire a per-cache-key mutex around the miss path.

    On Postgres: ``pg_advisory_xact_lock(bigint)`` — blocking, auto-released
    when the caller's transaction COMMITs or ROLLBACKs. The lock key is
    derived from a SHA256 of the cache_key so different keys never collide
    with each other or with the per-user keys used by credits.py.

    On SQLite (dev): falls through to a process-local threading.Lock keyed
    on the same string. Production runs on Postgres so this is acceptable
    as a documented fallback.

    The caller MUST commit (or rollback) before returning so the pg lock
    releases promptly. ``_persist_cache_key()`` issues the final commit on
    the success path; the ``finally`` block in ``process_upload_job`` only
    closes the session, so an exception that escapes without an explicit
    rollback will still release the lock when the session is closed
    (closing rolls back any in-flight tx).
    """
    key_int = _cache_key_to_lock_id(cache_key)
    if _is_postgres(db):
        try:
            db.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": int(key_int)},
            )
        except OperationalError as exc:
            log.warning(
                "branding._advisory_lock: pg lock failed for cache_key=%r "
                "(key_int=%d): %s; falling through unlocked",
                cache_key, key_int, exc,
            )
            yield
            return
        try:
            yield
        finally:
            # xact_lock auto-releases on commit/rollback; nothing to do here.
            # If the caller forgot to commit and the session closes via the
            # outer finally, SQLAlchemy issues a ROLLBACK which still
            # releases the lock.
            pass
    else:
        lk = _get_process_local_lock(cache_key)
        lk.acquire()
        try:
            yield
        finally:
            lk.release()


# ─── Filter-graph construction ─────────────────────────────────────────────
#
# We deliberately do NOT import from pipeline_v4/watermark.py. The
# logic below is a copy-with-isolation: the render pipeline must remain
# read-only INPUT (brief §0). When pipeline_v4 evolves it must not
# break this branding service.


def _probe_source(path: str) -> tuple[int, int, float]:
    """Probe (width, height, duration_seconds) from the source video.

    Duration is essential: our filter graph has ``-loop 1`` PNG
    overlays which produce infinite streams. Without an explicit
    ``-t <duration>`` clamp, ``-shortest`` only kicks in if the
    source has an audio stream — test masters with no audio cause
    ffmpeg to encode forever.

    Stdout is captured as bytes to dodge a Windows-only hang in
    ``subprocess.run(capture_output=True, text=True)``. Defaults are
    1920x1080 + 30s — sensible last-resort values that never produce
    a runaway encode even if probing fails entirely.
    """
    try:
        import json
        r = subprocess.run(
            [
                _ffprobe_bin(), "-v", "error",
                "-show_entries",
                "stream=width,height,codec_type,duration:format=duration",
                "-of", "json", path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if r.returncode == 0:
            data = json.loads((r.stdout or b"").decode("utf-8", errors="replace") or "{}")
            streams = data.get("streams") or []
            video_stream = next(
                (s for s in streams if s.get("codec_type") == "video"),
                streams[0] if streams else {},
            )
            w = int(video_stream.get("width") or 1920)
            h = int(video_stream.get("height") or 1080)
            # Duration: prefer container format duration; fall back to stream.
            dur_str = (data.get("format") or {}).get("duration") or video_stream.get("duration")
            try:
                duration = float(dur_str) if dur_str else 30.0
            except Exception:
                duration = 30.0
            if duration <= 0 or duration > 7200:
                duration = 30.0
            return w, h, duration
    except Exception as exc:
        log.warning("branding: ffprobe failed (%s); defaulting to 1920x1080 / 30s", exc)
    return 1920, 1080, 30.0


def _source_has_audio(path: str) -> bool:
    """True iff the file carries at least one audio stream.

    Best-effort and only consulted when the A/V nudge is active (it decides
    whether to apply ``atempo`` + AAC re-encode vs. keep ``-c:a copy``). On
    ANY probe failure we assume audio IS present — production masters always
    carry audio, so a flaky probe must never silently drop the audio track.
    """
    try:
        r = subprocess.run(
            [
                _ffprobe_bin(), "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0", path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if r.returncode == 0:
            return bool((r.stdout or b"").strip())
    except Exception as exc:
        log.warning("branding: audio probe failed (%s); assuming audio present", exc)
    return True


def _video_dimensions(path: str) -> tuple[int, int]:
    """Backward-compatibility shim — kept for callers that only need
    (width, height). Prefer ``_probe_source`` going forward."""
    w, h, _ = _probe_source(path)
    return w, h


def _watermark_overlay_xy(
    position: str, canvas_w: int = 0, canvas_h: int = 0, margin: int = 40,
) -> tuple[str, str]:
    """ffmpeg overlay x,y expressions for the watermark plate.

    Two fixes over the old version:
      1. ALL nine positions are honoured. The old code collapsed every
         corner (top-right, top-left, bottom-*) to lower-center, so a
         channel set to 'top-right' actually rendered lower-center.
      2. PORTRAIT (Shorts) awareness. On a 9:16 short the lower ~45% is
         the news-card / caption strip — a watermark placed there is
         invisible against the busy card. So on portrait we lift any
         non-top position to upper-center (clear of a top-right logo
         bug), keeping it on the visible video. Landscape honours the
         configured spot (the bottom of a 16:9 frame is fine there).
    """
    p = (position or "lower-center").lower().replace("_", "-")
    portrait = bool(canvas_w) and bool(canvas_h) and canvas_h > canvas_w * 1.2
    if portrait and p not in ("top-left", "top-center", "upper-center"):
        p = "upper-center"
    M = int(margin)
    table = {
        "top-left":     (f"{M}", f"{M}"),
        "top-right":    (f"W-w-{M}", f"{M}"),
        "top-center":   ("(W-w)/2", f"{M}"),
        "upper-center": ("(W-w)/2", "H/8"),
        "center":       ("(W-w)/2", "(H-h)/2"),
        "center-left":  (f"{M}", "(H-h)/2"),
        "center-right": (f"W-w-{M}", "(H-h)/2"),
        "bottom-left":  (f"{M}", f"H-h-{M}"),
        "bottom-right": (f"W-w-{M}", f"H-h-{M}"),
        "lower-center": ("(W-w)/2", f"H-h-{M}"),
    }
    return table.get(p, ("(W-w)/2", "H/8"))  # default: upper-center (visible)


def _render_plate_png(
    *,
    text: str,
    canvas_h: int,
    opacity: float,
    out_path: str,
) -> str:
    """Copied from pipeline_v4/watermark.py:_render_plate_png for
    refactor isolation. Text-only watermark plate."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font_size = max(28, int(canvas_h * 0.045))
        font = ImageFont.load_default()
        try:
            from pipeline_core.pipeline import FONTS_DIR  # type: ignore
            candidate = os.path.join(FONTS_DIR, "NotoSans-Bold.ttf")
            if os.path.isfile(candidate):
                font = ImageFont.truetype(candidate, font_size)
        except Exception:
            pass
        bbox = font.getbbox(text[:30])
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        font = ImageFont.load_default()
        text_w, text_h = 240, 40

    pad_x, pad_y = 16, 8
    plate_w = max(40, text_w + 2 * pad_x)
    plate_h = max(20, text_h + 2 * pad_y)
    alpha = max(0, min(255, int(round(opacity * 255))))

    img = Image.new("RGBA", (plate_w, plate_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if text:
        # 1-px shadow for legibility, both fade with opacity.
        d.text((pad_x + 1, pad_y + 1), text[:30],
               font=font, fill=(0, 0, 0, alpha))
        d.text((pad_x, pad_y), text[:30],
               font=font, fill=(255, 255, 255, alpha))
    img.save(out_path, "PNG")
    return out_path


def _prepare_logo_png(
    logo_local_path: str,
    canvas_w: int,
    canvas_h: int,
    work_dir: str,
    rect: Optional[dict] = None,
) -> str:
    """Pre-resize the logo to a bug PNG. Default = 10% of canvas height
    (top-right bug). When ``rect`` (a template's logo slot, output-canvas px)
    is given, FIT the logo inside that rect (contain) so it lands in the
    designed spot instead of the default corner."""
    from PIL import Image as _PI
    with _PI.open(logo_local_path) as logo:
        logo = logo.convert("RGBA")
        if rect and rect.get("w") and rect.get("h"):
            rw, rh = max(8, int(rect["w"])), max(8, int(rect["h"]))
            ratio = min(rw / max(1, logo.width), rh / max(1, logo.height))
            bug_w = max(8, int(logo.width * ratio))
            bug_h = max(8, int(logo.height * ratio))
        else:
            bug_h = max(48, int(canvas_h * 0.10))
            ratio = bug_h / max(1, logo.height)
            bug_w = max(48, int(logo.width * ratio))
        bug_png = os.path.join(work_dir, "_wm_bug.png")
        logo.resize((bug_w, bug_h), _PI.LANCZOS).save(bug_png, "PNG")
    return bug_png


def _build_ffmpeg_command(
    *,
    source_path: str,
    out_path: str,
    work_dir: str,
    apply_logo: bool,
    apply_text: bool,
    logo_local_path: Optional[str],
    text: str,
    opacity: float,
    position: str,
    canvas_w: int,
    canvas_h: int,
    source_duration: float,
    nudge_factor: float = 1.0,
    has_audio: bool = True,
    zoom_factor: float = 1.0,
    logo_rect: Optional[dict] = None,
    wm_rect: Optional[dict] = None,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
    grade_b: float = 0.0,
    grade_s: float = 1.0,
) -> list[str]:
    """Build the ffmpeg argv list for the overlay pass.

    Filter chain mirrors pipeline_v4/watermark.py:stamp_for_channel
    (lines 262-307):

      - Input 0: source mp4
      - Input 1 (optional): logo PNG (top-right bug at full opacity)
      - Input 2 (optional): text plate PNG (positioned by ``position``)

    Encoder: GPU (h264_nvenc / qsv / amf) when available, libx264
    fallback — via ``pipeline_core.hw_accel.h264_args``. Only the ENCODE
    moves to the GPU; the overlay filter still runs on CPU (NVENC accepts
    the CPU-composited frames). This is the expensive part: a 12-min
    1080p re-encode drops from >10 min on CPU (which blew the 600s
    timeout for long-form) to ~1-2 min on the RTX. ``-c:a copy`` is
    preserved (zero audio re-encode → no lipsync drift).
    """
    inputs: list[str] = ["-i", source_path]
    chain_parts: list[str] = []
    last_label = "0:v"
    next_input_idx = 1

    if apply_logo and logo_local_path:
        inputs += ["-loop", "1", "-i", logo_local_path]
        if logo_rect:
            # Center the (slot-fitted) logo inside the template's marked rect.
            rx, ry = int(logo_rect["x"]), int(logo_rect["y"])
            rw, rh = int(logo_rect["w"]), int(logo_rect["h"])
            _logo_xy = f"x={rx}+({rw}-w)/2:y={ry}+({rh}-h)/2"
        else:
            margin = max(20, int(canvas_w * 0.015))
            _logo_xy = f"x=W-w-{margin}:y={margin}"
        chain_parts.append(
            f"[{last_label}][{next_input_idx}:v]"
            f"overlay={_logo_xy}:format=auto[bug]"
        )
        last_label = "bug"
        next_input_idx += 1

    if apply_text and text:
        plate_path = os.path.join(work_dir, "_wm_plate.png")
        _render_plate_png(
            text=text, canvas_h=canvas_h,
            opacity=opacity, out_path=plate_path,
        )
        inputs += ["-loop", "1", "-i", plate_path]
        if wm_rect:
            # Center the text plate inside the template's marked watermark rect.
            rx, ry = int(wm_rect["x"]), int(wm_rect["y"])
            rw, rh = int(wm_rect["w"]), int(wm_rect["h"])
            wx, wy = f"{rx}+({rw}-w)/2", f"{ry}+({rh}-h)/2"
        else:
            wx, wy = _watermark_overlay_xy(position, canvas_w, canvas_h)
        chain_parts.append(
            f"[{last_label}][{next_input_idx}:v]"
            f"overlay=x={wx}:y={wy}:format=auto[outv]"
        )
        last_label = "outv"
        next_input_idx += 1

    nudge_on = abs(float(nudge_factor) - 1.0) > 1e-6
    zoom_on = abs(float(zoom_factor) - 1.0) > 1e-6
    grade_on = abs(float(grade_b)) > 1e-6 or abs(float(grade_s) - 1.0) > 1e-6

    if not chain_parts and not nudge_on and not zoom_on and not grade_on:
        # Shouldn't be reachable — caller short-circuits to pure copy.
        raise BrandingError("branding: no overlays to apply; caller bug")

    # ── Per-channel anti-duplicate video filters ───────────────────────
    # ZOOM (picture half): scale up by Z then crop back to the EXACT canvas with a
    # per-channel PAN offset (framing differs, not just scale). GRADE: an invisible
    # per-channel colour micro-shift (eq). NUDGE (timing half): setpts=PTS/F speeds the
    # video; the audio gets atempo=F (same F) below, so lip-sync stays frame-exact.
    # Stacked, each channel's audio+video fingerprint diverges. (last_label may be
    # "0:v" when there are no overlays — a pure anti-dup re-encode.)
    vmap = last_label
    _vf: list[str] = []
    if zoom_on:
        Z = float(zoom_factor)
        zw = int(round(int(canvas_w) * Z)); zw += zw % 2
        zh = int(round(int(canvas_h) * Z)); zh += zh % 2
        cx = max(0, int(round((zw - int(canvas_w)) * float(pan_x))))
        cy = max(0, int(round((zh - int(canvas_h)) * float(pan_y))))
        _vf.append(f"scale={zw}:{zh}")
        _vf.append(f"crop={int(canvas_w)}:{int(canvas_h)}:{cx}:{cy}")
    if grade_on:
        _vf.append(f"eq=brightness={float(grade_b):.3f}:saturation={float(grade_s):.3f}")
    if nudge_on:
        _vf.append(f"setpts=PTS/{float(nudge_factor):.6f}")
    if _vf:
        chain_parts.append(f"[{last_label}]{','.join(_vf)}[vout]")
        vmap = "vout"

    # atempo needs a real audio stream AND an audio re-encode (it is
    # mutually exclusive with '-c:a copy'). When the source has no audio
    # (synthetic test masters), skip it and keep the copy path.
    if nudge_on and has_audio:
        F = float(nudge_factor)
        chain_parts.append(f"[0:a]atempo={F:.6f}[aout]")
        audio_args = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    else:
        audio_args = ["-map", "0:a?", "-c:a", "copy"]

    # ``-t {source_duration}`` is the hard clamp that stops ffmpeg
    # from running forever when the source has no audio stream — the
    # PNG inputs are looped infinitely with ``-loop 1`` and ``-shortest``
    # only triggers when there's a finite audio stream sharing the
    # output. Production masters always have audio so this is belt-
    # and-braces, but the smoke test's synthetic master proves we need
    # the clamp regardless. (A nudge makes the real output ~source/F long;
    # this clamp sits above that and ``-shortest`` still ends at the
    # nudged stream end, so it stays correct.)
    # A small +0.5s pad protects against rounding in ffprobe's reported
    # duration; ``-shortest`` still clamps to the source's actual end.
    return [
        _ffmpeg_bin(), "-y", "-v", "error",
        *inputs,
        "-filter_complex", ";".join(chain_parts),
        "-map", f"[{vmap}]",
        # GPU encode (NVENC/QSV/AMF) when available, libx264 fallback.
        # h264_args() brings -pix_fmt + -profile + -movflags faststart.
        *_video_encode_args(),
        *audio_args,
        "-t", f"{max(0.5, source_duration + 0.5):.3f}",
        "-shortest",
        out_path,
    ]


# ── Superfast GPU overlay (full-CUDA pipeline) ──────────────────────
# The CPU path above decodes + composites overlays on the CPU, then NVENC-
# encodes — and benchmarking showed the NVENC PRESET is NOT the bottleneck
# (p5->p1 barely moved a 284s 1080p job: 52s->48s). The cost is the CPU
# decode + CPU overlay + GPU<->CPU frame copies. Moving the WHOLE chain onto
# the GPU (NVDEC decode -> scale_cuda -> overlay_cuda -> NVENC) cut the same
# job to ~28s (~1.85x). overlay_cuda pads the coded height to a /16 multiple
# (1080->1088), so we hwdownload + crop back to the EXACT canvas dims at the
# end (verified: exact 1080, no distortion, audio copied). Flag-gated OFF;
# any failure auto-falls-back to the CPU command (so nothing breaks).

_GPU_OVERLAY_CAPS: Optional[bool] = None


def _gpu_overlay_caps() -> bool:
    """True iff this host can run the full-CUDA overlay: NVENC is the active
    encoder AND ffmpeg has the cuda overlay/scale/upload filters. Cached."""
    global _GPU_OVERLAY_CAPS
    if _GPU_OVERLAY_CAPS is not None:
        return _GPU_OVERLAY_CAPS
    ok = False
    try:
        from pipeline_core.hw_accel import ACTIVE_ENCODER
        if ACTIVE_ENCODER == "h264_nvenc":
            r = subprocess.run(
                [_ffmpeg_bin(), "-hide_banner", "-filters"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, timeout=15,
            )
            f = r.stdout or ""
            ok = ("overlay_cuda" in f and "scale_cuda" in f
                  and "hwupload_cuda" in f)
    except Exception:
        ok = False
    _GPU_OVERLAY_CAPS = ok
    log.info("branding: GPU overlay caps = %s", ok)
    return ok


def _gpu_encode_args() -> list[str]:
    """Fast NVENC args for the GPU overlay pass. p1/low-latency is fine for a
    logo-stamp re-encode of already-finished video; env-tunable."""
    preset = (os.environ.get("KAIZER_BRAND_NVENC_PRESET", "").strip() or "p1")
    cq = (os.environ.get("KAIZER_BRAND_NVENC_CQ", "").strip() or "25")
    return [
        "-c:v", "h264_nvenc", "-preset", preset, "-tune", "ll",
        "-rc", "vbr", "-cq", cq, "-b:v", "8M", "-maxrate", "10M",
        "-bufsize", "16M", "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-movflags", "+faststart",
    ]


def _png_size(path: str) -> tuple[int, int]:
    from PIL import Image as _PI
    with _PI.open(path) as im:
        return int(im.width), int(im.height)


def _overlay_xy_numeric(
    position: str, W: int, H: int, w: int, h: int, margin: int = 40,
) -> tuple[int, int]:
    """Numeric (integer) twin of _watermark_overlay_xy — overlay_cuda takes
    plain x/y, so we resolve the position table to pixels (W/H=canvas,
    w/h=overlay) instead of ffmpeg W/w/H/h expressions."""
    p = (position or "lower-center").lower().replace("_", "-")
    portrait = H > W * 1.2
    if portrait and p not in ("top-left", "top-center", "upper-center"):
        p = "upper-center"
    M = int(margin)
    table = {
        "top-left":     (M, M),
        "top-right":    (W - w - M, M),
        "top-center":   ((W - w) // 2, M),
        "upper-center": ((W - w) // 2, H // 8),
        "center":       ((W - w) // 2, (H - h) // 2),
        "center-left":  (M, (H - h) // 2),
        "center-right": (W - w - M, (H - h) // 2),
        "bottom-left":  (M, H - h - M),
        "bottom-right": (W - w - M, H - h - M),
        "lower-center": ((W - w) // 2, H - h - M),
    }
    x, y = table.get(p, ((W - w) // 2, H // 8))
    return max(0, int(x)), max(0, int(y))


def _build_gpu_ffmpeg_command(
    *,
    source_path: str,
    out_path: str,
    work_dir: str,
    apply_logo: bool,
    apply_text: bool,
    logo_local_path: Optional[str],
    text: str,
    opacity: float,
    position: str,
    canvas_w: int,
    canvas_h: int,
    source_duration: float,
    nudge_factor: float = 1.0,
    has_audio: bool = True,
    zoom_factor: float = 1.0,
    logo_rect: Optional[dict] = None,   # accepted for **kwargs parity; slot renders force CPU
    wm_rect: Optional[dict] = None,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
    grade_b: float = 0.0,
    grade_s: float = 1.0,
) -> list[str]:
    """Full-CUDA twin of _build_ffmpeg_command: NVDEC decode -> scale_cuda ->
    overlay_cuda (numeric positions) -> hwdownload+crop to exact canvas dims
    -> NVENC. Same logo/plate prep + same output contract (-c:a copy, -t,
    -shortest) as the CPU path."""
    inputs: list[str] = [
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", source_path,
    ]
    chain: list[str] = ["[0:v]scale_cuda=format=yuv420p[m]"]
    last = "m"
    idx = 1
    margin = max(20, int(canvas_w * 0.015))

    if apply_logo and logo_local_path:
        lw, lh = _png_size(logo_local_path)
        x = max(0, canvas_w - lw - margin)
        y = margin
        inputs += ["-loop", "1", "-i", logo_local_path]
        chain.append(f"[{idx}:v]format=yuva420p,hwupload_cuda[l{idx}]")
        chain.append(f"[{last}][l{idx}]overlay_cuda=x={x}:y={y}[bug]")
        last = "bug"
        idx += 1

    if apply_text and text:
        plate_path = os.path.join(work_dir, "_wm_plate.png")
        _render_plate_png(text=text, canvas_h=canvas_h,
                          opacity=opacity, out_path=plate_path)
        pw, ph = _png_size(plate_path)
        x, y = _overlay_xy_numeric(position, canvas_w, canvas_h, pw, ph, 40)
        inputs += ["-loop", "1", "-i", plate_path]
        chain.append(f"[{idx}:v]format=yuva420p,hwupload_cuda[p{idx}]")
        chain.append(f"[{last}][p{idx}]overlay_cuda=x={x}:y={y}[over]")
        last = "over"
        idx += 1

    if last == "m":
        raise BrandingError("branding(gpu): no overlays to apply; caller bug")

    # overlay_cuda pads coded height to /16 — bring back to CPU and crop to the
    # EXACT canvas size so a 1080 source never becomes 1088 (QC would reject).
    # The frames are already in system memory after hwdownload, so the A/V
    # pacing nudge (setpts) folds straight onto this same node — no extra
    # GPU<->CPU detour (the detour the nudge would otherwise need is already here).
    nudge_on = abs(float(nudge_factor) - 1.0) > 1e-6
    zoom_on = abs(float(zoom_factor) - 1.0) > 1e-6
    grade_on = abs(float(grade_b)) > 1e-6 or abs(float(grade_s) - 1.0) > 1e-6
    tail = (
        f"[{last}]hwdownload,format=yuv420p,"
        f"crop={int(canvas_w)}:{int(canvas_h)}:0:0"
    )
    # Anti-dup, on the sysmem frames after hwdownload (so eq/offset-crop work here too):
    # ZOOM scale-up + per-channel PAN-offset crop back to the exact canvas; GRADE = invisible
    # per-channel colour micro-shift; NUDGE (setpts) folds on last — no extra GPU<->CPU detour.
    if zoom_on:
        Z = float(zoom_factor)
        zw = int(round(int(canvas_w) * Z)); zw += zw % 2
        zh = int(round(int(canvas_h) * Z)); zh += zh % 2
        cx = max(0, int(round((zw - int(canvas_w)) * float(pan_x))))
        cy = max(0, int(round((zh - int(canvas_h)) * float(pan_y))))
        tail += f",scale={zw}:{zh},crop={int(canvas_w)}:{int(canvas_h)}:{cx}:{cy}"
    if grade_on:
        tail += f",eq=brightness={float(grade_b):.3f}:saturation={float(grade_s):.3f}"
    if nudge_on:
        tail += f",setpts=PTS/{float(nudge_factor):.6f}"
    tail += "[outv]"
    chain.append(tail)

    # Audio decodes to the CPU normally (hwaccel only touches video), so
    # atempo applies cleanly. Re-encode (AAC) only when nudging a real
    # audio stream; otherwise copy as before.
    if nudge_on and has_audio:
        chain.append(f"[0:a]atempo={float(nudge_factor):.6f}[aout]")
        audio_args = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    else:
        audio_args = ["-map", "0:a?", "-c:a", "copy"]

    return [
        _ffmpeg_bin(), "-y", "-v", "error",
        *inputs,
        "-filter_complex", ";".join(chain),
        "-map", "[outv]",
        *_gpu_encode_args(),
        *audio_args,
        "-t", f"{max(0.5, source_duration + 0.5):.3f}",
        "-shortest",
        out_path,
    ]


def _qc_branded_artifact(
    source_local: str, out_local: str, expected_factor: float = 1.0,
) -> None:
    """Post-overlay QC gate (Wave 4 item H): verify the branded artifact
    BEFORE the R2 upload so a truncated / silent / runaway-encoded file
    never gets cached and published.

    Cheap by design: exactly one ffprobe of the input + one of the
    output. Checks:
      - branded output has >= 1 video stream
      - audio stream survived (when the master had one — '-c:a copy'
        must not lose it)
      - output duration equals the master's duration +/- 0.5s

    Raises :class:`FFmpegOverlayError` on violation — the existing error
    path (retry / cleanup / last_error mapping) handles it unchanged.
    Skippable via KAIZER_V4_QC=0 (same flag as the render-side gate).
    """
    if (os.environ.get("KAIZER_V4_QC", "1") or "1").strip() == "0":
        return
    try:
        # Lazy import — pipeline_v4.qc is a leaf module (json/subprocess
        # only); keeping the import here avoids any render-pipeline
        # import surface at branding module load.
        from pipeline_v4.qc import probe_media
    except Exception as exc:
        log.warning("branding: QC probe unavailable (%s); skipping QC gate", exc)
        return
    try:
        src = probe_media(source_local)
        out = probe_media(out_local)
    except Exception as exc:
        raise FFmpegOverlayError(
            f"branded artifact QC probe failed: {exc}",
            returncode=-1,
            stderr_tail=str(exc)[-500:],
        ) from exc

    problems: list[str] = []
    if out.get("n_video", 0) < 1:
        problems.append("branded artifact has no video stream")
    if src.get("n_audio", 0) >= 1 and out.get("n_audio", 0) < 1:
        problems.append("audio stream lost during overlay (master had audio)")
    sd = src.get("duration")
    od = out.get("duration")
    if sd is not None and od is not None:
        # When the per-channel A/V nudge sped the clip up by ``expected_factor``,
        # the branded artifact is LEGITIMATELY ~sd/F long — compare against that
        # expected duration, not the raw master, or a 2% speed-up falsely reads
        # as a truncated encode (the bug the map flagged). With factor 1.0 this
        # is exactly the original behaviour (exp == sd, same 0.5s/2% tolerances).
        try:
            f = float(expected_factor)
        except (TypeError, ValueError):
            f = 1.0
        if not (f > 0):
            f = 1.0
        exp = sd / f
        nudged = abs(f - 1.0) > 1e-6
        # Asymmetric tolerance. The corruption signal QC exists for is
        # the branded output coming out SHORTER (truncated encode,
        # killed ffmpeg). Slightly LONGER is benign: AAC priming/padding
        # frames add a roughly constant (~0.5s worst case) tail. Keep the
        # original tight 0.5s shortfall bound when un-nudged; allow a small
        # proportional margin when nudging (setpts/atempo round a few frames).
        short_tol = 0.5 if not nudged else max(0.75, 0.01 * exp)
        if (exp - od) > short_tol:
            problems.append(
                f"branded artifact TRUNCATED: expected ~{exp:.2f}s "
                f"(master {sd:.2f}s / nudge {f:.4f}) vs branded {od:.2f}s "
                f"(shorter by {exp - od:.2f}s)"
            )
        elif (od - exp) > max(1.0, 0.02 * exp):
            problems.append(
                f"duration drift: expected ~{exp:.2f}s (master {sd:.2f}s) vs "
                f"branded {od:.2f}s (longer by {od - exp:.2f}s)"
            )
    if problems:
        msg = "branded artifact failed QC: " + "; ".join(problems)
        log.error("branding: %s", msg)
        raise FFmpegOverlayError(msg, returncode=0, stderr_tail="; ".join(problems))


def _video_encode_args() -> list[str]:
    """H.264 encode args for the overlay re-encode — GPU (NVENC/QSV/AMF)
    when available, libx264 fallback. Delegates to the shared
    ``pipeline_core.hw_accel.h264_args`` (same encoder the render
    pipeline uses) with a fast CPU preset so GPU-less hosts stay quick
    too. Lazy import so a minimal test env without pipeline_core still
    brands (degrades to a safe CPU default)."""
    try:
        from pipeline_core.hw_accel import h264_args
        return h264_args(cpu_preset="veryfast")
    except Exception:
        return [
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]


def _encode_timeout(duration_seconds: float) -> int:
    """ffmpeg wall-clock budget for the overlay encode. The GPU finishes
    a 12-min 1080p in ~1-2 min, but the CPU fallback (GPU-less hosts)
    needs far longer for long-form — so scale with duration to stop a
    genuine longform video from falsely timing out (the old fixed 600s
    killed the 12.7-min master). Floor 600s, ceiling 1 hour."""
    try:
        d = float(duration_seconds or 0)
    except (TypeError, ValueError):
        d = 0.0
    return max(600, min(3600, int(d * 4) + 180))


def _run_ffmpeg(cmd: list[str], *, timeout: int = 600) -> None:
    """Run ffmpeg, raising FFmpegOverlayError on non-zero exit.

    Stdout is discarded; stderr is captured as bytes (then decoded
    once on error) to dodge a known Windows hang in
    ``subprocess.run(capture_output=True, text=True)`` where the
    decoder waits indefinitely on a tiny trailing buffer when the
    child has already exited.
    """
    cmd_str = " ".join(repr(a) if " " in a else a for a in cmd)
    _set_snap("last_ffmpeg_command", cmd_str)
    log.info("branding: ffmpeg cmd: %s", cmd_str)
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegOverlayError(
            f"ffmpeg timed out after {timeout}s",
            returncode=-1,
            stderr_tail=str(exc)[-500:],
        ) from exc
    dt = time.monotonic() - t0
    _set_snap("last_ffmpeg_duration_seconds", round(dt, 3))
    _bump("ffmpeg_invocations")
    if r.returncode != 0:
        try:
            tail = (r.stderr or b"").decode("utf-8", errors="replace")[-500:]
        except Exception:
            tail = repr((r.stderr or b"")[-500:])
        raise FFmpegOverlayError(
            f"ffmpeg failed (rc={r.returncode})",
            returncode=r.returncode,
            stderr_tail=tail,
        )


def _prepend_intro(
    *, intro_local: str, body_local: str, out_path: str,
    canvas_w: int, canvas_h: int,
) -> None:
    """Prepend a per-channel INTRO clip to the branded body via concat.

    Two-step + robust:
      1. Normalise the intro to the body's canvas — scale (preserve aspect)
         + pad to exact WxH, setsar=1, fps=30, yuv420p, and a GUARANTEED
         stereo 48k audio track (the intro's own audio if it has one, else
         silent ``anullsrc`` so the concat filter always has both streams).
      2. Concat [intro][body], normalising the body in-graph too so both
         segments share dims/sar/fps/pixfmt + audio params (the concat
         filter requires it), then re-encode (concat can't ``-c:a copy``).

    Raises FFmpegOverlayError on ffmpeg failure — the caller wraps this so a
    failed intro degrades to "no intro" rather than a failed branded artifact.
    """
    work = os.path.dirname(out_path)
    intro_norm = os.path.join(work, "_intro_norm.mp4")
    W, H, FPS = int(canvas_w), int(canvas_h), "30"
    _vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p"
    )
    if _source_has_audio(intro_local):
        cmd_norm = [
            _ffmpeg_bin(), "-y", "-v", "error",
            "-i", intro_local,
            "-vf", _vf, "-af", "aresample=48000", "-ar", "48000", "-ac", "2",
            *_video_encode_args(), "-c:a", "aac", "-b:a", "192k",
            intro_norm,
        ]
    else:
        cmd_norm = [
            _ffmpeg_bin(), "-y", "-v", "error",
            "-i", intro_local,
            "-f", "lavfi", "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", _vf, "-map", "0:v", "-map", "1:a", "-shortest",
            *_video_encode_args(), "-c:a", "aac", "-b:a", "192k",
            intro_norm,
        ]
    _run_ffmpeg(cmd_norm, timeout=600)

    _bv = (
        f"[1:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p[bv]"
    )
    fc = (
        f"{_bv};"
        "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[ia];"
        "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[ba];"
        "[0:v][ia][bv][ba]concat=n=2:v=1:a=1[v][a]"
    )
    cmd_cat = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-i", intro_norm, "-i", body_local,
        "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]",
        *_video_encode_args(), "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]
    _run_ffmpeg(cmd_cat, timeout=900)


# ─── DB helpers (terse — branding only needs to read MasterVideo + write a key) ─


def _commit_job_status(db: Session, job: models.UploadJobV2, status: str) -> None:
    job.status = status
    db.add(job)
    db.commit()


def _persist_cache_key(
    db: Session, job: models.UploadJobV2, cache_key: str,
) -> None:
    """Atomic: write branded_artifact_r2_key + flip status to
    ready_to_upload + commit."""
    job.branded_artifact_r2_key = cache_key
    job.status = "ready_to_upload"
    db.add(job)
    db.commit()


# ─── Main entry point ──────────────────────────────────────────────────────


def brand_local_for_preview(source_path: str, out_path: str, resolved, *,
                            work_dir: Optional[str] = None) -> str:
    """Fully brand a LOCAL rendered file for ONE channel — the SAME transforms
    the publish pass applies (logo + watermark + per-channel zoom + audio
    nudge + intro concat) — but on a local file with NO R2 / DB-job / cache-key
    involvement. Powers the V4 editor's per-channel VIDEO preview so the
    operator sees exactly what will publish.

    READ-ONLY w.r.t. the publish pipeline: this does NOT touch
    ``process_upload_job``, ``UploadJobV2``, ``MasterVideo`` or the R2 cache.
    It reuses the same low-level helpers so the preview matches publish output.

    ``resolved`` is a pre-computed ``ResolvedBrand`` (from
    ``resolve_brand_profile``). Returns ``out_path``; copies the source
    verbatim when the channel has no branding configured.
    """
    own_work = work_dir is None
    if own_work:
        work_dir = tempfile.mkdtemp(prefix="kaizer_preview_brand_")
    try:
        apply_logo = bool(getattr(resolved, "logo_local_path", None))
        apply_text = bool(getattr(resolved, "watermark_text", None))
        nudge_factor = float(getattr(resolved, "nudge_factor", 1.0) or 1.0)
        nudge_on = abs(nudge_factor - 1.0) > 1e-6
        zoom_factor = float(getattr(resolved, "zoom_factor", 1.0) or 1.0)
        zoom_on = abs(zoom_factor - 1.0) > 1e-6

        body_out = os.path.join(work_dir, "_preview_body.mp4")
        if not (apply_logo or apply_text or nudge_on or zoom_on):
            shutil.copy2(source_path, body_out)
        else:
            canvas_w, canvas_h, source_duration = _probe_source(source_path)
            has_audio = _source_has_audio(source_path) if nudge_on else True
            logo_path_for_cmd: Optional[str] = None
            if apply_logo and getattr(resolved, "logo_local_path", None):
                try:
                    logo_path_for_cmd = _prepare_logo_png(
                        resolved.logo_local_path, canvas_w, canvas_h, work_dir,
                    )
                except Exception:
                    apply_logo = False
                    logo_path_for_cmd = None
            if not (apply_logo or apply_text or nudge_on or zoom_on):
                shutil.copy2(source_path, body_out)
            else:
                _ov = dict(
                    source_path=source_path, out_path=body_out, work_dir=work_dir,
                    apply_logo=apply_logo, apply_text=apply_text,
                    logo_local_path=logo_path_for_cmd,
                    text=resolved.watermark_text,
                    opacity=resolved.watermark_opacity,
                    position=resolved.watermark_position,
                    canvas_w=canvas_w, canvas_h=canvas_h,
                    source_duration=source_duration,
                    nudge_factor=nudge_factor, has_audio=has_audio,
                    zoom_factor=zoom_factor,
                )
                _timeout = _encode_timeout(source_duration)
                _did_gpu = False
                if (os.environ.get("KAIZER_BRAND_GPU_OVERLAY", "0").strip() == "1"
                        and (apply_logo or apply_text) and _gpu_overlay_caps()):
                    try:
                        _run_ffmpeg(_build_gpu_ffmpeg_command(**_ov), timeout=_timeout)
                        _did_gpu = True
                    except FFmpegOverlayError:
                        pass
                if not _did_gpu:
                    _run_ffmpeg(_build_ffmpeg_command(**_ov), timeout=_timeout)
                # QC is advisory here — a preview should still render even if a
                # cosmetic QC check trips; the publish path enforces it strictly.
                try:
                    _qc_branded_artifact(source_path, body_out, expected_factor=nudge_factor)
                except Exception:
                    pass

        # Per-channel intro concat (same as publish).
        final_out = body_out
        intro_path = getattr(resolved, "intro_local_path", None)
        if intro_path and os.path.isfile(intro_path):
            try:
                _iw, _ih, _ = _probe_source(body_out)
                intro_out = os.path.join(work_dir, "_preview_intro.mp4")
                _prepend_intro(
                    intro_local=intro_path, body_local=body_out,
                    out_path=intro_out, canvas_w=_iw, canvas_h=_ih,
                )
                if os.path.isfile(intro_out) and os.path.getsize(intro_out) > 0:
                    final_out = intro_out
            except Exception:
                pass

        if not os.path.isfile(final_out) or os.path.getsize(final_out) == 0:
            raise FFmpegOverlayError(
                f"preview output missing/empty at {final_out!r}",
                returncode=-1, stderr_tail="",
            )
        if os.path.abspath(final_out) != os.path.abspath(out_path):
            shutil.move(final_out, out_path)
        return out_path
    finally:
        if own_work and work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


def process_upload_job(upload_job_id: int) -> str:
    """Produce (or cache-hit) the branded artifact for an UploadJobV2.

    Returns the R2 cache key. Idempotent: a re-run with the same row +
    same brand inputs is a cache hit and returns the same key without
    invoking ffmpeg.

    State machine touched on the UploadJobV2 row:
        queued → branding → ready_to_upload

    Raises (each maps to a stable ``code`` field for last_error):
        MasterVideoMissingError   - the row is gone
        MasterVideoNotReadyError  - status != 'ready'
        BrandResolveError         - brand_resolver raised
        FFmpegOverlayError        - ffmpeg returned non-zero
        StorageWriteError         - R2 upload/download failed
    """
    _bump("in_flight")
    db: Session = SessionLocal()
    work_dir: Optional[str] = None
    try:
        job = (
            db.query(models.UploadJobV2)
            .filter(models.UploadJobV2.id == int(upload_job_id))
            .first()
        )
        if job is None:
            raise BrandingError(
                f"upload_job_id={upload_job_id} not found",
            )
        master = (
            db.query(models.MasterVideo)
            .filter(models.MasterVideo.id == int(_resolve_master_video_id(db, job)))
            .first()
        )
        if master is None:
            raise MasterVideoMissingError(
                f"MasterVideo for upload_job_id={upload_job_id} not found",
            )
        if (master.status or "").strip() != "ready":
            raise MasterVideoNotReadyError(
                f"MasterVideo id={master.id} status={master.status!r} (need 'ready')",
            )

        # Flip to 'branding' so observers know we're working on it.
        _commit_job_status(db, job, "branding")

        # ── 1. Resolve the brand ───────────────────────────────────────
        # Per-CHANNEL intro override: if the source Job set an intro for THIS
        # publishing channel (inline per-channel picker at creation), it wins
        # over the channel's own assigned intro. Absent = unchanged behaviour.
        _job_intro = None
        try:
            _src_job = (
                db.query(models.Job)
                .filter(models.Job.id == int(master.source_upload_id))
                .first()
            )
            _ov_raw = getattr(_src_job, "intro_overrides", None) if _src_job else None
            if _ov_raw:
                import json as _json
                _ov = _json.loads(_ov_raw) or {}
                _job_intro = _ov.get(str(job.channel_id))
        except Exception:
            _job_intro = None
        try:
            resolved = resolve_brand_profile(
                db, int(job.channel_id), job_intro_asset_id=_job_intro,
            )
        except Exception as exc:
            raise BrandResolveError(
                f"brand_resolver failed for channel_id={job.channel_id}: {exc}",
            ) from exc

        # ── 2. Cache key + optimistic hit check (fast path, no lock) ───
        cache_key = brand_artifact_cache_key(master.id, resolved.version)
        if _cache_hit(cache_key):
            log.info(
                "branding: CACHE HIT upload_job_id=%d master_video_id=%d "
                "key=%r",
                upload_job_id, master.id, cache_key,
            )
            _persist_cache_key(db, job, cache_key)
            _bump("hits")
            return cache_key

        # ── 2b. Cache miss — serialize concurrent miss-path runners ────
        # Without this guard, N concurrent jobs sharing the same
        # brand_profile_version all observe cache_miss=True and run
        # ffmpeg redundantly (the bug Phase 2 exit test caught). We hold
        # the advisory lock across the ENTIRE miss-path work — download,
        # ffmpeg, R2 upload, _persist_cache_key (commit). The pg lock
        # auto-releases when the transaction commits (success) or rolls
        # back. On error escape, ``db.close()`` in the outer finally
        # rolls back any in-flight tx, which still releases the lock.
        with _advisory_lock(db, cache_key):
            # Double-checked locking: another worker may have just
            # finished while we waited on the mutex. Re-probe under the
            # lock so the loser of the race becomes a HIT not a MISS.
            if _cache_hit(cache_key):
                log.info(
                    "branding: CACHE HIT (post-lock) upload_job_id=%d "
                    "master_video_id=%d key=%r",
                    upload_job_id, master.id, cache_key,
                )
                _persist_cache_key(db, job, cache_key)
                _bump("hits")
                return cache_key

            log.info(
                "branding: CACHE MISS upload_job_id=%d master_video_id=%d "
                "key=%r — running overlay pass (mutex held)",
                upload_job_id, master.id, cache_key,
            )
            _bump("misses")

            # ── 3. Decide what to overlay (Decision 12 transitional) ──
            # Legacy-branded master means the render already baked the
            # logo; we must NOT add a second logo bug, but the text
            # watermark still wants applying because the legacy render
            # didn't do it.
            apply_logo = bool(resolved.logo_local_path)
            if not master.clean_master:
                if apply_logo:
                    log.warning(
                        "branding: legacy branded master detected "
                        "(MasterVideo id=%d, clean_master=False) — "
                        "skipping logo overlay (text watermark still applied)",
                        master.id,
                    )
                apply_logo = False
            apply_text = bool(resolved.watermark_text)

            work_dir = tempfile.mkdtemp(prefix="kaizer_branding_")
            source_local = os.path.join(work_dir, "_source.mp4")

            # ── 4. Download the clean master locally ─────────────────
            try:
                _storage_provider().download(master.r2_key, source_local)
            except Exception as exc:
                raise StorageWriteError(
                    f"download failed for master_video.r2_key={master.r2_key!r}: {exc}",
                ) from exc
            if not os.path.isfile(source_local) or os.path.getsize(source_local) == 0:
                raise StorageWriteError(
                    f"downloaded master is missing/empty at {source_local!r}",
                )

            # ── Template branding slots (custom templates) ────────────
            # A custom template marks WHERE its logo/watermark go; that rect is
            # recorded in a <master>.slots.json sidecar at render. Pull it next
            # to the master so we drop each channel's logo INTO the designed
            # spot (not a default corner). A marked slot also proves the master
            # is clean there, so re-enable the overlay even if the transitional
            # clean_master gate had turned it off.
            import json as _json
            _sidecar = source_local + ".slots.json"
            try:
                _storage_provider().download(master.r2_key + ".slots.json", _sidecar)
            except Exception:
                pass
            _has_logo_slot = _has_wm_slot = False
            try:
                with open(_sidecar, encoding="utf-8") as _sf:
                    _sd = _json.load(_sf)
                _has_logo_slot = bool(_sd.get("logo"))
                _has_wm_slot = bool(_sd.get("watermark"))
            except Exception:
                pass
            if _has_logo_slot and resolved.logo_local_path:
                apply_logo = True
            if _has_wm_slot and resolved.watermark_text:
                apply_text = True

            out_local = os.path.join(work_dir, f"_branded_{master.id}.mp4")

            # Per-channel A/V pacing nudge (anti-duplicate). >1.0 ⇒ the
            # branding pass MUST re-encode (setpts+atempo) even with no
            # logo/text overlay, so it counts toward "needs ffmpeg".
            nudge_factor = float(getattr(resolved, "nudge_factor", 1.0) or 1.0)
            nudge_on = abs(nudge_factor - 1.0) > 1e-6
            # Per-channel visual zoom (picture-half anti-dup). Like the nudge,
            # it forces a re-encode even with no logo/text overlay.
            zoom_factor = float(getattr(resolved, "zoom_factor", 1.0) or 1.0)
            zoom_on = abs(zoom_factor - 1.0) > 1e-6
            # Per-channel anti-dup forensics: log what makes THIS channel's render
            # distinct, so a multi-channel publish is verifiable after the fact.
            # Extra invisible per-channel anti-dup: framing PAN + colour micro-GRADE
            # (on top of nudge+zoom), derived from the channel id like the others.
            try:
                from services import brand_resolver as _br
                _cid_ad = getattr(job, "channel_id", 0) or 0
                _pan_x, _pan_y = _br._derive_pan_offset(_cid_ad)
                _grade_b, _grade_s = _br._derive_grade(_cid_ad)
            except Exception:
                _pan_x, _pan_y, _grade_b, _grade_s = 0.5, 0.5, 0.0, 1.0
            log.info(
                "branding: anti-dup channel_id=%s nudge=%.4f zoom=%.4f pan=%.2f,%.2f grade=%.3f,%.3f logo=%s intro=%s watermark=%s",
                getattr(job, "channel_id", "?"), nudge_factor, zoom_factor, _pan_x, _pan_y, _grade_b, _grade_s,
                bool(resolved.logo_local_path), bool(getattr(resolved, "intro_local_path", None)),
                bool(resolved.watermark_text),
            )

            if not (apply_logo or apply_text or nudge_on or zoom_on):
                # No overlays AND no nudge/zoom → cache the clean source bytes
                # verbatim. Documented short-circuit (no ffmpeg invocation).
                log.info(
                    "branding: NO overlays/nudge/zoom needed — uploading "
                    "source verbatim as cache entry",
                )
                shutil.copy2(source_local, out_local)
            else:
                canvas_w, canvas_h, source_duration = _probe_source(source_local)
                # Scaled slot rects (output-canvas px) for placement; only when
                # the template marked a slot, else default corner/named position.
                _logo_rect = _wm_rect = None
                if _has_logo_slot or _has_wm_slot:
                    try:
                        from pipeline_v4.watermark import _read_slot_rects
                        _sr = _read_slot_rects(source_local, canvas_w, canvas_h)
                        _logo_rect = _sr.get("logo")
                        _wm_rect = _sr.get("watermark")
                    except Exception:
                        _logo_rect = _wm_rect = None
                if (getattr(job, "brand_placement", "template") or "template") == "channel":
                    # User chose their channel position over the template's slot:
                    # keep the logo/watermark (slot proved the master is clean) but
                    # drop the rects so it lands at the channel position / corner.
                    _logo_rect = _wm_rect = None
                # Audio presence only matters when nudging (decides atempo vs
                # copy); skip the extra probe otherwise.
                has_audio = _source_has_audio(source_local) if nudge_on else True
                # If applying the logo, pre-resize it to a bug PNG so
                # ffmpeg can overlay without scale logic in the filter graph.
                logo_path_for_cmd: Optional[str] = None
                if apply_logo and resolved.logo_local_path:
                    try:
                        logo_path_for_cmd = _prepare_logo_png(
                            resolved.logo_local_path, canvas_w, canvas_h, work_dir,
                            rect=_logo_rect,
                        )
                    except Exception as exc:
                        log.warning(
                            "branding: logo prep failed (%s) — falling "
                            "back to text-only overlay",
                            exc,
                        )
                        apply_logo = False
                        logo_path_for_cmd = None

                if not (apply_logo or apply_text or nudge_on or zoom_on):
                    # Logo prep failed AND no text AND no nudge/zoom — copy source.
                    shutil.copy2(source_local, out_local)
                else:
                    _ov_kwargs = dict(
                        source_path=source_local,
                        out_path=out_local,
                        work_dir=work_dir,
                        apply_logo=apply_logo,
                        apply_text=apply_text,
                        logo_local_path=logo_path_for_cmd,
                        text=resolved.watermark_text,
                        opacity=resolved.watermark_opacity,
                        position=resolved.watermark_position,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                        source_duration=source_duration,
                        nudge_factor=nudge_factor,
                        has_audio=has_audio,
                        zoom_factor=zoom_factor,
                        logo_rect=_logo_rect,
                        wm_rect=_wm_rect,
                        pan_x=_pan_x,
                        pan_y=_pan_y,
                        grade_b=_grade_b,
                        grade_s=_grade_s,
                    )
                    _timeout = _encode_timeout(source_duration)
                    if nudge_on:
                        log.info(
                            "branding: A/V nudge ACTIVE factor=%.4f "
                            "(has_audio=%s, overlays: logo=%s text=%s)",
                            nudge_factor, has_audio, apply_logo, apply_text,
                        )
                    # Superfast GPU overlay (full-CUDA) when enabled + supported.
                    # Only worthwhile when there's an OVERLAY to keep on-GPU; a
                    # pure-nudge re-encode (no overlays) goes through the CPU
                    # builder, which still NVENC-encodes via h264_args.
                    # ANY GPU failure falls back to the proven CPU command, so
                    # this can never break branding — worst case = current behavior.
                    _did_gpu = False
                    if (os.environ.get("KAIZER_BRAND_GPU_OVERLAY", "0").strip() == "1"
                            and (apply_logo or apply_text)
                            and not (_logo_rect or _wm_rect)   # slot placement uses the proven CPU path
                            and _gpu_overlay_caps()):
                        try:
                            _run_ffmpeg(_build_gpu_ffmpeg_command(**_ov_kwargs),
                                        timeout=_timeout)
                            _did_gpu = True
                        except FFmpegOverlayError as exc:
                            log.warning(
                                "branding: GPU overlay failed (%s) — falling "
                                "back to CPU overlay", exc,
                            )
                    if not _did_gpu:
                        _run_ffmpeg(_build_ffmpeg_command(**_ov_kwargs),
                                    timeout=_timeout)
                    # Wave 4 QC gate: verify the overlay output BEFORE
                    # the R2 upload. Raises FFmpegOverlayError on
                    # violation → existing retry/cleanup path applies.
                    # ``expected_factor`` tells QC the output is legitimately
                    # ~source/F long so the nudge isn't read as a truncation.
                    _qc_branded_artifact(
                        source_local, out_local, expected_factor=nudge_factor,
                    )

            # ── 4b. Per-channel INTRO concat (anti-duplicate) ───────────
            # If this channel registered an intro video, prepend it to the
            # branded body. ADDITIVE + INERT: skipped entirely unless the
            # resolver materialised an intro (no channel has one until the
            # registration UI ships). Wrapped so an intro hiccup degrades to
            # "no intro" rather than failing the whole branded artifact. The
            # intro is already folded into resolved.version, so an intro'd
            # channel gets its own cache key.
            _intro_path = getattr(resolved, "intro_local_path", None)
            if _intro_path and os.path.isfile(_intro_path):
                try:
                    _iw, _ih, _ = _probe_source(out_local)
                    _intro_out = os.path.join(
                        work_dir, f"_intro_branded_{master.id}.mp4"
                    )
                    _prepend_intro(
                        intro_local=_intro_path,
                        body_local=out_local,
                        out_path=_intro_out,
                        canvas_w=_iw, canvas_h=_ih,
                    )
                    if os.path.isfile(_intro_out) and os.path.getsize(_intro_out) > 0:
                        out_local = _intro_out
                        log.info(
                            "branding: intro prepended (intro_asset_id=%s)",
                            getattr(resolved, "intro_asset_id", None),
                        )
                    else:
                        log.warning(
                            "branding: intro concat produced empty output — "
                            "shipping branded body WITHOUT intro",
                        )
                except Exception as exc:
                    log.warning(
                        "branding: intro prepend failed (%s) — shipping "
                        "branded body WITHOUT intro", exc,
                    )

            if not os.path.isfile(out_local) or os.path.getsize(out_local) == 0:
                raise FFmpegOverlayError(
                    f"output missing/empty at {out_local!r}",
                    returncode=-1,
                    stderr_tail="",
                )

            # ── 5. Upload to R2 at the cache key ─────────────────────
            try:
                _storage_provider().upload(
                    out_local, cache_key, content_type="video/mp4",
                )
            except Exception as exc:
                raise StorageWriteError(
                    f"R2 upload failed for key={cache_key!r}: {exc}",
                ) from exc

            # ── 6. Persist + flip status (this commits → releases lock) ─
            _persist_cache_key(db, job, cache_key)
            log.info(
                "branding: SUCCESS upload_job_id=%d → cache_key=%r "
                "(bytes=%d, ffmpeg=%s)",
                upload_job_id, cache_key, os.path.getsize(out_local),
                "yes" if (apply_logo or apply_text) else "skipped",
            )
            return cache_key

    finally:
        _bump("in_flight", -1)
        if work_dir is not None:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
        try:
            db.close()
        except Exception:
            pass


def _resolve_master_video_id(db: Session, job: models.UploadJobV2) -> int:
    """Find the MasterVideo id for a given UploadJobV2 via its PublishTask.

    The UploadJobV2 schema (CONTRACTS §3.3) doesn't carry a direct
    master_video_id — it goes UploadJobV2 → PublishTask → MasterVideo.
    """
    pt = (
        db.query(models.PublishTask)
        .filter(models.PublishTask.id == int(job.publish_task_id))
        .first()
    )
    if pt is None:
        raise BrandingError(
            f"PublishTask id={job.publish_task_id} not found for upload_job_id={job.id}",
        )
    if not pt.master_video_id:
        raise BrandingError(
            f"PublishTask id={pt.id} has no master_video_id",
        )
    return int(pt.master_video_id)


# ─── Cleanup (TTL sweep) ───────────────────────────────────────────────────
#
# We use a fresh boto3 client constructed from the same env vars rather
# than touching pipeline_core/storage.py (read-only contract per task
# spec). The upload/download path STILL goes through the provider so we
# don't duplicate that logic — only the list+delete missing pieces.


def _boto3_r2_client():
    """Construct a fresh S3 client for the R2 endpoint. Returns
    ``(client, bucket, key_prefix)`` or ``(None, '', '')`` when not
    configured."""
    bucket = os.environ.get("R2_BUCKET", "")
    endpoint = os.environ.get("R2_ENDPOINT", "")
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    raw_prefix = os.environ.get("R2_KEY_PREFIX", "") or ""
    # Mirror R2Storage._normalize_prefix in pipeline_core/storage.py.
    prefix = raw_prefix.strip().replace("\\", "/").lstrip("/")
    while "//" in prefix:
        prefix = prefix.replace("//", "/")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    if not (bucket and endpoint and access_key and secret_key):
        log.info("branding.cleanup_expired: R2 not configured; nothing to do")
        return None, "", ""
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        log.error("branding.cleanup_expired: boto3 not installed: %s", exc)
        return None, "", ""
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    return client, bucket, prefix


def cleanup_expired() -> int:
    """Delete all ``branded/...`` R2 objects older than
    ``KAIZER_BRANDED_ARTIFACT_TTL_HOURS`` (default 24).

    Returns the count of objects removed. Logs a warning and returns 0
    when R2 is not configured (dev with STORAGE_BACKEND=local).

    Intended to be wired to an hourly cron by the F-agent (CONTRACTS §4
    cleanup_expired surface; cron infrastructure is theirs to wire).
    """
    ttl_hours = _ttl_hours()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
    log.info(
        "branding.cleanup_expired: ttl_hours=%d cutoff=%s",
        ttl_hours, cutoff.isoformat(),
    )

    # When STORAGE_BACKEND != 'r2', the production cleanup is a no-op
    # (local dev uses LocalStorage which never expires). We do not
    # delete the local files because they survive as long as the dev
    # process needs them.
    storage_backend = (os.environ.get("STORAGE_BACKEND", "local") or "local").lower()
    if storage_backend != "r2":
        log.info(
            "branding.cleanup_expired: STORAGE_BACKEND=%r (not 'r2'); skipping",
            storage_backend,
        )
        _set_snap("last_cleanup_at", datetime.now(timezone.utc).isoformat())
        _set_snap("last_cleanup_removed_count", 0)
        return 0

    client, bucket, prefix = _boto3_r2_client()
    if client is None:
        _set_snap("last_cleanup_at", datetime.now(timezone.utc).isoformat())
        _set_snap("last_cleanup_removed_count", 0)
        return 0

    # ``branded/`` prefix per Decision 11, applied AFTER the env key prefix.
    full_prefix = f"{prefix}branded/" if prefix else "branded/"
    removed = 0
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []) or []:
                last_modified = obj.get("LastModified")
                key = obj.get("Key", "")
                if last_modified is None or not key:
                    continue
                # last_modified is timezone-aware UTC from boto3.
                if last_modified < cutoff:
                    try:
                        client.delete_object(Bucket=bucket, Key=key)
                        removed += 1
                        log.info(
                            "branding.cleanup_expired: deleted %r (age=%s)",
                            key, datetime.now(timezone.utc) - last_modified,
                        )
                    except Exception as exc:
                        log.warning(
                            "branding.cleanup_expired: delete failed for %r: %s",
                            key, exc,
                        )
    except Exception as exc:
        log.exception(
            "branding.cleanup_expired: list/delete sweep failed: %s", exc,
        )

    _set_snap("last_cleanup_at", datetime.now(timezone.utc).isoformat())
    _set_snap("last_cleanup_removed_count", removed)
    log.info("branding.cleanup_expired: removed=%d", removed)
    return removed


__all__ = [
    "brand_artifact_cache_key",
    "process_upload_job",
    "cleanup_expired",
    "snapshot",
    "BrandingError",
    "MasterVideoMissingError",
    "MasterVideoNotReadyError",
    "BrandResolveError",
    "FFmpegOverlayError",
    "StorageWriteError",
]
