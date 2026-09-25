"""Broadcast loudness conform — every final output lands at -14 LUFS.

Why this exists: the compose/stitch chain encodes audio as plain AAC with
no normalization, and the one-pass ``loudnorm`` in the V1 short-form args
is dynamic-mode — on short clips it routinely misses the target by 2-3 LU
(the integration smoke measured -11 LUFS against a -14 target). News
channels and YouTube both normalize to -14 LUFS; being 3 LU hot means the
platform turns the video down and it still *feels* inconsistent next to
other content.

The fix is the standard two-pass conform, run ONCE on each FINAL output
(bulletin.mp4 / short_NN.mp4) after all compositing:

  pass 1: measure  (loudnorm print_format=json over the real audio)
  pass 2: apply    (loudnorm with measured_* + linear=true — a pure gain
                    ramp, no dynamic pumping) — AUDIO ONLY, video stream
                    copied, so it costs ~seconds and never touches NVENC.

Fail-soft everywhere: any error leaves the original file untouched — a
paid job never fails because of loudness. Skips when the file is already
within tolerance (re-renders of conformed files are no-ops).

Env (DEV .env):
  KAIZER_V4_LOUDNORM      1|0   master switch (default 1)
  KAIZER_V4_LOUDNORM_I    target integrated LUFS (default -14)
  KAIZER_V4_LOUDNORM_TP   target true peak dBTP  (default -1.5)
  KAIZER_V4_LOUDNORM_LRA  target loudness range  (default 11)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _enabled() -> bool:
    return (os.environ.get("KAIZER_V4_LOUDNORM", "1") or "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _target(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _ffmpeg_bin() -> str:
    return os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg")


def measure_loudness(path: str, *, i: float = -14.0, tp: float = -1.5,
                     lra: float = 11.0, timeout: int = 300) -> Optional[dict]:
    """Pass 1: run loudnorm in analysis mode and return its JSON block
    (input_i / input_tp / input_lra / input_thresh / target_offset as
    floats). None on any failure (no audio stream, unreadable file…)."""
    try:
        proc = subprocess.run(
            [_ffmpeg_bin(), "-hide_banner", "-nostats", "-i", path,
             "-map", "0:a:0",
             "-af", f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as exc:
        print(f"[v4/loudnorm] measure failed for {path}: {exc}", flush=True)
        return None
    # The JSON block is the last {...} on stderr.
    m = None
    for m in re.finditer(r"\{[^{}]+\}", proc.stderr or ""):
        pass
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
        return {k: float(raw[k]) for k in
                ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def conform_loudness(path: str, *, tolerance_lu: float = 0.75,
                     timeout: int = 900) -> bool:
    """Two-pass loudness conform of ``path`` IN PLACE (video stream
    copied, audio re-encoded with a linear loudnorm). Returns True when
    the file was rewritten, False when skipped (disabled / already in
    tolerance / no audio / any error). Never raises."""
    try:
        if not _enabled():
            return False
        p = Path(path)
        if not p.is_file():
            return False
        i = _target("KAIZER_V4_LOUDNORM_I", -14.0)
        tp = _target("KAIZER_V4_LOUDNORM_TP", -1.5)
        lra = _target("KAIZER_V4_LOUDNORM_LRA", 11.0)

        measured = measure_loudness(str(p), i=i, tp=tp, lra=lra)
        if measured is None:
            print(f"[v4/loudnorm] no measurable audio in {p.name} — skipping",
                  flush=True)
            return False
        if (abs(measured["input_i"] - i) <= tolerance_lu
                and measured["input_tp"] <= tp + 0.1):
            print(f"[v4/loudnorm] {p.name} already at "
                  f"{measured['input_i']:.1f} LUFS (target {i:.0f}) — skipping",
                  flush=True)
            return False

        af = (f"loudnorm=I={i}:TP={tp}:LRA={lra}"
              f":measured_I={measured['input_i']}"
              f":measured_TP={measured['input_tp']}"
              f":measured_LRA={measured['input_lra']}"
              f":measured_thresh={measured['input_thresh']}"
              f":offset={measured['target_offset']}"
              f":linear=true")
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=p.suffix, dir=str(p.parent))
        os.close(tmp_fd)
        try:
            proc = subprocess.run(
                [_ffmpeg_bin(), "-y", "-v", "error", "-i", str(p),
                 "-map", "0:v?", "-map", "0:a:0",
                 "-c:v", "copy",
                 "-af", af,
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-movflags", "+faststart",
                 tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
            if proc.returncode != 0 or not (os.path.isfile(tmp_path)
                                            and os.path.getsize(tmp_path) > 0):
                print(f"[v4/loudnorm] apply failed for {p.name}: "
                      f"{(proc.stderr or '')[-300:]}", flush=True)
                return False
            os.replace(tmp_path, str(p))
        finally:
            try:
                if os.path.isfile(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
        print(f"[v4/loudnorm] {p.name}: {measured['input_i']:.1f} → {i:.0f} LUFS "
              f"(TP {measured['input_tp']:.1f} → ≤{tp})", flush=True)
        return True
    except Exception as exc:
        print(f"[v4/loudnorm] conform failed for {path} (soft-skip): {exc}",
              flush=True)
        return False
