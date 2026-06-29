"""Composer: place real video(s) into a template's slot rects + overlay the design PNG.

Takes the transparent design PNG from renderer.py and a list of placements (each a slot
rectangle + the video that fills it) and builds one FFmpeg filtergraph:
  black canvas -> for each placement: scale clip to COVER the rect, crop to fit, overlay
  at (x,y) -> finally overlay the design PNG (alpha) on top so the design's graphics/text
  sit above the clips while its transparent holes reveal them.

Audio is taken from the primary clip. Output duration = primary clip duration.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class Placement:
    x: int
    y: int
    w: int
    h: int
    video: str               # path to the clip that fills this slot
    background: bool = False  # True => composited at the BOTTOM (behind other tiles)


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe() -> str | None:
    return shutil.which("ffprobe")


def probe_duration(path: str) -> float:
    fp = _ffprobe()
    if not fp:
        return 0.0
    try:
        out = subprocess.run(
            [fp, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, timeout=30, text=True,
        )
        return float(json.loads(out.stdout or "{}").get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def compose(design, placements: list[Placement], out_path: str,
            canvas: tuple[int, int], *, fps: int = 30, duration: float | None = None) -> str:
    """Render the final mp4. Returns out_path. Raises RuntimeError on ffmpeg failure.

    ``design`` is either a single PNG path (static overlay) OR a dict describing an
    animated frame sequence: ``{"frames_dir", "pattern", "fps"}`` (transparent PNGs)."""
    cw, ch = int(canvas[0]), int(canvas[1])
    if duration is None:
        duration = (probe_duration(placements[0].video) if placements else 0.0) or 6.0
    dur = max(0.5, float(duration))

    cmd: list[str] = [_ffmpeg(), "-y"]
    for pl in placements:
        # A BACKGROUND clip loops for the whole output (-stream_loop -1) so a short video
        # background fills the full duration instead of freezing on its last frame. The main /
        # foreground clips play once (they drive the duration); -t caps the total below.
        if pl.background:
            cmd += ["-stream_loop", "-1", "-i", pl.video]
        else:
            cmd += ["-i", pl.video]
    if isinstance(design, dict):                     # animated frame sequence
        dfps = int(design.get("fps") or fps)
        cmd += ["-framerate", str(dfps), "-i",
                os.path.join(design["frames_dir"], design["pattern"])]
    else:                                            # single still PNG, looped
        cmd += ["-loop", "1", "-i", design]
    design_idx = len(placements)

    fc: list[str] = []
    for i, pl in enumerate(placements):
        fc.append(
            f"[{i}:v]scale={pl.w}:{pl.h}:force_original_aspect_ratio=increase,"
            f"crop={pl.w}:{pl.h},setsar=1,fps={fps}[v{i}]"
        )
    fc.append(f"color=c=black:s={cw}x{ch}:r={fps}:d={dur:.3f}[bg]")
    # Overlay order: background placements first (bottom), then foreground tiles.
    order = sorted(range(len(placements)), key=lambda i: 0 if placements[i].background else 1)
    prev = "bg"
    for n, i in enumerate(order):
        pl = placements[i]
        nxt = f"t{n}"
        fc.append(f"[{prev}][v{i}]overlay={pl.x}:{pl.y}:eof_action=repeat[{nxt}]")
        prev = nxt
    fc.append(f"[{design_idx}:v]format=rgba[png]")
    fc.append(f"[{prev}][png]overlay=0:0:format=auto:eof_action=repeat[outv]")

    # Audio from the first NON-background clip (the main content), not the bg loop.
    audio_idx = next((i for i, pl in enumerate(placements) if not pl.background),
                     0 if placements else None)
    cmd += ["-filter_complex", ";".join(fc), "-map", "[outv]"]
    if audio_idx is not None:
        cmd += ["-map", f"{audio_idx}:a?"]           # audio from the main clip
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-t", f"{dur:.3f}",
        "-c:a", "aac", "-b:a", "160k",
        out_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, timeout=900)
    if proc.returncode != 0 or not os.path.isfile(out_path):
        err = (proc.stderr or b"").decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"ffmpeg composite failed: {err}")
    return out_path


def _has_audio(path: str) -> bool:
    fp = _ffprobe()
    if not fp:
        return False
    try:
        out = subprocess.run(
            [fp, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "csv=p=0", path],
            capture_output=True, timeout=30, text=True)
        return bool((out.stdout or "").strip())
    except Exception:
        return False


def _normalize_for_concat(src: str, dst: str, cw: int, ch: int, fps: int) -> None:
    """Re-encode ``src`` to canvas size + fps + AAC stereo (silent if it has no audio)
    so two clips can be concat-demuxed losslessly."""
    vf = (f"scale={cw}:{ch}:force_original_aspect_ratio=increase,"
          f"crop={cw}:{ch},setsar=1,fps={fps}")
    common = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
              "-r", str(fps), "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k"]
    if _has_audio(src):
        cmd = [_ffmpeg(), "-y", "-i", src, "-vf", vf, *common, dst]
    else:
        cmd = [_ffmpeg(), "-y", "-i", src,
               "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
               "-vf", vf, "-map", "0:v", "-map", "1:a", "-shortest", *common, dst]
    proc = subprocess.run(cmd, capture_output=True, timeout=900)
    if proc.returncode != 0 or not os.path.isfile(dst):
        err = (proc.stderr or b"").decode("utf-8", "replace")[-600:]
        raise RuntimeError(f"normalize failed: {err}")


def prepend_intro(intro_path: str, main_path: str, out_path: str,
                  canvas: tuple[int, int], *, fps: int = 30) -> str:
    """Play ``intro_path`` before ``main_path`` (cold-open). Both are normalized to the
    canvas + fps + stereo audio, then concat-demuxed. Returns out_path; on any failure
    falls back to the main video unchanged (intro is non-essential)."""
    cw, ch = int(canvas[0]), int(canvas[1])
    work = tempfile.mkdtemp(prefix="kx_intro_")
    try:
        a = os.path.join(work, "a.mp4")
        b = os.path.join(work, "b.mp4")
        _normalize_for_concat(intro_path, a, cw, ch, fps)
        _normalize_for_concat(main_path, b, cw, ch, fps)
        lst = os.path.join(work, "list.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write(f"file '{a}'\nfile '{b}'\n")
        cmd = [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", lst,
               "-c", "copy", "-movflags", "+faststart", out_path]
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
        if proc.returncode != 0 or not os.path.isfile(out_path):
            raise RuntimeError("concat failed")
        return out_path
    except Exception as exc:
        # Intro is optional — never fail the whole render over it.
        print(f"[custom_templates] intro prepend failed ({exc}); using main only", flush=True)
        if main_path != out_path:
            shutil.copyfile(main_path, out_path)
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
