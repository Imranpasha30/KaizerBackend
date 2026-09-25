"""Responsive multi-video auto-grid (Phase-1 engine capability #4).

Variable speaker/source count → deterministic layout that never breaks
(spec 3.11): 1 → full-frame, 2 → side-by-side split, 3 → two on top +
one centered below, 4 → 2×2, ≥5 → pages of up to 4 that each hold the
screen for an equal share of the duration.

Two layers, deliberately separate:

  * ``plan_grid``  — PURE math. Returns ``GridPage`` rows (cell rects +
    page time windows). No ffmpeg, no I/O; unit-testable exhaustively.
  * ``compose_grid`` — turns a plan + source videos into one rendered
    clip via the same scale/crop/pad + enable-overlay idiom
    ``v1_bridge._compose_v4_bulletin_story`` uses (no ``xstack`` — cells
    need borders and page enable windows).

Cells are laid inside a SAFE RECT (defaults to the bulletin's tile area
above the lower-third strip) so grid video can never cover the
headline/ticker text — layout safety (Unit 10) supplies the rect.

Consumers: Phase-2 scenario templates (multi-speaker). Until then the
engine ships dark — nothing on the render hot path calls it. For manual
DEV experiments set ``KAIZER_V4_EXTRA_VIDEOS`` (pipe-joined paths) and
drive ``compose_grid`` from a script/test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default safe rect mirrors the V4 bulletin tile area: x/y margins and
# the 800px-tall band above the lower-third strip (y 890) minus padding.
DEFAULT_SAFE_RECT = (30, 50, 1860, 800)   # (x, y, w, h) on a 1920x1080 canvas


@dataclass
class GridPage:
    """One screenful of the grid: which sources show where, and when."""
    cells: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h) OUTER incl. border
    sources: list[int] = field(default_factory=list)                      # index into the sources list
    t_start: float = 0.0
    t_end: float = 0.0


def _cells_for_count(n: int, rect: tuple[int, int, int, int],
                     gap: int) -> list[tuple[int, int, int, int]]:
    """Cell rects for one page of 1..4 sources inside ``rect``.
    Deterministic; every rect stays inside ``rect`` with ``gap`` spacing."""
    x, y, w, h = rect
    if n <= 1:
        return [(x, y, w, h)]
    cw2 = (w - gap) // 2
    ch2 = (h - gap) // 2
    if n == 2:
        return [(x, y, cw2, h), (x + cw2 + gap, y, cw2, h)]
    if n == 3:
        return [
            (x, y, cw2, ch2), (x + cw2 + gap, y, cw2, ch2),
            (x + (w - cw2) // 2, y + ch2 + gap, cw2, ch2),   # centered bottom
        ]
    return [
        (x, y, cw2, ch2), (x + cw2 + gap, y, cw2, ch2),
        (x, y + ch2 + gap, cw2, ch2), (x + cw2 + gap, y + ch2 + gap, cw2, ch2),
    ]


def plan_grid(
    n_sources: int,
    *,
    duration: float,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
    safe_rect: tuple[int, int, int, int] = None,
    gap_px: int = 30,
    per_page: int = 4,
) -> list[GridPage]:
    """Pure grid plan: pages of up to ``per_page`` sources, each page
    holding the screen for an equal share of ``duration``. Returns []
    when there is nothing to lay out."""
    if n_sources <= 0 or duration <= 0:
        return []
    rect = tuple(safe_rect) if safe_rect else (
        DEFAULT_SAFE_RECT if (canvas_w, canvas_h) == (1920, 1080)
        else (int(canvas_w * 0.015), int(canvas_h * 0.046),
              int(canvas_w * 0.97), int(canvas_h * 0.74))
    )
    per_page = max(1, min(int(per_page), 4))
    source_ids = list(range(n_sources))
    chunks = [source_ids[i:i + per_page] for i in range(0, n_sources, per_page)]
    page_dur = duration / len(chunks)
    pages: list[GridPage] = []
    for p, chunk in enumerate(chunks):
        pages.append(GridPage(
            cells=_cells_for_count(len(chunk), rect, gap_px),
            sources=chunk,
            t_start=round(p * page_dur, 3),
            t_end=round(duration if p == len(chunks) - 1 else (p + 1) * page_dur, 3),
        ))
    return pages


def extra_videos_from_env() -> list[str]:
    """DEV hook: pipe-joined extra source paths (default off/empty)."""
    raw = (os.environ.get("KAIZER_V4_EXTRA_VIDEOS") or "").strip()
    return [p for p in (s.strip() for s in raw.split("|")) if p and Path(p).is_file()]


def compose_grid(
    *,
    sources: list[str],
    pages: list[GridPage],
    out_path: str,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
    bg_color: str = "black",
    border_px: int = 3,
    border_color: str = "white",
    fps: int = 30,
    timeout: int = 900,
) -> str:
    """Render a planned grid into one clip.

    Every cell: scale-to-cover + crop + white border pad (the bulletin
    tile aesthetic), overlaid with ``enable='between(t,page)'`` so pages
    swap without re-encoding per page. Audio = amix of every source
    (a speaker grid should hear all speakers). Raises on failure."""
    from pipeline_v4.v1_bridge import _enc_args, _ffmpeg_bin
    from pipeline_v4.ffmpeg_exec import run_ffmpeg

    if not sources or not pages:
        raise ValueError("compose_grid needs at least one source and one page")
    duration = max(p.t_end for p in pages)

    cmd: list[str] = [_ffmpeg_bin(), "-y", "-v", "error"]
    for s in sources:
        cmd += ["-i", s]

    # Each (page, cell) is one use of a source stream; ffmpeg forbids
    # reusing an input label, so split each source into exactly as many
    # copies as it appears across pages.
    uses: dict[int, int] = {}
    for pg in pages:
        for src in pg.sources:
            uses[src] = uses.get(src, 0) + 1
    fc: list[str] = [f"color=c={bg_color}:s={canvas_w}x{canvas_h}:r={fps}[bg]"]
    counters: dict[int, int] = {}
    for src, n in sorted(uses.items()):
        if n == 1:
            continue   # single use → [i:v] consumed directly
        outs = "".join(f"[s{src}_{k}]" for k in range(n))
        fc.append(f"[{src}:v]split={n}{outs}")

    def _src_label(src: int) -> str:
        if uses[src] == 1:
            return f"[{src}:v]"
        k = counters.get(src, 0)
        counters[src] = k + 1
        return f"[s{src}_{k}]"

    cursor = "bg"
    for p, pg in enumerate(pages):
        for c, (src, cell) in enumerate(zip(pg.sources, pg.cells)):
            ox, oy, ow, oh = cell
            iw = max(1, ow - 2 * border_px)
            ih = max(1, oh - 2 * border_px)
            lbl = f"p{p}c{c}"
            fc.append(
                f"{_src_label(src)}scale={iw}:{ih}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={iw}:{ih},setsar=1,"
                f"pad={ow}:{oh}:{border_px}:{border_px}:color={border_color}[{lbl}]"
            )
            nxt = f"st_{p}_{c}"
            fc.append(
                f"[{cursor}][{lbl}]overlay=x={ox}:y={oy}:"
                f"enable='between(t,{pg.t_start:.3f},{pg.t_end:.3f})'[{nxt}]"
            )
            cursor = nxt

    # Audio: hear every speaker. amix when >1 source has audio;
    # single source passes through.
    if len(sources) > 1:
        amix_in = "".join(f"[{i}:a]" for i in range(len(sources)))
        fc.append(
            f"{amix_in}amix=inputs={len(sources)}:"
            f"duration=longest:dropout_transition=0[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = "0:a?"

    cmd += [
        "-filter_complex", ";".join(fc),
        "-map", f"[{cursor}]",
        "-map", audio_map,
        "-t", f"{duration:.3f}",
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
        "-r", str(fps), "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        out_path,
    ]
    run_ffmpeg(cmd, timeout=timeout, log_label="grid_compose")
    return out_path
