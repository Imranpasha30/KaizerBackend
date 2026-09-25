"""Unit 9 — responsive multi-video auto-grid (pure planner + ffmpeg smoke)."""
from __future__ import annotations

import subprocess

import pytest

from pipeline_v4 import grid_engine as ge


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 9])
def test_plan_grid_properties(n):
    pages = ge.plan_grid(n, duration=60.0)
    assert pages, f"no pages for n={n}"
    # Every source appears exactly once across pages, in order.
    all_sources = [s for p in pages for s in p.sources]
    assert all_sources == list(range(n))
    # Page windows tile [0, 60] with no gaps or overlaps.
    assert pages[0].t_start == 0.0
    assert pages[-1].t_end == 60.0
    for a, b in zip(pages, pages[1:]):
        assert a.t_end == b.t_start
    sx, sy, sw, sh = ge.DEFAULT_SAFE_RECT
    for p in pages:
        assert len(p.cells) == len(p.sources) <= 4
        for (x, y, w, h) in p.cells:
            assert w > 0 and h > 0
            assert x >= sx and y >= sy
            assert x + w <= sx + sw and y + h <= sy + sh
        for i in range(len(p.cells)):
            for j in range(i + 1, len(p.cells)):
                assert not _overlaps(p.cells[i], p.cells[j]), \
                    f"n={n}: cells {i} and {j} overlap"


def test_plan_grid_shapes():
    one = ge.plan_grid(1, duration=10.0)[0]
    assert one.cells == [ge.DEFAULT_SAFE_RECT]
    two = ge.plan_grid(2, duration=10.0)[0]
    assert two.cells[0][1] == two.cells[1][1]          # side-by-side, same y
    assert two.cells[0][3] == ge.DEFAULT_SAFE_RECT[3]  # full height
    three = ge.plan_grid(3, duration=10.0)[0]
    assert three.cells[2][1] > three.cells[0][1]       # third row below
    five = ge.plan_grid(5, duration=10.0)
    assert len(five) == 2 and len(five[0].cells) == 4 and len(five[1].cells) == 1


def test_plan_grid_deterministic_and_custom_rect():
    a = ge.plan_grid(4, duration=20.0)
    b = ge.plan_grid(4, duration=20.0)
    assert a == b
    rect = (100, 100, 800, 600)
    pages = ge.plan_grid(2, duration=8.0, safe_rect=rect)
    for (x, y, w, h) in pages[0].cells:
        assert x >= 100 and y >= 100 and x + w <= 900 and y + h <= 700


def test_plan_grid_degenerate():
    assert ge.plan_grid(0, duration=10.0) == []
    assert ge.plan_grid(3, duration=0.0) == []


def test_extra_videos_env(monkeypatch, tmp_path):
    monkeypatch.delenv("KAIZER_V4_EXTRA_VIDEOS", raising=False)
    assert ge.extra_videos_from_env() == []
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    monkeypatch.setenv("KAIZER_V4_EXTRA_VIDEOS", f"{f}|Z:/missing.mp4|")
    assert ge.extra_videos_from_env() == [str(f)]


@pytest.mark.slow
def test_compose_grid_two_sources(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")

    def make(color, path):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=30:d=6",
             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
             "-t", "6", "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "aac", "-shortest", str(path)],
            check=True, capture_output=True)

    red = tmp_path / "red.mp4"
    green = tmp_path / "green.mp4"
    make("red", red)
    make("green", green)

    pages = ge.plan_grid(2, duration=6.0)
    out = tmp_path / "grid.mp4"
    ge.compose_grid(sources=[str(red), str(green)], pages=pages,
                    out_path=str(out))
    assert out.is_file() and out.stat().st_size > 0

    frame = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "3.0",
                    "-i", str(out), "-frames:v", "1", str(frame)],
                   check=True, capture_output=True)
    from PIL import Image
    with Image.open(frame) as im:
        im = im.convert("RGB")
        (lx, ly, lw, lh), (rx, ry, rw, rh) = pages[0].cells
        left = im.crop((lx + lw // 2 - 8, ly + lh // 2 - 8,
                        lx + lw // 2 + 8, ly + lh // 2 + 8))
        right = im.crop((rx + rw // 2 - 8, ry + rh // 2 - 8,
                         rx + rw // 2 + 8, ry + rh // 2 + 8))
        l = [sum(p[i] for p in left.getdata()) / 256 for i in range(3)]
        r = [sum(p[i] for p in right.getdata()) / 256 for i in range(3)]
    assert l[0] > 150 and l[1] < 90       # left cell red
    assert r[1] > 120 and r[0] < 90       # right cell green
