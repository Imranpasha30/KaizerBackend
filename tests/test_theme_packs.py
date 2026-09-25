"""THEME PACKS Unit T1 — registry integrity + deterministic backdrops."""
import hashlib
from pathlib import Path

from pipeline_v4 import theme_packs as tp


def test_registry_integrity():
    assert set(tp.THEMES) == {"obsidian", "ivory", "nightline", "verde"}
    for t in tp.THEMES.values():
        assert t.tile_border and t.ticker_bg.startswith("#")
        assert t.accent.startswith("#") and t.bg_color.startswith("#")
    assert tp.get_theme("OBSIDIAN ").key == "obsidian"
    assert tp.get_theme("") is None and tp.get_theme("nope") is None


def test_backdrops_render_and_are_deterministic(tmp_path):
    for key in tp.THEMES:
        p1 = tp.ensure_backdrop(key, tmp_path)
        assert p1 and Path(p1).is_file(), key
        from PIL import Image
        with Image.open(p1) as im:
            assert im.size == (1920, 1080), key
        h1 = hashlib.sha256(Path(p1).read_bytes()).hexdigest()
        Path(p1).unlink()                       # force a repaint
        p2 = tp.ensure_backdrop(key, tmp_path)
        h2 = hashlib.sha256(Path(p2).read_bytes()).hexdigest()
        assert h1 == h2, f"{key}: backdrop not deterministic"


def test_backdrop_cached_and_fail_soft(tmp_path):
    p1 = tp.ensure_backdrop("verde", tmp_path)
    m1 = Path(p1).stat().st_mtime_ns
    p2 = tp.ensure_backdrop("verde", tmp_path)
    assert p1 == p2 and Path(p2).stat().st_mtime_ns == m1   # reused, not repainted
    assert tp.ensure_backdrop("unknown", tmp_path) is None
