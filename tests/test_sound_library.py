"""UG.6 — sound library: every recipe synthesizes audibly on host."""
from __future__ import annotations

import json
import subprocess

import pytest

from pipeline_v4 import sound_library as sl


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path], capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)["format"]


def _mean_volume_db(path: str) -> float:
    out = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60)
    for line in out.stderr.splitlines():
        if "mean_volume" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    return -99.0


def test_catalog_shape():
    assert len(sl.LIBRARY) >= 50, f"only {len(sl.LIBRARY)} sounds"
    fams = {s.family for s in sl.LIBRARY.values()}
    assert {"impacts", "whooshes", "risers", "stingers", "ui", "tension",
            "ambience", "transition", "musical", "misc"} <= fams
    cat = sl.sound_catalog()
    assert len(cat) == len(sl.LIBRARY)
    for row in cat:
        assert row["key"] and row["label"] and row["used_for"]
    assert sl.ensure_sound("no_such_sound", None) is None


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
@pytest.mark.parametrize("key", sorted(sl.LIBRARY))
def test_every_sound_synthesizes(key, tmp_path):
    """Each recipe must render AND carry real energy (not silence)."""
    p = sl.ensure_sound(key, tmp_path)
    assert p, f"{key} failed to synthesize"
    dur = float(_probe(p)["duration"])
    assert abs(dur - sl.LIBRARY[key].dur) < 0.35, f"{key} duration {dur}"
    assert _mean_volume_db(p) > -55.0, f"{key} is (near) silent"


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_one_shot_preview_is_hearable(tmp_path):
    """One-shots preview as double-play ≥3 s; ambiences pass through."""
    p = sl.preview_sound("ui_tick", tmp_path)      # 0.09s blip
    assert p and float(_probe(p)["duration"]) >= 3.0
    amb = sl.preview_sound("amb_rain", tmp_path)
    assert amb and float(_probe(amb)["duration"]) >= 5.5


def test_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "impact_deep.wav"
    fake.write_bytes(b"RIFF")
    monkeypatch.setattr(sl, "_override_path",
                        lambda key: fake if key == "impact_deep" else None)
    assert sl.ensure_sound("impact_deep", tmp_path) == str(fake)
