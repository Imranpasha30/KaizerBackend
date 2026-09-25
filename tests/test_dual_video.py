"""Unit E — dual-video shorts layout + diarization word tags."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_v4 import v1_bridge as vb


def test_dual_video_is_a_supported_layout():
    assert "dual_video" in vb.SUPPORTED_SHORTS_LAYOUTS
    # legacy set untouched
    for k in ("torn_card", "clean_card", "split_frame", "follow_bar"):
        assert k in vb.SUPPORTED_SHORTS_LAYOUTS


def test_dual_video_fallback_without_second_source(tmp_path):
    inputs = SimpleNamespace(
        trimmed_short_path="a.mp4", title_text="t",
        output_path=str(tmp_path / "o.mp4"), work_dir=tmp_path,
        second_video_path=None,
    )
    assert vb._render_dual_video_short(inputs, None) is False
    inputs.second_video_path = str(tmp_path / "missing.mp4")
    assert vb._render_dual_video_short(inputs, None) is False


def test_diarized_word_shape_is_additive():
    """spk rides the word dicts ONLY when diarization produced it — the
    sidecar/prompt shape for news jobs is unchanged."""
    from pipeline_v4.orchestrator import _identity_words
    plain = _identity_words([{"w": "hi", "s": 1.0, "e": 1.3}], 0, 10)
    assert "spk" not in plain[0]
    tagged = _identity_words([{"w": "hi", "s": 1.0, "e": 1.3, "spk": 1}], 0, 10)
    assert tagged[0]["spk"] == 1


def test_flatten_carries_speaker_tags(tmp_path, monkeypatch):
    """run_step1's story-relative remap keeps spk on diarized words."""
    import contextlib
    from pipeline_v4 import trim_engine
    words = [{"i": 0, "w": "host", "s": 1.0, "e": 1.5, "spk": 0},
             {"i": 1, "w": "guest", "s": 2.0, "e": 2.5, "spk": 1}]
    plan = [{"title_native": "t", "title_english": "t", "summary": "",
             "kept_spans": [{"start_sec": 0.0, "end_sec": 5.0}]}]
    monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", "news")
    monkeypatch.setenv("KAIZER_V4_TIGHTEN_SILENCE", "0")
    monkeypatch.setattr(trim_engine, "_extract_audio_mp3", lambda s, d, **k: d)
    monkeypatch.setattr(trim_engine, "_deepgram_words",
                        lambda p, language="multi": (words, 10.0))
    monkeypatch.setattr(trim_engine, "_select_planner",
                        lambda: ("test", lambda **k: (plan, 0.0)))
    monkeypatch.setattr(trim_engine, "_atomic_trim_concat", lambda **k: None)
    monkeypatch.setattr(trim_engine, "_encode_gate",
                        lambda: contextlib.nullcontext())
    monkeypatch.setattr(trim_engine, "_belt", lambda *a, **k: None)
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    res = trim_engine.run_step1(source_video=str(src),
                                output_dir=str(tmp_path / "o"), language="en")
    got = res.stories[0].words
    assert [w.get("spk") for w in got] == [0, 1]


@pytest.mark.slow
def test_dual_video_renders_two_cams_stacked(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")

    def make(color, path, dur=4):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=30:d={dur}",
             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
             "-t", str(dur), "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "aac", "-shortest", str(path)],
            check=True, capture_output=True)

    main = tmp_path / "main.mp4"
    second = tmp_path / "second.mp4"
    make("red", main, dur=4)
    make("green", second, dur=2)        # shorter → must loop under the main

    out = tmp_path / "dual.mp4"
    inputs = SimpleNamespace(
        trimmed_short_path=str(main), title_text="రెండు కెమెరాలు test",
        output_path=str(out), work_dir=tmp_path,
        second_video_path=str(second),
    )
    assert vb._render_dual_video_short(inputs, None) is True
    assert out.stat().st_size > 0

    frame = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "3.0",
                    "-i", str(out), "-frames:v", "1", str(frame)],
                   check=True, capture_output=True)
    from PIL import Image
    with Image.open(frame) as im:
        im = im.convert("RGB")
        assert im.size == (1080, 1920)
        top = im.crop((500, 300, 580, 380))          # video A region
        bot = im.crop((500, 1400, 580, 1480))        # video B region
        t = [sum(p[i] for p in top.getdata()) / 6400 for i in range(3)]
        b = [sum(p[i] for p in bot.getdata()) / 6400 for i in range(3)]
    assert t[0] > 150 and t[1] < 90, f"top cam not red: {t}"
    assert b[1] > 120 and b[0] < 90, f"bottom cam not green (loop failed?): {b}"
