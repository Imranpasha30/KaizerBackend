"""Unit B — TV-style profanity bleep (detect → mute + tone → report)."""
from __future__ import annotations

import json
import re
import subprocess
from types import SimpleNamespace

import pytest

from pipeline_v4 import bleep


def _w(word, s, e):
    return {"w": word, "s": float(s), "e": float(e)}


# ─── Detection ──────────────────────────────────────────────────────


def test_detects_english_hindi_telugu_and_translit():
    words = [
        _w("This", 0.0, 0.2), _w("fucking", 0.3, 0.7), _w("news", 0.8, 1.0),
        _w("चूतिया", 2.0, 2.4),
        _w("bhosdike!", 4.0, 4.5),          # punctuation stripped
        _w("లంజ", 6.0, 6.3),
        _w("SHIT,", 8.0, 8.2),              # case-insensitive
    ]
    spans = bleep.detect_bleep_spans(words)
    flat = [w for sp in spans for w in sp["words"]]
    assert flat == ["fucking", "चूतिया", "bhosdike!", "లంజ", "SHIT,"]
    # padding applied
    assert spans[0]["start"] == pytest.approx(0.26, abs=0.01)
    assert spans[0]["end"] == pytest.approx(0.74, abs=0.01)


def test_no_substring_false_positives():
    """'assessment' must never trigger; exact normalized tokens only."""
    words = [_w("assessment", 0, 1), _w("class", 1, 2), _w("Scunthorpe", 2, 3),
             _w("shitake", 3, 4), _w("gander", 4, 5), _w("pukka", 5, 6)]
    assert bleep.detect_bleep_spans(words) == []


def test_adjacent_hits_merge_into_one_beep():
    words = [_w("fucking", 1.0, 1.4), _w("shit", 1.45, 1.8)]
    spans = bleep.detect_bleep_spans(words)
    assert len(spans) == 1
    assert spans[0]["words"] == ["fucking", "shit"]
    assert spans[0]["start"] < 1.0 and spans[0]["end"] > 1.8


def test_extra_words_env_and_param(monkeypatch):
    monkeypatch.setenv("KAIZER_BLEEP_EXTRA_WORDS", "badbrand, नकली")
    words = [_w("badbrand", 0, 1), _w("नकली", 2, 3), _w("custom", 4, 5)]
    spans = bleep.detect_bleep_spans(words)
    assert len(spans) == 2
    spans = bleep.detect_bleep_spans(words, extra_words=("custom",))
    assert len(spans) == 3


def test_clean_speech_yields_nothing():
    words = [_w("హైదరాబాద్", 0, 1), _w("వరదలు", 1, 2), _w("rescue", 2, 3)]
    assert bleep.detect_bleep_spans(words) == []


# ─── Orchestrator pass + report ─────────────────────────────────────


def test_run_bleep_pass_report_and_disable(tmp_path, monkeypatch):
    stories = [SimpleNamespace(video_t_start=10.0,
                               words=[_w("clean", 0.5, 1.0), _w("shit", 2.0, 2.4)])]
    tr = SimpleNamespace(stories=stories, trimmed_path=str(tmp_path / "absent.mp4"))
    report = bleep.run_bleep_pass(trim_result=tr, out_dir=tmp_path)
    # span remapped to ABSOLUTE timeline (10.0 + 2.0)
    assert report["spans"][0]["start"] == pytest.approx(11.96, abs=0.01)
    assert report["applied"] is False           # no real file → apply soft-fails
    on_disk = json.loads((tmp_path / bleep.BLEEP_REPORT_NAME).read_text(encoding="utf-8"))
    assert on_disk["spans"] == report["spans"]
    # master switch off → None, no report
    monkeypatch.setenv("KAIZER_V4_BLEEP", "0")
    assert bleep.run_bleep_pass(trim_result=tr, out_dir=tmp_path / "x") is None


def test_load_report_spans_only_when_applied(tmp_path):
    p = tmp_path / bleep.BLEEP_REPORT_NAME
    p.write_text(json.dumps({"applied": False,
                             "spans": [{"start": 1.0, "end": 2.0, "words": ["x"]}]}),
                 encoding="utf-8")
    assert bleep.load_report_spans(tmp_path) == []       # not applied → no hash impact
    p.write_text(json.dumps({"applied": True,
                             "spans": [{"start": 1.0, "end": 2.0, "words": ["x"]}]}),
                 encoding="utf-8")
    assert bleep.load_report_spans(tmp_path) == [(1.0, 2.0)]
    assert bleep.load_report_spans(tmp_path / "missing") == []


def test_bleep_hash_ingredient_conditional():
    from pipeline_v4 import v1_bridge as vb
    story = SimpleNamespace(title_native="t", title_english="t", summary="s",
                            video_t_start=0.0, video_t_end=10.0,
                            story_index=0, total_stories=1)
    def _h(spans):
        return vb._per_story_cache_hash(
            story=story, ticker_path="", sidebar_path="", layout=None,
            channel_bug_path="", watermark_path="", watermark_position="",
            font_path="", bg_video_abs=None, bg_video_volume=0.0,
            language_code="te", images=None, pool_dir=None,
            bleep_spans=spans)
    assert _h(()) == _h(())
    assert _h(()) != _h(((1.0, 2.0),))
    assert _h(((1.0, 2.0),)) == _h(((1.0, 2.0),))


# ─── ffmpeg smoke: word muted, beep audible, rest untouched ─────────


def _band_level(path, freq, t0, t1):
    """Mean volume (dB) of a narrow band around ``freq`` in [t0, t1]."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats",
         "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", str(path),
         "-af", f"bandpass=f={freq}:w=200,volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", proc.stderr or "")
    assert m, f"volumedetect gave no reading: {proc.stderr[-300:]}"
    return float(m.group(1))


@pytest.mark.slow
def test_apply_bleep_mutes_word_and_plays_tone(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    clip = tmp_path / "speech.mp4"
    # 10s clip whose "speech" is a steady 400 Hz tone.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30:d=10",
         "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=48000",
         "-af", "volume=6dB", "-t", "10",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(clip)],
        check=True, capture_output=True)

    assert bleep.apply_bleep(str(clip), [{"start": 4.0, "end": 6.0, "words": ["x"]}])

    in_400 = _band_level(clip, 400, 4.3, 5.7)     # speech inside the bleep → gone
    in_1k = _band_level(clip, 1000, 4.3, 5.7)     # tone inside the bleep → present
    out_400 = _band_level(clip, 400, 7.0, 9.0)    # speech outside → untouched
    out_1k = _band_level(clip, 1000, 7.0, 9.0)    # no tone outside
    assert in_400 < out_400 - 25, f"word not muted ({in_400} vs {out_400} dB)"
    assert in_1k > out_1k + 25, f"beep not audible ({in_1k} vs {out_1k} dB)"
    assert out_400 > -30, "programme audio outside the bleep was damaged"
