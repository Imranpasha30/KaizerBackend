"""Podcast pipeline (ported from kaizer-platform@d5fd482) — pure planners.

Offline only: cutlist filler-drop + keep-range coalescing, punch-in plan in
edited-timeline coordinates, promo highlight selection, the extraction-EDL
builder (the one pipeline_v2 module copied locally), the Remotion bridge's
dir resolution / availability gate, the router's upload-filename sanitizer,
the -filter_complex_script command builders, and the Deepgram diarize/conf
parameter plumbing (faked client — no network). No ffmpeg, no node, no LLM
calls."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from pipeline_core.podcast.cutlist import CutlistConfig, build_keep_ranges
from pipeline_core.podcast.edl_builder import build_extraction_edl
from pipeline_core.podcast.promo import build_promo_plan
from pipeline_core.podcast.punchin import build_punch_in_plan
from pipeline_core.podcast import remotion_bridge


@dataclass
class W:
    w: str
    s: float
    e: float
    speaker: int | None = 0
    confidence: float | None = 0.95


def _speech(tokens, start=0.0, step=0.4):
    """Evenly spaced words, 0.3s voiced each."""
    return [W(w=t, s=start + i * step, e=start + i * step + 0.3)
            for i, t in enumerate(tokens)]


_CLEAN = ("the market opened sharply higher today with banking stocks "
          "leading the rally across every major index in the country").split()


# ── cutlist ─────────────────────────────────────────────────────────

def test_cutlist_empty_input():
    r = build_keep_ranges([])
    assert r.keep_ranges == () and r.kept_seconds == 0.0


def test_cutlist_keeps_clean_speech():
    words = _speech(_CLEAN)
    r = build_keep_ranges(words)
    assert len(r.keep_ranges) >= 1
    # Clean speech: nearly everything survives.
    assert r.kept_seconds > 0.8 * r.source_duration_s


def test_cutlist_drops_hard_fillers():
    words = _speech(_CLEAN[:8]) + _speech(
        ["um", "uh", "umm", "erm", "uhh", "hmm"], start=4.0) + _speech(
        _CLEAN[8:], start=8.0)
    r = build_keep_ranges(words)
    clean = build_keep_ranges(_speech(_CLEAN[:8]) + _speech(_CLEAN[8:], start=8.0))
    # The filler block must not inflate the kept runtime.
    assert r.kept_seconds <= clean.kept_seconds + 1.0
    assert r.removed_seconds > 0.0
    assert any(d for d in r.dropped)          # removals are reported


def test_cutlist_ranges_ordered_and_positive():
    r = build_keep_ranges(_speech(_CLEAN * 3))
    prev_end = -1.0
    for s, e in r.keep_ranges:
        assert e > s >= 0.0
        assert s >= prev_end          # non-overlapping, playback order
        prev_end = e


# ── punch-in ────────────────────────────────────────────────────────

def test_punchin_empty_inputs():
    assert build_punch_in_plan([], _speech(_CLEAN)) == []
    assert build_punch_in_plan([(0.0, 5.0)], []) == []


def test_punchin_times_are_edited_timeline():
    words = _speech(_CLEAN * 6)
    r = build_keep_ranges(words)
    plan = build_punch_in_plan(r.keep_ranges, words)
    edited = sum(e - s for s, e in r.keep_ranges)
    for p in plan:
        assert 0.0 <= p.start_s < edited + 0.01
        assert p.end_s > p.start_s
        assert p.zoom >= 1.0 and p.mode in ("snap", "slow_push")


# ── promo ───────────────────────────────────────────────────────────

def test_promo_empty_without_words():
    # NOTE vendor semantics: empty keep_ranges = whole source eligible
    # (not "nothing kept") — only empty WORDS yields an empty plan.
    plan = build_promo_plan([], [])
    assert plan.segments == ()


def test_promo_selects_within_keep_ranges():
    words = _speech(_CLEAN * 20)          # ~2.5 min of speech
    r = build_keep_ranges(words)
    plan = build_promo_plan(words, r.keep_ranges)
    assert plan.segments, "long clean source must yield promo segments"
    src_lo = min(s for s, _ in r.keep_ranges)
    src_hi = max(e for _, e in r.keep_ranges)
    for seg in plan.segments:            # PromoSegment is SOURCE coords
        assert seg.start_s >= src_lo - 0.5
        assert seg.end_s <= src_hi + 0.5
        assert seg.end_s > seg.start_s


# ── extraction EDL (the copied pipeline_v2 module) ──────────────────

def test_edl_concat_mode():
    edl = build_extraction_edl([(0.0, 4.0), (6.0, 10.0)])
    assert "concat" in edl.filter_complex
    assert len(edl.outputs) == 1
    o = edl.outputs[0]
    assert o.role == "bulletin" and o.v_label and o.a_label
    assert o.duration_s == pytest.approx(8.0, abs=0.2)


def test_edl_per_story_mode():
    edl = build_extraction_edl([(0.0, 4.0), (6.0, 10.0)],
                               bulletin_mode="per_story")
    assert [(o.role, o.index) for o in edl.outputs] == \
        [("bulletin_story", 1), ("bulletin_story", 2)]


def test_edl_rejects_empty_and_bad_mode():
    with pytest.raises(ValueError):
        build_extraction_edl([], ())
    with pytest.raises(ValueError):
        build_extraction_edl([(0.0, 1.0)], bulletin_mode="nope")


# ── remotion bridge ─────────────────────────────────────────────────

def test_remotion_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIZER_REMOTION_DIR", str(tmp_path))
    assert remotion_bridge._remotion_dir() == tmp_path
    # No node_modules in the tmp dir → not available.
    assert remotion_bridge.remotion_available() is False


def test_remotion_default_dir_is_repo_relative(monkeypatch):
    """No hardcoded DEV path: the default must be computed from __file__
    (<KaizerBackend>/engines/remotion — engines/ is gitignored + excluded
    from promote), so a promoted LIVE tree renders from ITS OWN engines
    checkout, never the DEV one."""
    monkeypatch.delenv("KAIZER_REMOTION_DIR", raising=False)
    expected = (Path(remotion_bridge.__file__).resolve().parents[2]
                / "engines" / "remotion")
    assert remotion_bridge._remotion_dir() == expected
    # And the module source itself carries no literal dev-tree path.
    src = Path(remotion_bridge.__file__).read_text(encoding="utf-8")
    assert r"e:\kaizer-dev" not in src.lower()


# ── router upload-filename sanitizer ────────────────────────────────

def test_upload_filename_sanitizer():
    from routers.podcast import _safe_upload_name

    assert _safe_upload_name("clip.mp4") == "clip.mp4"
    assert _safe_upload_name(None) == "source.mp4"
    assert _safe_upload_name("") == "source.mp4"
    # Path traversal + absolute-path override are neutralized to a basename.
    assert _safe_upload_name("..\\..\\evil.mp4") == "evil.mp4"
    assert _safe_upload_name("../../evil.mov") == "evil.mov"
    assert _safe_upload_name("/etc/passwd.mkv") == "passwd.mkv"
    assert _safe_upload_name("C:\\Windows\\System32\\x.webm") == "x.webm"
    # Shell-ish / unicode chars are flattened to '_'.
    cleaned = _safe_upload_name("my clip$(rm)!.m4a")
    assert cleaned.endswith(".m4a")
    assert all(ch.isalnum() or ch in "._-" for ch in cleaned)
    # Extension whitelist: everything else is rejected (endpoint -> 422).
    for bad in ("evil.exe", "noext", "page.html", "script.mp4.bat"):
        with pytest.raises(ValueError):
            _safe_upload_name(bad)


# ── render command builders use -filter_complex_script ──────────────

def _synthetic_captions(n=200):
    """Enough per-word captions that an inline graph would blow Windows'
    32,767-char CreateProcess limit — the whole point of the script file."""
    return tuple(
        {"w": f"word{i}", "start_s": i * 0.4, "end_s": i * 0.4 + 0.3}
        for i in range(n)
    )


def test_main_command_uses_filter_complex_script(tmp_path):
    from pipeline_core.podcast.render import RenderRequest, _build_main_command

    req = RenderRequest(
        source_path="src.mp4",
        keep_ranges=((0.0, 40.0), (45.0, 90.0)),
        captions=_synthetic_captions(),
        out_path=str(tmp_path / "podcast_edit.mp4"),
    )
    cmd, dur = _build_main_command(req, font=None)
    assert "-filter_complex" not in cmd          # inline graph is gone
    assert "-filter_complex_script" in cmd
    script = cmd[cmd.index("-filter_complex_script") + 1]
    assert Path(script).is_file()
    assert Path(script).parent == tmp_path       # lives next to the output
    graph = Path(script).read_text(encoding="utf-8")
    assert "concat" in graph and "drawtext" in graph
    # The command line itself stays tiny; the bulk lives in the file.
    assert len(" ".join(cmd)) < 2000
    assert dur == pytest.approx(85.0, abs=0.5)


def test_promo_command_uses_filter_complex_script(tmp_path):
    from pipeline_core.podcast.render import _build_promo_command

    cmd, _ = _build_promo_command(
        "src.mp4", [(0.0, 10.0)], _synthetic_captions(25),
        str(tmp_path / "promo_916.mp4"), vertical=True, font=None,
    )
    assert "-filter_complex" not in cmd
    assert "-filter_complex_script" in cmd
    script = cmd[cmd.index("-filter_complex_script") + 1]
    assert Path(script).is_file()
    assert Path(script).name == "filtergraph_promo_916.txt"


# ── Deepgram shim: diarize param + per-word confidence ──────────────

class _FakeDGWord:
    punctuated_word = "Hello."
    word = "hello"
    start = 0.0
    end = 0.4
    speaker = 1
    confidence = 0.91


def _install_fake_deepgram(monkeypatch, captured):
    """Fake `deepgram` module capturing transcribe_file kwargs — no network."""
    def _transcribe(**kwargs):
        captured["kwargs"] = kwargs
        alt = types.SimpleNamespace(words=[_FakeDGWord()])
        chan = types.SimpleNamespace(alternatives=[alt])
        return types.SimpleNamespace(
            metadata=types.SimpleNamespace(duration=0.4),
            results=types.SimpleNamespace(channels=[chan]),
        )

    class DeepgramClient:
        def __init__(self, api_key):
            self.listen = types.SimpleNamespace(
                v1=types.SimpleNamespace(
                    media=types.SimpleNamespace(transcribe_file=_transcribe)))

    mod = types.ModuleType("deepgram")
    mod.DeepgramClient = DeepgramClient
    monkeypatch.setitem(sys.modules, "deepgram", mod)


def test_deepgram_diarize_param_and_conf(monkeypatch, tmp_path):
    from pipeline_v4 import trim_engine

    captured: dict = {}
    _install_fake_deepgram(monkeypatch, captured)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00")

    # Default (diarize=None) keeps the env gate — unset env => False.
    monkeypatch.delenv("KAIZER_V4_DIARIZE", raising=False)
    words, dur = trim_engine._deepgram_words(str(audio))
    assert captured["kwargs"]["diarize"] is False
    # Default still honors the env when it is set (V4 news behavior intact).
    monkeypatch.setenv("KAIZER_V4_DIARIZE", "1")
    trim_engine._deepgram_words(str(audio))
    assert captured["kwargs"]["diarize"] is True
    # Explicit False beats env=on; explicit True works without env.
    trim_engine._deepgram_words(str(audio), diarize=False)
    assert captured["kwargs"]["diarize"] is False
    monkeypatch.delenv("KAIZER_V4_DIARIZE", raising=False)
    trim_engine._deepgram_words(str(audio), diarize=True)
    assert captured["kwargs"]["diarize"] is True

    # Word entries carry spk + the new conf (conditional-key pattern).
    assert dur == pytest.approx(0.4)
    assert words and words[0]["w"] == "Hello."
    assert words[0]["spk"] == 1
    assert words[0]["conf"] == pytest.approx(0.91)


def test_podcast_shim_maps_conf_and_requests_diarize(monkeypatch, tmp_path):
    """The router's STT shim must pass diarize=True and map conf->confidence."""
    from routers import podcast as pod

    seen: dict = {}

    def fake_extract(src, out, **kw):
        return out

    def fake_words(audio_path, language="multi", *, diarize=None):
        seen["diarize"] = diarize
        return ([{"i": 0, "w": "hi", "s": 0.0, "e": 0.3, "spk": 2,
                  "conf": 0.88}], 0.3)

    import pipeline_v4.trim_engine as te
    monkeypatch.setattr(te, "_extract_audio_mp3", fake_extract)
    monkeypatch.setattr(te, "_deepgram_words", fake_words)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

    out = pod._transcribe_via_stt("src.mp4", "deepgram", "en", str(tmp_path))
    assert seen["diarize"] is True
    assert out[0].speaker == 2
    assert out[0].confidence == pytest.approx(0.88)
