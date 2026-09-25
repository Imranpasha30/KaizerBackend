"""Trailer engine — planner sanitizing, synth SFX, full render smoke."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_v4 import trailer as tr


# ─── Moment sanitizer ───────────────────────────────────────────────


def test_sanitize_bounds_and_sorts():
    raw = [
        {"t_start": 50.0, "t_end": 51.0, "punch": 0.4},        # short → padded
        {"t_start": 5.0, "t_end": 30.0, "punch": 0.9},         # long → capped 4.5
        {"t_start": -3.0, "t_end": 2.0, "punch": 2.5},         # clamped, punch capped
        {"t_start": 5.5, "t_end": 8.0},                         # overlaps #2 → dropped
        "garbage", {"t_start": "x", "t_end": 9},                # invalid → dropped
    ]
    out = tr.sanitize_moments(raw, duration=60.0)
    assert [m["t_start"] for m in out] == sorted(m["t_start"] for m in out)
    for m in out:
        assert 0.0 <= m["t_start"] < m["t_end"] <= 60.0
        assert 1.6 - 0.01 <= m["t_end"] - m["t_start"] <= 6.0 + 0.01
        assert 0.0 <= m["punch"] <= 1.0
    starts = [m["t_start"] for m in out]
    assert 5.0 in starts and 5.5 not in starts                  # earlier overlap wins


def test_sanitize_caps_count_and_total(monkeypatch):
    # proper-teaser defaults: 16 moments / target-sec total (default 60)
    raw = [{"t_start": i * 8.0, "t_end": i * 8.0 + 5.5} for i in range(30)]
    out = tr.sanitize_moments(raw, duration=400.0)
    assert len(out) <= 16
    assert sum(m["t_end"] - m["t_start"] for m in out) <= 60.0
    # operator can lengthen the teaser via env
    monkeypatch.setenv("KAIZER_V4_TRAILER_TARGET_SEC", "90")
    out90 = tr.sanitize_moments(raw, duration=400.0)
    assert sum(m["t_end"] - m["t_start"] for m in out90) <= 90.0
    assert (sum(m["t_end"] - m["t_start"] for m in out90)
            >= sum(m["t_end"] - m["t_start"] for m in out))


def test_sanitize_keeps_valid_moods_only():
    """Mixed-mood: valid moods survive (normalized to pack keys),
    invalid ones become '' so the cut's own pack applies."""
    out = tr.sanitize_moments([
        {"t_start": 1.0, "t_end": 3.0, "mood": "comedy"},
        {"t_start": 5.0, "t_end": 7.0, "mood": "MURDER_HOMICIDE"},  # alias
        {"t_start": 9.0, "t_end": 11.0, "mood": "not_a_mood"},
        {"t_start": 13.0, "t_end": 15.0},
    ], duration=30.0)
    assert [m["mood"] for m in out] == ["comedy", "crime", "", ""]


def test_length_guarantee_expands_stingy_plans(monkeypatch):
    """Operator-reported: a 2-moment plan yielded a 6s trailer that was
    mostly cards. The expander must widen + sample to a REAL teaser."""
    monkeypatch.setenv("KAIZER_V4_TRAILER_TARGET_SEC", "60")
    stingy = [{"t_start": 10.0, "t_end": 12.0, "overlay": "", "punch": 0.8},
              {"t_start": 100.0, "t_end": 102.0, "overlay": "", "punch": 0.9}]
    out = tr.ensure_min_content(stingy, duration=300.0)
    total = sum(m["t_end"] - m["t_start"] for m in out)
    assert total >= 45.0, f"only {total:.1f}s of content"
    # sorted, non-overlapping, in-bounds
    for i, m in enumerate(out):
        assert 0.0 <= m["t_start"] < m["t_end"] <= 300.0
        if i:
            assert m["t_start"] >= out[i - 1]["t_end"] - 0.01
    # SHORT source: target is capped by the source itself (~55%)
    short = tr.ensure_min_content(
        [{"t_start": 2.0, "t_end": 4.0, "overlay": "", "punch": 0.8}],
        duration=30.0)
    s_total = sum(m["t_end"] - m["t_start"] for m in short)
    assert 10.0 <= s_total <= 18.0, f"short-source content {s_total:.1f}s"
    # deterministic + no-op on an already-full plan
    assert tr.ensure_min_content(stingy, duration=300.0) == out
    assert tr.ensure_min_content([], duration=100.0) == []


def test_structures_registry_and_transforms():
    """UG.6b: 10 assembly architectures, each transforming the same
    moments into a distinct concrete plan."""
    from pipeline_v4.trailer_styles import TRAILER_STRUCTURES
    assert len(TRAILER_STRUCTURES) == 10
    ms = [{"t_start": float(i * 10), "t_end": float(i * 10 + 3),
           "punch": p, "overlay": ("hook line" if i == 1 else "")}
          for i, p in enumerate([0.4, 0.9, 0.6, 1.0, 0.5])]
    classic = tr.apply_structure("classic", ms)
    assert classic["hook_first"] is True and not classic["stinger"]
    assert [m["t_start"] for m in classic["moments"]] == [0, 10, 20, 30, 40]
    cold = tr.apply_structure("cold_open", ms)
    assert cold["hook_first"] is False
    assert cold["moments"][0]["punch"] == 1.0          # hardest first
    cres = tr.apply_structure("crescendo", ms)
    assert [m["punch"] for m in cres["moments"]] == sorted(m["punch"] for m in ms)
    quote = tr.apply_structure("quote_led", ms)
    assert quote["moments"][0]["overlay"] == "hook line"
    flash = tr.apply_structure("flash_forward", ms)
    assert len(flash["stinger"]) == 3
    cd = tr.apply_structure("countdown_led", ms)
    assert list(cd["number_before"].values()) == ["3", "2", "1"]
    two = tr.apply_structure("two_act", ms)
    assert two["mid_card"] and two["mid_index"] == 2
    book = tr.apply_structure("bookend", ms)
    assert book["reprise"] and book["reprise"]["t_start"] == 0.0
    rapid = tr.apply_structure("rapid_montage", ms)
    assert rapid["hook_first"] is None and rapid["pace_mult"] < 1.0
    q = tr.apply_structure("question_led", ms)
    assert q["hook_hold"] > 2.0
    assert tr.apply_structure("no_such", ms)["key"] == "classic"


def test_fallback_moments_from_story_openings():
    stories = [SimpleNamespace(video_t_start=0.0, story_index=0),
               SimpleNamespace(video_t_start=30.0, story_index=1),
               SimpleNamespace(video_t_start=61.0, story_index=2)]
    out = tr.fallback_moments(stories, duration=90.0)
    assert len(out) == 3
    assert out[0]["t_start"] == pytest.approx(0.3)
    assert out[1]["t_start"] == pytest.approx(30.3)


def test_plan_falls_back_when_llm_dies(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "_TRAILER_SYSTEM", tr._TRAILER_SYSTEM)  # no-op guard
    # Force the LLM path to explode → fallback moments + title hook.
    import seo.generator as sg
    monkeypatch.setattr(sg, "_gemini_client",
                        lambda: (_ for _ in ()).throw(RuntimeError("no api")))
    stories = [SimpleNamespace(video_t_start=0.0, story_index=0,
                               title_native="పెద్ద వార్త")]
    plan = tr.plan_trailer_moments(stories=stories, words_by_story={},
                                   duration=30.0, language="te")
    assert plan["moments"], "fallback must produce moments"
    assert plan["hook_text"] == "పెద్ద వార్త"


# ─── Style pack registry ────────────────────────────────────────────


def test_style_registry_30_categories_50_transitions():
    from pipeline_v4.trailer_styles import (DEFAULT_STYLE, STYLES,
                                            TRANSITION_CATALOG, get_style,
                                            style_catalog)
    assert len(STYLES) >= 30, f"spec says 30 categories, got {len(STYLES)}"
    assert len(TRANSITION_CATALOG) > 50, "spec says more than 50 transitions"
    assert len(set(TRANSITION_CATALOG)) == len(TRANSITION_CATALOG)
    for key, p in STYLES.items():
        assert p.key == key and p.label
        assert p.grade.startswith("eq=")
        assert p.transitions and all(t in TRANSITION_CATALOG for t in p.transitions)
        assert 0.1 <= p.pace <= 0.6
        assert 30 <= p.hit_freq <= 300
        assert len(p.card_bg) == 3 and len(p.card_accent) == 3
    # the operator's named examples exist with their OWN sound design
    assert STYLES["horror"].hit_freq < STYLES["news"].hit_freq
    assert STYLES["horror"].bed is not None
    assert STYLES["crime"].bed is not None
    assert STYLES["crime"].riser_lp < STYLES["news"].riser_lp
    assert get_style("nonsense").key == DEFAULT_STYLE
    assert len(style_catalog()) == len(STYLES)


# ─── Synth SFX (style-parameterized) ────────────────────────────────


@pytest.mark.slow
def test_sfx_synthesized(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    sfx = tr.ensure_sfx(tmp_path)
    assert set(sfx) == {"hit", "whoosh", "riser"}
    for p in sfx.values():
        assert Path(p).stat().st_size > 1000
    # cached second call returns the same files
    again = tr.ensure_sfx(tmp_path)
    assert again == sfx
    # a different pack synthesizes its OWN files (horror boom != news boom)
    from pipeline_v4.trailer_styles import get_style
    horror = tr.ensure_sfx(tmp_path, get_style("horror"))
    assert horror["hit"] != sfx["hit"]
    assert Path(horror["hit"]).stat().st_size != Path(sfx["hit"]).stat().st_size


@pytest.mark.slow
def test_mood_bed_synthesized(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    from pipeline_v4.trailer_styles import get_style
    bed = tr.synth_bed(tmp_path, get_style("horror"), 8.0)
    assert bed and Path(bed).stat().st_size > 1000
    # packs without a bed → None
    assert tr.synth_bed(tmp_path, get_style("news"), 8.0) is None


# ─── Full render smoke (both aspects) ───────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("aspect,dims,style,structure", [
    ("16:9", (1920, 1080), "news", "classic"),
    # 9:16 leg runs the HORROR pack + FLASH-FORWARD structure — proves
    # the styled path AND the structure assembly (stingers before the
    # title) end-to-end.
    ("9:16", (1080, 1920), "horror", "flash_forward"),
])
def test_render_trailer_end_to_end(tmp_path, aspect, dims, style, structure):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")

    # 30s "bulletin" master: color + a tone so the audio chain is real.
    src = tmp_path / "master.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=red:s=640x360:r=30:d=30",
         "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=48000",
         "-af", "volume=10dB", "-t", "30",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(src)],
        check=True, capture_output=True)

    stories = [SimpleNamespace(video_t_start=0.0, story_index=0,
                               title_native="టెస్ట్ ట్రైలర్")]
    # mixed-mood: moment 2 carries its own pack (comedy inside the cut) —
    # exercises per-moment grade + per-joint transition/whoosh selection
    plan = {"hook_text": "పెద్ద వార్త వస్తోంది", "moments": [
        {"t_start": 2.0, "t_end": 4.5, "overlay": "మొదటి క్షణం", "punch": 0.7},
        {"t_start": 12.0, "t_end": 14.0, "overlay": "", "punch": 0.8,
         "mood": "comedy"},
        {"t_start": 24.0, "t_end": 27.0, "overlay": "చివరి దెబ్బ", "punch": 1.0,
         "mood": "no_such_mood"},   # unknown → cut's own pack, never dies
    ]}
    out = tmp_path / f"trailer_{aspect.replace(':', 'x')}.mp4"
    res = tr.render_trailer(
        source_path=str(src), stories=stories, words_by_story={},
        out_path=str(out), work_dir=tmp_path / "_tw", aspect=aspect,
        language="te", channel_name="KAIZER X", plan=plan, style=style,
        structure=structure,
    )
    assert Path(res).is_file() and Path(res).stat().st_size > 0

    # dimensions + duration ≈ cards(2.0+1.8) + moments(2.5+2+3) − 5 joints*0.25
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries",
         "format=duration", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True).stdout.split()
    w, h = (int(x) for x in probe[0].split(",")[:2])
    dur = float(probe[-1])
    assert (w, h) == dims
    # classic ≈ 9-12s (VISIBLE 0.45-0.9s fades eat more than the old
    # 4-8-frame micro-fades); flash_forward adds 3 stingers (~2.7s)
    assert 8.4 <= dur <= 16.0, f"unexpected trailer length {dur}"

    if structure == "classic":
        # first frame = the dark hook card, not the red source
        frame = tmp_path / f"f0_{w}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "0.9",
                        "-i", str(out), "-frames:v", "1", str(frame)],
                       check=True, capture_output=True)
        from PIL import Image
        with Image.open(frame) as im:
            px = list(im.convert("RGB").resize((8, 8)).getdata())
        mean = [sum(p[i] for p in px) / len(px) for i in range(3)]
        assert mean[0] < 100, f"hook card missing (frame too red: {mean})"
    else:
        # flash_forward opens on STINGERS (source footage), so the cut is
        # longer than the classic assembly of the same moments
        assert dur >= 11.0, f"stingers missing (only {dur}s)"

    # audio present and loudness-conformed near -14 LUFS
    from pipeline_v4.audio_conform import measure_loudness
    m = measure_loudness(str(out))
    assert m is not None
    assert -16.5 <= m["input_i"] <= -11.5, f"trailer loudness {m['input_i']}"


# ─── trailer-only output format is accepted through the chain ───────


def test_trailer_only_format_accepted_everywhere():
    """The new output format must be valid at every validation gate —
    a rejected value silently degrades to 'both' and renders the full
    pipeline instead of the trailer."""
    import re
    for path in ("main.py", "runner.py", "pipeline_v4/orchestrator.py"):
        src = Path(path).read_text(encoding="utf-8")
        gates = re.findall(r'not in \{[^}]*"shorts-only"[^}]*\}', src)
        assert gates, f"{path}: output-format gate not found"
        for g in gates:
            assert "trailer-only" in g, f"{path}: gate missing trailer-only: {g}"
    orch = Path("pipeline_v4/orchestrator.py").read_text(encoding="utf-8")
    # shorts are skipped and the bulletin slot is rendered by the engine
    assert 'output_format in ("full-only", "trailer-only")' in orch
    assert 'output_format == "trailer-only"' in orch
    assert "trailer_engine" in orch


def test_compose_pack_mixing():
    """User style mixing: look/motion/sound/cards each from a different
    built-in pack; invalid components fall back to news, never raise."""
    from pipeline_v4.trailer_styles import STYLES, compose_pack
    p = compose_pack({"look": "horror", "motion": "action",
                      "sound": "devotional", "cards": "gaming"}, name="My mix")
    assert p.grade == STYLES["horror"].grade
    assert p.extra_vf == STYLES["horror"].extra_vf
    assert p.transitions == STYLES["action"].transitions
    assert p.pace == STYLES["action"].pace
    assert p.hit_freq == STYLES["devotional"].hit_freq
    assert p.bed == STYLES["devotional"].bed
    assert p.card_accent == STYLES["gaming"].card_accent
    assert p.label == "My mix" and p.key == "user_mix"
    fb = compose_pack({"look": "nope"}, name="")
    assert fb.grade == STYLES["news"].grade and fb.label == "My style"
