"""Custom-template EFFECTS: the Director's grade reaches template-job FOOTAGE
while the template DESIGN stays crisp.

Proves the safety properties this feature was built on:
  1. effects_vf omitted / "" -> the compose filtergraph is CHARACTER-IDENTICAL
     to the pre-effects engine (every existing template job renders unchanged).
  2. a non-empty LINEAR chain is appended to every per-clip chain AFTER fps=
     (post scale/crop so vignettes fit the slot's visible rect) and NEVER to
     the design-PNG overlay chain.
  3. the sanitizer rejects filtergraph fragments (";" / "[" / "]") outright —
     fail-soft to an ungraded render instead of a corrupted graph.
  4. RenderRequest grew the field with a "" default, so old callers that never
     pass it keep working (and keep byte-identical output).
  5. engine.render_template threads req.effects_vf into compose.compose.

NOTE: there is deliberately no cache test — the custom-template path has NO
render cache (render_template captures frames + runs ffmpeg fresh every call),
so effects_vf needs no cache fingerprint, unlike v1_bridge's per-story cache.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from services.custom_templates import RenderRequest
from services.custom_templates import compose as ct_compose
from services.custom_templates import engine as ct_engine

# A realistic Director grade: a plain LINEAR chain (no ";", no labels).
_FX = "eq=contrast=1.06:saturation=1.08,vignette=PI/6"


# ── helpers ───────────────────────────────────────────────────────────

def _fake_run_factory(captured: dict):
    """Stand-in for subprocess.run inside compose: records the ffmpeg cmd and
    writes the output file so compose's isfile success-check passes — the chain
    STRING is what we assert on, no real encode needed."""
    def _fake_run(cmd, capture_output=True, timeout=None, **kw):
        captured.setdefault("cmds", []).append(list(cmd))
        out = cmd[-1]
        with open(out, "wb") as fh:
            fh.write(b"\x00" * 2048)
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
    return _fake_run


def _compose_graph(monkeypatch, tmp_path, name: str, **kwargs) -> str:
    """Run compose() with a stubbed ffmpeg and return the -filter_complex string."""
    captured: dict = {}
    monkeypatch.setattr(ct_compose, "subprocess",
                        SimpleNamespace(run=_fake_run_factory(captured)))
    placements = [
        ct_compose.Placement(x=10, y=20, w=640, h=360, video="main.mp4"),
        ct_compose.Placement(x=0, y=0, w=1920, h=1080, video="bg.mp4",
                             background=True),
    ]
    out = str(tmp_path / f"{name}.mp4")
    # duration passed explicitly so compose never shells out to ffprobe.
    ct_compose.compose("design.png", placements, out, canvas=(1920, 1080),
                       fps=30, duration=6.0, **kwargs)
    cmd = captured["cmds"][-1]
    return cmd[cmd.index("-filter_complex") + 1]


# ── 1+2: chain string with / without effects_vf ───────────────────────

def test_empty_effects_vf_is_byte_identical(monkeypatch, tmp_path):
    legacy = _compose_graph(monkeypatch, tmp_path, "legacy")            # omitted
    off = _compose_graph(monkeypatch, tmp_path, "off", effects_vf="")   # explicit ""
    assert off == legacy
    # Lock the exact legacy per-clip chain characters (byte-compat contract).
    assert ("[0:v]scale=640:360:force_original_aspect_ratio=increase,"
            "crop=640:360,setsar=1,fps=30[v0]") in legacy
    assert ("[1:v]scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080,setsar=1,fps=30[v1]") in legacy
    assert "eq=" not in legacy and "vignette" not in legacy


def test_effects_vf_appended_post_fps_on_all_clips(monkeypatch, tmp_path):
    graph = _compose_graph(monkeypatch, tmp_path, "fx", effects_vf=_FX)
    # Appended AFTER fps= (post scale/crop) on EVERY video placement — main
    # and background alike — so edge effects fit each slot's visible rect.
    assert ("[0:v]scale=640:360:force_original_aspect_ratio=increase,"
            f"crop=640:360,setsar=1,fps=30,{_FX}[v0]") in graph
    assert ("[1:v]scale=1920:1080:force_original_aspect_ratio=increase,"
            f"crop=1920:1080,setsar=1,fps=30,{_FX}[v1]") in graph
    assert graph.count(_FX) == 2                    # once per placement, no more
    # The design PNG overlay chain stays UNGRADED (template graphics stay crisp).
    assert "format=rgba[png]" in graph
    assert f"format=rgba,{_FX}" not in graph and f"{_FX},format=rgba" not in graph


def test_bad_fragment_fails_soft_to_legacy_graph(monkeypatch, tmp_path):
    legacy = _compose_graph(monkeypatch, tmp_path, "legacy2")
    # A graph fragment must NOT corrupt the composite — footage renders ungraded.
    bad = _compose_graph(monkeypatch, tmp_path, "bad",
                         effects_vf="split[a][b];[a][b]overlay")
    assert bad == legacy


# ── 3: sanitizer ──────────────────────────────────────────────────────

def test_sanitizer_accepts_linear_and_trims():
    assert ct_compose.sanitize_effects_vf(_FX) == _FX
    # stray commas / whitespace are trimmed so the join never emits ',,'
    assert ct_compose.sanitize_effects_vf(f" ,{_FX}, ") == _FX


def test_sanitizer_rejects_graph_fragments():
    assert ct_compose.sanitize_effects_vf("") == ""
    assert ct_compose.sanitize_effects_vf(None) == ""
    assert ct_compose.sanitize_effects_vf("   , ,  ") == ""
    assert ct_compose.sanitize_effects_vf("eq=contrast=1.1;vignette") == ""     # ';'
    assert ct_compose.sanitize_effects_vf("[0:v]eq=contrast=1.1") == ""         # '['
    assert ct_compose.sanitize_effects_vf("split[a][b]") == ""                  # labels
    assert ct_compose.sanitize_effects_vf("overlay]") == ""                     # ']'


# ── 4: RenderRequest back-compat ──────────────────────────────────────

def test_render_request_default_keeps_old_callers_working():
    # Old-style construction (no effects_vf) must still work and default to "".
    req = RenderRequest(videos={"video": "clip.mp4"}, texts={"headline": "hi"},
                        fps=30, main_slot="video", literal=True)
    assert req.effects_vf == ""
    assert RenderRequest().effects_vf == ""
    # New-style construction carries the chain through.
    assert RenderRequest(effects_vf=_FX).effects_vf == _FX
    # compose() itself defaults the kwarg, so pre-effects callers of compose
    # (and duck-typed requests without the field) stay valid too.
    assert inspect.signature(ct_compose.compose).parameters["effects_vf"].default == ""


# ── 5: engine threads the field into compose ──────────────────────────

def _render_via_engine(monkeypatch, tmp_path, req) -> str:
    """Run render_template with the browser capture + ffmpeg compose stubbed,
    returning the effects_vf that reached compose."""
    captured: dict = {}

    def fake_compose(design, placements, out_path, canvas, *, fps=30,
                     duration=None, effects_vf=""):
        captured["effects_vf"] = effects_vf
        with open(out_path, "wb") as fh:
            fh.write(b"\x00" * 2048)
        return out_path

    fake_ar = SimpleNamespace(
        video_rects=[{"key": "video", "x": 0, "y": 0, "w": 320, "h": 180,
                      "kind": "video"}],
        frames_dir=str(tmp_path), frame_pattern="design_%04d.png", fps=30,
        animated=False, frame_count=1, warnings=[], anim_rects=None,
        ticker_rect=None, carousel_rects=[], brand_rects={},
    )
    monkeypatch.setattr(ct_engine._compose, "compose", fake_compose)
    monkeypatch.setattr(ct_engine._renderer, "render_animated",
                        lambda *a, **k: fake_ar)
    monkeypatch.delenv("KAIZER_TEMPLATE_ANIMATE", raising=False)  # still mode

    bundle = SimpleNamespace(root_dir=str(tmp_path), entry_rel="index.html")
    contract = SimpleNamespace(canvas_w=1920, canvas_h=1080, slots=[],
                               text_slots=[], image_slots=[], video_slots=[],
                               warnings=[])
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 64)
    req.videos = {"video": str(clip)}
    req.duration = 5.0                       # skip the ffprobe duration probe
    ct_engine.render_template(bundle, contract, req,
                              work_dir=str(tmp_path / "w"),
                              out_path=str(tmp_path / "out.mp4"))
    return captured["effects_vf"]


def test_engine_threads_effects_vf_to_compose(monkeypatch, tmp_path):
    assert _render_via_engine(monkeypatch, tmp_path,
                              RenderRequest(effects_vf=_FX)) == _FX


def test_engine_default_reaches_compose_as_empty(monkeypatch, tmp_path):
    assert _render_via_engine(monkeypatch, tmp_path, RenderRequest()) == ""
