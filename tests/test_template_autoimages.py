"""Custom-template AUTO-IMAGES — slot-fill decision logic (pure functions).

Covers orchestrator._empty_image_slots / _slot_fill_value /
_merge_template_media / _active_custom_template_ids /
_template_autoimages_on. NO image generation or DB here — the pure
decision layer is what decides which slots get filled and what lands in
Job.template_media, so it's what must be airtight.
"""
from __future__ import annotations

import pytest

from pipeline_v4.orchestrator import (
    _active_custom_template_ids,
    _empty_image_slots,
    _merge_template_media,
    _slot_fill_value,
    _template_autoimages_on,
)


def _contract(*slots):
    return {"slots": list(slots)}


def _img(name, carousel=False, **extra):
    return {"kind": "image", "name": name, "raw": f"image:{name}",
            "tag": "img", "carousel": carousel, **extra}


# ─── _empty_image_slots ─────────────────────────────────────────────

def test_empty_slots_basic_image_only():
    cj = _contract(
        _img("side"),
        _img("gallery", carousel=True),
        {"kind": "text", "name": "headline"},     # non-image kinds never fill
        {"kind": "video", "name": "main"},
    )
    assert _empty_image_slots([cj], {}, 4) == [("side", False), ("gallery", True)]


def test_empty_slots_skips_user_filled():
    cj = _contract(_img("side"), _img("gallery", carousel=True))
    # scalar-filled AND carousel-struct-filled slots both count as "taken"
    tm = {"side": 123, "gallery": {"carousel": [{"id": 9}], "fit": "cover"}}
    assert _empty_image_slots([cj], tm, 4) == []
    assert _empty_image_slots([cj], {"side": 123}, 4) == [("gallery", True)]


def test_empty_slots_unnamed_slot_keys_on_kind():
    # contract.Slot.key = name or kind — an unnamed image slot maps to "image"
    cj = _contract(_img(""))
    assert _empty_image_slots([cj], {}, 4) == [("image", False)]
    assert _empty_image_slots([cj], {"image": 7}, 4) == []


def test_empty_slots_dedupes_across_templates_first_wins():
    # One shared template_media map serves BOTH the fullform and short custom
    # templates — a colliding key is planned once, first template's flag wins.
    ff = _contract(_img("hero"), _img("strip", carousel=True))
    sh = _contract(_img("hero", carousel=True), _img("side"))
    out = _empty_image_slots([ff, sh], {}, 4)
    assert out == [("hero", False), ("strip", True), ("side", False)]


def test_empty_slots_max_slots_cap_and_floor():
    cj = _contract(*[_img(f"s{i}") for i in range(10)])
    assert len(_empty_image_slots([cj], {}, 4)) == 4
    # floor: a busted 0/negative env value still allows one slot
    assert len(_empty_image_slots([cj], {}, 0)) == 1
    assert len(_empty_image_slots([cj], {}, -3)) == 1


def test_empty_slots_survives_garbage_shapes():
    # poisoned/legacy rows must never raise — the caller is mid-render
    assert _empty_image_slots(None, {}, 4) == []
    assert _empty_image_slots(["nope", 42, {}], {}, 4) == []
    cj = {"slots": ["nope", None, _img("ok"), {"kind": "image"}]}
    assert _empty_image_slots([cj], "not-a-dict", 4) == [("ok", False), ("image", False)]


# ─── _slot_fill_value ───────────────────────────────────────────────

def test_fill_value_single_slot_scalar_id():
    # scalar id — the exact shape _load_template_media resolves for singles
    assert _slot_fill_value(False, [5]) == 5
    assert _slot_fill_value(False, [5, 6]) == 5


def test_fill_value_carousel_struct():
    val = _slot_fill_value(True, [1, 2])
    assert val == {
        "carousel": [
            {"id": 1, "duration_s": 3.0, "effect": "fade", "effect_duration": 0.4},
            {"id": 2, "duration_s": 3.0, "effect": "fade", "effect_duration": 0.4},
        ],
        "fit": "cover",
    }


def test_fill_value_empty_or_falsy_ids():
    # no surviving assets -> slot stays EMPTY (positional pool fill applies)
    assert _slot_fill_value(False, []) is None
    assert _slot_fill_value(True, []) is None
    assert _slot_fill_value(True, [0, None]) is None
    assert _slot_fill_value(True, [0, None, 3])["carousel"] == [
        {"id": 3, "duration_s": 3.0, "effect": "fade", "effect_duration": 0.4}]


def test_fill_value_carousel_50_frame_cap():
    # same DoS guard as _load_template_media / create_job
    val = _slot_fill_value(True, list(range(1, 61)))
    assert len(val["carousel"]) == 50


# ─── _merge_template_media ──────────────────────────────────────────

def test_merge_user_filled_always_wins():
    existing = {"side": 111}
    filled = {"side": 999, "gallery": {"carousel": [{"id": 1}], "fit": "cover"}}
    merged = _merge_template_media(existing, filled)
    assert merged["side"] == 111
    assert merged["gallery"] == filled["gallery"]


def test_merge_returns_fresh_dict():
    # JSON column: in-place mutation silently no-ops the commit — must be new
    existing = {"a": 1}
    merged = _merge_template_media(existing, {"b": 2})
    assert merged is not existing
    assert existing == {"a": 1}
    assert _merge_template_media(None, {"x": 1}) == {"x": 1}
    assert _merge_template_media({"x": 1}, None) == {"x": 1}


# ─── end-to-end decision: contract + media + generated -> media map ─

def test_decision_pipeline_composed():
    """The requested shape: given contract slots + template_media + generated
    assets, the resulting media map fills only the empty slots."""
    cj = _contract(_img("hero"), _img("strip", carousel=True), _img("side"))
    tm = {"side": 42}                                    # user already picked
    generated = {"hero": [201], "strip": [301, 302, 303]}
    filled = {}
    for key, is_car in _empty_image_slots([cj], tm, 4):
        val = _slot_fill_value(is_car, generated.get(key, []))
        if val:
            filled[key] = val
    merged = _merge_template_media(tm, filled)
    assert merged["side"] == 42                          # untouched
    assert merged["hero"] == 201
    assert [f["id"] for f in merged["strip"]["carousel"]] == [301, 302, 303]


# ─── env gates ──────────────────────────────────────────────────────

def test_autoimages_gate_default_on(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_TEMPLATE_AUTOIMAGES", raising=False)
    assert _template_autoimages_on() is True
    for off in ("0", "off", "FALSE", "no"):
        monkeypatch.setenv("KAIZER_V4_TEMPLATE_AUTOIMAGES", off)
        assert _template_autoimages_on() is False
    monkeypatch.setenv("KAIZER_V4_TEMPLATE_AUTOIMAGES", "1")
    assert _template_autoimages_on() is True


@pytest.mark.parametrize("fmt,expect", [
    ("both", [7, 9]),
    ("full-only", [7]),          # shorts env ignored
    ("shorts-only", [9]),        # fullform env ignored
    ("trailer-only", []),        # no bulletin compose at all
])
def test_active_template_ids_respect_output_format(monkeypatch, fmt, expect):
    monkeypatch.setenv("KAIZER_V4_FULLFORM_LAYOUT", "custom:7")
    monkeypatch.setenv("KAIZER_V4_SHORT_LAYOUT", "custom:9")
    assert _active_custom_template_ids(fmt) == expect


def test_active_template_ids_dedupe_and_non_custom(monkeypatch):
    # same template on both envs -> planned once
    monkeypatch.setenv("KAIZER_V4_FULLFORM_LAYOUT", "custom:7")
    monkeypatch.setenv("KAIZER_V4_SHORT_LAYOUT", "custom:7")
    assert _active_custom_template_ids("both") == [7]
    # library / builtin layouts never trigger auto-images
    monkeypatch.setenv("KAIZER_V4_FULLFORM_LAYOUT", "lib:aurora_glass")
    monkeypatch.setenv("KAIZER_V4_SHORT_LAYOUT", "")
    assert _active_custom_template_ids("both") == []
    monkeypatch.setenv("KAIZER_V4_SHORT_LAYOUT", "custom:notanint")
    assert _active_custom_template_ids("both") == []
