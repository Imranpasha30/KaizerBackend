"""Item 104 -- transition catalog tests.

Exercises ``pipeline_v2.transitions`` (the catalog + lookups) plus the
end-to-end plumbing into ``Stage4Render`` so the operator's choice
actually reaches the renderer. The 11 tests in this module cover:

  Catalog shape:
    1.  All 7 transitions present in TRANSITIONS_ORDERED.
    2.  DEFAULT_TRANSITION_NAME resolves to a real entry in the catalog.
    3.  smart_cut is implemented; smart_cut.duration_s == 0.

  Lookup helpers:
    4.  get_transition(name) returns the named entry.
    5.  get_transition(unknown) falls back to default.
    6.  get_transition(None) / "" / blank falls back to default.
    7.  is_valid_transition() accepts catalog names, rejects unknown.
    8.  resolve_for_render() returns smart_cut for non-implemented
        entries (preserves operator's choice on the Job row but
        guarantees a renderable selection).

  Plumbing:
    9.  Stage4Render dataclass exposes transition_style with default
        "smart_cut".
    10. The runner.event_data payload carries a transition_style key.
    11. main.create_job's transition_style form field defaults to
        "smart_cut" (introspected via inspect.signature).
"""

from __future__ import annotations

# Make the in-tree package importable when pytest collects.
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))
_BACKEND_ROOT = _PIPELINE_V2_ROOT.parent
sys.path.insert(0, str(_BACKEND_ROOT))


# ----- Catalog shape ---------------------------------------------------


def test_01_all_seven_transitions_present_in_catalog():
    from pipeline_v2.transitions import TRANSITIONS, TRANSITIONS_ORDERED
    expected_names = {
        "smart_cut", "crossfade", "fade_to_black", "dip_to_white",
        "slide_left", "wipe_right", "dissolve",
    }
    assert set(TRANSITIONS.keys()) == expected_names
    assert len(TRANSITIONS_ORDERED) == 7
    # Ordering matters for the UI dropdown: smart_cut comes first so
    # the default selection is the catalog's first entry.
    assert TRANSITIONS_ORDERED[0].name == "smart_cut"
    # No duplicate names across the ordered tuple.
    assert len(set(t.name for t in TRANSITIONS_ORDERED)) == 7


def test_02_default_transition_name_resolves_to_real_catalog_entry():
    from pipeline_v2.transitions import (
        DEFAULT_TRANSITION_NAME, TRANSITIONS, SMART_CUT,
    )
    assert DEFAULT_TRANSITION_NAME == "smart_cut"
    assert DEFAULT_TRANSITION_NAME in TRANSITIONS
    assert TRANSITIONS[DEFAULT_TRANSITION_NAME] is SMART_CUT


def test_03_smart_cut_is_implemented_with_zero_duration():
    """Smart cut is a hard cut: no overlap window, no re-encode cost,
    fully implemented today. The other six are reserved slots."""
    from pipeline_v2.transitions import SMART_CUT
    assert SMART_CUT.implemented is True
    assert SMART_CUT.duration_s == 0.0
    assert SMART_CUT.name == "smart_cut"
    assert SMART_CUT.display_name == "Smart Cut"
    # description is non-empty (used by the UI tooltip).
    assert isinstance(SMART_CUT.description, str)
    assert SMART_CUT.description.strip()


# ----- Lookup helpers --------------------------------------------------


def test_04_get_transition_returns_named_entry():
    from pipeline_v2.transitions import (
        get_transition, SMART_CUT, CROSSFADE, DISSOLVE,
    )
    assert get_transition("smart_cut") is SMART_CUT
    assert get_transition("crossfade") is CROSSFADE
    assert get_transition("dissolve") is DISSOLVE


def test_05_get_transition_unknown_falls_back_to_default():
    from pipeline_v2.transitions import get_transition, SMART_CUT
    assert get_transition("not_a_transition") is SMART_CUT
    assert get_transition("xyz") is SMART_CUT
    # Catalog typos with similar names also fall back -- no fuzzy match.
    assert get_transition("smartcut") is SMART_CUT


def test_06_get_transition_blank_inputs_fall_back_to_default():
    """The DB column is nullable on pre-item-104 rows; blank/None
    must coerce silently to the default (otherwise legacy jobs would
    crash at re-render)."""
    from pipeline_v2.transitions import get_transition, SMART_CUT
    assert get_transition(None) is SMART_CUT
    assert get_transition("") is SMART_CUT


def test_07_is_valid_transition_accepts_only_catalog_names():
    """Used by the create-job endpoint to choose whether to coerce
    a stale-frontend value to the default. Must be strict (no
    fallback) -- that's the lookup's job."""
    from pipeline_v2.transitions import is_valid_transition
    assert is_valid_transition("smart_cut") is True
    assert is_valid_transition("crossfade") is True
    assert is_valid_transition("fade_to_black") is True
    assert is_valid_transition("dip_to_white") is True
    assert is_valid_transition("slide_left") is True
    assert is_valid_transition("wipe_right") is True
    assert is_valid_transition("dissolve") is True
    # Strict-mode: unknown rejected.
    assert is_valid_transition("not_a_transition") is False
    assert is_valid_transition("") is False


def test_08_resolve_for_render_falls_back_for_non_implemented():
    """The renderer-side resolver returns smart_cut whenever the
    selected transition is not yet ffmpeg-backed. This is distinct
    from get_transition which preserves the operator's choice for
    UI display."""
    from pipeline_v2.transitions import (
        resolve_for_render, get_transition, SMART_CUT,
    )
    # smart_cut is implemented -> resolve returns it.
    assert resolve_for_render("smart_cut") is SMART_CUT
    # Other six entries are not yet implemented -> resolve falls back.
    for name in ("crossfade", "fade_to_black", "dip_to_white",
                 "slide_left", "wipe_right", "dissolve"):
        sel = get_transition(name)
        assert sel.implemented is False
        assert resolve_for_render(name) is SMART_CUT
    # Unknown / blank also falls back to smart_cut (through
    # get_transition -> default -> SMART_CUT).
    assert resolve_for_render("xyz") is SMART_CUT
    assert resolve_for_render(None) is SMART_CUT


# ----- Plumbing --------------------------------------------------------


def test_09_stage4render_dataclass_exposes_transition_style_default():
    """Stage4Render must accept transition_style with the catalog
    default. The orchestrator constructs Stage4Render from event
    envelope data; a missing field would otherwise crash render."""
    from dataclasses import fields
    from pipeline_v2.stages.stage_4_render import Stage4Render
    f = next(
        f for f in fields(Stage4Render)
        if f.name == "transition_style"
    )
    assert f.default == "smart_cut"


def test_10_runner_event_payload_includes_transition_style():
    """Smoke-test that the event payload helper actually puts
    transition_style on the wire. We monkey-patch the inngest client
    so no real event is sent; we just want to inspect event_data."""
    import runner as _runner
    captured = {}

    class _StubEvent:
        def __init__(self, name=None, data=None, id=None):
            captured["name"] = name
            captured["data"] = data
            captured["id"] = id

    class _StubClient:
        def send_sync(self, events):
            # send_sync receives the Event instance -- just record it
            # so the test can assert against the captured data dict.
            captured["sent"] = events

    # Monkey-patch the lazy imports inside _dispatch_v2_inngest_event.
    import inngest as _inngest_mod
    import pipeline_v2.inngest_client as _ic_mod
    orig_event = _inngest_mod.Event
    orig_get_client = _ic_mod.get_client
    _inngest_mod.Event = _StubEvent
    _ic_mod.get_client = lambda: _StubClient()

    # Stub the DB write so we don't need an active SessionLocal.
    class _StubDbSession:
        def query(self, *_a, **_kw): return self
        def filter(self, *_a, **_kw): return self
        def update(self, *_a, **_kw): return None
        def commit(self): return None
        def close(self): return None
    db_factory = lambda: _StubDbSession()

    try:
        _runner._dispatch_v2_inngest_event(
            job_id=42,
            video_path="/tmp/test.mp4",
            language="te",
            platform="full_video_shorts_v2",
            frame="torn_card",
            stt_provider="deepgram",
            db_session_factory=db_factory,
            transition_style="crossfade",
        )
    finally:
        _inngest_mod.Event = orig_event
        _ic_mod.get_client = orig_get_client

    assert captured["name"] == "video/v2/uploaded"
    assert captured["data"]["transition_style"] == "crossfade"
    assert captured["data"]["job_id"] == 42


def test_11_create_job_endpoint_has_transition_style_form_field():
    """Wire-level lock: introspect the FastAPI handler signature to
    confirm transition_style is a Form() field defaulting to
    smart_cut. Catches a future refactor that drops the field or
    changes its default."""
    import inspect
    from main import create_job
    sig = inspect.signature(create_job)
    assert "transition_style" in sig.parameters
    param = sig.parameters["transition_style"]
    # The default for Form(default) is wrapped in a FieldInfo-like
    # object -- read .default off it. Falls back to str() on
    # opaque types.
    raw_default = getattr(param.default, "default", param.default)
    assert raw_default == "smart_cut"
