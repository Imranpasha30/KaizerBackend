"""_loads_lenient repair tiers (trim_engine cut-plan JSON robustness).

Job #500: trailing commas / fences. Job #599: Gemini dropped a comma
between two members and the model self-repair ALSO failed — the new
missing-comma-at-structural-newline tier fixes that class locally.
A literal newline can never occur inside a valid JSON string, so
token-end + newline + token-start is always a member boundary.
"""
import json

import pytest

from pipeline_v4.trim_engine import _loads_lenient


def test_strict_json_untouched():
    assert _loads_lenient('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_code_fences_and_prose():
    raw = 'Here you go:\n```json\n{"a": 1}\n```'
    assert _loads_lenient(raw) == {"a": 1}


def test_trailing_comma():
    assert _loads_lenient('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


def test_missing_comma_between_string_members():
    # the job#599 shape: value string, newline, next key — no comma
    raw = '{\n"title": "some text"\n"summary": "more text"\n}'
    assert _loads_lenient(raw) == {"title": "some text", "summary": "more text"}


def test_missing_comma_between_objects_in_array():
    raw = '{"stories": [\n{"i": 1}\n{"i": 2}\n]}'
    assert _loads_lenient(raw) == {"stories": [{"i": 1}, {"i": 2}]}


def test_missing_comma_after_number_and_bool():
    raw = '{\n"n": 5\n"ok": true\n"label": "x"\n}'
    assert _loads_lenient(raw) == {"n": 5, "ok": True, "label": "x"}


def test_valid_json_with_commas_never_double_repaired():
    raw = '{\n"a": "x",\n"b": [1, 2],\n"c": {"d": true}\n}'
    assert _loads_lenient(raw) == {"a": "x", "b": [1, 2], "c": {"d": True}}


def test_closing_then_closing_not_touched():
    # }\n] is legal — must NOT gain a comma
    raw = '{"a": [\n{"b": 1}\n]\n}'
    assert _loads_lenient(raw) == {"a": [{"b": 1}]}


def test_escaped_quotes_inside_strings_survive():
    raw = '{\n"a": "he said \\"hi\\""\n"b": 2\n}'
    assert _loads_lenient(raw) == {"a": 'he said "hi"', "b": 2}


def test_missing_comma_after_indented_array_close():
    # the Director's job-603 shape: ] on its own line, then the next key —
    # the regex tiers missed this; the position-guided repair catches it.
    raw = ('{\n  "stories": [\n    {\n      "overlays": [\n'
           '        {"id": "x", "t": 1.5}\n      ]\n'
           '      "grade": "mono"\n    }\n  ]\n}')
    out = _loads_lenient(raw)
    assert out["stories"][0]["grade"] == "mono"
    assert out["stories"][0]["overlays"][0]["id"] == "x"


def test_missing_comma_same_line_members():
    assert _loads_lenient('{"a": 1 "b": 2}') == {"a": 1, "b": 2}


def test_multiple_missing_commas_across_array():
    raw = '{"xs": [1 2 3 4]}'
    assert _loads_lenient(raw) == {"xs": [1, 2, 3, 4]}


def test_missing_comma_between_story_objects():
    # the real Director shape: a top-level object whose "stories" array has
    # missing commas between the story objects (job 603's actual failure).
    raw = '{"stories": [{"index":0} {"index":1} {"index":2}]}'
    assert _loads_lenient(raw) == {
        "stories": [{"index": 0}, {"index": 1}, {"index": 2}]}


def test_hopeless_json_still_raises():
    with pytest.raises(json.JSONDecodeError):
        _loads_lenient('{"a": [1, 2')
    # a genuinely truncated string is NOT a comma error → still raises
    # (so the caller's model-repair retry / formula fallback engages)
    with pytest.raises(json.JSONDecodeError):
        _loads_lenient('{"a": "unterminated')
