"""corpus._loads_lenient — truncated-JSON repair so a Gemini response cut off
at the output cap loses only its tail instead of failing the whole corpus
refresh (the study-channel public-data learn path depends on this)."""
import json

import pytest

from learning.corpus import _loads_lenient


def test_clean_json_parses():
    assert _loads_lenient('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_strips_code_fences():
    assert _loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_truncated_array_tail_recovered():
    # Cut off mid-string inside the last array element — the earlier, complete
    # structure must still parse (losing only the truncated tail).
    truncated = '{"power_words": ["LIVE", "హై టెన్షన్"], "hooks": ["a", "b'
    out = _loads_lenient(truncated)
    assert out["power_words"] == ["LIVE", "హై టెన్షన్"]


def test_unrecoverable_raises():
    with pytest.raises(json.JSONDecodeError):
        _loads_lenient("not json at all {{{")
