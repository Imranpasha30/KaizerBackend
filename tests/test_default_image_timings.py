"""_default_image_timings: the heuristic fallback must not explode one
image into many windows (job 608 root cause: 1 image on a 96s story
became 24 windows → a 24-layer carousel)."""
from pipeline_v4.orchestrator import _default_image_timings


def test_single_image_is_one_window():
    out = _default_image_timings(96.0, 1)
    assert out == [(0, 0.0, 96.0)]          # was 24 windows before


def test_empty_pool_or_zero_duration():
    assert _default_image_timings(96.0, 0) == []
    assert _default_image_timings(0.0, 3) == []


def test_multi_image_reasonable_count_and_cover():
    out = _default_image_timings(96.0, 6)
    # each image shown once..twice → 6..12 windows, never the old 24
    assert 6 <= len(out) <= 12
    # windows tile the whole story with no gaps/overlaps
    assert out[0][1] == 0.0
    assert abs(out[-1][2] - 96.0) < 0.01
    for (a, b) in zip(out, out[1:]):
        assert abs(a[2] - b[1]) < 0.01     # contiguous
    # every pool image appears at least once
    assert {e[0] for e in out} == set(range(6))


def test_never_exceeds_two_cycles():
    out = _default_image_timings(300.0, 3)   # long story, few images
    assert len(out) <= 3 * 2                 # at most 2 cycles


def test_dwell_env_tunable(monkeypatch):
    monkeypatch.setenv("KAIZER_V4_IMG_DWELL_S", "20")
    out = _default_image_timings(120.0, 4)
    # ~20s dwell → ~6 windows, clamped to [4, 8]
    assert 4 <= len(out) <= 8
