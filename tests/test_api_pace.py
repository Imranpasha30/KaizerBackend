"""api_pace: burst-prone Gemini calls get spaced request starts instead
of colliding with Vertex's per-minute quota (job 611: 16-story job spent
~13min in 429 retry-sleeps and fell back to heuristics)."""
import importlib

import pipeline_v4.api_pace as ap


def _fresh(monkeypatch, rpm):
    monkeypatch.setenv("KAIZER_GEMINI_RPM", str(rpm))
    importlib.reload(ap)          # reset the module clock between tests
    slept = []
    monkeypatch.setattr(ap.time, "sleep", lambda s: slept.append(s))
    return slept


def test_calls_are_spaced_by_gap(monkeypatch):
    slept = _fresh(monkeypatch, 60)          # gap = 1s
    w1 = ap.pace()                            # first call: no wait
    w2 = ap.pace()                            # ~1s behind
    w3 = ap.pace()                            # ~2s behind
    assert w1 == 0.0
    assert 0.9 <= w2 <= 1.1
    assert 1.8 <= w3 <= 2.2
    assert slept == [w2, w3]                  # only the waiters slept


def test_zero_rpm_disables_pacing(monkeypatch):
    slept = _fresh(monkeypatch, 0)
    assert ap.pace() == 0.0
    assert ap.pace() == 0.0
    assert slept == []


def test_bad_env_falls_back_to_default(monkeypatch):
    slept = _fresh(monkeypatch, 10)
    monkeypatch.setenv("KAIZER_GEMINI_RPM", "garbage")
    assert ap._rpm() == 10.0
