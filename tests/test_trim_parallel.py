"""Parallel two-chunk trim (job 611: the 28.5-min single-pass trim encode
took 6m48s and was decode/filter-bound — preset levers moved nothing, so
the fix is encoding two contiguous span-halves CONCURRENTLY and joining
losslessly). These tests pin the split balance, the chunk/join command
shapes, the short-source bypass, and the single-pass fallback."""
import os

import pytest

from pipeline_v4 import trim_engine as te
from pipeline_v4.trim_engine import KeptSpan


def _spans(*durs, start=0.0):
    out, t = [], start
    for d in durs:
        out.append(KeptSpan(start_sec=t, end_sec=t + d))
        t += d + 1.0
    return out


# ── _split_spans_balanced ──────────────────────────────────────────

def test_split_balances_duration_and_preserves_order():
    spans = _spans(100, 100, 100, 100)
    a, b = te._split_spans_balanced(spans)
    assert len(a) == 2 and len(b) == 2
    assert a + b == spans                        # contiguous, ordered
    assert abs(sum(s.duration for s in a)
               - sum(s.duration for s in b)) < 1.0


def test_split_never_empty_even_when_first_span_dominates():
    spans = _spans(1000, 5)
    a, b = te._split_spans_balanced(spans)
    assert a and b                               # both non-empty
    assert a + b == spans


def test_split_two_spans():
    spans = _spans(30, 500)
    a, b = te._split_spans_balanced(spans)
    assert len(a) == 1 and len(b) == 1


# ── atomic_trim_concat routing ─────────────────────────────────────

def _fake_run_factory(calls, fail_labels=()):
    def fake_run(cmd, timeout=0, log_label=""):
        calls.append({"cmd": list(cmd), "label": log_label})
        if log_label in fail_labels:
            raise RuntimeError(f"boom in {log_label}")
        # materialize the output file (last arg) like ffmpeg would
        out = cmd[-1]
        if not str(out).endswith(".txt"):
            with open(out, "wb") as fh:
                fh.write(b"x" * 2048)
    return fake_run


def test_long_source_uses_two_chunks_and_join(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_TRIM_PARALLEL", "1")   # opt-in
    calls = []
    monkeypatch.setattr(te, "_run_ffmpeg", _fake_run_factory(calls))
    monkeypatch.setattr(te, "_bad_color_trc", lambda p: False)
    out = str(tmp_path / "trimmed.mp4")
    te.atomic_trim_concat(source_video="src.mp4",
                          spans=_spans(400, 400), output_path=out)
    labels = [c["label"] for c in calls]
    assert sorted(labels[:2]) == ["trim_chunk_0", "trim_chunk_1"]
    assert labels[-1] == "trim_chunk_join"
    join = calls[-1]["cmd"]
    assert "-c" in join and join[join.index("-c") + 1] == "copy"
    assert os.path.isfile(out)
    # chunk temps cleaned up
    assert not os.path.isfile(out + ".chunk0.mp4")
    assert not os.path.isfile(out + ".chunk1.mp4")
    assert not os.path.isfile(out + ".concat.txt")


def test_short_source_keeps_single_pass(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(te, "_run_ffmpeg", _fake_run_factory(calls))
    monkeypatch.setattr(te, "_bad_color_trc", lambda p: False)
    out = str(tmp_path / "short.mp4")
    te.atomic_trim_concat(source_video="src.mp4",
                          spans=_spans(30, 20), output_path=out)
    assert [c["label"] for c in calls] == ["atomic_trim_concat"]


def test_default_is_single_pass_even_for_long_sources(tmp_path, monkeypatch):
    # Chunking is OPT-IN: job 613 measured it WORSE (11m32s vs 10m09s) on
    # this single-NVENC-engine GPU — splitting a shared encoder/decoder
    # adds contention. Unset env must mean the single pass.
    monkeypatch.delenv("KAIZER_V4_TRIM_PARALLEL", raising=False)
    calls = []
    monkeypatch.setattr(te, "_run_ffmpeg", _fake_run_factory(calls))
    monkeypatch.setattr(te, "_bad_color_trc", lambda p: False)
    out = str(tmp_path / "kill.mp4")
    te.atomic_trim_concat(source_video="src.mp4",
                          spans=_spans(400, 400), output_path=out)
    assert [c["label"] for c in calls] == ["atomic_trim_concat"]


def test_chunk_failure_falls_back_to_single_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_TRIM_PARALLEL", "1")   # opt-in
    calls = []
    monkeypatch.setattr(
        te, "_run_ffmpeg",
        _fake_run_factory(calls, fail_labels=("trim_chunk_1",)))
    monkeypatch.setattr(te, "_bad_color_trc", lambda p: False)
    out = str(tmp_path / "fb.mp4")
    te.atomic_trim_concat(source_video="src.mp4",
                          spans=_spans(400, 400), output_path=out)
    # chunks attempted, then the single pass rescued the render
    assert calls[-1]["label"] == "atomic_trim_concat"
    assert os.path.isfile(out)
    assert not os.path.isfile(out + ".chunk0.mp4")   # temps cleaned


def test_chunk_cmds_carry_same_encoder_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_TRIM_PARALLEL", "1")   # opt-in
    calls = []
    monkeypatch.setattr(te, "_run_ffmpeg", _fake_run_factory(calls))
    monkeypatch.setattr(te, "_bad_color_trc", lambda p: False)
    out = str(tmp_path / "enc.mp4")
    te.atomic_trim_concat(source_video="src.mp4",
                          spans=_spans(400, 400), output_path=out)
    single = te._trim_cmd(source_video="src.mp4", spans=_spans(400, 400),
                          output_path=out, ffmpeg_bin="ffmpeg", norm=False)
    enc_tail = single[single.index("-map"):]  # everything after inputs
    for c in calls[:2]:
        cmd = c["cmd"]
        # identical encoder/mux args (only filter graph + output differ)
        assert cmd[cmd.index("-pix_fmt"):][:8] == \
            enc_tail[enc_tail.index("-pix_fmt"):][:8]
