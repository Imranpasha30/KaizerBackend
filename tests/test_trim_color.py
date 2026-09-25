"""Invalid color-transfer normalization in the trim pass (jobs 601/602).

Browser-compressed uploads carried trc:reserved; ffmpeg 8's swscaler
refuses the nv12→yuv420p conversion with it (Error -129). The trim pass
probes the source and, ONLY for invalid tags, heads each video chain
with a metadata-only setparams rewrite to bt709 — valid sources keep
the exact legacy filter string.
"""
from types import SimpleNamespace

from pipeline_v4 import trim_engine as te
from pipeline_v4.trim_engine import KeptSpan, _build_filter_complex


def _span(a, b):
    return KeptSpan(start_sec=a, end_sec=b)


def test_legacy_filter_string_unchanged_without_flag():
    fc = _build_filter_complex([_span(1.0, 3.0)])
    assert fc.startswith("[0:v]trim=start=1.000:end=3.000,setpts=PTS-STARTPTS[v0]")
    assert "setparams" not in fc


def test_normalize_heads_every_video_chain():
    fc = _build_filter_complex([_span(1.0, 3.0), _span(5.0, 8.0)],
                               normalize_trc=True)
    assert fc.count("setparams=color_trc=bt709,trim=") == 2
    # audio chains stay untouched
    assert "setparams" not in fc.split(";")[1]


def _probe_result(monkeypatch, stdout):
    monkeypatch.setattr(te.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0,
                                                        stdout=stdout,
                                                        stderr=""))


def test_bad_trc_detects_reserved(monkeypatch, tmp_path):
    _probe_result(monkeypatch, "reserved\n")
    assert te._bad_color_trc(str(tmp_path / "x.mp4")) is True
    _probe_result(monkeypatch, "unknown\n")
    assert te._bad_color_trc(str(tmp_path / "x.mp4")) is True


def test_bad_trc_accepts_valid_and_fails_soft(monkeypatch, tmp_path):
    _probe_result(monkeypatch, "bt709\n")
    assert te._bad_color_trc(str(tmp_path / "x.mp4")) is False
    _probe_result(monkeypatch, "")
    assert te._bad_color_trc(str(tmp_path / "x.mp4")) is False

    def boom(*a, **k):
        raise OSError("no ffprobe")

    monkeypatch.setattr(te.subprocess, "run", boom)
    assert te._bad_color_trc(str(tmp_path / "x.mp4")) is False
