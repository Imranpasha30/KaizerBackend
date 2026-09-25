"""Guard the NVENC ``-tune`` default.

The trim/shorts intermediate encode opts into ``-tune ll`` (faster, skips
NVENC's HQ rate-control analysis) — safe ONLY because that asset is always
re-encoded downstream. Every FINAL composite call site must stay on ``-tune
hq``. That safety rests entirely on the DEFAULT of video_encoder_args being
``hq``: no v1_bridge composite passes ``tune=``, so a plumbing slip that
flipped the default would silently degrade every deliverable. These tests
lock the default.
"""
import pytest

from pipeline_v4 import encoder as enc


def _nv(monkeypatch):
    monkeypatch.setenv("KAIZER_FORCE_ENCODER", "h264_nvenc")


def _after(args, flag):
    return args[args.index(flag) + 1]


def test_default_tune_is_hq(monkeypatch):
    _nv(monkeypatch)
    args = enc.video_encoder_args(crf=20, preset_hint="medium")
    assert "-c:v" in args and _after(args, "-c:v") == "h264_nvenc"
    assert _after(args, "-tune") == "hq"   # every FINAL composite relies on this


def test_explicit_ll_only_when_asked(monkeypatch):
    _nv(monkeypatch)
    assert _after(enc.video_encoder_args(tune="ll"), "-tune") == "ll"
    assert _after(enc.video_encoder_args(tune="ull"), "-tune") == "ull"


def test_empty_tune_falls_back_to_hq(monkeypatch):
    _nv(monkeypatch)
    assert _after(enc.video_encoder_args(tune=""), "-tune") == "hq"
    assert _after(enc.video_encoder_args(tune=None), "-tune") == "hq"


def test_veryfast_maps_to_p1(monkeypatch):
    _nv(monkeypatch)
    assert _after(enc.video_encoder_args(preset_hint="veryfast"), "-preset") == "p1"


def test_libx264_fallback_has_no_tune(monkeypatch):
    # The CPU fallback never emits -tune, so the new param can't corrupt it.
    monkeypatch.setenv("KAIZER_FORCE_ENCODER", "libx264")
    args = enc.video_encoder_args(crf=19, preset_hint="veryfast", tune="ll")
    assert _after(args, "-c:v") == "libx264"
    assert "-tune" not in args


# ── decoder policy: SOFTWARE decode by default ─────────────────────
# hwaccel-cuda's per-frame GPU->CPU transfer capped decode at ~134fps vs
# ~644fps software on the real job-611 source (probe in encoder.py's
# docstring) — it throttled the trim AND every compose. cuda is opt-in.

def test_decoder_default_is_software(monkeypatch):
    monkeypatch.delenv("KAIZER_VIDEO_DECODER", raising=False)
    monkeypatch.setenv("KAIZER_FORCE_ENCODER", "h264_nvenc")
    assert enc.video_decoder_args() == []     # even with NVENC active


def test_decoder_cuda_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("KAIZER_VIDEO_DECODER", "cuda")
    assert enc.video_decoder_args() == ["-hwaccel", "cuda"]
    monkeypatch.setenv("KAIZER_VIDEO_DECODER", "cpu")
    assert enc.video_decoder_args() == []
