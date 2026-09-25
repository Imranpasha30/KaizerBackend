"""run_ffmpeg OOM rung (job 600): a filtergraph that dies on memory is
retried single-threaded (-filter_complex_threads 1) with a stretched
timeout — ffmpeg 8's threaded executor queues frames per link, so huge
graphs OOM even with plenty of free RAM."""
from types import SimpleNamespace

from pipeline_v4 import ffmpeg_exec as fx


def _fake_runs(monkeypatch, results):
    calls = []

    def fake_run(cmd, **kw):
        calls.append({"cmd": list(cmd), "timeout": kw.get("timeout")})
        rc, err = results[min(len(calls) - 1, len(results) - 1)]
        return SimpleNamespace(returncode=rc, stderr=err, stdout="")

    monkeypatch.setattr(fx.subprocess, "run", fake_run)
    return calls


BASE = ["ffmpeg", "-y", "-i", "in.mp4",
        "-filter_complex", "[0:v]scale=64:64[v]",
        "-map", "[v]", "-c:v", "h264_nvenc", "out.mp4"]

OOM = "[fc#0 @ 0x1] Error while filtering: Cannot allocate memory\n" \
      "[fc#0 @ 0x1] Task finished with error code: -12 (Cannot allocate memory)"


def test_oom_retries_single_threaded_with_stretched_timeout(monkeypatch):
    calls = _fake_runs(monkeypatch, [(1, OOM), (0, "")])
    fx.run_ffmpeg(list(BASE), timeout=900, log_label="t", retry=1)
    assert len(calls) == 2
    c2 = calls[1]["cmd"]
    i = c2.index("-filter_complex_threads")
    assert c2[i + 1] == "1" and i < c2.index("-filter_complex")
    assert calls[1]["timeout"] == 3600            # 900 * 4
    # the rung keeps GPU encode — no CPU rewrite on an OOM
    assert "h264_nvenc" in c2


def test_oom_rung_applied_once_then_cascades(monkeypatch):
    calls = _fake_runs(monkeypatch, [(1, OOM), (1, OOM), (0, "")])
    fx.run_ffmpeg(list(BASE), timeout=100, log_label="t", retry=2)
    assert len(calls) == 3
    # attempt 3 must NOT stack a second threads flag — it cascades to the
    # CPU-rewrite rung instead
    assert calls[2]["cmd"].count("-filter_complex_threads") == 1
    assert "libx264" in calls[2]["cmd"]


BASE_CAPPED = ["ffmpeg", "-y", "-i", "in.mp4",
               "-filter_complex_threads", "6",
               "-filter_complex", "[0:v]scale=64:64[v]",
               "-map", "[v]", "-c:v", "h264_nvenc", "out.mp4"]


def test_oom_lowers_proactive_cap_to_one(monkeypatch):
    # callers now set a moderate proactive cap (6) — OOM must REWRITE it to
    # 1, not add a second flag.
    calls = _fake_runs(monkeypatch, [(1, OOM), (0, "")])
    fx.run_ffmpeg(list(BASE_CAPPED), timeout=900, log_label="t", retry=1)
    c2 = calls[1]["cmd"]
    assert c2.count("-filter_complex_threads") == 1
    i = c2.index("-filter_complex_threads")
    assert c2[i + 1] == "1"
    assert calls[1]["timeout"] == 3600


def test_oom_capped_then_cascades_to_cpu(monkeypatch):
    # cap→1 on the first OOM, then (already at 1) cascade to the CPU rewrite.
    calls = _fake_runs(monkeypatch, [(1, OOM), (1, OOM), (0, "")])
    fx.run_ffmpeg(list(BASE_CAPPED), timeout=100, log_label="t", retry=2)
    assert len(calls) == 3
    assert calls[2]["cmd"].count("-filter_complex_threads") == 1
    assert "libx264" in calls[2]["cmd"]


def test_oom_without_filter_complex_falls_through(monkeypatch):
    cmd = ["ffmpeg", "-y", "-i", "a.mp4", "-c:v", "h264_nvenc", "o.mp4"]
    calls = _fake_runs(monkeypatch, [(1, OOM), (0, "")])
    fx.run_ffmpeg(cmd, timeout=60, log_label="t", retry=1)
    assert len(calls) == 2
    assert "-filter_complex_threads" not in calls[1]["cmd"]
    assert "libx264" in calls[1]["cmd"]           # rung 3 CPU fallback
