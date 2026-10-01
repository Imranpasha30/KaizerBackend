"""Three faults the operator found on a real broadcast.

1. A VIDEO-ONLY FILE WAS CALLED STREAM-READY. The checker appended a warning --
   "no audio track; YouTube accepts it but viewers get silence" -- and let the
   file through. For YouTube LIVE that is wrong: ingest does not come up on a
   video-only stream, enableAutoStart never fires, and the broadcast sits in
   `ready` forever while everything on our side reads healthy. Both of that
   day's test uploads were silent.

   And the repair path could not have saved it: `-map 0:a:0?` makes the audio
   mapping OPTIONAL, so re-encoding a silent file produced another silent file.

2. CONFIRMATION GAVE UP PERMANENTLY. `_confirm` returned early unless
   youtube == "pending", and the give-up branch set it to "not_live" -- the same
   value YouTube's own "complete". So a broadcast that went live a few minutes
   after the first checks was displayed as broken for its entire run.

3. THE HEALTH SERVER SHOUTED. Every probe that hung up printed a ten-line
   traceback into the log kept for relay errors.
"""
import subprocess

import pytest

from kaizer_live import checker, ffcmd
from kaizer_live.config import Settings


# ───────────────────────── 1 · silent audio ─────────────────────────

def _probe(has_audio: bool):
    streams = [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720,
                "pix_fmt": "yuv420p", "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
                "bit_rate": "2000000"}]
    if has_audio:
        streams.append({"codec_type": "audio", "codec_name": "aac",
                        "sample_rate": "48000", "channels": 2})
    return {"streams": streams,
            "format": {"format_name": "mov,mp4,m4a", "duration": "60", "bit_rate": "2000000"}}


def test_a_video_only_file_is_not_stream_ready():
    """The exact shape of the uploads that could never go live."""
    v = checker.evaluate(_probe(False), [0.0, 2.0, 4.0], True)
    assert v.action != "none", "a silent file was called stream-ready"
    assert any("audio" in r.lower() for r in v.reasons), v.reasons


def test_a_file_with_audio_is_untouched_by_this_rule():
    v = checker.evaluate(_probe(True), [0.0, 2.0, 4.0], True)
    assert not any("no audio track" in r for r in v.reasons), v.reasons


def test_the_repair_synthesises_audio_rather_than_mapping_one_that_is_absent():
    """`-map 0:a:0?` is optional: without anullsrc the 'fixed' file is silent too."""
    cmd = ffcmd.normalize("ffmpeg", "in.mp4", "out.mp4", silent_audio=True)
    j = " ".join(cmd)
    assert "anullsrc=r=48000:cl=stereo" in j, j
    assert "-map 1:a:0" in j, "the synthesised track is not mapped"
    assert "-shortest" in j, "anullsrc is infinite; without -shortest the output never ends"
    assert "-c:a aac" in j


def test_a_file_that_has_audio_keeps_the_optional_mapping():
    j = " ".join(ffcmd.normalize("ffmpeg", "in.mp4", "out.mp4"))
    assert "anullsrc" not in j
    assert "-map 0:a:0?" in j


def test_has_audio_treats_an_unreadable_file_as_having_audio():
    """Unknown must not mean 'synthesise': overwriting a real track with silence
    would break a file that was fine."""
    assert ffcmd.has_audio("definitely-not-ffprobe", "nope.mp4") is True


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg")
                    or not __import__("shutil").which("ffprobe"),
                    reason="needs ffmpeg + ffprobe on PATH")
def test_end_to_end_a_silent_file_comes_out_with_audio(tmp_path):
    """The whole point, against real ffmpeg."""
    src = tmp_path / "silent.mp4"
    out = tmp_path / "fixed.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    assert ffcmd.has_audio("ffprobe", str(src)) is False
    subprocess.run(ffcmd.normalize("ffmpeg", str(src), str(out), video=False,
                                   silent_audio=True), check=True)
    assert ffcmd.has_audio("ffprobe", str(out)) is True
    def dur(p):
        return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                     "format=duration", "-of", "csv=p=0", str(p)],
                                    capture_output=True, text=True).stdout.strip())
    assert abs(dur(out) - dur(src)) < 1.0, "-shortest did not bound the output"


# ──────────────────── 2 · confirmation must retry ────────────────────

def test_giving_up_is_recorded_as_unconfirmed_not_as_not_live():
    """They are different facts: one says YouTube ended it, the other says we
    stopped asking."""
    import inspect

    from kaizer_live import service

    src = inspect.getsource(service.LiveService._confirm)
    assert '"unconfirmed"' in src, "giving up still records not_live"
    assert 'yt_state not in ("pending", "unconfirmed")' in src, (
        "a channel we gave up on is still never re-examined")


def test_a_recheck_is_bounded_and_paid_for():
    """Unbounded re-asking would drain the day's quota on a stuck broadcast."""
    import inspect

    from kaizer_live import service

    src = inspect.getsource(service.LiveService._confirm)
    assert "confirm_recheck_max" in src, "the recheck has no ceiling"
    assert "confirm_recheck_after_s" in src, "the recheck has no backoff"
    assert "spend_extra_check" in src, "a recheck is 1 unit and must go through the ledger"


def test_the_recheck_settings_exist_and_are_sane():
    s = Settings(fernet_key="x")
    assert s.confirm_recheck_after_s >= 60, s.confirm_recheck_after_s
    assert 1 <= s.confirm_recheck_max <= 10, s.confirm_recheck_max


def test_youtubes_own_terminal_answer_is_still_terminal():
    """complete/revoked must NOT be retried — that broadcast is over."""
    import inspect

    from kaizer_live import service

    src = inspect.getsource(service.LiveService._confirm)
    assert '("complete", "revoked")' in src
    assert src.count('spec["youtube"] = "not_live"') >= 1


# ─────────────────── 3 · the health server is quiet ───────────────────

def test_a_probe_hanging_up_does_not_print_a_traceback():
    import inspect

    from kaizer_live import health

    src = inspect.getsource(health)
    assert "handle_error" in src, "socketserver will still dump a traceback per disconnect"
    for exc in ("ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError"):
        assert exc in src, f"{exc} is not treated as a normal disconnect"


def test_a_real_handler_error_is_still_reported():
    """Silencing disconnects must not silence genuine faults."""
    import inspect

    from kaizer_live import health

    src = inspect.getsource(health)
    assert "[health] handler error" in src, "real errors became invisible"
