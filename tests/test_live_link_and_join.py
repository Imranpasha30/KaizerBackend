"""A link is one file for every channel, and open-vs-join is decided once.

Two production failures on 2026-10-02, both invisible until a seven-channel batch ran:

  * a YouTube link was handed to ffprobe as if it were a file, so every channel failed with
    "could not prepare the file";
  * start_through_engine fell back to go_live on ANY add_channel refusal, so the customer was
    shown "already live; add channels to it instead" rather than the real reason, and two
    channels that finished preparing together could both try to open the same video.

No Redis, no YouTube, no network: the engine is a fake that records what it was asked.
Runs under pytest, or directly:  python tests/test_live_link_and_join.py
"""
from __future__ import annotations

import threading
import types

import live_integration as li
from kaizer_live.service import LiveError


class FakeSvc:
    """Just enough of LiveService: a video is unknown until go_live, then 'live'."""

    def __init__(self, refuse_join: str = ""):
        self.state = None
        self.opened, self.joined = [], []
        self.refuse_join = refuse_join
        self._l = threading.Lock()

    def status(self, vid):
        if self.state is None:
            raise LiveError("no such video", 404)
        return {"state": self.state, "channels": []}

    def go_live(self, vid, user_id, source, channels, **kw):
        with self._l:
            if self.state == "live":
                raise LiveError(f"video {vid} is already live; add channels to it instead", 409)
            self.state = "live"
            self.opened.append((vid, source, [c.channel_id for c in channels]))
        return {"state": "live"}

    def add_channel(self, vid, req):
        if self.refuse_join:
            raise LiveError(self.refuse_join, 409)
        with self._l:
            self.joined.append(req.channel_id)
        return {"state": "on"}


class FakeDb:
    def query(self, _model):
        return types.SimpleNamespace(get=lambda _id: types.SimpleNamespace(yt_stream_id="s"))


def _row(channel_id, **kw):
    base = dict(batch_id=77, video_slot=0, channel_id=channel_id, user_id=2, title="t",
                description="", privacy="unlisted", thumbnail_path="", target_hours=1.0,
                source_url="https://youtu.be/abc", upload_path="/tmp/kaizer-live-studio/url-77-0.mp4")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _with_svc(svc):
    li._SERVICE_BACKUP = li.get_live_service
    li.get_live_service = lambda: svc


def _restore():
    li.get_live_service = li._SERVICE_BACKUP


def test_seven_channels_open_once_and_six_join():
    svc = FakeSvc()
    _with_svc(svc)
    try:
        errors = []

        def start(cid):
            try:
                li.start_through_engine(FakeDb(), _row(cid))
            except Exception as exc:               # pragma: no cover - the assertion reports it
                errors.append(exc)

        threads = [threading.Thread(target=start, args=(c,)) for c in range(1, 8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors, errors
        assert len(svc.opened) == 1, svc.opened
        assert len(svc.joined) == 6, svc.joined
    finally:
        _restore()


def test_the_engine_is_given_the_file_not_the_link():
    svc = FakeSvc()
    _with_svc(svc)
    try:
        li.start_through_engine(FakeDb(), _row(1))
        assert svc.opened[0][1] == "/tmp/kaizer-live-studio/url-77-0.mp4"
    finally:
        _restore()


def test_a_refused_join_reports_its_own_reason():
    svc = FakeSvc(refuse_join="channel 9 is already on this video")
    svc.state = "live"
    _with_svc(svc)
    try:
        try:
            li.start_through_engine(FakeDb(), _row(9))
        except LiveError as exc:
            assert "already on this video" in str(exc), str(exc)
            assert "add channels to it instead" not in str(exc)
        else:
            raise AssertionError("a refused join must raise")
    finally:
        _restore()


def test_one_fetched_file_per_engine_video():
    a = li.fetched_path_for(77, 0)
    assert a == li.fetched_path_for(77, 0)
    assert a != li.fetched_path_for(77, 1) and a != li.fetched_path_for(78, 0)
    assert a.endswith("url-77-0.mp4")


# ─── a link that is LIVE right now ────────────────────────────────────────────
import time as _t

MANIFEST = ("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/{exp}/ip/2a09:bac1::1/"
            "id/abc.1/itag/95/file/index.m3u8")


def _fmt(fid, height, **kw):
    base = dict(format_id=str(fid), protocol="m3u8_native", vcodec="avc1", acodec="mp4a",
                height=height, tbr=height, url=MANIFEST.format(exp=int(_t.time()) + 21600) + f"#{fid}")
    base.update(kw)
    return base


def test_pick_hls_prefers_combined_up_to_720p():
    fmts = [_fmt(93, 360), _fmt(94, 480), _fmt(95, 720), _fmt(96, 1080),
            _fmt(301, 720, acodec="none"), _fmt(140, 0, vcodec="none")]
    assert li.pick_hls(fmts)["format_id"] == "95"
    assert li.pick_hls([_fmt(96, 1080)])["format_id"] == "96"          # nothing smaller exists
    assert li.pick_hls([_fmt(301, 720, acodec="none")]) is None        # no combined stream at all


def test_classifying_a_link():
    live = li.info_to_link({"live_status": "is_live", "title": "Press meet", "formats": [_fmt(95, 720)]})
    assert live["manifest"].startswith("https://manifest.googlevideo.com/")
    assert 21000 < live["expires_at"] - _t.time() <= 21600
    assert li.info_to_link({"live_status": "not_live", "formats": [_fmt(95, 720)]}) is None
    assert li.info_to_link({"live_status": "was_live"}) is None
    for status, word in (("is_upcoming", "not started"), ("post_live", "processing")):
        try:
            li.info_to_link({"live_status": status, "title": "t"})
        except li.NotStreamReady as exc:
            assert word in str(exc), str(exc)
        else:
            raise AssertionError(status)


def test_a_live_manifest_is_recognised_and_a_file_or_page_is_not():
    assert li.is_live_manifest(MANIFEST.format(exp=1))
    assert not li.is_live_manifest("/tmp/kaizer-live-studio/stream-1.mp4")
    assert not li.is_live_manifest("https://youtu.be/abc")
    assert not li.is_live_manifest("https://evil.example/googlevideo.com/x")   # host, not substring


def test_a_live_relay_runs_until_the_source_ends_not_for_live_hours():
    rec = {"expires_at": _t.time() + 3 * 3600, "manifest": "m", "title": "t"}
    h, note = li.live_relay_hours(rec)
    assert h == float(li.MAX_LIVE_HOURS) and "until the source broadcast ends" in note
    # an address with 3 hours left is NOT a reason to stop at 3 hours: the wrapper renews it
    assert h > 100
    try:
        li.live_relay_hours({"expires_at": _t.time() + 60, "manifest": "m", "title": "t"})
    except li.NotStreamReady:
        pass
    else:
        raise AssertionError("an address about to expire must be refused at start")


def test_a_live_link_is_relayed_not_looped_and_every_channel_joins():
    class Svc(FakeSvc):
        def go_live(self, vid, user_id, source, channels, **kw):
            self.kw = kw
            return super().go_live(vid, user_id, source, channels, **kw)

    svc = Svc()
    rec = {"expires_at": _t.time() + 21600, "manifest": MANIFEST.format(exp=1), "title": "Press meet"}
    saved = li.live_link_for
    li.live_link_for = lambda vid: dict(rec)
    _with_svc(svc)
    try:
        errors = []

        def start(cid):
            try:
                li.start_through_engine(FakeDb(), _row(cid, upload_path="", target_hours=1.0))
            except Exception as exc:                      # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=start, args=(c,)) for c in range(1, 8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors, errors
        assert len(svc.opened) == 1 and len(svc.joined) == 6
        assert svc.opened[0][1] == rec["manifest"], "the engine must be given the live address"
        assert svc.kw["loop"] is False, "a live source must not be looped"
        assert svc.kw["duration_s"] == float(li.MAX_LIVE_HOURS) * 3600.0, "live hours must not cap a live relay"
    finally:
        li.live_link_for = saved
        _restore()


def test_the_address_is_minted_once_for_a_batch():
    calls = []
    saved = li.probe_link

    def fake_probe(url):
        calls.append(url)
        return {"manifest": MANIFEST.format(exp=int(_t.time()) + 21600), "title": "t",
                "expires_at": _t.time() + 21600}

    li.probe_link = fake_probe
    li._LIVE_LINKS.pop("77-5", None)
    try:
        a = li.resolve_live_link("https://youtu.be/x", "77-5")
        b = li.resolve_live_link("https://youtu.be/x", "77-5")
        assert a["manifest"] == b["manifest"] and len(calls) == 1
    finally:
        li.probe_link = saved
        li._LIVE_LINKS.pop("77-5", None)


# ─── a channel YouTube refused must not be reported as streaming ────────────────
REASON = "YouTube broadcast start failed: Failed to refresh token for channel 5 (reconnect from the Channels page)"


def test_a_refused_first_channel_is_reported_not_swallowed():
    class Refusing(FakeSvc):
        def go_live(self, vid, user_id, source, channels, **kw):
            super().go_live(vid, user_id, source, channels, **kw)
            return {"state": "live", "channels": [{"channel_id": str(channels[0].channel_id),
                                                   "state": "failed", "error": REASON}]}

    svc = Refusing()
    _with_svc(svc)
    try:
        try:
            li.start_through_engine(FakeDb(), _row(5))
        except RuntimeError as exc:
            assert "reconnect from the Channels page" in str(exc), str(exc)
        else:
            raise AssertionError("a refused channel must raise, or the row says 'streaming'")
    finally:
        _restore()


def test_a_refused_joining_channel_is_reported():
    class Refusing(FakeSvc):
        def add_channel(self, vid, req):
            return {"channel_id": req.channel_id, "state": "failed", "error": REASON}

    svc = Refusing()
    svc.state = "live"
    _with_svc(svc)
    try:
        try:
            li.start_through_engine(FakeDb(), _row(6))
        except RuntimeError as exc:
            assert "Failed to refresh token" in str(exc)
        else:
            raise AssertionError("a refused join must raise")
    finally:
        _restore()


def test_only_the_failed_channel_is_blamed():
    out = {"state": "live", "channels": [{"channel_id": "1", "state": "on"},
                                         {"channel_id": "2", "state": "failed", "error": "e2"}]}
    assert li.engine_refusal(out, 1) == ""
    assert li.engine_refusal(out, 2) == "e2"
    assert li.engine_refusal({"state": "queued"}, 1) == ""
    assert li.engine_refusal(None, 1) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
