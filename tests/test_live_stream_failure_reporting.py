"""A failed live stream must say WHY -- to the log and to the operator.

Both defects these cover were real and both were found from a production
screenshot, because production is the one box we cannot attach a debugger to.

  1. The reason was written to Postgres and nowhere else. A hosted deployment's
     logs come from stdout, so the operator could see "failed" and had no way
     to learn what failed.

  2. The UI renders `message`, and a failure that passed only status+error left
     the PREVIOUS step's message on screen. A stream that failed at the OAuth
     step still read "queued - waiting for an available broadcast slot" -- a
     row confidently describing a state it had already left.

These run against a fake session rather than a database: `_update` opens its
own SessionLocal, and binding a real one in a test is how an earlier run in
this project ended up writing to DEV Postgres.
"""
import io
import contextlib

import pytest

from live_studio import orchestrator


class _Row:
    """Just enough LiveStream for _update to patch and read back."""
    def __init__(self, **kw):
        self.id = kw.get("id", 7)
        self.status = kw.get("status", "queued")
        self.error = kw.get("error")
        self.message = kw.get("message")
        self.channel_id = kw.get("channel_id", 42)
        self.batch_id = kw.get("batch_id", "C3FpvNqlsUg")
        self.source_url = kw.get("source_url")
        self.yt_broadcast_id = kw.get("yt_broadcast_id")


class _Query:
    def __init__(self, row): self._row = row
    def get(self, _id): return self._row


class _Session:
    def __init__(self, row): self._row = row; self.committed = False
    def query(self, _model): return _Query(self._row)
    def commit(self): self.committed = True
    def close(self): pass


@pytest.fixture
def patched(monkeypatch):
    """Return (row, run) where run(**fields) calls _update and captures stdout."""
    def _make(**row_kw):
        row = _Row(**row_kw)
        monkeypatch.setattr(orchestrator, "SessionLocal", lambda: _Session(row))

        def run(**fields):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                orchestrator._update(row.id, **fields)
            return buf.getvalue()
        return row, run
    return _make


# ── 1 · the reason reaches the log ───────────────────────────────────

def test_failure_prints_the_reason(patched):
    row, run = patched(message="queued — waiting for an available broadcast slot")
    out = run(status="failed", error="YouTube OAuth failed: invalid_grant")
    assert "[live_studio] STREAM 7 -> failed" in out
    assert "invalid_grant" in out, "the actual reason must be in the log line"


def test_the_log_line_is_one_line(patched):
    """Log collectors split on newlines; a reason spread over five lines is a
    reason nobody greps out. A traceback-shaped error must be flattened."""
    _, run = patched()
    out = run(status="failed", error="Traceback:\n  line one\n  line two\nBoom")
    assert len([l for l in out.splitlines() if l.strip()]) == 1
    assert "Boom" in out


def test_log_carries_the_identifiers_needed_to_find_the_row(patched):
    _, run = patched(channel_id=42, batch_id="C3FpvNqlsUg",
                     source_url="https://www.youtube.com/live/V_b7R6a5D_w")
    out = run(status="failed", error="nope")
    assert "channel=42" in out and "batch=C3FpvNqlsUg" in out
    assert "youtube.com/live/V_b7R6a5D_w" in out


def test_a_failure_with_no_error_still_logs(patched):
    """Silence here would be indistinguishable from the bug being fixed."""
    _, run = patched()
    out = run(status="failed")
    assert "STREAM 7 -> failed" in out and "(none recorded)" in out


def test_canceled_and_done_are_logged_too(patched):
    _, run = patched()
    assert "-> canceled" in run(status="canceled")
    assert "-> done" in run(status="done")


def test_ordinary_progress_updates_stay_quiet(patched):
    """Every step calls _update. Logging non-terminal ones would bury the
    terminal ones in noise, which is the problem we are fixing."""
    _, run = patched()
    assert run(status="live", message="pushing to YouTube") == ""
    assert run(progress=50) == ""


# ── 2 · the UI message stops describing a state it has left ──────────

def test_failure_replaces_the_stale_queue_message(patched):
    """THE PHOTOGRAPHED BUG: failed row, still reading 'queued - waiting...'."""
    row, run = patched(message="queued — waiting for an available broadcast slot")
    run(status="failed", error="YouTube OAuth failed: invalid_grant")
    assert "queued" not in row.message
    assert "invalid_grant" in row.message


def test_an_explicit_message_is_respected(patched):
    row, run = patched()
    run(status="failed", error="raw detail", message="couldn't reach YouTube")
    assert row.message == "couldn't reach YouTube"


def test_message_is_bounded(patched):
    row, run = patched()
    run(status="failed", error="x" * 5000)
    assert len(row.message) <= 512


def test_a_successful_finish_keeps_its_own_message(patched):
    row, run = patched(message="broadcast complete")
    run(status="done")
    assert row.message == "broadcast complete"


def test_the_error_itself_is_still_stored(patched):
    """The message is a summary for the row; `error` stays the full text."""
    row, run = patched()
    run(status="failed", error="YouTube OAuth failed: invalid_grant (token revoked)")
    assert row.error == "YouTube OAuth failed: invalid_grant (token revoked)"


def test_a_missing_row_is_survivable(patched):
    """_update races with deletion; it must not take the worker down."""
    row, run = patched()
    orchestrator.SessionLocal = lambda: _Session(None)
    orchestrator._update(999, status="failed", error="x")   # must not raise
