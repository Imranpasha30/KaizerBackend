"""Autonomous watcher: re-invokes the agent on EITHER event —
  (1) the publish queue drains  -> time to build the pipeline, or
  (2) a NEW reply arrives from the operator -> time to read + act + reply.

Polls every POLL_S seconds (IMAP for new mail FROM the operator, DB for
active jobs). Exits (which re-wakes the agent) on the first trigger, or
on a long timeout. The agent handles the trigger, then RE-LAUNCHES this
to keep the watch alive — that's how we "don't sleep" between events.

Secrets (SMTP/IMAP password) are read from env and never printed.

Args:
  --drain 1|0   watch the publish drain too (default 1; set 0 once the
                pipeline is built and we only need the email channel).
"""
import os
import sys
import time
import email
from email.header import decode_header

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
sys.path.insert(0, _BACKEND)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# load .env
_envp = os.path.join(_BACKEND, ".env")
if os.path.isfile(_envp):
    for _line in open(_envp, encoding="utf-8", errors="replace"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import imaplib
from sqlalchemy import text
from database import SessionLocal

OPERATOR = "imranpasha.ahmed@gmail.com"
IMAP_USER = os.environ.get("KAIZER_SMTP_USER", "")
IMAP_PASS = os.environ.get("KAIZER_SMTP_PASS", "")
ACTIVE = ("queued", "claimed", "branding", "ready_to_upload", "uploading", "running")
POLL_S = 60
MAX_POLLS = 240  # ~4h ceiling, then re-wake to re-arm

WATCH_DRAIN = "--drain" not in sys.argv or (
    sys.argv[sys.argv.index("--drain") + 1] != "0"
    if "--drain" in sys.argv and len(sys.argv) > sys.argv.index("--drain") + 1
    else True
)


def _active_count() -> int:
    db = SessionLocal()
    try:
        return int(db.execute(text(
            "SELECT count(*) FROM upload_jobs_v2 WHERE status = ANY(:s)"),
            {"s": list(ACTIVE)}).scalar() or 0)
    finally:
        db.close()


def _operator_uids():
    """Return (imap_conn, set_of_uids) for mail FROM the operator. Caller
    must logout. Returns (None, set()) on failure."""
    try:
        M = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        M.login(IMAP_USER, IMAP_PASS)
        M.select("INBOX")
        typ, data = M.uid("search", None, f'(FROM "{OPERATOR}")')
        uids = set(data[0].split()) if data and data[0] else set()
        return M, uids
    except Exception as exc:
        print(f"[watch] IMAP poll error: {repr(exc)[:160]}", flush=True)
        return None, set()


def _decode(s) -> str:
    if not s:
        return ""
    out = []
    for part, enc in decode_header(s):
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(part)
    return "".join(out)


def _body_text(msg) -> str:
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() == "text/plain":
                try:
                    return p.get_payload(decode=True).decode(
                        p.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    pass
        return ""
    try:
        return msg.get_payload(decode=True).decode(
            msg.get_content_charset() or "utf-8", errors="replace")
    except Exception:
        return str(msg.get_payload())


# Baseline: existing mail from operator (only NEWER ones are "replies").
M, baseline = _operator_uids()
if M is not None:
    try:
        M.logout()
    except Exception:
        pass
print(f"[watch] armed. drain_watch={WATCH_DRAIN} baseline_operator_mails={len(baseline)}", flush=True)

for i in range(MAX_POLLS):
    # 1) New mail from operator?
    M, cur = _operator_uids()
    if M is not None:
        new = cur - baseline
        if new:
            uid = sorted(new, key=lambda x: int(x))[-1]
            try:
                typ, md = M.uid("fetch", uid, "(RFC822)")
                msg = email.message_from_bytes(md[0][1])
                subj = _decode(msg.get("Subject"))
                body = _body_text(msg).strip()
            except Exception as exc:
                subj, body = "(parse error)", repr(exc)[:200]
            try:
                M.logout()
            except Exception:
                pass
            try:
                from email_comms_log import log_email
                log_email("received", OPERATOR, subj, body, "ok")
            except Exception:
                pass
            print("TRIGGER=EMAIL", flush=True)
            print(f"SUBJECT: {subj}", flush=True)
            print("BODY_START", flush=True)
            print(body[:4000], flush=True)
            print("BODY_END", flush=True)
            sys.exit(0)
        try:
            M.logout()
        except Exception:
            pass
    # 2) Publish drained?
    if WATCH_DRAIN:
        try:
            if _active_count() == 0:
                print("TRIGGER=DRAIN", flush=True)
                sys.exit(0)
        except Exception as exc:
            print(f"[watch] DB poll error: {repr(exc)[:160]}", flush=True)
    time.sleep(POLL_S)

print("TRIGGER=TIMEOUT", flush=True)
sys.exit(0)
