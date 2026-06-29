"""Durable log of operator email communication (sent updates + received
commands). Append-only JSONL at logs/email_comms.jsonl. Surfaced in the
admin panel via GET /api/admin/email-comms. Both the standalone sender
(scripts/send_update_email.py) and the inbox watcher
(scripts/_watch_and_mail.py) append here, so there's one durable record
of every message in/out regardless of which process produced it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "email_comms.jsonl")


def log_email(direction: str, peer: str, subject: str, body: str,
              status: str = "ok") -> dict:
    """Append one record. direction = 'sent' | 'received'. Best-effort —
    never raises into the caller (logging must not break mail flow)."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "peer": peer or "",
        "subject": (subject or "")[:300],
        "body_preview": (body or "")[:2000],
        "status": status,
    }
    try:
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec


def read_email_log(limit: int = 200) -> list[dict]:
    """Most-recent-last list of records (capped). Empty if no log yet."""
    if not os.path.isfile(_LOG):
        return []
    try:
        lines = open(_LOG, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return []
    out: list[dict] = []
    for ln in lines[-int(limit):]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out
