"""Transactional email, through ZeptoMail when it is configured.

WHY THIS EXISTS. ``auth.send_reset_email`` already spoke SMTP, and the app
needs a second sender now that sign-in can happen by emailed code. Two
copies of "compose a message and hope it left" is how one of them quietly
stops working, so both go through here.

ZEPTOMAIL FIRST, SMTP SECOND, LOG ALWAYS.

  * The operator's credential is a ZeptoMail **sendmail token** -- the
    ``Zoho-enczapikey ...`` kind -- which is an HTTP API credential, not an
    OAuth one. It authenticates a POST to the send endpoint and nothing
    else: it cannot sign anybody in, and no "Sign in with Zoho" is possible
    with it. What it buys is a deliverable From: address on his own domain.
  * SMTP stays as the fallback because it already worked and costs nothing
    to keep.
  * THE LINK OR CODE IS ALWAYS PRINTED, whatever happens. Pre-launch, on a
    box with no credential, or when Zoho is down, the operator can still
    read it out of the admin Logs tab and get in. An auth flow whose only
    exit is a third party is an auth flow that locks you out of your own
    product on the day that third party has an outage.

THE CREDENTIAL IS NEVER IN THIS FILE, never logged, and never returned to a
caller. It is read from the environment at CALL time, not at import, because
a key saved into a running process's environment would otherwise never be
seen.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

#: ZeptoMail's Indian data centre, which is where this account lives (the
#: console URL is cpaas.zoho.in). The .com endpoint rejects an .in token.
ZEPTO_URL = "https://api.zeptomail.in/v1.1/email"

#: Environment names. The token is the whole credential; the rest is display.
ENV_TOKEN = "ZEPTOMAIL_TOKEN"
ENV_FROM = "KAIZER_MAIL_FROM"
ENV_FROM_NAME = "KAIZER_MAIL_FROM_NAME"

DEFAULT_FROM_NAME = "Kaizer X"
TIMEOUT_S = 20.0

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _token() -> str:
    """The ZeptoMail send token, read at call time. Never logged."""
    raw = str(os.environ.get(ENV_TOKEN) or "").strip()
    if not raw:
        return ""
    # The console hands it out already prefixed; accept either form so a
    # paste that keeps the prefix and one that strips it both work.
    return raw if raw.lower().startswith("zoho-enczapikey") else f"Zoho-enczapikey {raw}"


def configured() -> bool:
    return bool(_token()) and bool(sender_address())


def sender_address() -> str:
    return str(os.environ.get(ENV_FROM) or "").strip()


def valid_email(addr: str) -> bool:
    return bool(_EMAIL_RE.match(str(addr or "").strip()))


def redact(text: str) -> str:
    """Strip anything that looks like the token out of a string.

    Used on error bodies before they are printed. ZeptoMail echoes request
    details on failure, and this project has already written one live key
    into a job log.
    """
    s = str(text or "")
    s = re.sub(r"Zoho-enczapikey\s+\S+", "Zoho-enczapikey <redacted>", s, flags=re.I)
    tok = _token().split(" ", 1)[-1]
    if tok and len(tok) > 8:
        s = s.replace(tok, "<redacted>")
    return s


def send(*, to_email: str, subject: str, text: str, html: str = "",
         to_name: str = "") -> bool:
    """Send one message. Returns True only if it actually left.

    Never raises: a failed send must not take down a sign-in request, and the
    caller has already printed whatever the recipient needs.
    """
    if not valid_email(to_email):
        print(f"[mail] refusing to send to an invalid address: {to_email!r}")
        return False
    if _send_zepto(to_email=to_email, subject=subject, text=text, html=html,
                   to_name=to_name):
        return True
    return _send_smtp(to_email=to_email, subject=subject, text=text, html=html)


def _send_zepto(*, to_email: str, subject: str, text: str, html: str,
                to_name: str) -> bool:
    tok, frm = _token(), sender_address()
    if not tok or not frm:
        return False
    body = {
        "from": {"address": frm,
                 "name": os.environ.get(ENV_FROM_NAME) or DEFAULT_FROM_NAME},
        "to": [{"email_address": {"address": to_email,
                                  "name": to_name or to_email.split("@")[0]}}],
        "subject": subject,
        "textbody": text,
    }
    if html:
        body["htmlbody"] = html
    req = urllib.request.Request(
        ZEPTO_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json",
                 "Authorization": tok},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            ok = 200 <= r.status < 300
            if ok:
                print(f"[mail] sent via ZeptoMail to {to_email}")
            return ok
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = redact(exc.read().decode("utf-8", "replace"))[:300]
        except Exception:                                      # noqa: BLE001
            pass
        print(f"[mail] ZeptoMail refused ({exc.code}): {detail}")
        return False
    except Exception as exc:                                   # noqa: BLE001
        print(f"[mail] ZeptoMail unreachable: {redact(str(exc))[:200]}")
        return False


def _send_smtp(*, to_email: str, subject: str, text: str, html: str) -> bool:
    """The pre-existing path, kept as the fallback."""
    host = os.environ.get("KAIZER_SMTP_HOST", "")
    if not host:
        return False
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    try:
        port = int(os.environ.get("KAIZER_SMTP_PORT", "587"))
        user = os.environ.get("KAIZER_SMTP_USER", "")
        pwd = os.environ.get("KAIZER_SMTP_PASS", "")
        frm = (os.environ.get("KAIZER_SMTP_FROM") or sender_address()
               or user or "noreply@kaizer.local")
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, frm, to_email
        msg.attach(MIMEText(text, "plain", "utf-8"))
        if html:
            msg.attach(MIMEText(html, "html", "utf-8"))
        with smtplib.SMTP(host, port, timeout=TIMEOUT_S) as srv:
            if os.environ.get("KAIZER_SMTP_TLS", "true").lower() in ("1", "true", "yes"):
                srv.starttls()
            if user:
                srv.login(user, pwd)
            srv.sendmail(frm, [to_email], msg.as_string())
        print(f"[mail] sent via SMTP to {to_email}")
        return True
    except Exception as exc:                                   # noqa: BLE001
        print(f"[mail] SMTP failed: {redact(str(exc))[:200]}")
        return False


def status() -> dict:
    """What the admin screen can show WITHOUT revealing the credential."""
    tok = _token()
    return {
        "zeptomail": bool(tok),
        "token_tail": (tok[-4:] if len(tok) > 12 else ""),
        "from": sender_address(),
        "from_name": os.environ.get(ENV_FROM_NAME) or DEFAULT_FROM_NAME,
        "smtp_fallback": bool(os.environ.get("KAIZER_SMTP_HOST")),
        "endpoint": ZEPTO_URL,
    }
