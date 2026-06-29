"""Reusable SMTP sender — same transport/creds the password-reset email
uses (KAIZER_SMTP_* env). Used to email the operator build/publish
updates. Secrets are read from env and NEVER printed.

Usage:
  python scripts/send_update_email.py <to_email> "<subject>" "<body text>"
  # or pipe a longer body on stdin:
  echo "long body" | python scripts/send_update_email.py <to_email> "<subject>" -
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
sys.path.insert(0, _BACKEND)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Load .env the same way the app does (best-effort).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND, ".env"))
except Exception:
    # Manual minimal parse if python-dotenv isn't importable here.
    envp = os.path.join(_BACKEND, ".env")
    if os.path.isfile(envp):
        for line in open(envp, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def smtp_status() -> dict:
    """Presence-only view of the SMTP config (no secret values)."""
    return {
        "host_set":  bool(os.environ.get("KAIZER_SMTP_HOST")),
        "host":      os.environ.get("KAIZER_SMTP_HOST", ""),  # host is not secret
        "port":      os.environ.get("KAIZER_SMTP_PORT", "587"),
        "user_set":  bool(os.environ.get("KAIZER_SMTP_USER")),
        "pass_set":  bool(os.environ.get("KAIZER_SMTP_PASS")),
        "from":      os.environ.get("KAIZER_SMTP_FROM", ""),
        "tls":       os.environ.get("KAIZER_SMTP_TLS", "true"),
    }


def send(to_email: str, subject: str, body_text: str, body_html: str | None = None) -> bool:
    host = os.environ.get("KAIZER_SMTP_HOST")
    if not host:
        print("SMTP_NOT_CONFIGURED: KAIZER_SMTP_HOST is not set in .env")
        return False
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    port = int(os.environ.get("KAIZER_SMTP_PORT", "587"))
    user = os.environ.get("KAIZER_SMTP_USER", "")
    pwd = os.environ.get("KAIZER_SMTP_PASS", "")
    sender = os.environ.get("KAIZER_SMTP_FROM", user or "noreply@kaizer.local")
    use_tls = os.environ.get("KAIZER_SMTP_TLS", "true").lower() in ("1", "true", "yes")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    with smtplib.SMTP(host, port, timeout=20) as s:
        if use_tls:
            s.starttls()
        if user:
            s.login(user, pwd)
        s.sendmail(sender, [to_email], msg.as_string())
    print(f"SENT ok -> {to_email} via {host}:{port} (from {sender})")
    # Durable comms log (admin-visible).
    try:
        from email_comms_log import log_email
        log_email("sent", to_email, subject, body_text, "ok")
    except Exception:
        pass
    return True


if __name__ == "__main__":
    print("SMTP status:", smtp_status())
    if len(sys.argv) >= 4:
        to_addr, subject, body = sys.argv[1], sys.argv[2], sys.argv[3]
        if body == "-":
            body = sys.stdin.read()
        try:
            ok = send(to_addr, subject, body)
            sys.exit(0 if ok else 2)
        except Exception as exc:
            print(f"SMTP send FAILED: {exc}")
            sys.exit(3)
    else:
        print("(no recipient/subject/body args — status check only)")
