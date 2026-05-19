"""Preflight check for the V2 Beta launch (Phase 14 / D-13.2).

Run this on the production host BEFORE flipping V2 traffic on. Exits
non-zero if any required dependency is missing or misconfigured.

Checks
------
1. Required env vars present + non-empty.
2. ``INNGEST_DEV`` is unset or ``0`` (must be off in production so
   Inngest Cloud authenticates the webhook signatures).
3. ``KAIZER_V2_ENABLED`` is truthy.
4. Database connection works (lazy import; uses the existing
   ``database.py`` engine).
5. Inngest Cloud is reachable via TCP/TLS on ``api.inngest.com:443``.
   We don't authenticate the call -- just confirm the host is
   resolvable + the operator's egress allows it.

Usage
-----
    python pipeline_v2/scripts/preflight_v2_launch.py

Exit codes
----------
* 0 -- every check passed; safe to launch.
* 1 -- one or more checks failed; do NOT launch.

Test hook
---------
Set ``KAIZER_PREFLIGHT_SKIP_NETWORK=1`` to skip the network probe
(used by the unit test).
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Tuple, List


# ── Required env vars (D-13.2 locked list) ───────────────────────────

REQUIRED_ENV_VARS: List[str] = [
    "KAIZER_V2_ENABLED",
    "INNGEST_EVENT_KEY",
    "INNGEST_SIGNING_KEY",
    "DEEPGRAM_API_KEY",
    "GEMINI_API_KEY",
    "KAIZER_STT_DEFAULT_PROVIDER",
]

# Optional but commonly required for Stage 4 image sourcing. Surfaces
# as a warning instead of a hard failure so a host that doesn't use
# image fallback can still pass.
OPTIONAL_ENV_VARS: List[str] = [
    "OPENAI_API_KEY",
    "CSE_API_KEY",
    "CSE_CX",
]


def check_env_vars() -> Tuple[bool, List[str]]:
    """Verify every REQUIRED_ENV_VARS entry is set + non-empty.

    Returns (ok, lines) where lines is the per-var report.
    """
    lines: List[str] = []
    ok = True
    for name in REQUIRED_ENV_VARS:
        val = os.environ.get(name, "")
        if not val:
            lines.append(f"  FAIL  {name}: missing or empty")
            ok = False
        else:
            lines.append(f"  PASS  {name}: set ({len(val)} chars)")
    for name in OPTIONAL_ENV_VARS:
        val = os.environ.get(name, "")
        if not val:
            lines.append(f"  WARN  {name}: missing (Stage 4 image sourcing may degrade)")
        else:
            lines.append(f"  PASS  {name}: set")
    return ok, lines


def check_inngest_dev_off() -> Tuple[bool, str]:
    """INNGEST_DEV must be unset or 0 in production."""
    raw = (os.environ.get("INNGEST_DEV", "") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return True, f"  PASS  INNGEST_DEV='{raw}' (dev mode off)"
    return False, f"  FAIL  INNGEST_DEV='{raw}' must be 0 or unset in production"


def check_v2_enabled() -> Tuple[bool, str]:
    """KAIZER_V2_ENABLED must be truthy."""
    raw = (os.environ.get("KAIZER_V2_ENABLED", "") or "").strip().lower()
    truthy = raw not in ("0", "false", "no", "off", "")
    if truthy:
        return True, f"  PASS  KAIZER_V2_ENABLED='{raw}' (V2 traffic enabled)"
    return False, f"  FAIL  KAIZER_V2_ENABLED='{raw}' must be truthy to accept V2 jobs"


def check_database() -> Tuple[bool, str]:
    """Open a connection through the existing engine to confirm reachability."""
    try:
        # Lazy import so the script can be unit-tested without a real DB.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"  PASS  Database: connected ({engine.url.drivername})"
    except Exception as exc:
        return False, f"  FAIL  Database: {exc.__class__.__name__}: {exc}"


def check_inngest_cloud_reachable() -> Tuple[bool, str]:
    """TCP/TLS reachability probe to api.inngest.com:443. Skip when
    KAIZER_PREFLIGHT_SKIP_NETWORK=1 (unit-test hook)."""
    if os.environ.get("KAIZER_PREFLIGHT_SKIP_NETWORK", "").strip() in ("1", "true", "yes"):
        return True, "  SKIP  Inngest Cloud reachability (KAIZER_PREFLIGHT_SKIP_NETWORK set)"
    try:
        with socket.create_connection(("api.inngest.com", 443), timeout=5.0):
            pass
        return True, "  PASS  Inngest Cloud: TCP connect to api.inngest.com:443 succeeded"
    except Exception as exc:
        return False, f"  FAIL  Inngest Cloud: {exc.__class__.__name__}: {exc}"


# ── Runner ───────────────────────────────────────────────────────────


def run_all() -> Tuple[bool, str]:
    """Execute every check; aggregate result + a multi-line report."""
    parts: List[str] = []
    overall_ok = True

    parts.append("[1/5] Required env vars")
    env_ok, env_lines = check_env_vars()
    parts.extend(env_lines)
    overall_ok = overall_ok and env_ok

    parts.append("")
    parts.append("[2/5] Inngest dev mode")
    dev_ok, dev_line = check_inngest_dev_off()
    parts.append(dev_line)
    overall_ok = overall_ok and dev_ok

    parts.append("")
    parts.append("[3/5] V2 feature flag")
    flag_ok, flag_line = check_v2_enabled()
    parts.append(flag_line)
    overall_ok = overall_ok and flag_ok

    parts.append("")
    parts.append("[4/5] Database")
    db_ok, db_line = check_database()
    parts.append(db_line)
    overall_ok = overall_ok and db_ok

    parts.append("")
    parts.append("[5/5] Inngest Cloud reachability")
    net_ok, net_line = check_inngest_cloud_reachable()
    parts.append(net_line)
    overall_ok = overall_ok and net_ok

    parts.append("")
    if overall_ok:
        parts.append("RESULT: ALL CHECKS PASSED -- safe to launch V2 Beta")
    else:
        parts.append("RESULT: ONE OR MORE CHECKS FAILED -- do not launch")
    return overall_ok, "\n".join(parts)


def main() -> int:
    ok, report = run_all()
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
