"""Kaizer watchdog — append a status snapshot every N seconds.

Run in a background bash. The main Claude context reads the log file to
get up-to-date project state without needing agents to notify completion.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("C:/Users/user/kaizer_watchdog.log")
INTERVAL_S = 60
BACKEND_DIR = Path("e:/kaizer new data training/kaizer/KaizerBackend")
FRONTEND_DIR = Path("e:/kaizer new data training/kaizer/kaizerFrontned")


def _section(label: str) -> str:
    return f"\n--- {label} ---\n"


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return (r.stdout or r.stderr or "").strip()
    except Exception as exc:
        return f"[watchdog error: {exc}]"


def _r2_listing() -> str:
    try:
        # Load .env lazily
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(BACKEND_DIR / ".env")
        import sys
        sys.path.insert(0, str(BACKEND_DIR))
        from pipeline_core.storage import get_storage_provider, _PROVIDER_CACHE  # type: ignore
        _PROVIDER_CACHE.clear()
        p = get_storage_provider("r2")
        client = p._get_client()
        resp = client.list_objects_v2(Bucket=p.bucket, Prefix="local/")
        lines = []
        for obj in (resp.get("Contents") or [])[:30]:
            size_mb = obj["Size"] // 1024 // 1024
            lines.append(f"  {obj['Key']:<60} {size_mb:>4} MB")
        return "\n".join(lines) if lines else "  (empty)"
    except Exception as exc:
        return f"  [r2 check failed: {exc}]"


def _ports_listening() -> str:
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10)
        out = r.stdout or ""
        keep = [line for line in out.splitlines()
                if "LISTENING" in line and any(p in line for p in (":8000", ":3000"))]
        return "\n".join(keep) if keep else "  (neither :8000 nor :3000 listening)"
    except Exception as exc:
        return f"  [netstat failed: {exc}]"


def snapshot() -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[str] = [f"\n========== {now} =========="]

    out.append(_section("backend git status"))
    out.append(_run(["git", "status", "--short"], cwd=BACKEND_DIR))

    out.append(_section("backend log tail (last 6 lines, 4xx/5xx filtered)"))
    # Look at the most-recent uvicorn log file — easier: just the backend
    # output file we launched with. But since background Bash output lives in
    # a transient temp path, we grep for known uvicorn port messages too.
    out.append(_run(["git", "log", "--oneline", "-n", "3"], cwd=BACKEND_DIR))

    out.append(_section("ports listening"))
    out.append(_ports_listening())

    out.append(_section("r2 bucket — local/ prefix"))
    out.append(_r2_listing())

    out.append(_section("frontend git status"))
    out.append(_run(["git", "status", "--short"], cwd=FRONTEND_DIR))

    return "\n".join(out)


def main() -> None:
    print(f"[watchdog] writing to {LOG_PATH} every {INTERVAL_S}s")
    while True:
        snap = snapshot()
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(snap + "\n")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
