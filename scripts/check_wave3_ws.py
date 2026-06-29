"""Wave 3 check — app imports with the WS router mounted and both
WebSocket routes exist at the /api-prefixed paths the frontend uses.

Run from KaizerBackend/:  python scripts/check_wave3_ws.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    from main import app

    ws_paths = sorted(
        r.path for r in app.routes
        if r.__class__.__name__ == "APIWebSocketRoute"
    )
    print("WebSocket routes:", ws_paths)
    expected = {"/api/ws/jobs/{job_id}", "/api/ws/uploads"}
    missing = expected - set(ws_paths)
    if missing:
        print("MISSING:", sorted(missing))
        return 1
    print("WS ROUTES OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
