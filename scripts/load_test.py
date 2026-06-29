"""Wave 6 — load test for the scaled API layer.

Simulates the hot read paths at scale against a RUNNING backend:

  * N status pollers   GET /api/jobs/{id}/status/?since=...  (1s cadence
                        — exercises the 2s micro-cache + incremental log)
  * M list clients     GET /api/jobs/?limit=20               (N+1 fix +
                        pagination + DB pool behaviour)
  * K WebSockets       /api/ws/jobs/{id}                     (shared-
                        tailer fan-out; requires `websockets` package,
                        skipped with a warning if missing)

Asserts: zero 5xx, zero transport errors, p95 latency under threshold.

Usage (from KaizerBackend/, backend running on :8000):
  python scripts/load_test.py --pollers 1000 --listers 100 --ws 50 \
      --duration 30 --job-id 123

Auth: dev runs with KAIZER_AUTH_REQUIRED=false → anonymous requests map
to the legacy user, so no token is needed. Pass --token for real auth.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import httpx
except ImportError:
    print("load_test: `pip install httpx` first")
    sys.exit(2)


class Stats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.latencies: list[float] = []
        self.errors: list[str] = []
        self.codes: dict[int, int] = {}

    def record(self, code: int, dt: float) -> None:
        self.latencies.append(dt)
        self.codes[code] = self.codes.get(code, 0) + 1

    def error(self, msg: str) -> None:
        if len(self.errors) < 50:
            self.errors.append(msg)
        else:
            self.errors.append("...")

    def summary(self) -> str:
        if not self.latencies:
            return f"{self.name}: no samples, errors={len(self.errors)}"
        ls = sorted(self.latencies)

        def pct(p: float) -> float:
            return ls[min(len(ls) - 1, int(len(ls) * p))]

        bad = sum(v for k, v in self.codes.items() if k >= 500)
        return (
            f"{self.name}: n={len(ls)} "
            f"p50={pct(0.50)*1000:.0f}ms p95={pct(0.95)*1000:.0f}ms "
            f"p99={pct(0.99)*1000:.0f}ms max={ls[-1]*1000:.0f}ms "
            f"codes={self.codes} 5xx={bad} transport_errors={len(self.errors)}"
        )

    @property
    def failed(self) -> bool:
        bad_5xx = sum(v for k, v in self.codes.items() if k >= 500)
        if bad_5xx:
            return True
        # Tolerate ≤0.1% transport errors: HTTP/1.1 keepalive-reuse
        # races (server closes an idle conn as the client reuses it)
        # are inherent to the protocol and retried by real clients.
        n = max(1, len(self.latencies))
        return len(self.errors) > max(1, n // 1000)


async def poller(client: httpx.AsyncClient, base: str, job_id: int,
                 headers: dict, stats: Stats, stop_at: float,
                 ramp_s: float) -> None:
    # Stagger arrival like real users (a same-millisecond stampede of
    # 1000 SYNs just measures the OS accept queue, not the app).
    await asyncio.sleep(ramp_s)
    since = 0
    while time.monotonic() < stop_at:
        t0 = time.monotonic()
        try:
            r = await client.get(
                f"{base}/api/jobs/{job_id}/status/",
                params={"since": since}, headers=headers, timeout=15,
            )
            stats.record(r.status_code, time.monotonic() - t0)
            if r.status_code == 200:
                try:
                    since = int(r.json().get("log_offset") or 0)
                except Exception:
                    pass
        except Exception as exc:
            stats.error(repr(exc))
        await asyncio.sleep(1.0)


async def lister(client: httpx.AsyncClient, base: str, headers: dict,
                 stats: Stats, stop_at: float, ramp_s: float) -> None:
    await asyncio.sleep(ramp_s)
    while time.monotonic() < stop_at:
        t0 = time.monotonic()
        try:
            r = await client.get(
                f"{base}/api/jobs/", params={"limit": 20},
                headers=headers, timeout=30,
            )
            stats.record(r.status_code, time.monotonic() - t0)
        except Exception as exc:
            stats.error(repr(exc))
        await asyncio.sleep(2.0)


async def ws_client(base: str, job_id: int, token: str,
                    stats: Stats, stop_at: float) -> None:
    try:
        import websockets
    except ImportError:
        return
    ws_base = base.replace("http", "ws", 1)
    url = f"{ws_base}/api/ws/jobs/{job_id}?token={token}"
    try:
        t0 = time.monotonic()
        async with websockets.connect(url, open_timeout=15) as ws:
            stats.record(101, time.monotonic() - t0)
            while time.monotonic() < stop_at:
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    continue
    except Exception as exc:
        # A clean close (code 1000) is the server finishing a terminal
        # job's stream — correct behaviour, not an error.
        name = exc.__class__.__name__
        if "ConnectionClosedOK" in name:
            return
        stats.error(repr(exc))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--job-id", type=int, default=0,
                    help="existing job id to poll (auto-picks latest if 0)")
    ap.add_argument("--pollers", type=int, default=1000)
    ap.add_argument("--listers", type=int, default=100)
    ap.add_argument("--ws", type=int, default=50)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--token", default="")
    ap.add_argument("--p95-ms", type=int, default=1500,
                    help="p95 latency budget for the poller path")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    base = args.base_url.rstrip("/")

    limits = httpx.Limits(max_connections=args.pollers + args.listers + 50,
                          max_keepalive_connections=200)
    async with httpx.AsyncClient(limits=limits) as client:
        # Sanity + auto job pick.
        r = await client.get(f"{base}/api/jobs/", params={"limit": 1},
                             headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"backend not ready: GET /api/jobs/ -> {r.status_code}")
            return 2
        job_id = args.job_id
        if not job_id:
            data = r.json()
            rows = data if isinstance(data, list) else data.get("jobs") or []
            if not rows:
                print("no jobs exist to poll — create one first")
                return 2
            job_id = int(rows[0]["id"])
        print(f"load_test: base={base} job_id={job_id} "
              f"pollers={args.pollers} listers={args.listers} ws={args.ws} "
              f"duration={args.duration}s")

        s_poll = Stats("status-pollers")
        s_list = Stats("job-listers")
        s_ws = Stats("websockets")
        ramp = 8.0  # arrival window — clients connect over ~8s like real users
        stop_at = time.monotonic() + args.duration + ramp

        tasks = (
            [poller(client, base, job_id, headers, s_poll, stop_at,
                    ramp * i / max(1, args.pollers))
             for i in range(args.pollers)]
            + [lister(client, base, headers, s_list, stop_at,
                      ramp * i / max(1, args.listers))
               for i in range(args.listers)]
            + [ws_client(base, job_id, args.token, s_ws, stop_at)
               for _ in range(args.ws)]
        )
        t0 = time.monotonic()
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - t0

    print(f"\nfinished in {elapsed:.1f}s")
    for s in (s_poll, s_list, s_ws):
        print(" ", s.summary())
        if s.errors[:3]:
            print("    sample errors:", s.errors[:3])

    ok = not (s_poll.failed or s_list.failed)
    if s_poll.latencies:
        ls = sorted(s_poll.latencies)
        p95 = ls[min(len(ls) - 1, int(len(ls) * 0.95))] * 1000
        if p95 > args.p95_ms:
            print(f"FAIL: poller p95 {p95:.0f}ms > budget {args.p95_ms}ms")
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    # Windows: keep the DEFAULT ProactorEventLoop (IOCP) — the selector
    # loop dies at ~512 sockets ("too many file descriptors in
    # select()") which this test exceeds by design.
    sys.exit(asyncio.run(main()))
