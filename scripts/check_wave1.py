"""Wave 1 sanity check — syntax + real imports of every touched module.

Run from KaizerBackend/:  python scripts/check_wave1.py
"""
import ast
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILES = [
    "services/job_queue.py",
    "services/publish_worker.py",
    "services/cron_runner.py",
    "services/upload_dispatch.py",
    "services/fanout.py",
    "youtube/uploader_v2.py",
    "youtube/rtmp_provider.py",
    "database.py",
    "models.py",
    "main.py",
]

MODULES = [
    "database",
    "models",
    "services.job_queue",
    "services.upload_dispatch",
    "services.publish_worker",
    "services.cron_runner",
    "services.fanout",
    "youtube.uploader_v2",
    "youtube.rtmp_provider",
]


def main() -> int:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in FILES:
        with open(os.path.join(base, f), encoding="utf-8") as fh:
            ast.parse(fh.read())
    print(f"AST OK: {len(FILES)} files")

    for mod in MODULES:
        importlib.import_module(mod)
        print(f"import OK: {mod}")

    # Contract checks: the new public surfaces exist with the expected
    # signatures.
    from services import job_queue, upload_dispatch, publish_worker, cron_runner
    assert callable(job_queue.claim_jobs)
    assert callable(job_queue.requeue_for_retry)
    assert callable(job_queue.reap_expired)
    assert callable(job_queue.unpark_batch)
    assert callable(upload_dispatch.process)
    assert callable(upload_dispatch.fail_terminal)
    assert upload_dispatch.Outcome.RETRY.value == "retry"
    assert callable(publish_worker.start)
    assert callable(cron_runner.cron_leader_loop)
    assert len(cron_runner.CRONS) == 7

    import inspect
    sig = inspect.signature(upload_dispatch.process)
    assert list(sig.parameters) == ["upload_job_id", "worker_id", "deadline"], (
        f"process() signature drifted: {list(sig.parameters)}"
    )
    print("contract OK: queue/dispatch/worker/cron surfaces verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
