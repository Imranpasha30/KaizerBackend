"""desktop_entry — single entry point for the Kaizer X DESKTOP build.

The desktop app is the SAME backend the SaaS runs, booted in a local,
single-user shape. This module is what the frozen executable (PyInstaller)
runs; it is also runnable from a dev checkout for testing:

    python desktop_entry.py serve --port 8765 --data-dir C:/Users/me/KaizerX
    python desktop_entry.py render --job-id 5 --source a.mp4 --output-dir d --language te

Subcommands
-----------
serve   Boot the FastAPI backend on 127.0.0.1 only. Sets the desktop
        environment (KAIZER_DESKTOP=1, SQLite DB + output + .env all inside
        --data-dir) BEFORE importing main, so every import-time consumer
        (database.py engine, config.py ENV_PATH, main.py gates) sees it.
render  Forward the remaining argv verbatim to pipeline_v4.orchestrator's
        CLI. This exists because a frozen build has no venv python for
        runner.py to spawn — runner re-invokes THIS executable with
        ``render`` + the exact same flags it would have passed to
        ``python -m pipeline_v4.orchestrator`` (see
        runner.build_v4_spawn_cmd — the two argv tails are identical and
        tests/test_desktop_mode.py string-compares them).

Frozen-build resource layout (SHIPPED by the Electron builder — the backend
exe is nested TWO levels below resources/):
    <resources>/python/kaizer_backend/kaizer_backend.exe   (sys.executable)
    <resources>/bin/            ffmpeg.exe, ffprobe.exe, yt-dlp.exe ...
    <resources>/ms-playwright/  bundled Playwright browsers

``bin`` is PREPENDED to PATH because many call sites resolve ffmpeg via
``shutil.which("ffmpeg")`` (pipeline_v4/orchestrator.py, pipeline_v4/
encoder.py, services/branding.py, pipeline_core/*, ...) — PATH is the one
fix that covers all of them without touching each site.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# The exact flag vocabulary of pipeline_v4.orchestrator._cli's argparse.
# runner.build_v4_spawn_cmd must only ever emit these after "render";
# tests/test_desktop_mode.py enforces it so the contract can't drift.
ORCHESTRATOR_ARGV_CONTRACT = (
    "--job-id", "--source", "--output-dir", "--language", "--brand-logo",
)


def desktop_mode() -> bool:
    """True when this process runs as the desktop app (KAIZER_DESKTOP=1)."""
    return (os.environ.get("KAIZER_DESKTOP", "") or "").strip() == "1"


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _resources_path() -> Optional[Path]:
    """Frozen build: Electron's resourcesPath. The SHIPPED layout nests the
    backend exe at <resources>/python/kaizer_backend/kaizer_backend.exe, so
    resources/ is the exe's great-grandparent (parents[2]) — bin/ and
    ms-playwright/ live directly under it. Harmless when the Electron shell
    already exported PATH/PLAYWRIGHT_BROWSERS_PATH: _apply_frozen_paths only
    fills in what the environment hasn't set (env wins)."""
    if not _is_frozen():
        return None
    exe = Path(sys.executable).resolve()
    try:
        return exe.parents[2]
    except IndexError:
        # Exe sits too close to the drive root for the shipped layout —
        # fall back to its own folder rather than crash.
        return exe.parent


def _apply_frozen_paths(resources: Optional[Path] = None) -> None:
    """Wire the frozen build's bundled tools into the environment.

    - PREPEND <resources>/bin to PATH so every shutil.which("ffmpeg"/
      "ffprobe"/"yt-dlp") call site in the pipeline finds the bundled
      binaries first.
    - Point PLAYWRIGHT_BROWSERS_PATH at the bundled browsers so template
      rendering works without a per-user browser download.

    No-op in a dev checkout (not frozen). Idempotent — safe to call from
    both serve and render.
    """
    if resources is None:
        resources = _resources_path()
    if resources is None:
        return
    bin_dir = resources / "bin"
    if bin_dir.is_dir():
        entry = str(bin_dir)
        current = os.environ.get("PATH", "")
        if entry not in current.split(os.pathsep):
            os.environ["PATH"] = entry + os.pathsep + current
    browsers = resources / "ms-playwright"
    if browsers.is_dir() and not (os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers)


def _database_url(data_dir: Path) -> str:
    """sqlite URL for the desktop DB — ALWAYS forward slashes (SQLAlchemy
    on Windows chokes on sqlite:///E:\\path style URLs)."""
    return "sqlite:///" + data_dir.resolve().as_posix() + "/kaizer.db"


def _default_data_dir() -> str:
    """Where user data lives if the launcher didn't pass --data-dir.
    The Electron shell always passes app.getPath('userData'); this default
    only serves dev-checkout runs."""
    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    base = Path(local) if local else (Path.home() / "AppData" / "Local")
    return str(base / "KaizerX")


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Connect-probe: True when something already accepts on host:port."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except OSError:
        return False


def _provision_jwt_secret(data_dir: Path) -> None:
    """Keep local sign-ins alive across restarts.

    auth._jwt_secret falls back to a NEW random per-boot secret when
    KAIZER_JWT_SECRET is unset — every restart would invalidate the local
    token and silently sign the user out. Generate the secret ONCE and
    persist it in the userData .env (mirrors config._provision_encryption_key:
    re-read the file first in case an earlier boot already wrote one).
    A secret already present in the process environment always wins."""
    if (os.environ.get("KAIZER_JWT_SECRET", "") or "").strip():
        return
    env_path = data_dir / ".env"
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("KAIZER_JWT_SECRET="):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        os.environ["KAIZER_JWT_SECRET"] = val
                        return
    except OSError:
        pass

    import secrets
    fresh = secrets.token_urlsafe(48)
    os.environ["KAIZER_JWT_SECRET"] = fresh
    try:
        with env_path.open("a", encoding="utf-8") as f:
            if env_path.stat().st_size > 0:
                f.seek(0, 2)
                f.write("\n")
            f.write(f"KAIZER_JWT_SECRET={fresh}\n")
    except OSError as exc:
        print(f"[desktop] WARN: could not save the sign-in secret ({exc}). "
              "You may need to sign in again after restarting the app.",
              flush=True)


def _apply_desktop_env(data_dir: Path) -> None:
    """Set the desktop environment. MUST run before importing main —
    database.py builds its engine and config.py resolves ENV_PATH at
    import time."""
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = data_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAIZER_DESKTOP"] = "1"
    os.environ["DATABASE_URL"] = _database_url(data_dir)
    os.environ["KAIZER_OUTPUT_ROOT"] = str(output_dir)
    # Uploads/media must land in userData too — the frozen install dir may
    # be read-only (main.py's MEDIA_ROOT honors this override).
    os.environ["KAIZER_MEDIA_ROOT"] = str(media_dir)
    # config.py relocates ENV_PATH here so the auto-generated Fernet key
    # (and the API keys routers/desktop_local.py writes) persist in
    # userData — never inside the (possibly read-only) install dir.
    os.environ["KAIZER_ENV_DIR"] = str(data_dir)
    # Stable JWT secret so local sign-ins survive restarts.
    _provision_jwt_secret(data_dir)


def _start_parent_watchdog(parent_pid: int) -> None:
    """Exit when the launching Electron shell dies.

    Without this, closing the app window can leave an orphaned backend
    holding the port and the SQLite DB. When --parent-pid is given, a
    daemon thread watches that process and hard-exits (os._exit(0)) the
    moment it is gone. No-op when the flag is absent (dev runs).

    Windows: OpenProcess+WaitForSingleObject — a real handle wait, immune
    to PID reuse. Fallback (handle unavailable / non-Windows): poll
    psutil.pid_exists every 5s."""
    try:
        parent_pid = int(parent_pid or 0)
    except (TypeError, ValueError):
        return
    if parent_pid <= 0:
        return

    import threading

    def _watch() -> None:
        if os.name == "nt":
            try:
                import ctypes
                SYNCHRONIZE = 0x00100000
                WAIT_TIMEOUT = 0x00000102
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
                if handle:
                    try:
                        while True:
                            rc = kernel32.WaitForSingleObject(handle, 5000)
                            if rc != WAIT_TIMEOUT:
                                break  # parent exited (or wait failed hard)
                    finally:
                        kernel32.CloseHandle(handle)
                    os._exit(0)
            except Exception:
                pass  # fall through to the psutil poll below
        try:
            import psutil
        except Exception:
            return  # no way to watch — behave like the flag was absent
        import time
        while True:
            try:
                if not psutil.pid_exists(parent_pid):
                    os._exit(0)
            except Exception:
                pass
            time.sleep(5)

    threading.Thread(target=_watch, name="kaizer-parent-watchdog",
                     daemon=True).start()


def _cmd_serve(args: argparse.Namespace) -> int:
    port = int(args.port)
    if _port_in_use(port):
        # Guard: never boot a second backend onto the same port — the OS
        # would refuse the bind and a second DB writer is never wanted.
        print(
            f"Kaizer X is already running on port {port}. "
            "Please close the other Kaizer X window first (or pick a "
            "different port with --port).",
            flush=True,
        )
        return 2

    data_dir = Path(args.data_dir).expanduser()
    _apply_desktop_env(data_dir)
    _apply_frozen_paths()
    # Die with the Electron shell (no-op when --parent-pid wasn't given).
    _start_parent_watchdog(int(getattr(args, "parent_pid", 0) or 0))

    # Load the API keys the user saved earlier (userData .env). override
    # stays False so the desktop values set above always win.
    try:
        from dotenv import load_dotenv
        load_dotenv(data_dir / ".env", override=False)
    except Exception:
        pass

    import uvicorn
    # Import the module and pass the APP OBJECT, not the "main:app" string:
    # PyInstaller's analysis only follows real import statements — the
    # string form froze a bundle without main.py in it ("Could not import
    # module 'main'" on the first frozen boot). The import sits HERE, after
    # _apply_desktop_env, so the env-before-import contract holds.
    import main as _main
    uvicorn.run(_main.app, host="127.0.0.1", port=port)
    return 0


def _cmd_render(rest: list) -> int:
    """Forward argv verbatim to pipeline_v4.orchestrator's CLI.

    Spawned by runner.py in the frozen build (see build_v4_spawn_cmd).
    All configuration (DATABASE_URL, KAIZER_* vars, planner choice) is
    inherited from the parent serve process's environment — exactly like
    the venv ``python -m pipeline_v4.orchestrator`` spawn it replaces.
    """
    _apply_frozen_paths()
    sys.argv = ["pipeline_v4.orchestrator"] + list(rest)
    from pipeline_v4.orchestrator import _cli
    return int(_cli())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaizer-desktop",
        description="Kaizer X desktop backend (local, single-user).",
    )
    sub = parser.add_subparsers(dest="command")
    p_serve = sub.add_parser(
        "serve", help="Start the local Kaizer X backend on 127.0.0.1.")
    p_serve.add_argument("--port", type=int, default=8765,
                         help="Local port to listen on (default 8765).")
    p_serve.add_argument("--data-dir", default=_default_data_dir(),
                         help="Folder for the desktop database, rendered "
                              "videos and saved settings.")
    p_serve.add_argument("--parent-pid", type=int, default=0,
                         help="PID of the launching shell (Electron). When "
                              "given, the backend exits automatically if "
                              "that process dies, so closing the app never "
                              "leaves a hidden server running.")
    # NOTE: "render" is listed here ONLY for --help output. Its argv tail is
    # peeled off in split_command() BEFORE argparse ever runs — argparse's
    # REMAINDER cannot reliably capture leading --flags on a subparser
    # (Python 3.10, argparse gh-61252), and the tail must reach
    # pipeline_v4.orchestrator VERBATIM.
    sub.add_parser(
        "render", help="Run one render job (used internally by serve); "
                       "all following arguments are forwarded to the "
                       "render pipeline (--job-id, --source, ...).")
    return parser


def split_command(argv: list) -> tuple:
    """Peel the subcommand. ('render', rest) forwards rest VERBATIM —
    tests string-compare this against runner.build_v4_spawn_cmd's tail."""
    if argv and argv[0] == "render":
        return "render", list(argv[1:])
    return None, list(argv)


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    command, rest = split_command(argv)
    if command == "render":
        return _cmd_render(rest)
    parser = build_parser()
    args = parser.parse_args(rest)
    if args.command == "serve":
        return _cmd_serve(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
