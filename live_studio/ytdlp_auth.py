"""Authentication options for yt-dlp, for the case production actually hits.

THE PRODUCTION FAILURE (2026-09-26), once yt-dlp was finally installed:

    ERROR: [youtube] V_b7R6a5D_w: Sign in to confirm you're not a bot.
    Use --cookies-from-browser or --cookies for the authentication.

This is not about the video and not about the channel — extraction never got
that far. YouTube refuses anonymous extraction from datacenter IP ranges, and
every cloud host is a datacenter IP range. It therefore cannot be reproduced
on the operator's desktop, where the same URL fetches perfectly, which is
exactly why it needs to be configurable rather than coded around.

THREE KNOBS, all optional, all no-ops when unset — so the behaviour on a
machine that never sets them is byte-identical to before:

  YTDLP_COOKIES_FILE   path to a Netscape cookies.txt on disk.
  YTDLP_COOKIES        the cookie file's CONTENT, for a PaaS where you set a
                       secret but cannot upload a file. Written once to a
                       0600 temp file and reused.
  YTDLP_PLAYER_CLIENT  e.g. "tv" or "web_safari". YouTube's bot check varies
                       by client, so this can sometimes avoid it without
                       credentials at all. Worth trying FIRST.
  YTDLP_PROXY          route extraction through a non-datacenter IP.

Named without the KAIZER_ prefix on purpose: these configure an external tool,
matching the existing FFMPEG_BIN / YTDLP_BIN convention rather than the
product's own KAIZER_* settings registry.

COOKIES ARE CREDENTIALS. Nothing here logs, echoes, or returns their content —
only the path to them — and the temp file is written 0600.
"""
from __future__ import annotations

import os
import tempfile
import threading

_LOCK = threading.Lock()
_CACHED: str | None = None


def _cookie_file() -> str | None:
    """Path to a cookies.txt, or None. Never returns the contents."""
    path = (os.environ.get("YTDLP_COOKIES_FILE") or "").strip()
    if path:
        return path if os.path.isfile(path) else None

    blob = os.environ.get("YTDLP_COOKIES") or ""
    if not blob.strip():
        return None

    global _CACHED
    with _LOCK:
        if _CACHED and os.path.isfile(_CACHED):
            return _CACHED
        fd, tmp = tempfile.mkstemp(prefix="ytdlp_cookies_", suffix=".txt")
        # A secret pasted into a PaaS field usually arrives with its newlines
        # escaped; yt-dlp needs a real Netscape file, one cookie per line.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(blob.replace("\\n", "\n"))
        try:
            os.chmod(tmp, 0o600)          # it is a credential on disk
        except OSError:
            pass                           # best effort; Windows has no mode
        _CACHED = tmp
        return tmp


def ytdlp_auth_args() -> list:
    """Extra yt-dlp flags for this environment. Empty when nothing is set."""
    args: list = []

    cookies = _cookie_file()
    if cookies:
        args += ["--cookies", cookies]

    client = (os.environ.get("YTDLP_PLAYER_CLIENT") or "").strip()
    if client:
        args += ["--extractor-args", f"youtube:player_client={client}"]

    proxy = (os.environ.get("YTDLP_PROXY") or "").strip()
    if proxy:
        args += ["--proxy", proxy]

    return args


def describe() -> str:
    """One line for the log saying WHICH of these are active — never what
    they contain. Without this, 'it still fails' cannot be told apart from
    'the secret never reached the container'."""
    bits = []
    if _cookie_file():
        bits.append("cookies=yes")
    if (os.environ.get("YTDLP_PLAYER_CLIENT") or "").strip():
        bits.append(f"player_client={os.environ['YTDLP_PLAYER_CLIENT'].strip()}")
    if (os.environ.get("YTDLP_PROXY") or "").strip():
        bits.append("proxy=yes")
    return ", ".join(bits) or "no yt-dlp auth configured"
